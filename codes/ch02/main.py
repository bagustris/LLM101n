import math

class Value:
    """A scalar value that tracks its computation graph for backprop."""

    def __init__(self, data: float, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0           # ∂loss/∂self, initialised to 0
        self._backward = lambda: None   # filled in by each operation
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------ #
    # Forward operations                                                   #
    # ------------------------------------------------------------------ #

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad  += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, exponent: float):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad   # d(e^x)/dx = e^x
        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, "log of non-positive number"
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Convenience wrappers so Python operators work symmetrically          #
    # ------------------------------------------------------------------ #

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __sub__(self, other):  return self + (-1 * other)
    def __rsub__(self, other): return other + (-1 * self)
    def __truediv__(self, other): return self * other**-1
    def __neg__(self): return self * -1

    # ------------------------------------------------------------------ #
    # Backpropagation                                                      #
    # ------------------------------------------------------------------ #

    def backward(self):
        """Compute gradients for all ancestors via reverse-mode autodiff."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0          # d(loss)/d(loss) = 1
        for node in reversed(topo):
            node._backward()


# Manual check: f(x, y) = (x + y) * x
x = Value(2.0, label="x")
y = Value(3.0, label="y")
z = (x + y) * x      # z = x² + xy = 4 + 6 = 10
z.backward()

# Analytically: dz/dx = 2x + y = 7, dz/dy = x = 2
print(f"z  = {z.data}")          # 10.0
print(f"dz/dx = {x.grad}")       # 7.0 ✓
print(f"dz/dy = {y.grad}")       # 2.0 ✓
assert abs(x.grad - 7.0) < 1e-9
assert abs(y.grad - 2.0) < 1e-9
print("Gradient check: PASSED")


import random

class Neuron:
    def __init__(self, n_in: int):
        # Weights and bias initialised from U(-1, 1)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        # Weighted sum + bias, then ReLU activation
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in: int, layer_sizes: list[int]):
        sizes = [n_in] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        # Return scalar if last layer has one neuron
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# XOR dataset: inputs → target
XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [0.0,    1.0,    1.0,    0.0]   # XOR labels

random.seed(42)
model = MLP(n_in=2, layer_sizes=[4, 4, 1])  # 2 hidden layers
print(f"Parameter count: {len(model.parameters())}")

learning_rate = 0.1
losses = []

for step in range(200):
    # ----- Forward pass -----
    preds = [model([Value(xi) for xi in x]) for x in XOR_X]

    # Mean squared error loss
    loss = sum((pred - y) ** 2 for pred, y in zip(preds, XOR_Y))
    loss = loss * (1.0 / len(XOR_Y))   # average

    # ----- Backward pass -----
    model.zero_grad()
    loss.backward()

    # ----- Gradient descent update -----
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    losses.append(loss.data)
    if step % 40 == 0:
        print(f"Step {step:3d}  Loss: {loss.data:.6f}")

print("\nFinal predictions:")
for x, y in zip(XOR_X, XOR_Y):
    pred = model([Value(xi) for xi in x])
    print(f"  XOR{x} = {y:.0f}  →  model: {pred.data:.4f}")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Training step")
plt.ylabel("MSE Loss")
plt.title("Micrograd MLP training on XOR")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../data/micrograd_loss.png", dpi=100)
print("Saved → data/micrograd_loss.png")


def print_graph(v, indent=0, visited=None):
    """Print the computation graph recursively."""
    if visited is None:
        visited = set()
    if id(v) in visited:
        return
    visited.add(id(v))
    label = f"[{v._op}] " if v._op else ""
    name  = v.label if v.label else ""
    print(" " * indent + f"{label}{name} data={v.data:.3f} grad={v.grad:.3f}")
    for child in v._prev:
        print_graph(child, indent + 2, visited)

# Re-run forward/backward on a tiny example for visualisation
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(5.0, label="c")
e = a * b; e.label = "e"
d = e + c; d.label = "d"
f = d.relu(); f.label = "f"
f.backward()
print("Computation graph (with gradients):")
print_graph(f)
