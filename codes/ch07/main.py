import torch
import torch.nn as nn
import math


def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return tensor.uniform_(-a, a)


def kaiming_normal_(tensor: torch.Tensor, mode: str = "fan_in",
                    nonlinearity: str = "relu"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std  = gain / math.sqrt(fan)
    return tensor.normal_(0, std)


def trace_activations(init_fn, activation, n_layers=10, n_units=512):
    x = torch.randn(256, n_units)
    stds = [x.std().item()]
    for _ in range(n_layers):
        W = torch.empty(n_units, n_units)
        init_fn(W)
        x = activation(x @ W)
        stds.append(x.std().item())
    return stds


class SGD:
    def __init__(self, params, lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class SGDMomentum:
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9):
        self.params   = list(params)
        self.lr       = lr
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            v.mul_(self.momentum).add_(p.grad)
            p.data.sub_(self.lr * v)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr     = lr
        self.b1, self.b2 = betas
        self.eps    = eps
        self.t      = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(self.b1).add_((1 - self.b1) * g)
            v.mul_(self.b2).add_((1 - self.b2) * g * g)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params       = list(params)
        self.lr           = lr
        self.b1, self.b2  = betas
        self.eps          = eps
        self.wd           = weight_decay
        self.t            = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            p.data.mul_(1 - self.lr * self.wd)
            m.mul_(self.b1).add_((1 - self.b1) * g)
            v.mul_(self.b2).add_((1 - self.b2) * g * g)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def cosine_lr_schedule(step: int, warmup_steps: int,
                        max_steps: int, max_lr: float,
                        min_lr: float = None) -> float:
    if min_lr is None:
        min_lr = max_lr / 10
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    relu = torch.relu
    tanh = torch.tanh

    results = {
        "Random N(0,1) + ReLU":  trace_activations(lambda W: W.normal_(0, 1), relu),
        "Xavier Uniform + Tanh": trace_activations(xavier_uniform_, tanh),
        "Kaiming Normal + ReLU": trace_activations(kaiming_normal_, relu),
        "Std=0.01 + ReLU":       trace_activations(lambda W: W.normal_(0, 0.01), relu),
    }

    plt.figure(figsize=(10, 5))
    for label, stds in results.items():
        plt.plot(stds, marker="o", label=label)
    plt.xlabel("Layer depth")
    plt.ylabel("Activation std")
    plt.title("Activation variance under different initialisations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../data/ch07_init.png", dpi=100)
    print("Saved -> data/ch07_init.png")

    steps = list(range(3000))
    lrs = [cosine_lr_schedule(s, warmup_steps=200, max_steps=3000, max_lr=3e-4) for s in steps]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, lrs, color="coral")
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("Cosine LR schedule with linear warmup")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../data/ch07_lr_schedule.png", dpi=100)
    print("Saved -> data/ch07_lr_schedule.png")

    optimiser_classes = {
        "SGD (lr=0.001)":         lambda p: SGD(p, lr=0.001),
        "SGD+Momentum (lr=0.01)": lambda p: SGDMomentum(p, lr=0.01),
        "Adam (lr=0.01)":         lambda p: Adam(p, lr=0.01),
        "AdamW (lr=0.01)":        lambda p: AdamW(p, lr=0.01, weight_decay=0.0),
    }

    STEPS_OPT = 500
    for name, make_opt in optimiser_classes.items():
        x = nn.Parameter(torch.tensor([-1.0]))
        y = nn.Parameter(torch.tensor([1.0]))
        opt = make_opt([x, y])
        losses_opt = []
        for _ in range(STEPS_OPT):
            loss = rosenbrock(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses_opt.append(loss.item())
        print(f"{name}: final loss = {losses_opt[-1]:.6f}")
