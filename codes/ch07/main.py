import torch
import torch.nn as nn
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def xavier_uniform_(tensor: torch.Tensor, gain: float = 1.0):
    """
    Xavier uniform: U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
    Keeps activation variance constant for symmetric activations (tanh, sigmoid).
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return tensor.uniform_(-a, a)

def kaiming_normal_(tensor: torch.Tensor, mode: str = "fan_in",
                    nonlinearity: str = "relu"):
    """
    Kaiming (He) normal: N(0, std²) where std = sqrt(2 / fan)
    Accounts for the variance-halving effect of ReLU.
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    # Gain for relu = sqrt(2), for linear = 1
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std  = gain / math.sqrt(fan)
    return tensor.normal_(0, std)

# Compare activations at initialisation with different schemes
torch.manual_seed(42)
n_layers = 10
n_units  = 512

def trace_activations(init_fn, activation):
    x = torch.randn(256, n_units)
    stds = [x.std().item()]
    for _ in range(n_layers):
        W = torch.empty(n_units, n_units)
        init_fn(W)
        x = activation(x @ W)
        stds.append(x.std().item())
    return stds

relu = torch.relu
tanh = torch.tanh

results = {
    "Random N(0,1) + ReLU":     trace_activations(lambda W: W.normal_(0, 1), relu),
    "Xavier Uniform + Tanh":    trace_activations(xavier_uniform_,            tanh),
    "Kaiming Normal + ReLU":    trace_activations(kaiming_normal_,            relu),
    "Std=0.01 + ReLU":          trace_activations(lambda W: W.normal_(0, 0.01), relu),
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
print("Saved → data/ch07_init.png")


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
        # Velocity buffers initialised to zero
        self.velocity = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            v.mul_(self.momentum).add_(p.grad)      # v = β*v + g
            p.data.sub_(self.lr * v)                # θ -= lr * v

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class Adam:
    """Adam: Adaptive Moment Estimation (Kingma & Ba, 2015)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr     = lr
        self.b1, self.b2 = betas
        self.eps    = eps
        self.t      = 0
        self.m = [torch.zeros_like(p) for p in self.params]  # 1st moment
        self.v = [torch.zeros_like(p) for p in self.params]  # 2nd moment

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(self.b1).add_((1 - self.b1) * g)            # m = β1*m + (1-β1)*g
            v.mul_(self.b2).add_((1 - self.b2) * g * g)        # v = β2*v + (1-β2)*g²
            # Bias correction
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


class AdamW:
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""
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
            # Weight decay applied directly (decoupled from adaptive step)
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


import numpy as np

def cosine_lr_schedule(step: int, warmup_steps: int,
                        max_steps: int, max_lr: float,
                        min_lr: float = None) -> float:
    """
    Linear warmup followed by cosine decay.
    Used by GPT-2, GPT-3, and most modern LLMs.
    """
    if min_lr is None:
        min_lr = max_lr / 10

    if step < warmup_steps:
        # Linear ramp-up
        return max_lr * step / warmup_steps

    if step >= max_steps:
        return min_lr

    # Cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

# Visualise the schedule
steps   = list(range(3000))
lrs     = [cosine_lr_schedule(s, warmup_steps=200, max_steps=3000,
                               max_lr=3e-4) for s in steps]

plt.figure(figsize=(8, 4))
plt.plot(steps, lrs, color="coral")
plt.axvline(200, linestyle="--", color="gray", label="End of warmup")
plt.xlabel("Step")
plt.ylabel("Learning rate")
plt.title("Cosine LR schedule with linear warmup")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../data/ch07_lr_schedule.png", dpi=100)
print("Saved → data/ch07_lr_schedule.png")


def rosenbrock(x, y):
    """Classic non-convex optimisation benchmark."""
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

optimiser_classes = {
    "SGD (lr=0.001)":         lambda p: SGD(p, lr=0.001),
    "SGD+Momentum (lr=0.01)": lambda p: SGDMomentum(p, lr=0.01),
    "Adam (lr=0.01)":         lambda p: Adam(p, lr=0.01),
    "AdamW (lr=0.01)":        lambda p: AdamW(p, lr=0.01, weight_decay=0.0),
}

results_opt = {}
STEPS_OPT   = 500

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
    results_opt[name] = losses_opt
    print(f"{name}: final loss = {losses_opt[-1]:.6f}  x={x.item():.4f} y={y.item():.4f}")

plt.figure(figsize=(10, 5))
for name, losses_opt in results_opt.items():
    plt.semilogy(losses_opt, label=name)
plt.xlabel("Step")
plt.ylabel("Rosenbrock loss (log scale)")
plt.title("Optimizer comparison on Rosenbrock function")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../data/ch07_optimisers.png", dpi=100)
print("Saved → data/ch07_optimisers.png")


# Gradient clipping prevents exploding gradients in early training.
# PyTorch's built-in implementation:

model = nn.Linear(512, 512)
dummy_loss = model(torch.randn(32, 512)).sum()
dummy_loss.backward()

total_norm_before = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm before clip: {total_norm_before:.4f}")
total_norm_after = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
print(f"Gradient norm after  clip: {total_norm_after:.4f}")
