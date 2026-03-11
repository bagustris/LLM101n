import torch, torch.nn as nn, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch07_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch07_main"] = _mod
_spec.loader.exec_module(_mod)

xavier_uniform_ = _mod.xavier_uniform_
kaiming_normal_ = _mod.kaiming_normal_
SGD = _mod.SGD
SGDMomentum = _mod.SGDMomentum
Adam = _mod.Adam
AdamW = _mod.AdamW
cosine_lr_schedule = _mod.cosine_lr_schedule
rosenbrock = _mod.rosenbrock

def test_xavier_uniform_range():
    W = torch.empty(64, 64)
    xavier_uniform_(W)
    a = (6 / (64 + 64)) ** 0.5
    assert W.min().item() >= -a - 1e-6
    assert W.max().item() <= a + 1e-6

def test_kaiming_normal_std():
    W = torch.empty(1000, 512)
    kaiming_normal_(W)
    expected_std = (2.0 / 512) ** 0.5
    assert abs(W.std().item() - expected_std) < 0.05

def test_sgd_step():
    p = nn.Parameter(torch.tensor([2.0]))
    p.grad = torch.tensor([1.0])
    opt = SGD([p], lr=0.1)
    opt.step()
    assert abs(p.data.item() - 1.9) < 1e-6

def test_sgd_zero_grad():
    p = nn.Parameter(torch.tensor([1.0]))
    p.grad = torch.tensor([5.0])
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    assert p.grad.item() == 0.0

def test_adam_step_decreases_loss():
    x = nn.Parameter(torch.tensor([-2.0]))
    opt = Adam([x], lr=0.1)
    losses = []
    for _ in range(50):
        loss = x ** 2
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    assert losses[-1] < losses[0]

def test_adamw_weight_decay():
    x = nn.Parameter(torch.tensor([1.0]))
    x.grad = torch.zeros(1)
    opt = AdamW([x], lr=0.1, weight_decay=0.1)
    opt.step()
    assert abs(x.data.item()) < 1.0

def test_cosine_lr_warmup():
    lr = cosine_lr_schedule(step=50, warmup_steps=100, max_steps=1000, max_lr=1e-3)
    assert abs(lr - 0.5e-3) < 1e-5

def test_cosine_lr_peak():
    lr = cosine_lr_schedule(step=100, warmup_steps=100, max_steps=1000, max_lr=1e-3)
    assert abs(lr - 1e-3) < 1e-5

def test_cosine_lr_min():
    lr = cosine_lr_schedule(step=1000, warmup_steps=100, max_steps=1000, max_lr=1e-3)
    assert abs(lr - 1e-4) < 1e-5

def test_rosenbrock_minimum():
    assert abs(rosenbrock(torch.tensor(1.0), torch.tensor(1.0)).item()) < 1e-9
