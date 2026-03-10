import importlib.util, os, sys, torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch15_main", os.path.join(os.path.dirname(__file__), "main.py"))
RewardModel = _mod.RewardModel
reward_model_loss = _mod.reward_model_loss
dpo_loss = _mod.dpo_loss


def test_reward_model_output_shape():
    model = RewardModel(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    ids = torch.randint(0, 50, (2, 8))
    rewards = model(ids)
    assert rewards.shape == (2,)

def test_reward_model_loss_positive():
    model = RewardModel(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    chosen = torch.randint(0, 50, (2, 8))
    rejected = torch.randint(0, 50, (2, 8))
    loss = reward_model_loss(model, chosen, rejected)
    assert loss.item() > 0

def test_dpo_loss_positive():
    B = 2
    policy_chosen_logps   = torch.randn(B)
    policy_rejected_logps = torch.randn(B)
    ref_chosen_logps      = torch.randn(B)
    ref_rejected_logps    = torch.randn(B)
    loss = dpo_loss(policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps, beta=0.1)
    assert loss.item() > 0

def test_dpo_loss_better_policy():
    B = 4
    high = torch.ones(B) * 5.0
    low  = torch.ones(B) * -5.0
    loss_good = dpo_loss(high, low, torch.zeros(B), torch.zeros(B))
    loss_bad  = dpo_loss(low, high, torch.zeros(B), torch.zeros(B))
    assert loss_good < loss_bad
