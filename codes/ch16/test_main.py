import importlib.util, os, sys, torch
import torch.nn as nn

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch16_main", os.path.join(os.path.dirname(__file__), "main.py"))
load_custom_checkpoint = _mod.load_custom_checkpoint
GenerateRequest = _mod.GenerateRequest


def test_generate_request_model():
    req = GenerateRequest(prompt="Hello", max_new_tokens=10)
    assert req.prompt == "Hello"
    assert req.max_new_tokens == 10

def test_generate_request_defaults():
    req = GenerateRequest(prompt="Test")
    assert req.temperature == 0.8
    assert req.top_k == 40

def test_load_custom_checkpoint(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4))
    ckpt_path = tmp_path / "test.pt"
    torch.save(model.state_dict(), ckpt_path)
    model2 = nn.Sequential(nn.Linear(4, 4))
    load_custom_checkpoint(str(ckpt_path), model2)
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
