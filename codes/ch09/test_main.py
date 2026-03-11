import torch, torch.nn as nn, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch09_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch09_main"] = _mod
_spec.loader.exec_module(_mod)

float_bits = _mod.float_bits
model_memory_mb = _mod.model_memory_mb

def test_float_bits_fp32_length():
    bits = float_bits(1.0, torch.float32)
    assert len(bits) == 32

def test_float_bits_fp16_length():
    bits = float_bits(1.0, torch.float16)
    assert len(bits) == 16

def test_float_bits_bf16_length():
    bits = float_bits(1.0, torch.bfloat16)
    assert len(bits) == 16

def test_float_bits_zero():
    bits = float_bits(0.0, torch.float32)
    assert bits == "0" * 32

def test_model_memory_mb_fp32():
    model = nn.Linear(100, 100, bias=False)
    mb = model_memory_mb(model, torch.float32)
    assert abs(mb - 0.04) < 1e-3

def test_model_memory_mb_fp16_half():
    model = nn.Linear(100, 100, bias=False)
    mb32 = model_memory_mb(model, torch.float32)
    mb16 = model_memory_mb(model, torch.float16)
    assert abs(mb16 - mb32 / 2) < 1e-6
