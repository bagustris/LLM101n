import importlib.util, os, sys, torch
import torch.nn as nn

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch13_main", os.path.join(os.path.dirname(__file__), "main.py"))
quantize_absmax_int8 = _mod.quantize_absmax_int8
dequantize_absmax_int8 = _mod.dequantize_absmax_int8
quantize_per_channel = _mod.quantize_per_channel
quantize_groupwise = _mod.quantize_groupwise
quantize_int4_absmax = _mod.quantize_int4_absmax
model_size_mb = _mod.model_size_mb


def test_quantize_int8_dtype():
    W = torch.randn(32, 32)
    w_int8, scale = quantize_absmax_int8(W)
    assert w_int8.dtype == torch.int8

def test_quantize_int8_range():
    W = torch.randn(64, 64)
    w_int8, _ = quantize_absmax_int8(W)
    assert w_int8.min().item() >= -128
    assert w_int8.max().item() <= 127

def test_dequantize_roundtrip_mse():
    torch.manual_seed(0)
    W = torch.randn(128, 128)
    w_int8, scale = quantize_absmax_int8(W)
    W_rec = dequantize_absmax_int8(w_int8, scale)
    mse = ((W - W_rec) ** 2).mean().item()
    assert mse < 0.01

def test_per_channel_better_than_per_tensor():
    torch.manual_seed(1)
    W = torch.randn(64, 256) * 5.0
    w_pt, s_pt = quantize_absmax_int8(W)
    err_pt = ((W - dequantize_absmax_int8(w_pt, s_pt))**2).mean().item()
    w_pc, s_pc = quantize_per_channel(W)
    err_pc = ((W - w_pc.float() * s_pc)**2).mean().item()
    assert err_pc <= err_pt

def test_quantize_groupwise_shape():
    W = torch.randn(32, 128)
    w, s = quantize_groupwise(W, group_size=64)
    assert w.shape == (32, 2, 64)

def test_quantize_int4_range():
    W = torch.randn(64, 64)
    w_i4, _ = quantize_int4_absmax(W)
    assert w_i4.min().item() >= -8
    assert w_i4.max().item() <= 7

def test_model_size_mb():
    model = nn.Linear(100, 100, bias=False)
    mb = model_size_mb(model)
    assert abs(mb - 0.04) < 1e-3
