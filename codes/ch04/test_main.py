import torch, math, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch04_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch04_main"] = _mod
_spec.loader.exec_module(_mod)

scaled_dot_product_attention = _mod.scaled_dot_product_attention
causal_mask = _mod.causal_mask
SinusoidalPositionalEncoding = _mod.SinusoidalPositionalEncoding
MultiHeadAttention = _mod.MultiHeadAttention

def test_causal_mask_shape():
    mask = causal_mask(5)
    assert mask.shape == (5, 5)

def test_causal_mask_upper_triangle():
    mask = causal_mask(4)
    assert mask[0, 1] == True
    assert mask[0, 0] == False
    assert mask[2, 1] == False

def test_scaled_dot_product_attention_shape():
    B, H, T, hd = 2, 4, 6, 16
    Q = torch.randn(B, H, T, hd)
    K = torch.randn(B, H, T, hd)
    V = torch.randn(B, H, T, hd)
    out, weights = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (B, H, T, hd)
    assert weights.shape == (B, H, T, T)

def test_attention_weights_sum_to_one():
    B, H, T, hd = 1, 1, 4, 8
    Q = torch.randn(B, H, T, hd)
    K = torch.randn(B, H, T, hd)
    V = torch.randn(B, H, T, hd)
    _, weights = scaled_dot_product_attention(Q, K, V)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

def test_causal_attention_future_masked():
    B, H, T, hd = 1, 1, 4, 8
    Q = torch.randn(B, H, T, hd)
    K = torch.randn(B, H, T, hd)
    V = torch.randn(B, H, T, hd)
    mask = causal_mask(T)
    _, weights = scaled_dot_product_attention(Q, K, V, mask)
    for i in range(T):
        for j in range(i+1, T):
            assert weights[0, 0, i, j].item() < 1e-6

def test_sinusoidal_pe_shape():
    pe = SinusoidalPositionalEncoding(d_model=64, max_len=100)
    x = torch.randn(2, 10, 64)
    out = pe(x)
    assert out.shape == (2, 10, 64)

def test_sinusoidal_pe_adds_position():
    pe = SinusoidalPositionalEncoding(d_model=16, max_len=20)
    x = torch.zeros(1, 5, 16)
    out = pe(x)
    assert not torch.allclose(out, x)

def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    mask = causal_mask(10)
    out, weights = mha(x, mask)
    assert out.shape == (2, 10, 64)
    assert weights.shape == (2, 4, 10, 10)
