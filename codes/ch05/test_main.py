import torch, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch05_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch05_main"] = _mod
_spec.loader.exec_module(_mod)

LayerNorm = _mod.LayerNorm
FeedForward = _mod.FeedForward
TransformerBlock = _mod.TransformerBlock
GPT = _mod.GPT

def test_layernorm_output_shape():
    ln = LayerNorm(64)
    x = torch.randn(2, 10, 64)
    out = ln(x)
    assert out.shape == (2, 10, 64)

def test_layernorm_normalizes():
    ln = LayerNorm(64)
    x = torch.randn(2, 10, 64) * 100
    out = ln(x)
    assert abs(out.mean().item()) < 0.1
    assert abs(out.std().item() - 1.0) < 0.2

def test_feedforward_output_shape():
    ff = FeedForward(d_model=64)
    x = torch.randn(2, 10, 64)
    out = ff(x)
    assert out.shape == (2, 10, 64)

def test_transformer_block_output_shape():
    block = TransformerBlock(d_model=64, n_heads=4)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)

def test_gpt_forward_no_targets():
    model = GPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    idx = torch.randint(0, 50, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, 50)
    assert loss is None

def test_gpt_forward_with_targets():
    model = GPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    idx = torch.randint(0, 50, (2, 8))
    targets = torch.randint(0, 50, (2, 8))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0

def test_gpt_generate():
    model = GPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    prompt = torch.randint(0, 50, (1, 4))
    out = model.generate(prompt, max_new_tokens=5)
    assert out.shape[1] == 4 + 5

def test_gpt_weight_tying():
    model = GPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    assert model.lm_head.weight is model.token_emb.weight
