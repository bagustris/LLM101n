import importlib.util, os, sys, torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch12_main", os.path.join(os.path.dirname(__file__), "main.py"))
AttentionNoCache = _mod.AttentionNoCache
AttentionWithCache = _mod.AttentionWithCache
GPTWithCache = _mod.GPTWithCache
generate_no_cache = _mod.generate_no_cache
kv_cache_memory_mb = _mod.kv_cache_memory_mb


def test_attention_no_cache_shape():
    attn = AttentionNoCache(d_model=64, n_heads=4)
    x = torch.randn(2, 8, 64)
    out = attn(x)
    assert out.shape == (2, 8, 64)

def test_attention_with_cache_shape():
    attn = AttentionWithCache(d_model=64, n_heads=4)
    x = torch.randn(2, 8, 64)
    out, cache = attn(x, kv_cache={}, layer_idx=0)
    assert out.shape == (2, 8, 64)
    assert 0 in cache

def test_kv_cache_grows():
    attn = AttentionWithCache(d_model=64, n_heads=4)
    cache = {}
    x1 = torch.randn(1, 4, 64)
    _, cache = attn(x1, kv_cache=cache, layer_idx=0)
    seq_len_after_4 = cache[0][0].shape[2]
    x2 = torch.randn(1, 1, 64)
    _, cache = attn(x2, kv_cache=cache, layer_idx=0)
    seq_len_after_5 = cache[0][0].shape[2]
    assert seq_len_after_5 == seq_len_after_4 + 1

def test_gpt_with_cache_forward():
    model = GPTWithCache(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    idx = torch.randint(0, 50, (1, 8))
    logits, cache = model(idx)
    assert logits.shape == (1, 8, 50)
    assert isinstance(cache, dict)

def test_generate_no_cache():
    model = GPTWithCache(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    prompt = torch.randint(0, 50, (1, 4))
    result = generate_no_cache(model, prompt, n_new=5)
    assert len(result) == 4 + 5

def test_kv_cache_memory_mb():
    mb = kv_cache_memory_mb(batch_size=1, seq_len=512, n_layers=12, n_heads=12,
                            head_dim=64, dtype=torch.float16)
    assert mb > 0
    expected = 2 * 1 * 12 * 12 * 512 * 64 * 2 / 1e6
    assert abs(mb - expected) < 0.01
