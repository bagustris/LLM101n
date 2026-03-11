import torch
import pytest
import importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch01_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch01_main"] = _mod
_spec.loader.exec_module(_mod)

build_vocab = _mod.build_vocab
encode = _mod.encode
decode = _mod.decode
generate = _mod.generate
compute_nll = _mod.compute_nll
build_bigram_matrix = _mod.build_bigram_matrix

SAMPLE_TEXT = "hello world. hello again."

def test_build_vocab():
    stoi, itos, vocab_size = build_vocab(SAMPLE_TEXT)
    assert '.' in stoi
    assert stoi['.'] == 0
    assert len(stoi) == vocab_size
    assert all(itos[i] == c for c, i in stoi.items())

def test_encode_decode_roundtrip():
    stoi, itos, _ = build_vocab(SAMPLE_TEXT)
    s = "hello"
    ids = encode(s, stoi)
    recovered = decode(ids, itos)
    assert recovered == s

def test_generate_returns_string():
    stoi, itos, vocab_size = build_vocab(SAMPLE_TEXT)
    N = torch.ones((vocab_size, vocab_size), dtype=torch.int32)
    P = N.float() / N.float().sum(dim=1, keepdim=True)
    result = generate(P, itos, stoi, max_chars=10, seed=0)
    assert isinstance(result, str)

def test_generate_stops_at_end_token():
    stoi, itos, vocab_size = build_vocab("ab")
    P = torch.zeros((vocab_size, vocab_size))
    dot_idx = stoi['.']
    P[:, dot_idx] = 1.0
    result = generate(P, itos, stoi, max_chars=100, seed=0)
    assert result == ""

def test_compute_nll_returns_positive_float(tmp_path):
    stoi, itos, vocab_size = build_vocab("abc")
    N = torch.ones((vocab_size, vocab_size), dtype=torch.int32)
    P = N.float() / N.float().sum(dim=1, keepdim=True)
    tmp_file = tmp_path / "test.txt"
    tmp_file.write_text("abc\nabc\n")
    nll = compute_nll(P, stoi, str(tmp_file), max_chars=1000)
    assert nll > 0

def test_bigram_matrix_shape():
    stoi, itos, vocab_size = build_vocab("abc")
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("abc\nabc\n")
        fname = f.name
    try:
        N = build_bigram_matrix(fname, stoi, vocab_size)
        assert N.shape == (vocab_size, vocab_size)
        assert N.sum() > 0
    finally:
        os.unlink(fname)
