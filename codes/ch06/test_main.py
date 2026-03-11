import importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch06_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch06_main"] = _mod
_spec.loader.exec_module(_mod)

BPETokenizer = _mod.BPETokenizer

MINI_CORPUS = "the cat sat on the mat. the cat is fat. a cat sat."

def test_bpe_trains():
    tok = BPETokenizer()
    tok.train(MINI_CORPUS, vocab_size=30, verbose=False)
    assert len(tok.vocab) >= 20

def test_bpe_encode_returns_ints():
    tok = BPETokenizer()
    tok.train(MINI_CORPUS, vocab_size=30, verbose=False)
    ids = tok.encode("the cat")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

def test_bpe_decode_returns_string():
    tok = BPETokenizer()
    tok.train(MINI_CORPUS, vocab_size=30, verbose=False)
    ids = tok.encode("cat sat")
    result = tok.decode(ids)
    assert isinstance(result, str)
    assert len(result) > 0

def test_bpe_vocab_grows():
    tok = BPETokenizer()
    tok.train(MINI_CORPUS, vocab_size=40, verbose=False)
    assert len(tok.vocab) <= 40

def test_bpe_encode_known_text():
    tok = BPETokenizer()
    tok.train(MINI_CORPUS, vocab_size=30, verbose=False)
    ids = tok.encode("the")
    assert len(ids) >= 1

def test_bpe_stoi_itos_consistent():
    tok = BPETokenizer()
    tok.train(MINI_CORPUS, vocab_size=25, verbose=False)
    for token_id, token_str in tok.vocab.items():
        assert tok.stoi.get(token_str) == token_id
