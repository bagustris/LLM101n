import torch, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch03_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch03_main"] = _mod
_spec.loader.exec_module(_mod)

GELU = _mod.GELU
NGramMLP = _mod.NGramMLP
build_dataset = _mod.build_dataset
encode = _mod.encode
decode = _mod.decode
CONTEXT_LEN = _mod.CONTEXT_LEN

def make_vocab(text="hello world"):
    chars = ['.'] + sorted(set(text) - {'.'})
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)

def test_gelu_zero():
    gelu = GELU()
    x = torch.tensor([0.0])
    out = gelu(x)
    assert abs(out.item()) < 1e-5

def test_gelu_positive():
    gelu = GELU()
    x = torch.tensor([2.0])
    out = gelu(x)
    assert out.item() > 1.9

def test_gelu_negative():
    gelu = GELU()
    x = torch.tensor([-2.0])  # GELU(-2) ≈ -0.0455, clearly negative
    out = gelu(x)
    assert out.item() < 0

def test_ngram_mlp_output_shape():
    stoi, itos, vocab_size = make_vocab("abcde hello world")
    model = NGramMLP(vocab_size=vocab_size, emb_dim=8, context_len=4, hidden_dim=32)
    x = torch.randint(0, vocab_size, (2, 4))
    logits = model(x)
    assert logits.shape == (2, vocab_size)

def test_ngram_mlp_generate():
    stoi, itos, vocab_size = make_vocab("abcde hello world")
    model = NGramMLP(vocab_size=vocab_size, emb_dim=8, context_len=4, hidden_dim=32)
    model.eval()
    orig = _mod.CONTEXT_LEN
    _mod.CONTEXT_LEN = 4
    result = model.generate(stoi, itos, max_chars=10, seed=0)
    _mod.CONTEXT_LEN = orig
    assert isinstance(result, str)

def test_build_dataset_shapes():
    stoi, itos, vocab_size = make_vocab("abcde")
    orig_context_len = _mod.CONTEXT_LEN
    orig_stoi = _mod.stoi
    _mod.CONTEXT_LEN = 4
    _mod.stoi = stoi
    text = "abcdeabcde"
    X, Y = build_dataset(text)
    _mod.CONTEXT_LEN = orig_context_len
    _mod.stoi = orig_stoi
    assert X.shape[1] == 4
    assert len(X) == len(Y)
