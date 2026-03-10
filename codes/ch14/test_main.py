import importlib.util, os, sys, torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch14_main", os.path.join(os.path.dirname(__file__), "main.py"))
format_alpaca = _mod.format_alpaca
LoRALinear = _mod.LoRALinear
TinyGPT = _mod.TinyGPT
make_chat_prompt = _mod.make_chat_prompt


def test_format_alpaca_no_input():
    ex = {"instruction": "Write a poem.", "input": "", "output": "Roses are red."}
    result = format_alpaca(ex)
    assert "Write a poem." in result
    assert "Roses are red." in result
    assert "### Response:" in result

def test_format_alpaca_with_input():
    ex = {"instruction": "Translate to French.", "input": "Hello", "output": "Bonjour"}
    result = format_alpaca(ex)
    assert "Hello" in result
    assert "Translate to French." in result
    assert "### Input:" in result

def test_lora_linear_forward_shape():
    layer = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8.0)
    torch.nn.init.normal_(layer.weight)
    x = torch.randn(2, 64)
    out = layer(x)
    assert out.shape == (2, 32)

def test_lora_linear_trainable_params():
    layer = LoRALinear(in_features=64, out_features=32, rank=4, alpha=8.0)
    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
    assert "lora_A" in trainable
    assert "lora_B" in trainable
    assert "weight" not in trainable

def test_tiny_gpt_forward():
    model = TinyGPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, max_len=64)
    idx = torch.randint(0, 50, (2, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 8, 50)
    assert loss.item() > 0

def test_make_chat_prompt():
    prompt = make_chat_prompt("What is 2+2?")
    assert "2+2" in prompt
    assert "assistant" in prompt.lower() or "Assistant" in prompt
