import torch, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch08_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch08_main"] = _mod
_spec.loader.exec_module(_mod)

benchmark_matmul = _mod.benchmark_matmul
SmallTransformer = _mod.SmallTransformer
time_model = _mod.time_model

def test_benchmark_matmul_returns_float():
    t = benchmark_matmul(64, "cpu", n_runs=3)
    assert isinstance(t, float)
    assert t > 0

def test_small_transformer_output_shape():
    model = SmallTransformer()
    x = torch.randint(0, 1000, (1, 16))
    out = model(x)
    assert out.shape == (1, 16, 1000)

def test_time_model_returns_float():
    model = SmallTransformer()
    model.eval()
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        t = time_model(model, x, n=3)
    assert isinstance(t, float)
    assert t > 0
