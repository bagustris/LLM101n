import math, random, importlib.util, sys, os

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
_spec = importlib.util.spec_from_file_location("ch02_main", _path)
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["ch02_main"] = _mod
_spec.loader.exec_module(_mod)

Value = _mod.Value
Neuron = _mod.Neuron
Layer = _mod.Layer
MLP = _mod.MLP
print_graph = _mod.print_graph

def test_value_add():
    a = Value(2.0); b = Value(3.0); c = a + b
    assert c.data == 5.0

def test_value_mul():
    a = Value(3.0); b = Value(4.0); c = a * b
    assert c.data == 12.0

def test_value_pow():
    a = Value(3.0); b = a ** 2
    assert abs(b.data - 9.0) < 1e-9

def test_value_relu_positive():
    a = Value(2.0)
    assert a.relu().data == 2.0

def test_value_relu_negative():
    a = Value(-1.0)
    assert a.relu().data == 0.0

def test_value_exp():
    a = Value(1.0)
    assert abs(a.exp().data - math.e) < 1e-6

def test_value_log():
    a = Value(math.e)
    assert abs(a.log().data - 1.0) < 1e-6

def test_gradient_simple():
    x = Value(2.0); y = Value(3.0)
    z = (x + y) * x
    z.backward()
    assert abs(x.grad - 7.0) < 1e-9
    assert abs(y.grad - 2.0) < 1e-9

def test_gradient_chain():
    x = Value(3.0)
    y = x ** 2
    y.backward()
    assert abs(x.grad - 6.0) < 1e-9

def test_neuron_output_is_value():
    random.seed(0)
    n = Neuron(3)
    out = n([Value(1.0), Value(2.0), Value(3.0)])
    assert isinstance(out, Value)
    assert out.data >= 0

def test_mlp_parameter_count():
    random.seed(0)
    mlp = MLP(2, [4, 1])
    assert len(mlp.parameters()) == 17

def test_mlp_forward():
    random.seed(0)
    mlp = MLP(2, [3, 1])
    out = mlp([Value(0.5), Value(-0.5)])
    assert isinstance(out, Value)

def test_mlp_zero_grad():
    random.seed(0)
    mlp = MLP(2, [3, 1])
    for p in mlp.parameters():
        p.grad = 1.0
    mlp.zero_grad()
    assert all(p.grad == 0.0 for p in mlp.parameters())

def test_value_sub():
    a = Value(5.0); b = Value(3.0)
    assert (a - b).data == 2.0

def test_value_div():
    a = Value(6.0); b = Value(2.0)
    assert abs((a / b).data - 3.0) < 1e-9

def test_value_neg():
    a = Value(4.0)
    assert (-a).data == -4.0
