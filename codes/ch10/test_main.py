import importlib.util, os, sys

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch10_main", os.path.join(os.path.dirname(__file__), "main.py"))
simulate_allreduce = _mod.simulate_allreduce


def test_simulate_allreduce_returns_string():
    result = simulate_allreduce(1_000_000, world_size=4)
    assert isinstance(result, str)
    assert "MB" in result

def test_simulate_allreduce_scales_with_size():
    small = simulate_allreduce(1_000_000, world_size=8)
    large = simulate_allreduce(10_000_000, world_size=8)
    assert large != small

def test_simulate_allreduce_world_size_1():
    result = simulate_allreduce(1_000_000, world_size=1)
    assert "0.0 MB" in result

def test_simulate_allreduce_world_size_2():
    result = simulate_allreduce(1_000_000, world_size=2)
    assert isinstance(result, str)
