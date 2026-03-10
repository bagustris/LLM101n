import importlib.util, os, sys, torch, tempfile

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch11_main", os.path.join(os.path.dirname(__file__), "main.py"))
is_valid_story = _mod.is_valid_story
TinyStoriesDataset = _mod.TinyStoriesDataset
collate_fn = _mod.collate_fn

SAMPLE_STORIES = "Once there was a dog.\nThe dog ran fast.\nA cat slept.\n"

def make_temp_file(content=SAMPLE_STORIES):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name

def test_is_valid_story_short_rejected():
    assert is_valid_story("Hi") == False

def test_is_valid_story_long_accepted():
    assert is_valid_story("Once there was a very happy dog who loved to play.") == True

def test_is_valid_story_empty_rejected():
    assert is_valid_story("") == False

def test_tinystories_dataset_len():
    path = make_temp_file()
    try:
        ds = TinyStoriesDataset(path, block_size=16)
        assert len(ds) > 0
    finally:
        os.unlink(path)

def test_tinystories_dataset_item_shape():
    path = make_temp_file("abcdefghijklmnopqrstuvwxyz " * 20 + "\n")
    try:
        ds = TinyStoriesDataset(path, block_size=16)
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)
    finally:
        os.unlink(path)

def test_collate_fn():
    batch = [(torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)),
             (torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long))]
    x, y = collate_fn(batch)
    assert x.shape == (2, 8)
    assert y.shape == (2, 8)
