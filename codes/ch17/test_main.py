import importlib.util, os, sys, torch

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_mod = load_module("ch17_main", os.path.join(os.path.dirname(__file__), "main.py"))
VectorQuantizer = _mod.VectorQuantizer
ResBlock = _mod.ResBlock
VQVAE = _mod.VQVAE
denorm = _mod.denorm


def test_vector_quantizer_output_shape():
    vq = VectorQuantizer(num_embeddings=16, embedding_dim=8)
    z = torch.randn(2, 8, 4, 4)
    z_q, loss, indices = vq(z)
    assert z_q.shape == z.shape
    assert loss.item() >= 0

def test_vector_quantizer_commitment_loss():
    vq = VectorQuantizer(num_embeddings=16, embedding_dim=8)
    z = torch.randn(2, 8, 4, 4)
    _, loss, _ = vq(z)
    assert isinstance(loss.item(), float)

def test_resblock_output_shape():
    rb = ResBlock(channels=32)
    x = torch.randn(2, 32, 8, 8)
    out = rb(x)
    assert out.shape == x.shape

def test_vqvae_forward():
    model = VQVAE(in_channels=3, hidden_dim=32, num_embeddings=16, embedding_dim=8)
    x = torch.randn(2, 3, 32, 32)
    recon, vq_loss = model(x)
    assert recon.shape == x.shape
    assert vq_loss.item() >= 0

def test_denorm():
    x = torch.zeros(3, 4, 4)
    out = denorm(x)
    assert out.shape == (3, 4, 4)
    assert out.min().item() > -1e-3
