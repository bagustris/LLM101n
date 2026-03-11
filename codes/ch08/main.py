import torch
import torch.nn as nn
import time


def benchmark_matmul(size: int, device: str, n_runs: int = 50) -> float:
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    for _ in range(5):
        _ = A @ B
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        C = A @ B
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / n_runs
    return elapsed


class SmallTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=4, vocab=1000):
        super().__init__()
        self.emb    = nn.Embedding(vocab, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4*d_model,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.head(self.transformer(self.emb(x)))


def time_model(m, x, n=30):
    device_str = str(next(m.parameters()).device)
    for _ in range(5):
        m(x)
    if "cuda" in device_str:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        m(x)
    if "cuda" in device_str:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sizes = [128, 256, 512, 1024]
    print(f"{'Size':>6}  {'CPU (ms)':>10}")
    for s in sizes:
        cpu_ms = benchmark_matmul(s, "cpu", n_runs=10)
        print(f"{s:6d}  {cpu_ms:10.2f}")

    model_eager = SmallTransformer().to(device).eval()
    x = torch.randint(0, 1000, (8, 128), device=device)

    with torch.no_grad():
        eager_ms = time_model(model_eager, x)
    print(f"Eager model: {eager_ms:.2f} ms")
