import torch
import torch.nn as nn
import struct


def float_bits(x: float, dtype) -> str:
    t = torch.tensor(x, dtype=dtype)
    if dtype == torch.float32:
        bits = struct.unpack("I", struct.pack("f", t.item()))[0]
        return f"{bits:032b}"
    elif dtype == torch.float16:
        bits = t.view(torch.int16).item() & 0xFFFF
        return f"{bits:016b}"
    elif dtype == torch.bfloat16:
        bits = t.view(torch.int16).item() & 0xFFFF
        return f"{bits:016b}"


def model_memory_mb(model: nn.Module, dtype) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.finfo(dtype).bits // 8
    return n_params * bytes_per_param / 1e6


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    val = 3.14
    print(f"  float32 bits: {float_bits(val, torch.float32)}")
    print(f"  float16 bits: {float_bits(val, torch.float16)}")
    print(f" bfloat16 bits: {float_bits(val, torch.bfloat16)}")

    for dtype, label in [(torch.float32, "float32"),
                         (torch.float16, "float16"),
                         (torch.bfloat16, "bfloat16")]:
        info = torch.finfo(dtype)
        print(f"\n{label}: bits={info.bits} max={info.max:.2e}")

    model = nn.Sequential(
        nn.Linear(1024, 4096), nn.GELU(),
        nn.Linear(4096, 4096), nn.GELU(),
        nn.Linear(4096, 1024),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        mb = model_memory_mb(model, dtype)
        print(f"  {str(dtype):20s}: {mb:.1f} MB")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(64, 1024, device=device)

    model = model.float()
    t0 = time.perf_counter()
    for _ in range(20):
        out  = model(x)
        loss = out.sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
    if device == "cuda": torch.cuda.synchronize()
    fp32_ms = (time.perf_counter() - t0) * 1000 / 20
    print(f"FP32 step time: {fp32_ms:.2f} ms")

    x_vals = torch.linspace(-10, 10, 1000)
    fp16_vals = x_vals.half().float()
    bf16_vals = x_vals.bfloat16().float()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, (fp16_vals - x_vals).abs(), label="FP16 error", alpha=0.8)
    ax.plot(x_vals, (bf16_vals - x_vals).abs(), label="BF16 error", alpha=0.8)
    ax.set_xlabel("Value")
    ax.set_ylabel("Absolute quantisation error")
    ax.set_title("FP16 vs BF16 quantisation error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../data/ch09_precision.png", dpi=100)
    print("Saved -> data/ch09_precision.png")
