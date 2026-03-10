import torch
import torch.nn as nn
import numpy as np


def quantize_absmax_int8(weight: torch.Tensor):
    """Symmetric absmax quantisation to INT8."""
    scale = weight.abs().max() / 127.0
    w_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


def dequantize_absmax_int8(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Convert INT8 weights back to float32 for matmul."""
    return w_int8.to(torch.float32) * scale


def quantize_per_channel(weight: torch.Tensor):
    """Per-channel absmax: each output channel has its own scale."""
    scale = weight.abs().max(dim=1, keepdim=True).values / 127.0
    w_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


def quantize_groupwise(weight: torch.Tensor, group_size: int = 64):
    """Group-wise quantisation: each group gets its own scale."""
    out_f, in_f = weight.shape
    assert in_f % group_size == 0
    n_groups = in_f // group_size
    w_grouped = weight.view(out_f, n_groups, group_size)
    scale = w_grouped.abs().max(dim=2, keepdim=True).values / 127.0
    w_int8 = (w_grouped / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


def quantize_int4_absmax(weight: torch.Tensor):
    """4-bit symmetric absmax quantisation."""
    scale = weight.abs().max() / 7.0
    w_int4 = (weight / scale).round().clamp(-8, 7)
    w_int4_i8 = w_int4.to(torch.int8)
    return w_int4_i8, scale


def model_size_mb(model: nn.Module) -> float:
    """Return model size in MB (float32 equivalent)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    W_fp32 = torch.randn(256, 256)

    W_int8, scale = quantize_absmax_int8(W_fp32)
    W_reconstructed = dequantize_absmax_int8(W_int8, scale)

    mse   = ((W_fp32 - W_reconstructed) ** 2).mean().item()
    max_err = (W_fp32 - W_reconstructed).abs().max().item()
    print(f"INT8 Quantisation Error  --  MSE: {mse:.6f}  |  Max abs: {max_err:.4f}")

    W = torch.randn(512, 1024) * 0.02
    w_pt, s_pt = quantize_absmax_int8(W)
    err_pt = ((W - dequantize_absmax_int8(w_pt, s_pt)) ** 2).mean().item()
    w_pc, s_pc = quantize_per_channel(W)
    recon_pc = (w_pc.float() * s_pc)
    err_pc = ((W - recon_pc) ** 2).mean().item()
    w_gw, s_gw = quantize_groupwise(W, group_size=64)
    recon_gw = (w_gw.float() * s_gw).view(512, 1024)
    err_gw = ((W - recon_gw) ** 2).mean().item()

    print(f"Per-tensor MSE  : {err_pt:.8f}")
    print(f"Per-channel MSE : {err_pc:.8f}")
    print(f"Group-wise MSE  : {err_gw:.8f}")

    model = nn.Sequential(
        nn.Linear(512, 2048), nn.ReLU(),
        nn.Linear(2048, 2048), nn.ReLU(),
        nn.Linear(2048, 512),
    )
    print(f"FP32 model size: {model_size_mb(model):.1f} MB")

    model_int8 = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    print(f"INT8 model size: {model_size_mb(model_int8):.1f} MB")

    W = torch.randn(1024, 1024) * 0.02
    w_i8, s_i8 = quantize_absmax_int8(W)
    w_i4, s_i4 = quantize_int4_absmax(W)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title in [
        (axes[0], W.flatten().numpy(), "FP32 original"),
        (axes[1], (w_i8.float()*s_i8).flatten().numpy(), "INT8 reconstructed"),
        (axes[2], (w_i4.float()*s_i4).flatten().numpy(), "INT4 reconstructed"),
    ]:
        ax.hist(data, bins=80, density=True, color="steelblue", edgecolor="none")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("../data/ch13_quantisation.png", dpi=100)
    print("Saved -> data/ch13_quantisation.png")
