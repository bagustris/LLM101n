import torch
import torch.nn as nn
import numpy as np

def quantize_absmax_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric absmax quantisation to INT8.
    scale = max(|W|) / 127
    W_int8 = round(W / scale)

    Returns: (quantized_weight: int8, scale: float32)
    """
    scale = weight.abs().max() / 127.0
    w_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


def dequantize_absmax_int8(
    w_int8: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Convert INT8 weights back to float32 for matmul."""
    return w_int8.to(torch.float32) * scale


# Test on a random weight matrix
torch.manual_seed(42)
W_fp32 = torch.randn(256, 256)

W_int8, scale = quantize_absmax_int8(W_fp32)
W_reconstructed = dequantize_absmax_int8(W_int8, scale)

# Measure quantisation error
mse   = ((W_fp32 - W_reconstructed) ** 2).mean().item()
max_err = (W_fp32 - W_reconstructed).abs().max().item()
print(f"INT8 Quantisation Error  —  MSE: {mse:.6f}  |  Max abs: {max_err:.4f}")
print(f"Memory: {W_fp32.numel()*4} bytes (fp32) → {W_int8.numel()*1} bytes (int8)")
print(f"Compression: {W_fp32.numel()*4 / W_int8.numel():.0f}×")


def quantize_per_channel(weight: torch.Tensor):
    """
    Per-channel absmax: each output channel has its own scale.
    Much more accurate than per-tensor for weights with outlier channels.
    weight: (out_features, in_features)
    """
    # Scale per output channel (dim=1 across in_features)
    scale = weight.abs().max(dim=1, keepdim=True).values / 127.0
    w_int8 = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale


def quantize_groupwise(weight: torch.Tensor, group_size: int = 64):
    """
    Group-wise quantisation: split each row into groups of `group_size`.
    Each group gets its own scale — much better accuracy for INT4.
    weight: (out_features, in_features)
    """
    out_f, in_f = weight.shape
    assert in_f % group_size == 0, "in_features must be divisible by group_size"
    n_groups = in_f // group_size

    w_grouped = weight.view(out_f, n_groups, group_size)
    scale = w_grouped.abs().max(dim=2, keepdim=True).values / 127.0
    w_int8 = (w_grouped / scale).round().clamp(-128, 127).to(torch.int8)
    return w_int8, scale   # scale: (out_f, n_groups, 1)


W = torch.randn(512, 1024) * 0.02

# Per-tensor (worst)
w_pt, s_pt = quantize_absmax_int8(W)
err_pt = ((W - dequantize_absmax_int8(w_pt, s_pt)) ** 2).mean().item()

# Per-channel
w_pc, s_pc = quantize_per_channel(W)
recon_pc = (w_pc.float() * s_pc)
err_pc = ((W - recon_pc) ** 2).mean().item()

# Group-wise (group=64)
w_gw, s_gw = quantize_groupwise(W, group_size=64)
recon_gw = (w_gw.float() * s_gw).view(512, 1024)
err_gw = ((W - recon_gw) ** 2).mean().item()

print("MSE comparison:")
print(f"  Per-tensor  : {err_pt:.8f}")
print(f"  Per-channel : {err_pc:.8f}")
print(f"  Group-wise  : {err_gw:.8f}")


def quantize_int4_absmax(weight: torch.Tensor):
    """
    4-bit symmetric absmax quantisation.
    INT4 range: [-8, 7] (15 steps).
    Pack two INT4 values into one byte (2× compression vs INT8).
    """
    scale = weight.abs().max() / 7.0
    w_int4 = (weight / scale).round().clamp(-8, 7)
    # Packing two int4 into one int8 (for illustration)
    w_int4_i8 = w_int4.to(torch.int8)   # stored as int8, logically 4-bit
    return w_int4_i8, scale


W = torch.randn(256, 256)
w_i4, s_i4 = quantize_int4_absmax(W)
recon = w_i4.float() * s_i4
err = ((W - recon) ** 2).mean().item()
print(f"INT4 absmax MSE: {err:.6f}")
print("Logical memory (packed): {:.0f} bytes vs {:.0f} bytes (fp32)".format(
    W.numel() * 0.5, W.numel() * 4))


# PyTorch's built-in dynamic quantization:
# Weights are quantised to INT8 at load time; activations are quantised on-the-fly.
# Works out-of-the-box for nn.Linear — no calibration data needed.

model = nn.Sequential(
    nn.Linear(512, 2048), nn.ReLU(),
    nn.Linear(2048, 2048), nn.ReLU(),
    nn.Linear(2048, 512),
)

def model_size_mb(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6

print(f"FP32 model size: {model_size_mb(model):.1f} MB")

model_int8 = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},       # quantise all Linear layers
    dtype=torch.qint8,
)

print(f"INT8 model size: {model_size_mb(model_int8):.1f} MB")

# Verify outputs are close
x = torch.randn(8, 512)
with torch.no_grad():
    out_fp32 = model(x)
    out_int8 = model_int8(x)

print(f"Max output diff: {(out_fp32 - out_int8).abs().max().item():.4f}")


# bitsandbytes provides NF4 (Normal Float 4) quantisation
# used by QLoRA and many production fine-tuning pipelines
# Install: pip install bitsandbytes

try:
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit            = True,
        bnb_4bit_quant_type     = "nf4",      # Normal Float 4
        bnb_4bit_compute_dtype  = torch.bfloat16,
        bnb_4bit_use_double_quant = True,     # quantise the quantisation constants too
    )

    print("Loading GPT-2 in 4-bit …")
    model_4bit = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        quantization_config=quantization_config,
        device_map="auto",
    )

    fp32_size = sum(p.numel() * 4 for p in model_4bit.parameters()) / 1e6
    int4_size = sum(p.numel() * p.element_size()
                    for p in model_4bit.parameters()) / 1e6
    print(f"FP32 equivalent: {fp32_size:.0f} MB")
    print(f"INT4 actual:     {int4_size:.0f} MB")
    print(f"Compression:     {fp32_size/int4_size:.1f}×")

except ImportError:
    print("bitsandbytes not installed. Run: pip install bitsandbytes")
    print("Example would load GPT-2 in NF4 4-bit quantisation.")


from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
quantize_config = BaseQuantizeConfig(bits=4, group_size=128)
model = AutoGPTQForCausalLM.from_pretrained("meta-llama/Llama-2-7b", quantize_config)
model.quantize(calibration_data)
model.save_quantized("llama-2-7b-gptq")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = torch.randn(1024, 1024) * 0.02   # typical transformer weight scale

w_i8, s_i8 = quantize_absmax_int8(W)
w_i4, s_i4 = quantize_int4_absmax(W)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, data, title in [
    (axes[0], W.flatten().numpy(),             "FP32 original"),
    (axes[1], (w_i8.float()*s_i8).flatten().numpy(), "INT8 reconstructed"),
    (axes[2], (w_i4.float()*s_i4).flatten().numpy(), "INT4 reconstructed"),
]:
    ax.hist(data, bins=80, density=True, color="steelblue", edgecolor="none")
    ax.set_title(title)
    ax.set_xlabel("Weight value")

plt.tight_layout()
plt.savefig("../data/ch13_quantisation.png", dpi=100)
print("Saved → data/ch13_quantisation.png")
