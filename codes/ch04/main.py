import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out     = weights @ V
    return out, weights


def causal_mask(T: int, device="cpu") -> torch.Tensor:
    mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
    return mask


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.split(C, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)

        out, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.out_proj(out))
        return out, attn_weights


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = 6
    mask = causal_mask(T)
    print("Causal mask (True = masked out):")
    print(mask.int())

    torch.manual_seed(0)
    d_model = 64
    n_heads = 4
    T_demo  = 10
    B       = 1

    mha = MultiHeadAttention(d_model, n_heads)
    pos_enc = SinusoidalPositionalEncoding(d_model)

    x = torch.randn(B, T_demo, d_model)
    x = pos_enc(x)
    mask = causal_mask(T_demo)
    out, weights = mha(x, mask)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Attention weight shape: {weights.shape}")

    fig, axes = plt.subplots(1, n_heads, figsize=(14, 3))
    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(weights[0, h].detach().numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"Head {h}")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Multi-Head Attention Patterns (causal)")
    plt.tight_layout()
    plt.savefig("../data/ch04_attention.png", dpi=100)
    print("Saved -> data/ch04_attention.png")
