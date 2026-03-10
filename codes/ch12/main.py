import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class AttentionNoCache(nn.Module):
    """Standard causal attention -- recomputes everything every step."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out      = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, hd   = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        scale  = math.sqrt(hd)
        scores = Q @ K.transpose(-2, -1) / scale
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn   = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class AttentionWithCache(nn.Module):
    """Causal attention with KV cache for efficient autoregressive generation."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = d_model
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out      = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: dict = None, layer_idx: int = 0):
        B, T_new, C = x.shape
        H, hd       = self.n_heads, self.head_dim
        qkv = self.qkv(x).view(B, T_new, 3, H, hd).permute(2, 0, 3, 1, 4)
        Q, K_new, V_new = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None and layer_idx in kv_cache:
            K_cached, V_cached = kv_cache[layer_idx]
            K = torch.cat([K_cached, K_new], dim=2)
            V = torch.cat([V_cached, V_new], dim=2)
        else:
            K, V = K_new, V_new

        if kv_cache is not None:
            kv_cache[layer_idx] = (K.detach(), V.detach())

        T_total = K.shape[2]
        scale  = math.sqrt(hd)
        scores = Q @ K.transpose(-2, -1) / scale

        if T_new > 1:
            mask = torch.triu(
                torch.ones(T_new, T_total, device=x.device), diagonal=T_total - T_new + 1
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out  = (attn @ V).transpose(1, 2).reshape(B, T_new, C)
        return self.out(out), kv_cache


class GPTWithCache(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.layers    = nn.ModuleList([
            AttentionWithCache(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor, kv_cache: dict = None, start_pos: int = 0):
        B, T = idx.shape
        positions = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        if kv_cache is None:
            kv_cache = {}
        for i, layer in enumerate(self.layers):
            x, kv_cache = layer(x, kv_cache=kv_cache, layer_idx=i)
        x      = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, kv_cache

    @torch.no_grad()
    def generate_with_cache(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                            temperature: float = 1.0, itos: dict = None) -> list:
        self.eval()
        logits, kv_cache = self(prompt_ids, kv_cache={})
        next_logit = logits[:, -1, :] / temperature
        next_id    = torch.multinomial(F.softmax(next_logit, dim=-1), 1)
        generated = prompt_ids[0].tolist() + [next_id.item()]
        pos       = prompt_ids.shape[1]
        for step in range(max_new_tokens - 1):
            x              = next_id
            logits, kv_cache = self(x, kv_cache=kv_cache, start_pos=pos + step)
            next_logit     = logits[:, -1, :] / temperature
            next_id        = torch.multinomial(F.softmax(next_logit, dim=-1), 1)
            generated.append(next_id.item())
            if itos and itos.get(next_id.item()) == ".":
                break
        return generated


@torch.no_grad()
def generate_no_cache(model, prompt: torch.Tensor, n_new: int) -> list:
    """Naive generation: pass the full growing sequence each step."""
    ids = prompt.clone()
    for _ in range(n_new):
        logits, _ = model(ids, kv_cache=None)
        next_id   = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0].tolist()


def kv_cache_memory_mb(
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> float:
    """Estimate KV cache memory in MB."""
    bytes_per_elem = torch.finfo(dtype).bits // 8
    total_elements = 2 * batch_size * n_layers * n_heads * seq_len * head_dim
    return total_elements * bytes_per_elem / 1e6


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB   = 128
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 4
    MAX_LEN  = 256

    model_cache = GPTWithCache(VOCAB, D_MODEL, N_HEADS, N_LAYERS, MAX_LEN).to(device)
    prompt = torch.randint(0, VOCAB, (1, 16), device=device)
    N_NEW  = 64

    t0 = time.perf_counter()
    for _ in range(5):
        _ = generate_no_cache(model_cache, prompt, N_NEW)
    if device == "cuda": torch.cuda.synchronize()
    no_cache_ms = (time.perf_counter() - t0) / 5 * 1000

    t0 = time.perf_counter()
    for _ in range(5):
        _ = model_cache.generate_with_cache(prompt, N_NEW)
    if device == "cuda": torch.cuda.synchronize()
    cache_ms = (time.perf_counter() - t0) / 5 * 1000

    print(f"No cache : {no_cache_ms:.1f} ms")
    print(f"KV cache : {cache_ms:.1f} ms")
    print(f"Speedup  : {no_cache_ms / cache_ms:.2f}x")

    print("KV Cache Memory Estimates (FP16):")
    configs = [
        ("GPT-2 Small",  12,  12,   64),
        ("GPT-2 Large",  36,  20,   64),
        ("LLaMA-7B",     32,  32,  128),
    ]
    for name, n_layers, n_heads, head_dim in configs:
        for seq_len in [512, 2048]:
            mb = kv_cache_memory_mb(1, seq_len, n_layers, n_heads, head_dim)
            print(f"  {name:15s} seq={seq_len:5d}: {mb:6.1f} MB")
