import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from datasets import load_dataset

class LayerNorm(nn.Module):
    """
    Normalise across the feature (d_model) dimension.
    Works independently for each (batch, position) pair.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # learned scale
        self.beta  = nn.Parameter(torch.zeros(d_model))  # learned shift
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta


class FeedForward(nn.Module):
    """
    Two-layer MLP applied independently to each position.
    The inner dimension is 4× d_model, matching GPT-2.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One GPT-2 block using pre-norm (norm before each sublayer).
    x → LN → MHA → x (residual) → LN → FFN → x (residual)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout,
            batch_first=True, bias=False
        )
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, T, _ = x.shape

        # Self-attention with residual
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=mask, is_causal=True)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    Minimal GPT-2 style language model.
    """
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Learned positional embedding (GPT-2 style)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f  = LayerNorm(d_model)          # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and lm_head weights (saves params)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor = None):
        """
        idx:     (B, T) integer token indices
        targets: (B, T) shifted targets for teacher forcing
        Returns (logits, loss) where loss=None if targets not provided.
        """
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_emb(idx)                    # (B, T, d_model)
        pos     = torch.arange(T, device=device)
        pos_emb = self.pos_emb(pos)                      # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B*T, vocab_size) and (B*T,) for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """Autoregressive generation with optional top-k sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to model's max context length
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature   # last position

            if top_k is not None:
                # Keep only top-k logits; set rest to -inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


DATA_DIR = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")
os.makedirs(DATA_DIR, exist_ok=True)

def ensure_data():
    for split, path, n in [("train", TRAIN_FILE, 50_000),
                            ("validation", VAL_FILE, 5_000)]:
        if not os.path.exists(path):
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            with open(path, "w", encoding="utf-8") as f:
                for i, ex in enumerate(ds):
                    if i >= n: break
                    f.write(ex["text"].strip() + "\n")
            print(f"Saved {path}")

ensure_data()

with open(TRAIN_FILE) as f: train_text = f.read()
with open(VAL_FILE)   as f: val_text   = f.read()

chars = ['.'] + sorted(set(train_text) - {'.'})
stoi  = {c: i for i, c in enumerate(chars)}
itos  = {i: c for c, i in stoi.items()}
vocab_size = len(chars)
print(f"vocab_size = {vocab_size}")

def encode(s): return [stoi.get(c, 0) for c in s]

train_ids = torch.tensor(encode(train_text), dtype=torch.long)
val_ids   = torch.tensor(encode(val_text),   dtype=torch.long)
print(f"Train tokens: {len(train_ids):,}  Val tokens: {len(val_ids):,}")


device    = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 128   # context length
BATCH_SIZE = 32
LR         = 3e-4
STEPS      = 3_000

model = GPT(
    vocab_size = vocab_size,
    d_model    = 128,
    n_heads    = 4,
    n_layers   = 4,
    max_len    = BLOCK_SIZE,
    dropout    = 0.1,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optim = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                          weight_decay=0.1)

def get_batch(ids: torch.Tensor):
    """Sample a random batch of (input, target) pairs."""
    ix  = torch.randint(len(ids) - BLOCK_SIZE, (BATCH_SIZE,))
    x   = torch.stack([ids[i     : i + BLOCK_SIZE] for i in ix])
    y   = torch.stack([ids[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)

for step in range(STEPS + 1):
    model.train()
    xb, yb      = get_batch(train_ids)
    _, loss      = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
    optim.step()

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_ids)
            _, val_loss = model(xv, yv)
        print(f"Step {step:4d}  train={loss.item():.4f}  val={val_loss.item():.4f}")


model.eval()
context = torch.tensor([[stoi['.']]], device=device)

for seed in range(3):
    torch.manual_seed(seed)
    out = model.generate(context.clone(), max_new_tokens=300,
                         temperature=0.8, top_k=40)
    story = "".join(itos[i] for i in out[0].tolist())
    print(f"\n--- Story {seed + 1} ---\n{story}")

# Save checkpoint
torch.save(model.state_dict(), "../data/gpt_tinystories.pt")
print("\nCheckpoint saved → data/gpt_tinystories.pt")
