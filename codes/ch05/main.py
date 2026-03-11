import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.eps   = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta


class FeedForward(nn.Module):
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
        normed = self.ln1(x)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln_f  = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        device = idx.device
        tok_emb = self.token_emb(idx)
        pos     = torch.arange(T, device=device)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


DATA_DIR   = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")


def ensure_data():
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, path, n in [("train", TRAIN_FILE, 50_000),
                            ("validation", VAL_FILE, 5_000)]:
        if not os.path.exists(path):
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
            with open(path, "w", encoding="utf-8") as f:
                for i, ex in enumerate(ds):
                    if i >= n: break
                    f.write(ex["text"].strip() + "\n")
            print(f"Saved {path}")


def encode(s, stoi):
    return [stoi.get(c, 0) for c in s]


if __name__ == "__main__":
    ensure_data()

    with open(TRAIN_FILE) as f: train_text = f.read()
    with open(VAL_FILE)   as f: val_text   = f.read()

    chars = ['.'] + sorted(set(train_text) - {'.'})
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    print(f"vocab_size = {vocab_size}")

    train_ids = torch.tensor(encode(train_text, stoi), dtype=torch.long)
    val_ids   = torch.tensor(encode(val_text, stoi),   dtype=torch.long)
    print(f"Train tokens: {len(train_ids):,}  Val tokens: {len(val_ids):,}")

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    BLOCK_SIZE = 128
    BATCH_SIZE = 32
    LR         = 3e-4
    STEPS      = 100

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
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        out = model.generate(context.clone(), max_new_tokens=300, temperature=0.8, top_k=40)
        story = "".join(itos[i] for i in out[0].tolist())
        print(f"\n--- Story {seed + 1} ---\n{story}")

    torch.save(model.state_dict(), "../data/gpt_tinystories.pt")
    print("\nCheckpoint saved -> data/gpt_tinystories.pt")
