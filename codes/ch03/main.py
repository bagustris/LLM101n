import os
import torch
import torch.nn as nn
import math

DATA_DIR   = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")

CONTEXT_LEN = 8

stoi = {}
itos = {}


def download_tinystories():
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, path, n in [("train", TRAIN_FILE, 50_000),
                            ("validation", VAL_FILE, 5_000)]:
        if os.path.exists(path):
            continue
        print(f"Downloading TinyStories ({split}) ...")
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                if i >= n:
                    break
                f.write(ex["text"].strip() + "\n")
        print(f"  Saved -> {path}")


def encode(s: str) -> list:
    return [stoi.get(c, stoi.get('.', 0)) for c in s]


def decode(ids) -> str:
    return "".join(itos[i] for i in ids)


def build_dataset(text: str):
    X, Y = [], []
    ids = encode(text)
    for i in range(len(ids) - CONTEXT_LEN):
        X.append(ids[i : i + CONTEXT_LEN])
        Y.append(ids[i + CONTEXT_LEN])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            )
        )


class NGramMLP(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, context_len: int,
                 hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        in_dim = context_len * emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb = emb.view(emb.size(0), -1)
        return self.net(emb)

    @torch.no_grad()
    def generate(self, stoi, itos, max_chars=200, temperature=1.0, seed=0):
        torch.manual_seed(seed)
        self.eval()
        device = next(self.parameters()).device
        context = [stoi['.']] * CONTEXT_LEN
        result  = []
        for _ in range(max_chars):
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self(x)[0]
            probs  = torch.softmax(logits / temperature, dim=-1)
            idx    = torch.multinomial(probs, 1).item()
            if itos[idx] == '.':
                break
            result.append(itos[idx])
            context = context[1:] + [idx]
        return "".join(result)


if __name__ == "__main__":
    download_tinystories()

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_text = f.read()

    print(f"Train chars: {len(train_text):,}")
    print(f"Val   chars: {len(val_text):,}")

    chars = ['.'] + sorted(set(train_text) - {'.'})
    stoi  = {c: i for i, c in enumerate(chars)}
    itos  = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    Xtr, Ytr = build_dataset(train_text)
    Xva, Yva = build_dataset(val_text)
    print(f"Train samples: {Xtr.shape}  Val samples: {Xva.shape}")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = NGramMLP(
        vocab_size   = vocab_size,
        emb_dim      = 32,
        context_len  = CONTEXT_LEN,
        hidden_dim   = 256,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    BATCH_SIZE = 256
    STEPS      = 200

    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)

    for step in range(STEPS):
        model.train()
        idx    = torch.randint(len(Xtr), (BATCH_SIZE,))
        xb, yb = Xtr[idx], Ytr[idx]
        logits = model(xb)
        loss   = loss_fn(logits, yb)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_idx    = torch.randint(len(Xva), (2048,))
                val_logits = model(Xva[val_idx])
                val_loss   = loss_fn(val_logits, Yva[val_idx]).item()
            print(f"Step {step:5d}  train loss: {loss.item():.4f}  val loss: {val_loss:.4f}")

    model.eval()
    for seed in range(3):
        print(f"\n--- Generated story {seed + 1} ---")
        print(model.generate(stoi, itos, max_chars=300, temperature=0.8, seed=seed))
