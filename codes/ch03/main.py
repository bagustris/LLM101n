import os
from datasets import load_dataset

DATA_DIR   = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")
os.makedirs(DATA_DIR, exist_ok=True)

def download_tinystories():
    for split, path, n in [("train", TRAIN_FILE, 50_000),
                            ("validation", VAL_FILE, 5_000)]:
        if os.path.exists(path):
            continue
        print(f"Downloading TinyStories ({split}) …")
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                if i >= n:
                    break
                f.write(ex["text"].strip() + "\n")
        print(f"  Saved → {path}")

download_tinystories()

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_text = f.read()
with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_text = f.read()

print(f"Train chars: {len(train_text):,}")
print(f"Val   chars: {len(val_text):,}")


import torch

# Build vocabulary from training text only
chars = ['.'] + sorted(set(train_text) - {'.'})
stoi  = {c: i for i, c in enumerate(chars)}
itos  = {i: c for c, i in stoi.items()}
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

def encode(s: str) -> list[int]:
    return [stoi.get(c, stoi['.']) for c in s]

def decode(ids) -> str:
    return "".join(itos[i] for i in ids)


CONTEXT_LEN = 8   # how many characters we condition on

def build_dataset(text: str):
    """
    Slide a window of CONTEXT_LEN over the text.
    X[i] = context characters (integers)
    Y[i] = next character (integer)
    """
    X, Y = [], []
    ids = encode(text)
    for i in range(len(ids) - CONTEXT_LEN):
        X.append(ids[i : i + CONTEXT_LEN])
        Y.append(ids[i + CONTEXT_LEN])
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y

Xtr, Ytr = build_dataset(train_text)
Xva, Yva = build_dataset(val_text)
print(f"Train samples: {Xtr.shape}  Val samples: {Xva.shape}")


import torch.nn as nn
import math

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    GELU(x) = x * Φ(x)  where Φ is the standard normal CDF.
    Approximated as: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
            )
        )

# Alternatively: nn.GELU() — built-in PyTorch implementation


class NGramMLP(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, context_len: int,
                 hidden_dim: int):
        super().__init__()
        # Embedding table: each character gets a dense vector of size emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # After embedding lookup and flattening, input dim = context_len * emb_dim
        in_dim = context_len * emb_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, context_len)  integer token indices
        returns: (B, vocab_size) logits
        """
        # Embed each token in the context window
        emb = self.embedding(x)            # (B, context_len, emb_dim)
        emb = emb.view(emb.size(0), -1)   # (B, context_len * emb_dim) — flatten
        return self.net(emb)

    @torch.no_grad()
    def generate(self, stoi, itos, max_chars=200, temperature=1.0, seed=0):
        """Autoregressively sample one story."""
        torch.manual_seed(seed)
        self.eval()
        device = next(self.parameters()).device
        context = [stoi['.']] * CONTEXT_LEN   # start with padding tokens
        result  = []
        for _ in range(max_chars):
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits = self(x)[0]               # (vocab_size,)
            probs  = torch.softmax(logits / temperature, dim=-1)
            idx    = torch.multinomial(probs, 1).item()
            if itos[idx] == '.':
                break
            result.append(itos[idx])
            context = context[1:] + [idx]     # slide window
        return "".join(result)


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

train_losses, val_losses = [], []

for step in range(STEPS):
    model.train()
    # Sample a random mini-batch
    idx    = torch.randint(len(Xtr), (BATCH_SIZE,))
    xb, yb = Xtr[idx], Ytr[idx]

    logits = model(xb)
    loss   = loss_fn(logits, yb)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    train_losses.append(loss.item())

    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            val_idx     = torch.randint(len(Xva), (2048,))
            val_logits  = model(Xva[val_idx])
            val_loss    = loss_fn(val_logits, Yva[val_idx]).item()
        val_losses.append(val_loss)
        print(f"Step {step:5d}  train loss: {loss.item():.4f}  val loss: {val_loss:.4f}")


model.eval()
for seed in range(3):
    print(f"\n--- Generated story {seed + 1} ---")
    print(model.generate(stoi, itos, max_chars=300, temperature=0.8, seed=seed))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Project character embeddings to 2-D for visualisation
emb_weights = model.embedding.weight.detach().cpu().numpy()
pca = PCA(n_components=2)
coords = pca.fit_transform(emb_weights)

fig, ax = plt.subplots(figsize=(10, 8))
for i, ch in itos.items():
    ax.scatter(coords[i, 0], coords[i, 1], s=20, color="steelblue")
    ax.annotate(repr(ch), (coords[i, 0], coords[i, 1]), fontsize=7)
ax.set_title("Character Embeddings (PCA)")
plt.tight_layout()
plt.savefig("../data/ch03_embeddings.png", dpi=100)
print("Saved → data/ch03_embeddings.png")
