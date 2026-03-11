import os
from datasets import load_dataset

DATA_DIR = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(TRAIN_FILE):
    print("Downloading TinyStories …")
    # streaming=True avoids loading all 2M stories into RAM at once
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    # We take a manageable slice (50 000 stories) for local experiments.
    # Remove the slice to use the full dataset (≈ 2 GB).
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds):
            if i >= 50_000:
                break
            f.write(example["text"].strip() + "\n")
    print(f"Saved training slice → {TRAIN_FILE}")
else:
    print(f"Training file already exists: {TRAIN_FILE}")

if not os.path.exists(VAL_FILE):
    ds_val = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for i, example in enumerate(ds_val):
            if i >= 5_000:
                break
            f.write(example["text"].strip() + "\n")
    print(f"Saved validation slice → {VAL_FILE}")
else:
    print(f"Validation file already exists: {VAL_FILE}")


# Read the training corpus
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Corpus length: {len(text):,} characters")

# Collect every unique character
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars[:40])} …")

# Build integer ↔ character mappings
stoi = {ch: i for i, ch in enumerate(chars)}  # string → index
itos = {i: ch for ch, i in stoi.items()}       # index → string

def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

# Sanity check
sample = "Once upon a time"
assert decode(encode(sample)) == sample
print("Encode/decode round-trip: OK")


import torch

# N[i, j] = number of times character i is followed by character j
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

# We add special start/end token '.' at index 0
# Rebuild stoi/itos with '.' prepended
chars_with_special = ['.'] + [c for c in chars if c != '.']
stoi = {ch: i for i, ch in enumerate(chars_with_special)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars_with_special)

N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

# Iterate over every story in the training file
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        story = line.strip()
        if not story:
            continue
        # Wrap each story with start/end sentinel '.'
        tokens = [stoi['.']] + [stoi[c] for c in story] + [stoi['.']]
        for ch1, ch2 in zip(tokens, tokens[1:]):
            N[ch1, ch2] += 1

print(f"Bigram count matrix shape: {N.shape}")
print(f"Total bigrams counted: {N.sum().item():,}")


# Convert counts to probabilities using Laplace (add-one) smoothing
# Smoothing ensures no probability is exactly zero (avoids -inf log-prob)
P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)  # row-normalise

print("Row sums (should all be 1.0):", P.sum(dim=1)[:5])


import random

def generate(P, itos, stoi, max_chars: int = 200, seed: int = 42) -> str:
    """Sample a story character-by-character from the bigram distribution."""
    torch.manual_seed(seed)
    idx = stoi['.']          # start token
    result = []

    for _ in range(max_chars):
        # P[idx] is the probability distribution over next characters
        probs = P[idx]
        # torch.multinomial draws a sample index according to `probs`
        idx = torch.multinomial(probs, num_samples=1).item()
        if itos[idx] == '.':  # end token reached
            break
        result.append(itos[idx])

    return "".join(result)

for seed in range(3):
    print(f"--- Sample {seed + 1} ---")
    print(generate(P, itos, stoi, seed=seed))
    print()


def compute_nll(P, stoi, filepath: str, max_chars: int = 100_000) -> float:
    """
    Negative log-likelihood: average -log P(next_char | current_char).
    Lower is better. A uniform model over V chars gives NLL = log(V).
    """
    total_log_prob = 0.0
    n = 0

    with open(filepath, "r", encoding="utf-8") as f:
        chars_seen = 0
        for line in f:
            story = line.strip()
            if not story:
                continue
            tokens = [stoi['.']] + [stoi.get(c, stoi['.']) for c in story] + [stoi['.']]
            for ch1, ch2 in zip(tokens, tokens[1:]):
                log_prob = torch.log(P[ch1, ch2]).item()
                total_log_prob += log_prob
                n += 1
                chars_seen += 1
            if chars_seen >= max_chars:
                break

    nll = -total_log_prob / n
    return nll

train_nll = compute_nll(P, stoi, TRAIN_FILE)
val_nll   = compute_nll(P, stoi, VAL_FILE)

print(f"Train NLL: {train_nll:.4f}")
print(f"Val   NLL: {val_nll:.4f}")
print(f"Baseline (uniform): {torch.log(torch.tensor(float(vocab_size))).item():.4f}")


import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt

# Show top-20 most frequent bigrams
counts_flat = N.view(-1)
top_indices = counts_flat.argsort(descending=True)[:20]
labels = []
values = []
for idx in top_indices:
    i, j = divmod(idx.item(), vocab_size)
    labels.append(f"{repr(itos[i])}→{repr(itos[j])}")
    values.append(counts_flat[idx].item())

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(range(len(labels)), values)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_title("Top-20 Most Frequent Bigrams (TinyStories)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("../data/bigram_frequencies.png", dpi=100)
print("Saved plot → data/bigram_frequencies.png")
