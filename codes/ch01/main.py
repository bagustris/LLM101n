import os
import torch
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR   = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")


def build_vocab(text: str):
    chars = ['.'] + sorted(set(text) - {'.'})
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars)


def build_bigram_matrix(filepath: str, stoi: dict, vocab_size: int, max_stories: int = 5000) -> torch.Tensor:
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    with open(filepath, "r", encoding="utf-8") as f:
        for count, line in enumerate(f):
            if count >= max_stories:
                break
            story = line.strip()
            if not story:
                continue
            tokens = [stoi["."]] + [stoi.get(c, stoi["."])  for c in story] + [stoi["."] ]
            for ch1, ch2 in zip(tokens, tokens[1:]):
                N[ch1, ch2] += 1
    return N


def encode(s: str, stoi: dict) -> list:
    return [stoi.get(c, stoi.get(".", 0)) for c in s]


def decode(ids: list, itos: dict) -> str:
    return "".join(itos[i] for i in ids)


def generate(P: torch.Tensor, itos: dict, stoi: dict, max_chars: int = 200, seed: int = 42) -> str:
    torch.manual_seed(seed)
    idx = stoi["."]
    result = []
    for _ in range(max_chars):
        probs = P[idx]
        idx = torch.multinomial(probs, num_samples=1).item()
        if itos[idx] == ".": break
        result.append(itos[idx])
    return "".join(result)


def compute_nll(P: torch.Tensor, stoi: dict, filepath: str, max_chars: int = 100_000) -> float:
    total_log_prob = 0.0
    n = 0
    with open(filepath, "r", encoding="utf-8") as f:
        chars_seen = 0
        for line in f:
            story = line.strip()
            if not story:
                continue
            tokens = [stoi["."]] + [stoi.get(c, stoi["."])  for c in story] + [stoi["."]]
            for ch1, ch2 in zip(tokens, tokens[1:]):
                log_prob = torch.log(P[ch1, ch2]).item()
                total_log_prob += log_prob
                n += 1
                chars_seen += 1
            if chars_seen >= max_chars:
                break
    return -total_log_prob / n


if __name__ == "__main__":
    from datasets import load_dataset
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_FILE):
        print("Downloading TinyStories …")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
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

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Corpus length: {len(text):,} characters")

    stoi, itos, vocab_size = build_vocab(text)
    print(f"Vocabulary size: {vocab_size}")

    N = build_bigram_matrix(TRAIN_FILE, stoi, vocab_size, max_stories=5000)
    print(f"Bigram count matrix shape: {N.shape}")

    P = (N + 1).float()
    P = P / P.sum(dim=1, keepdim=True)

    for seed in range(3):
        print(f"--- Sample {seed + 1} ---")
        print(generate(P, itos, stoi, seed=seed))
        print()

    train_nll = compute_nll(P, stoi, TRAIN_FILE, max_chars=10_000)
    val_nll   = compute_nll(P, stoi, VAL_FILE, max_chars=10_000)
    print(f"Train NLL: {train_nll:.4f}")
    print(f"Val   NLL: {val_nll:.4f}")

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
