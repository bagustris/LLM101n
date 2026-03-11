import os
import re
from collections import defaultdict

DATA_DIR   = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")


class BPETokenizer:
    def __init__(self):
        self.vocab  = {}
        self.merges = {}
        self.stoi   = {}

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.stoi  = {c: i for i, c in self.vocab.items()}

        word_freq = defaultdict(int)
        for word in text.split():
            word_freq[" ".join(list(word)) + " </w>"] += 1

        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")

        while len(self.vocab) < vocab_size:
            pair_counts = defaultdict(int)
            for word, freq in word_freq.items():
                symbols = word.split()
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            merged = "".join(best_pair)

            if verbose and len(self.merges) % 100 == 0:
                print(f"  Merge #{len(self.merges):4d}: {best_pair} -> '{merged}' (freq={best_count:,})")

            self.merges[best_pair] = merged
            new_id = len(self.vocab)
            self.vocab[new_id] = merged
            self.stoi[merged] = new_id

            bigram = re.escape(" ".join(best_pair))
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            word_freq = {
                pattern.sub(merged, word): freq
                for word, freq in word_freq.items()
            }

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    def _apply_merges(self, word: str) -> list:
        symbols = list(word) + ["</w>"]
        for (a, b), merged in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols = symbols[:i] + [merged] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode(self, text: str) -> list:
        ids = []
        for word in text.split():
            for sym in self._apply_merges(word):
                ids.append(self.stoi.get(sym, 0))
        return ids

    def decode(self, ids: list) -> str:
        tokens = [self.vocab.get(i, "?") for i in ids]
        text   = " ".join(tokens)
        text   = text.replace(" </w>", " ").replace("</w>", "")
        text   = re.sub(r" (?=[^<\s])", "", text)
        return text.strip()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_FILE):
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        with open(TRAIN_FILE, "w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                if i >= 50_000: break
                f.write(ex["text"].strip() + "\n")

    with open(TRAIN_FILE, encoding="utf-8") as f:
        corpus = f.read()

    corpus_small = corpus[:500_000]
    print(f"Corpus size: {len(corpus_small):,} chars")

    tokenizer = BPETokenizer()
    tokenizer.train(corpus_small, vocab_size=500, verbose=True)

    sample_texts = [
        "Once upon a time there was a little girl.",
        "The dog ran quickly through the forest.",
    ]

    for text in sample_texts:
        ids = tokenizer.encode(text)
        reconstructed = tokenizer.decode(ids)
        print(f"Original : {text}")
        print(f"Token IDs: {ids}")
        print(f"Decoded  : {reconstructed}")
        print()
