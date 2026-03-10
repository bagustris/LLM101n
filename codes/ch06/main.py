import os
import re
from collections import defaultdict

DATA_DIR   = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
os.makedirs(DATA_DIR, exist_ok=True)

# Download TinyStories if not present
if not os.path.exists(TRAIN_FILE):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= 50_000: break
            f.write(ex["text"].strip() + "\n")

with open(TRAIN_FILE, encoding="utf-8") as f:
    corpus = f.read()

# Use a smaller slice for BPE training (faster to demonstrate)
corpus_small = corpus[:500_000]
print(f"Corpus size: {len(corpus_small):,} chars")


class BPETokenizer:
    """
    Minimal Byte Pair Encoding tokeniser.
    Operates on characters (not bytes) for clarity.
    """

    def __init__(self):
        self.vocab  = {}          # token_id → string
        self.merges = {}          # (a, b) → merged token string
        self.stoi   = {}          # string → token_id

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        """
        Train BPE until the vocabulary reaches `vocab_size`.
        `vocab_size` must be > number of unique characters in text.
        """
        # Start with all unique characters as base vocabulary
        chars = sorted(set(text))
        self.vocab = {i: c for i, c in enumerate(chars)}
        self.stoi  = {c: i for i, c in self.vocab.items()}

        # Split corpus into words separated by whitespace
        # Each word is represented as a list of character tokens
        word_freq = defaultdict(int)
        for word in text.split():
            word_freq[" ".join(list(word)) + " </w>"] += 1

        print(f"Initial vocab size: {len(self.vocab)}")

        while len(self.vocab) < vocab_size:
            # Count all adjacent symbol pairs weighted by word frequency
            pair_counts = defaultdict(int)
            for word, freq in word_freq.items():
                symbols = word.split()
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += freq

            if not pair_counts:
                break

            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            merged = "".join(best_pair)

            if verbose and len(self.merges) % 100 == 0:
                print(f"  Merge #{len(self.merges):4d}: {best_pair} → '{merged}'"
                      f"  (freq={best_count:,})")

            # Record the merge rule
            self.merges[best_pair] = merged
            new_id = len(self.vocab)
            self.vocab[new_id] = merged
            self.stoi[merged] = new_id

            # Apply the merge to all words in the corpus
            bigram = re.escape(" ".join(best_pair))
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            word_freq = {
                pattern.sub(merged, word): freq
                for word, freq in word_freq.items()
            }

        print(f"Final vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _apply_merges(self, word: str) -> list[str]:
        """Apply all learned merge rules to a single word."""
        symbols = list(word) + ["</w>"]
        for (a, b), merged in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols = symbols[:i] + [merged] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode(self, text: str) -> list[int]:
        """Convert a string to a list of token IDs."""
        ids = []
        for word in text.split():
            for sym in self._apply_merges(word):
                ids.append(self.stoi.get(sym, 0))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to a string."""
        tokens = [self.vocab.get(i, "?") for i in ids]
        text   = " ".join(tokens)
        text   = text.replace(" </w>", " ").replace("</w>", "")
        # Remove spaces within merged subwords
        text   = re.sub(r" (?=[^<\s])", "", text)
        return text.strip()


tokenizer = BPETokenizer()
tokenizer.train(corpus_small, vocab_size=500, verbose=True)

print(f"\nVocabulary sample (last 20 tokens):")
ids = sorted(tokenizer.vocab.keys())
for i in ids[-20:]:
    print(f"  {i:4d}: {repr(tokenizer.vocab[i])}")


sample_texts = [
    "Once upon a time there was a little girl.",
    "The dog ran quickly through the forest.",
    "She loved to read stories about dragons.",
]

for text in sample_texts:
    ids = tokenizer.encode(text)
    reconstructed = tokenizer.decode(ids)
    print(f"Original : {text}")
    print(f"Token IDs: {ids}")
    print(f"Decoded  : {reconstructed}")
    print()


test_text = corpus_small[:10_000]

# Character-level tokenisation
char_tokens = len(test_text)

# BPE tokenisation
bpe_tokens = len(tokenizer.encode(test_text))

print(f"Test text: {char_tokens:,} characters")
print(f"Char-level tokens: {char_tokens:,}")
print(f"BPE tokens (v=500): {bpe_tokens:,}")
print(f"Compression ratio: {char_tokens / bpe_tokens:.2f}×")


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# Build a BPE tokenizer using the fast Rust-based HuggingFace library
hf_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
hf_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
hf_tokenizer.decoder       = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size     = 2_000,
    min_frequency  = 2,
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    show_progress  = True,
)

# Train on the TinyStories file directly
hf_tokenizer.train([TRAIN_FILE], trainer)

# Save tokenizer
hf_tokenizer.save("../data/tinystories_bpe_tokenizer.json")
print("Tokenizer saved → data/tinystories_bpe_tokenizer.json")

# Encode / decode test
sample = "Once upon a time, there was a brave little rabbit."
encoded = hf_tokenizer.encode(sample)
print(f"\nOriginal : {sample}")
print(f"Token IDs: {encoded.ids}")
print(f"Tokens   : {encoded.tokens}")
print(f"Decoded  : {hf_tokenizer.decode(encoded.ids)}")


from transformers import AutoTokenizer

# GPT-2 uses a 50 257-token BPE vocabulary trained on WebText
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

text = "Once upon a time there was a tiny dragon."
ids  = gpt2_tok.encode(text)
print(f"GPT-2 tokenisation of: {repr(text)}")
print(f"  Tokens  : {gpt2_tok.convert_ids_to_tokens(ids)}")
print(f"  IDs     : {ids}")
print(f"  Chars   : {len(text)}")
print(f"  Tokens  : {len(ids)}")
print(f"  Ratio   : {len(text)/len(ids):.1f}×")
