import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

DATA_DIR = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "tinystories_train.txt")
VAL_FILE   = os.path.join(DATA_DIR, "tinystories_val.txt")


def is_valid_story(text: str) -> bool:
    """Basic quality filter for TinyStories examples."""
    if len(text) < 20 or len(text) > 5000:
        return False
    if not re.search(r"[.!?]", text):
        return False
    n_ascii = sum(c.isascii() for c in text)
    if len(text) == 0 or n_ascii / len(text) < 0.8:
        return False
    return True


def tokenize_and_chunk(examples: dict, tokenizer, block_size: int = 512) -> dict:
    """Tokenise a batch of stories and concatenate into fixed-length blocks."""
    tokens = tokenizer(
        examples["text"],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    flat = []
    for ids in tokens:
        flat.extend(ids + [tokenizer.eos_token_id])
    input_ids = [flat[i : i + block_size]
                 for i in range(0, len(flat) - block_size, block_size)]
    return {"input_ids": input_ids}


class TinyStoriesDataset(Dataset):
    """
    PyTorch Dataset that reads a text file and returns (input, target) pairs
    for causal language modelling using character-level tokenization.
    Target is input shifted by one.
    """

    def __init__(self, filepath: str, block_size: int = 512):
        self.block_size = block_size
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # Character-level tokenization using ord values
        ids = [ord(c) % 256 for c in text]
        # Split into blocks of block_size + 1
        self.blocks = []
        for i in range(0, len(ids) - block_size, block_size):
            chunk = ids[i : i + block_size + 1]
            self.blocks.append(chunk)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        chunk = self.blocks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


def collate_fn(batch, pad_id: int = 0):
    """Pad variable-length sequences in a batch to the same length."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xs_padded = torch.stack([
        torch.cat([x, torch.full((max_len - x.size(0),), pad_id, dtype=torch.long)]) for x in xs
    ])
    ys_padded = torch.stack([
        torch.cat([y, torch.full((max_len - y.size(0),), -100, dtype=torch.long)]) for y in ys
    ])
    return xs_padded, ys_padded


class StreamingTinyStories(IterableDataset):
    def __init__(self, filepath, tokenizer, block_size=512):
        self.filepath   = filepath
        self.tokenizer  = tokenizer
        self.block_size = block_size
        self.buffer     = []

    def __iter__(self):
        with open(self.filepath, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
                self.buffer.extend(ids)
                while len(self.buffer) >= self.block_size + 1:
                    chunk = self.buffer[:self.block_size + 1]
                    self.buffer = self.buffer[self.block_size:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:],  dtype=torch.long)
                    yield x, y


if __name__ == "__main__":
    from datasets import load_dataset, DatasetDict, Dataset as HFDataset

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading TinyStories from local files ...")
    with open(TRAIN_FILE, encoding="utf-8") as f:
        train_stories = [{"text": line.strip()} for line in f if line.strip()]
    with open(VAL_FILE, encoding="utf-8") as f:
        val_stories = [{"text": line.strip()} for line in f if line.strip()]

    ds = DatasetDict({
        "train":      HFDataset.from_list(train_stories),
        "validation": HFDataset.from_list(val_stories),
    })

    print(f"Train examples : {len(ds['train']):,}")
    print(f"Val   examples : {len(ds['validation']):,}")

    train_ds = TinyStoriesDataset(TRAIN_FILE, block_size=512)
    print(f"Train blocks: {len(train_ds):,}")

    xb, yb = train_ds[0]
    print(f"x shape: {xb.shape}, y shape: {yb.shape}")

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    xb, yb = next(iter(loader))
    print(f"Batch x shape: {xb.shape}  dtype: {xb.dtype}")

    before = len(ds['train'])
    filtered = [ex for ex in ds['train'] if is_valid_story(ex['text'])]
    after = len(filtered)
    print(f"Before filtering: {before:,}  |  After: {after:,}")
