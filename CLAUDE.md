# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM101n: Let's build a Storyteller** — an educational course building a storytelling LLM from scratch. Each of the 17 chapters is a standalone markdown file (`chNN.md`) containing prose and embedded Python code blocks. These are extracted into runnable scripts under `codes/chNN/main.py`.

## Running Code

The virtual environment is at `codes/.venv/`. Activate it before running scripts:

```bash
source codes/.venv/bin/activate
```

Run any chapter's script from its directory (paths are relative to the chapter folder):

```bash
cd codes/ch01 && python main.py
```

Shared data (TinyStories dataset, model checkpoints, tokenizers) lives in `codes/data/` and is referenced as `../data/` from within each chapter directory.

## Extracting Code from Markdown

When a chapter markdown file is updated, re-extract the code with:

```bash
cd codes && python extract.py
```

`extract.py` scans `ch01.md`–`ch17.md` at the repo root, concatenates all ` ```python ` blocks from each file, rewrites `"data"` path references to `"../data"`, and writes to `codes/chNN/main.py`.

## Architecture

### Chapter Structure
Each chapter is self-contained:
- `chNN.md` — narrative + embedded Python code blocks (source of truth)
- `codes/chNN/main.py` — auto-extracted runnable script (do not manually edit; re-extract from markdown)
- `codes/chNN/run.log` — expected output from running the script

### Shared Data (`codes/data/`)
- `tinystories_train.txt` / `tinystories_val.txt` — 50K/5K children's stories (primary training corpus)
- `gpt_tinystories.pt` — pretrained GPT-2 checkpoint on TinyStories (used in later chapters)
- `tinystories_bpe_tokenizer.json` — BPE tokenizer (128 vocab)
- `lora_adapter/` — saved LoRA adapter (rank=8, alpha=16, targets: c_attn, c_proj)
- `vqvae_cifar10.pt` — pretrained VQVAE for CIFAR-10 (ch17)
- `cifar-10-batches-py/` — CIFAR-10 image dataset (ch17)

### Deployment (`codes/data/`)
- `server.py` — FastAPI server with streaming text generation endpoint
- `frontend.html` — web UI for the storyteller
- `Dockerfile` — containerizes the server
- `ds_config.json` — DeepSpeed config for distributed training (ch10)

### Chapter Progression
| Ch | Topic |
|----|-------|
| 01 | Bigram language model |
| 02 | Micrograd (scalar autodiff / backprop from scratch) |
| 03 | N-gram MLP |
| 04 | Attention, softmax, positional encoding |
| 05 | Full GPT-2 transformer |
| 06 | BPE tokenization |
| 07 | Optimization (AdamW, LR schedules) |
| 08 | Device (CPU/GPU) |
| 09 | Mixed precision (fp16/bf16) |
| 10 | Distributed training (DDP, ZeRO) |
| 11 | Datasets and synthetic data generation |
| 12 | KV-cache inference |
| 13 | Quantization |
| 14 | SFT + LoRA finetuning |
| 15 | RLHF (PPO, DPO) |
| 16 | Deployment (FastAPI web app) |
| 17 | Multimodal (VQVAE + diffusion transformer) |

## Key Conventions

- **Markdown is the source of truth.** `main.py` files are generated artifacts. When fixing bugs in chapter code, edit the markdown and re-run `extract.py`.
- All chapters share `codes/data/`; paths in scripts are always `../data/` relative to the chapter dir.
- No `requirements.txt` exists — the `.venv` at `codes/.venv/` has all dependencies installed.
