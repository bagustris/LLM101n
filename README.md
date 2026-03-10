# LLM101n: Let's build a Storyteller

![LLM101n header image](llm101n.jpg)

> What I cannot create, I do not understand. — Richard Feynman

In this course we build a Storyteller AI Large Language Model (LLM) from scratch — from a one-line bigram model all the way to a deployed, multimodal web app. Everything is implemented end-to-end in Python with minimal prerequisites. By the end you will have a deep, hands-on understanding of how modern LLMs work.

The training corpus throughout is [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — a dataset of short children's stories — keeping experiments fast enough to run on a laptop while still producing meaningful results.

---

## Syllabus

| # | Chapter | Key concepts |
|---|---------|--------------|
| 01 | **Bigram Language Model** | language modeling, NLL loss, character-level tokenization |
| 02 | **Micrograd** | scalar autodiff, backpropagation from scratch |
| 03 | **N-gram MLP** | multi-layer perceptron, matmul, GELU |
| 04 | **Attention** | self-attention, softmax, positional encoding |
| 05 | **Transformer** | GPT-2 architecture, residual connections, LayerNorm |
| 06 | **Tokenization** | Byte Pair Encoding (BPE), minBPE |
| 07 | **Optimization** | weight initialization, AdamW, LR schedules |
| 08 | **Need for Speed I: Device** | CPU vs GPU, device-agnostic PyTorch |
| 09 | **Need for Speed II: Precision** | mixed precision, fp16, bf16, fp8 |
| 10 | **Need for Speed III: Distributed** | DDP, ZeRO, DeepSpeed |
| 11 | **Datasets** | data loading, synthetic data generation |
| 12 | **Inference I: KV-Cache** | key-value cache, autoregressive generation |
| 13 | **Inference II: Quantization** | INT8/INT4 quantization |
| 14 | **Finetuning I: SFT** | supervised finetuning, PEFT, LoRA, chat format |
| 15 | **Finetuning II: RL** | RLHF, PPO, DPO |
| 16 | **Deployment** | FastAPI server, streaming, web UI |
| 17 | **Multimodal** | VQVAE, diffusion transformer, image+text |

---

## Repository Layout

```
LLM101n/
├── chNN.md          # Chapter narratives + embedded Python code (source of truth)
├── codes/
│   ├── extract.py   # Extracts Python blocks from chNN.md → codes/chNN/main.py
│   ├── chNN/
│   │   ├── main.py  # Auto-generated runnable script (do not edit manually)
│   │   └── run.log  # Expected output
│   └── data/        # Shared datasets, checkpoints, tokenizers
└── llm101n.jpg
```

> **Markdown is the source of truth.** `codes/chNN/main.py` files are generated artifacts — always edit the markdown chapter and re-run `extract.py`.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A GPU is helpful but not required for the early chapters

### Setup

```bash
git clone https://github.com/karpathy/LLM101n.git
cd LLM101n

# Create and activate the virtual environment
python -m venv codes/.venv
source codes/.venv/bin/activate   # Windows: codes\.venv\Scripts\activate

pip install torch datasets transformers tqdm fastapi uvicorn
```

### Extracting code from the markdown chapters

```bash
cd codes
python extract.py   # writes codes/chNN/main.py for all 17 chapters
```

### Running a chapter

```bash
source codes/.venv/bin/activate
cd codes/ch01
python main.py
```

Each chapter is self-contained. Chapter 01 downloads the TinyStories dataset on first run and saves it to `codes/data/` so subsequent chapters can reuse it without hitting the network again.

---

## Shared Data (`codes/data/`)

| File / Directory | Description |
|-----------------|-------------|
| `tinystories_train.txt` | 50 K training stories |
| `tinystories_val.txt` | 5 K validation stories |
| `gpt_tinystories.pt` | Pretrained GPT-2 checkpoint on TinyStories |
| `tinystories_bpe_tokenizer.json` | BPE tokenizer (128-token vocab) |
| `lora_adapter/` | Saved LoRA adapter (rank=8, alpha=16) |
| `vqvae_cifar10.pt` | Pretrained VQVAE for CIFAR-10 (ch17) |
| `cifar-10-batches-py/` | CIFAR-10 image dataset (ch17) |
| `server.py` | FastAPI streaming text-generation server (ch16) |
| `frontend.html` | Web UI (ch16) |
| `Dockerfile` | Container for the deployed server (ch16) |
| `ds_config.json` | DeepSpeed config for distributed training (ch10) |

---

## Appendix — Topics to Explore Further

- **Programming languages:** Assembly, C, Python internals
- **Data types:** Integer, Float, String (ASCII, Unicode, UTF-8)
- **Tensors:** shapes, views, strides, contiguous memory
- **Frameworks:** PyTorch, JAX
- **Architectures:** GPT-1/2/3/4, Llama (RoPE, RMSNorm, GQA), Mixture-of-Experts
- **Multimodal:** Images, Audio, Video, VQVAE, VQGAN, Diffusion models
