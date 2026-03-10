import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

DATA_DIR = "../data"


def format_alpaca(example: dict) -> str:
    """Format an Alpaca example into a prompt-response string."""
    if example.get("input", "").strip():
        prompt = (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. Write a response "
            "that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            "### Response:\n"
        )
    return prompt + example["output"] + "\n\n### End"


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA weights."""

    def __init__(self, in_features: int, out_features: int,
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.scaling      = alpha / rank

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora * self.scaling

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        lora = cls(linear.in_features, linear.out_features, rank=rank, alpha=alpha)
        lora.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora.bias.data.copy_(linear.bias.data)
        return lora

    def trainable_params(self) -> list:
        return [self.lora_A, self.lora_B]

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scaling={self.scaling:.3f}"


def inject_lora(model: nn.Module, target_modules: set,
                rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Replace target nn.Linear layers with LoRALinear. Freeze all others."""
    for param in model.parameters():
        param.requires_grad = False

    for name, module in list(model.named_modules()):
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model if not parent_name else dict(model.named_modules())[parent_name]
                lora_layer = LoRALinear.from_linear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = True

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
    return model


class TinyGPT(nn.Module):
    """Minimal GPT-style model for SFT demonstration."""

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        B, T = x.shape
        device = x.device
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        for layer in self.layers:
            normed = layer["ln1"](h)
            attn_out, _ = layer["attn"](normed, normed, normed, attn_mask=mask)
            h = h + attn_out
            h = h + layer["ffn"](layer["ln2"](h))
        h = self.ln_f(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss


SYSTEM_PROMPT = "You are a helpful assistant that tells creative children's stories."


def make_chat_prompt(user_message: str) -> str:
    """Format a user message using a simple ChatML-style template."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{user_message}\n"
        f"<|assistant|>\n"
    )


if __name__ == "__main__":
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading Alpaca instruction dataset ...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"Dataset size: {len(alpaca):,} examples")

    for i in range(2):
        print(f"--- Example {i+1} ---")
        print(format_alpaca(alpaca[i])[:400])
        print()

    print("\nLoading GPT-2 with PEFT LoRA ...")
    model_name = "gpt2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = 8,
        lora_alpha    = 16,
        lora_dropout  = 0.05,
        target_modules= ["c_attn", "c_proj"],
        bias          = "none",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_model = peft_model.to(device)
    optim = torch.optim.AdamW(peft_model.parameters(), lr=2e-4, weight_decay=0.0)

    texts = [format_alpaca(alpaca[i]) for i in range(8)]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    peft_model.train()
    outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    print(f"\nSFT loss (first batch): {loss.item():.4f}")

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
    optim.step()

    peft_model.save_pretrained("../data/lora_adapter")
    print("LoRA adapter saved -> data/lora_adapter/")

    prompt = make_chat_prompt("Tell me a short story about a dragon.")
    print(prompt)
    print(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
