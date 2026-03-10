import torch
import torch.nn as nn
import os
from pydantic import BaseModel


def load_model_for_inference(model_name: str = "gpt2", device: str = None,
                              dtype: torch.dtype = torch.float16):
    """Load a HuggingFace model optimised for inference. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} ({dtype}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    if not torch.cuda.is_available():
        model = model.to(device)
    return model, tokenizer


def generate_tokens(model, tokenizer, prompt: str, max_new_tokens: int = 200,
                    temperature: float = 0.8, top_k: int = 40,
                    top_p: float = 0.95) -> list:
    """Run sampling generation, return list of new token IDs."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens   = max_new_tokens,
            temperature      = temperature,
            top_k            = top_k,
            top_p            = top_p,
            do_sample        = True,
            pad_token_id     = tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:].tolist()
    return new_ids


class GenerateRequest(BaseModel):
    prompt:         str   = "Once upon a time"
    max_new_tokens: int   = 200
    temperature:    float = 0.8
    top_k:          int   = 40


def load_custom_checkpoint(checkpoint_path: str, model: nn.Module,
                            device: str = "cpu") -> nn.Module:
    """Load a locally saved model checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return model
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# FastAPI app (importable without running the server)
try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="LLM101n Story Generator", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate_endpoint(req: GenerateRequest):
        return {"generated_text": req.prompt, "tokens": 0}

except ImportError:
    print("FastAPI not available — skipping app definition")
    app = None


if __name__ == "__main__":
    import time
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs("../data", exist_ok=True)

    # Save server code
    SERVER_CODE = '''
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

app = FastAPI()
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()

class GenerateRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40

@app.post("/generate")
async def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=req.max_new_tokens,
                                 temperature=req.temperature, top_k=req.top_k,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}
'''
    with open("../data/server.py", "w") as f:
        f.write(SERVER_CODE)
    print("Saved → data/server.py")
    print("Run with: cd data && uvicorn server:app --host 0.0.0.0 --port 8000")

    gpt_checkpoint = "../data/gpt_tinystories.pt"
    if os.path.exists(gpt_checkpoint):
        print(f"Found Chapter 5 checkpoint: {gpt_checkpoint}")
    else:
        print("No Chapter 5 checkpoint found.")

    print("Loading GPT-2 for throughput benchmark …")
    tok   = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    mdl   = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl   = mdl.to(device)

    prompt = "Once upon a time there was a little"
    inputs = tok(prompt, return_tensors="pt").to(device)
    N_TOKENS = 50
    N_RUNS   = 2

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=N_TOKENS,
                               do_sample=False, pad_token_id=tok.eos_token_id)
        if device == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_s = sum(times) / N_RUNS
    print(f"Generated {N_TOKENS} tokens in {avg_s:.2f}s")
    print(f"Throughput: {N_TOKENS / avg_s:.1f} tokens/sec")

    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
