# model_utils.py — save this in your project directory

import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_for_inference(
    model_name: str = "gpt2",
    device: str = None,
    dtype: torch.dtype = torch.float16,
):
    """
    Load a HuggingFace model optimised for inference.
    Returns (model, tokenizer).
    """
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

    # Compile for faster inference (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    except Exception:
        pass   # compile not available in all environments

    return model, tokenizer


def generate_tokens(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
) -> list[int]:
    """Run greedy/sampling generation, return list of new token IDs."""
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
    # Only return newly generated tokens (not the prompt)
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:].tolist()
    return new_ids


print("model_utils.py: defined load_model_for_inference and generate_tokens")



import asyncio
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import os

app = FastAPI(title="LLM101n Story Generator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model/tokenizer (loaded once at startup)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt2")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    model = model.to(DEVICE)


class GenerateRequest(BaseModel):
    prompt:         str   = "Once upon a time"
    max_new_tokens: int   = 200
    temperature:    float = 0.8
    top_k:          int   = 40


@app.get("/")
async def index():
    """Serve the frontend HTML page."""
    return HTMLResponse(open("frontend.html").read())


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate text (non-streaming) — returns complete response at once."""
    inputs = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
            top_k          = req.top_k,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )
    new_ids  = outputs[0, inputs["input_ids"].shape[1]:]
    text_out = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {"generated_text": req.prompt + text_out, "tokens": len(new_ids)}


@app.post("/stream")
async def stream_generate(req: GenerateRequest):
    """
    Streaming generation via Server-Sent Events (SSE).
    The client receives tokens as they are produced.
    """
    inputs   = tokenizer(req.prompt, return_tensors="pt").to(DEVICE)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens = req.max_new_tokens,
        temperature    = req.temperature,
        top_k          = req.top_k,
        do_sample      = True,
        pad_token_id   = tokenizer.eos_token_id,
        streamer       = streamer,
    )

    # Run generation in a background thread (non-blocking)
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    async def event_generator():
        for token_text in streamer:
            yield f"data: {json.dumps({'token': token_text})}\n\n"
            await asyncio.sleep(0)   # yield control to event loop
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


import torch
import torch.nn as nn
import os

def load_custom_checkpoint(checkpoint_path: str, model: nn.Module,
                            device: str = "cpu") -> nn.Module:
    """
    Load a locally saved model checkpoint (from Chapter 5).
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the model in Chapter 5 first, or use a HuggingFace model.")
        return model

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# If you saved the Chapter 5 GPT, you can load and serve it like this:
gpt_checkpoint = "../data/gpt_tinystories.pt"
if os.path.exists(gpt_checkpoint):
    print(f"Found Chapter 5 checkpoint: {gpt_checkpoint}")
    print("To serve it, import your GPT class and call load_custom_checkpoint()")
else:
    print("No Chapter 5 checkpoint found — run ch05.md first to create one.")
    print("For now, the server uses GPT-2 from HuggingFace.")


import time
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading GPT-2 for throughput benchmark …")
tok   = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
mdl   = AutoModelForCausalLM.from_pretrained("gpt2").eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
mdl   = mdl.to(device)

prompt = "Once upon a time there was a little"
inputs = tok(prompt, return_tensors="pt").to(device)

N_TOKENS = 100
N_RUNS   = 3

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
