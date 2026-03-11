
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


_HERE = os.path.dirname(os.path.abspath(__file__))

@app.get("/")
async def index():
    """Serve the frontend HTML page."""
    frontend_path = os.path.join(_HERE, "frontend.html")
    return HTMLResponse(open(frontend_path).read())


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
