# Dockerfile for LLM101n Story Generator
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    transformers fastapi uvicorn[standard] pydantic

# Copy server files
COPY data/server.py   ./server.py
COPY data/frontend.html ./frontend.html

# Expose the API port
EXPOSE 8000

# Download model at container build time (optional — can also lazy-load)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('gpt2'); \
    AutoModelForCausalLM.from_pretrained('gpt2')"

ENV MODEL_NAME=gpt2

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
