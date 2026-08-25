# ── Backend Dockerfile (CPU) ──────────────────────────────────────────────
# For GPU support, use Dockerfile.gpu or the docker-compose.gpu.yml override.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OpenCV, git (for pip installs), and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "torch==2.6.0" "torchvision==0.21.0" \
        "transformers==4.48.3" "tokenizers==0.21.0" "sentencepiece==0.1.99" \
        shortuuid "accelerate==1.6.0" "peft>=0.10.0,<0.14.0" \
        pydantic "markdown2[all]" "numpy==1.26.4" "scikit-learn==1.2.2" \
        requests uvicorn fastapi \
        "einops==0.6.1" "einops-exts==0.0.4" "timm==1.0.15" \
        opencv-python-headless onnxruntime python-dotenv openai Pillow \
        yt-dlp \
        "typing_extensions>=4.8" \
        websockets

# Copy application source
COPY . .

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8000

# Default: run the backend server
CMD ["python", "-m", "backend.server", \
     "--model-path", "checkpoints/llava-fastvithd_0.5b_stage3", \
     "--host", "0.0.0.0", "--port", "8000"]
