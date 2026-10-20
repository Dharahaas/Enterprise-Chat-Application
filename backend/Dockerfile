# ============================================================
# Enterprise Knowledge Assistant — Backend Dockerfile
# Multi-stage build: install CPU-only PyTorch, then app deps
# ============================================================

FROM python:3.11-slim AS base

# Set environment vars for unbuffered Python + CPU-only torch
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_OFFLINE=0

# System deps for PDF parsing + chromadb native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Stage 2: dependencies ----------
# Install CPU-only PyTorch first (smaller), then the rest of requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 3: app code ----------
COPY . .

# Create data directory for Chroma DB (mounted as volume in production)
RUN mkdir -p /app/data/chroma_db /app/data/uploads

# Expose FastAPI port
EXPOSE 8000

# Health check via /api/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run with uvicorn workers (use --workers 4 in prod behind a load balancer)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
