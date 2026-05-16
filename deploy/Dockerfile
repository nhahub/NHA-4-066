# ─────────────────────────────────────────────────────────────────────────────
#  Customer Support RAG API — production image
#
#  Build (from project root):
#      docker build -f deploy/Dockerfile -t support-rag-api:latest .
#
#  Run locally (needs Mongo + Ollama reachable from the container):
#      docker run --rm -p 8000:8000 \
#          -e API_KEY=devkey \
#          -e MONGO_URI=mongodb://host.docker.internal:27017 \
#          -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
#          support-rag-api:latest
#
#  The image bundles the API and the BGE embedding code. It does NOT
#  bundle Ollama / Mistral — those run as a sibling container (see
#  deploy/docker-compose.full.yml) so the API image stays small.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder — install Python deps into a venv ───────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# System deps required by torch / transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Project pins torch==2.4.1; CPU-only wheel keeps the image manageable.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip install \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.6 \
        gunicorn==22.0.0


# ── Stage 2: runtime — slim image with venv + app only ───────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    LOG_LEVEL=INFO \
    FORCE_CPU=1

# Non-root user (ACA + most managed platforms require this).
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Copy only what the API needs at runtime.
COPY config/ ./config/
COPY src/ ./src/
COPY api/ ./api/
COPY data/preprocessed/ ./data/preprocessed/

# Pre-download the BGE-base model into the image so first request is
# fast and the container doesn't need internet at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5')"

RUN chown -R app:app /app
USER app

EXPOSE 8000

# Healthcheck hits /health (no auth required there).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"

# 2 workers default; tune via WORKERS env. Each worker loads its own
# BGE — for memory-constrained envs (≤2 vCPU) keep WORKERS=1.
ENV WORKERS=2
CMD ["sh", "-c", "gunicorn api.main:app \
        --workers ${WORKERS} \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind 0.0.0.0:8000 \
        --timeout 180 \
        --access-logfile -"]
