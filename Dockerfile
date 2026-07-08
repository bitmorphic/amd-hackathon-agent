# ============================================================
# Hybrid Token-Efficient Routing Agent — Dockerfile
# AMD Developer Hackathon: ACT II — Track 1
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): Install Python deps
#   Stage 2 (runtime): Slim image with only what's needed
#
# Build:
#   docker build -t amd-routing-agent .
#
# Run (hackathon evaluation):
#   docker run -v ./input:/input -v ./output:/output \
#     -e FIREWORKS_API_KEY=... \
#     -e FIREWORKS_BASE_URL=... \
#     -e ALLOWED_MODELS=... \
#     amd-routing-agent
#
# Run (local dev):
#   docker run --env-file .env \
#     -v ./eval:/input \
#     -v ./output:/output \
#     amd-routing-agent
# ============================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --------------- Stage 2: Runtime ---------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (no .env — harness injects env vars)
COPY agent/ ./agent/
COPY eval/ ./eval/
COPY run.py .
COPY main.py .
COPY requirements.txt .

# Create output directory and input mount point
RUN mkdir -p /input /output

# Default environment variables (can be overridden at runtime)
# NOTE: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS
# are injected by the hackathon harness — do NOT set them here.
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    LOCAL_MODEL_NAME=google/gemma-2-2b-it \
    LOCAL_MODEL_DEVICE=auto \
    ROUTER_COMPLEXITY_THRESHOLD=0.6 \
    ROUTER_CONFIDENCE_FALLBACK_THRESHOLD=0.2 \
    CACHE_ENABLED=true \
    COMPRESSION_ENABLED=true

# Health check — verify Python and imports work
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from agent.router import HeuristicRouter; print('OK')"

# Container entry point — reads /input/tasks.json, writes /output/results.json
ENTRYPOINT ["python", "run.py"]
