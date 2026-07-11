# ============================================================
# Hybrid Token-Efficient Routing Agent — Dockerfile
# AMD Developer Hackathon: ACT II — Track 1
# ============================================================

# Stage 1: Builder (compiles llama-cpp-python and downloads model)
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Install dependencies into venv
RUN pip install --no-cache-dir -r requirements.txt

# Download the model using huggingface-cli
RUN mkdir -p /models && \
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir /models --local-dir-use-symlinks False

# Stage 2: Final
FROM python:3.11-slim

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the downloaded model from builder
COPY --from=builder /models/qwen2.5-3b-instruct-q4_k_m.gguf /app/models/model.gguf

# Copy application code
COPY agent/ ./agent/
COPY eval/ ./eval/
COPY run.py .
COPY main.py .

# Create mount points
RUN mkdir -p /input /output

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO \
    CACHE_ENABLED=true \
    LOCAL_MODEL_PATH=/app/models/model.gguf

# Health check (verify Python and imports work)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from agent.router import HeuristicRouter; print('OK')"

# Container entry point
ENTRYPOINT ["python", "run.py"]
