# ============================================================
# Hybrid Token-Efficient Routing Agent — Dockerfile
# AMD Developer Hackathon: ACT II — Track 1
# ============================================================

# Stage 1: Builder (compiles llama-cpp-python and downloads model)
FROM python:3.11-slim AS builder

WORKDIR /build

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# Install dependencies into venv
RUN pip install --no-cache-dir -r requirements.txt



# Stage 2: Final
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies for llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

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
