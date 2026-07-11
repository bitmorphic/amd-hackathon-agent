# ============================================================
# Hybrid Token-Efficient Routing Agent — Dockerfile
# AMD Developer Hackathon: ACT II — Track 1
# ============================================================
# Single-stage slim build — openai + pydantic + dotenv
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
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
    CACHE_ENABLED=true

# Health check (verify Python and imports work)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from agent.router import HeuristicRouter; print('OK')"

# Container entry point
ENTRYPOINT ["python", "run.py"]
