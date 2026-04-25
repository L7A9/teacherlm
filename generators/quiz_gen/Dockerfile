# Build from the teacherlm/ repo root:
#   docker build -f generators/quiz_gen/Dockerfile -t teacherlm/quiz_gen:latest .
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl rustc cargo \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own cached layer. Source copy comes after
# so iterating on quiz_gen/*.py doesn't re-trigger pip install.
COPY packages/teacherlm_core /app/packages/teacherlm_core
RUN pip install --upgrade pip \
    && pip install -e /app/packages/teacherlm_core \
    && pip install \
        "fastapi>=0.135.0" \
        "uvicorn[standard]>=0.32.0" \
        "ollama>=0.4.0" \
        "pydantic>=2.12.0" \
        "pydantic-settings>=2.6.0" \
        "sse-starlette>=2.1.0" \
        "httpx>=0.27.0" \
        "fastembed>=0.4.0" \
        "minio>=7.2.0" \
        "numpy>=1.26.0"

COPY generators/quiz_gen /app/generators/quiz_gen

ENV PYTHONPATH=/app/generators \
    QUIZ_GEN_HOST=0.0.0.0 \
    QUIZ_GEN_PORT=8002 \
    HF_HOME=/opt/hf_cache

# Pre-download embedding model so first /run doesn't stall on HF Hub.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8002/health || exit 1

CMD ["uvicorn", "quiz_gen.app:app", "--host", "0.0.0.0", "--port", "8002"]
