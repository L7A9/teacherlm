# Build from the teacherlm/ repo root:
#   docker build -f generators/teacher_gen/Dockerfile -t teacherlm/teacher_gen:latest .
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl rustc cargo \
    && rm -rf /var/lib/apt/lists/*

COPY packages/teacherlm_core /app/packages/teacherlm_core
COPY generators/teacher_gen /app/generators/teacher_gen

RUN pip install --upgrade pip \
    && pip install -e /app/packages/teacherlm_core \
    && pip install \
        "fastapi>=0.135.0" \
        "uvicorn[standard]>=0.32.0" \
        "ollama>=0.4.0" \
        "pydantic>=2.12.0" \
        "pydantic-settings>=2.6.0" \
        "httpx>=0.27.0"

ENV PYTHONPATH=/app/generators \
    TEACHER_GEN_HOST=0.0.0.0 \
    TEACHER_GEN_PORT=8001 \
    HF_HOME=/opt/hf_cache

# Pre-download the cross-encoder reranker so first /run doesn't stall on HF Hub.
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='BAAI/bge-reranker-base')"

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "teacher_gen.app:app", "--host", "0.0.0.0", "--port", "8001"]
