# Build context: the `teacherlm/` repo root (see docker-compose.yml).
# This lets us install `packages/teacherlm_core` as an editable sibling dep.

FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps: libpq for asyncpg wheels, build tools for any source-only deps
# (tiktoken, fastembed's tokenizers, etc.), curl for healthchecks.
# Rust is required for py-rust-stemmers (fastembed dependency) to build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libpq-dev \
        rustc \
        cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the shared core first so its install is cached independently.
COPY packages/teacherlm_core /app/packages/teacherlm_core
RUN pip install -e /app/packages/teacherlm_core

# Backend dependencies next (separate layer for cache reuse).
COPY platform/backend/requirements.txt /app/platform/backend/requirements.txt
RUN pip install -r /app/platform/backend/requirements.txt

# Fastembed/HF cache lives on a named volume mounted by docker-compose so the
# embedding model (~80 MB) persists across container rebuilds — the Docker
# build sandbox on this host can't reach HF Hub, so we download lazily on
# first use from within the running container and keep it across recreations.
ENV HF_HOME=/opt/hf_cache \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

# Copy the rest of the backend source.
COPY platform/backend /app/platform/backend
COPY generators_registry.json /app/generators_registry.json

WORKDIR /app/platform/backend

EXPOSE 8000

# Default command starts the FastAPI app. The arq worker service overrides this.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
