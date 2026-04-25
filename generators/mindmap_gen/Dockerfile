# Build from the teacherlm/ repo root:
#   docker build -f generators/mindmap_gen/Dockerfile -t teacherlm/mindmap_gen:latest .
#
# Note: teacherlm_core's pyproject.toml requires fastembed + qdrant-client
# + numpy, but mindmap_gen never imports those modules — it only uses
# teacherlm_core.schemas, .llm, and .prompts. We install teacherlm_core with
# --no-deps and bring in only the runtime packages this generator actually
# needs, which avoids a multi-hundred-MB onnxruntime download per build.
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# build-essential covers any C-extension fallbacks (e.g. pydantic-core if no
# 3.14 wheel is published yet). curl is for the HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps in their own cached layer so iterating on source
# doesn't re-trigger pip install.
RUN pip install --upgrade pip \
    && pip install \
        "fastapi>=0.135.0" \
        "uvicorn[standard]>=0.32.0" \
        "ollama>=0.4.0" \
        "pydantic>=2.12.0" \
        "pydantic-settings>=2.6.0" \
        "sse-starlette>=2.1.0" \
        "httpx>=0.28.0" \
        "jinja2>=3.1.0"

# Install teacherlm_core WITHOUT its heavy ML deps. The runtime deps it
# actually exercises (pydantic, pydantic-settings, ollama, httpx) were
# installed above.
COPY packages/teacherlm_core /app/packages/teacherlm_core
RUN pip install --no-deps -e /app/packages/teacherlm_core

COPY generators/mindmap_gen /app/generators/mindmap_gen

ENV PYTHONPATH=/app/generators \
    MINDMAP_GEN_HOST=0.0.0.0 \
    MINDMAP_GEN_PORT=8008

RUN mkdir -p /app/generators/mindmap_gen/artifacts

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8008/health || exit 1

WORKDIR /app/generators/mindmap_gen

CMD ["uvicorn", "mindmap_gen.app:app", "--host", "0.0.0.0", "--port", "8008"]
