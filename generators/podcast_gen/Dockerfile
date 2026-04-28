# Build from the teacherlm/ repo root:
#   docker build -f generators/podcast_gen/Dockerfile -t teacherlm/podcast_gen:latest .
#
# pydub shells out to ffmpeg at runtime, and both pyttsx3's espeak driver
# AND piper-tts's phonemizer need espeak-ng / libespeak1 on Linux.
# kokoro-onnx + piper-tts wheels are pure-Python + onnxruntime, no system
# deps beyond what python:slim already has.
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ffmpeg \
        espeak-ng \
        libespeak1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own cached layer.
COPY packages/teacherlm_core /app/packages/teacherlm_core
RUN pip install --upgrade pip \
    && pip install --no-deps -e /app/packages/teacherlm_core \
    && pip install \
        "fastapi>=0.135.0" \
        "uvicorn[standard]>=0.32.0" \
        "ollama>=0.4.0" \
        "pydantic>=2.12.0" \
        "pydantic-settings>=2.6.0" \
        "sse-starlette>=2.1.0" \
        "httpx>=0.28.0" \
        "piper-tts>=1.2.0" \
        "kokoro-onnx>=0.4.0" \
        "onnxruntime>=1.20.0" \
        "pyttsx3>=2.99" \
        "pydub>=0.25.1" \
        "audioop-lts>=0.2.1" \
        "soundfile>=0.13.0" \
        "numpy>=1.26.0" \
        "nltk>=3.9.1" \
        "langdetect>=1.0.9" \
        "minio>=7.2.0"

# Pre-fetch the NLTK punkt tokenizer so first /run doesn't stall on download.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY generators/podcast_gen /app/generators/podcast_gen

ENV PYTHONPATH=/app/generators \
    PODCAST_GEN_HOST=0.0.0.0 \
    PODCAST_GEN_PORT=8007

RUN mkdir -p /app/generators/podcast_gen/artifacts \
    /app/generators/podcast_gen/models \
    /app/generators/podcast_gen/models/piper

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8007/health || exit 1

WORKDIR /app/generators/podcast_gen

CMD ["uvicorn", "podcast_gen.app:app", "--host", "0.0.0.0", "--port", "8007"]
