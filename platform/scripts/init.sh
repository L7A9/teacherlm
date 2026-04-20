#!/usr/bin/env bash
# Initialize a freshly-started TeacherLM stack:
#   1. Pull Ollama models required by teacher_gen (host-side Ollama).
#   2. Create the MinIO bucket.
#   3. Apply the latest database migrations.
#
# Run from the repo root or `platform/` directory after `docker compose up -d`.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(cd "${HERE}/.." && pwd)"

# Load the shared .env so we can read OLLAMA_HOST / MinIO credentials.
if [[ -f "${PLATFORM_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${PLATFORM_DIR}/.env"
  set +a
fi

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
MINIO_ENDPOINT="${MINIO_ENDPOINT:-localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"
MINIO_BUCKET="${MINIO_BUCKET:-teacherlm}"

log() { printf '\n\033[1;34m[init]\033[0m %s\n' "$*"; }

# -------- 1. Ollama models --------
log "Pulling Ollama models (host must be running Ollama at ${OLLAMA_HOST})."
if ! command -v ollama >/dev/null 2>&1; then
  echo "  ollama CLI not found on PATH — install from https://ollama.com and re-run." >&2
  exit 1
fi

for model in "llama3.1:8b" "nomic-embed-text"; do
  log "ollama pull ${model}"
  OLLAMA_HOST="${OLLAMA_HOST}" ollama pull "${model}"
done

# -------- 2. MinIO bucket --------
log "Ensuring MinIO bucket '${MINIO_BUCKET}' exists."
docker run --rm --network "platform_default" \
  --entrypoint sh \
  minio/mc:latest -c "
    mc alias set local http://minio:9000 '${MINIO_ACCESS_KEY}' '${MINIO_SECRET_KEY}' >/dev/null
    mc mb --ignore-existing local/${MINIO_BUCKET}
  " || {
    # Fallback: hit the host-exposed MinIO if the compose network name differs.
    log "Network 'platform_default' not reachable — falling back to host endpoint ${MINIO_ENDPOINT}."
    docker run --rm --entrypoint sh minio/mc:latest -c "
      mc alias set host http://${MINIO_ENDPOINT} '${MINIO_ACCESS_KEY}' '${MINIO_SECRET_KEY}' >/dev/null
      mc mb --ignore-existing host/${MINIO_BUCKET}
    "
  }

# -------- 3. Alembic migrations --------
log "Running alembic migrations inside the backend container."
docker compose -f "${PLATFORM_DIR}/docker-compose.yml" exec -T backend alembic upgrade head

log "Done. Visit http://localhost:${FRONTEND_PORT:-3000} to start using TeacherLM."
