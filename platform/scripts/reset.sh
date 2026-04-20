#!/usr/bin/env bash
# Destroy ALL TeacherLM data: Postgres tables, Qdrant collections, MinIO
# objects, Redis queues, and the compose volumes backing them.
#
# This is destructive. It will prompt for confirmation unless --yes is passed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM_DIR="$(cd "${HERE}/.." && pwd)"
COMPOSE_FILE="${PLATFORM_DIR}/docker-compose.yml"

log() { printf '\n\033[1;31m[reset]\033[0m %s\n' "$*"; }

if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "This will wipe all TeacherLM data (Postgres, Qdrant, MinIO, Redis). Continue? [y/N] " answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
fi

log "Stopping and removing containers + volumes."
docker compose -f "${COMPOSE_FILE}" down --volumes --remove-orphans

log "Pruning any dangling named volumes for this project (safe — only this project's)."
for vol in postgres_data qdrant_data minio_data; do
  # Compose prefixes named volumes with the project name (the directory name).
  project="$(basename "${PLATFORM_DIR}")"
  full="${project}_${vol}"
  if docker volume inspect "${full}" >/dev/null 2>&1; then
    docker volume rm "${full}"
    echo "  removed ${full}"
  fi
done

log "Done. Run 'docker compose -f ${COMPOSE_FILE} up -d' and then ./scripts/init.sh to bootstrap again."
