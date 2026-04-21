#!/usr/bin/env bash
# TeacherLM — start / stop / logs / rebuild the whole stack.
#
# Usage:
#   ./run.sh              # start everything (builds only if needed)
#   ./run.sh up           # same as above
#   ./run.sh rebuild      # force rebuild all images, then start
#   ./run.sh stop         # stop containers (keeps volumes)
#   ./run.sh down         # stop + remove containers (keeps volumes)
#   ./run.sh logs [svc]   # tail logs (all services or a specific one)
#   ./run.sh ps           # show container status
#   ./run.sh shell <svc>  # open a shell in a service container

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $ROOT/platform/docker-compose.yml"

cmd="${1:-up}"
shift || true

case "$cmd" in
  up)
    $COMPOSE up -d --build
    $COMPOSE ps
    echo
    echo "Frontend:     http://localhost:3000"
    echo "Backend API:  http://localhost:8000/api/health"
    echo "teacher_gen:  http://localhost:8001/health"
    echo "quiz_gen:     http://localhost:8002/health"
    echo "flashcards_gen: http://localhost:8005/health"
    ;;
  rebuild)
    $COMPOSE build --no-cache "$@"
    $COMPOSE up -d
    $COMPOSE ps
    ;;
  stop)
    $COMPOSE stop
    ;;
  down)
    $COMPOSE down
    ;;
  logs)
    $COMPOSE logs -f --tail=100 "$@"
    ;;
  ps)
    $COMPOSE ps
    ;;
  shell)
    svc="${1:?usage: ./run.sh shell <service>}"
    $COMPOSE exec "$svc" sh
    ;;
  *)
    echo "unknown command: $cmd" >&2
    echo "usage: ./run.sh [up|rebuild|stop|down|logs|ps|shell]" >&2
    exit 1
    ;;
esac
