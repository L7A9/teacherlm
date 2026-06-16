#!/usr/bin/env bash
# TeacherLM — start / stop / logs / rebuild the whole stack.
#
# Usage:
#   ./run.sh              # start everything and expose it on your LAN
#   ./run.sh up           # same as above
#   ./run.sh build        # rebuild images, start, and expose it on your LAN
#   ./run.sh rebuild      # force rebuild all images, then start
#   ./run.sh stop         # stop containers (keeps volumes)
#   ./run.sh down         # stop + remove containers (keeps volumes)
#   ./run.sh logs [svc]   # tail logs (all services or a specific one)
#   ./run.sh ps           # show container status
#   ./run.sh shell <svc>  # open a shell in a service container
#   ./run.sh report-charts # run report retrieval tests and regenerate chart SVG/PNG files

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE="docker compose -f $ROOT/platform/docker-compose.yml"

detect_lan_host() {
  if [ -n "${TEACHERLM_LAN_HOST:-}" ]; then
    printf '%s\n' "$TEACHERLM_LAN_HOST"
    return
  fi

  local ip=""

  if command -v ip >/dev/null 2>&1; then
    ip="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") { print $(i + 1); exit }}' || true)"
  fi

  if [ -z "$ip" ] && command -v hostname >/dev/null 2>&1; then
    ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^([0-9]{1,3}\.){3}[0-9]{1,3}$/ && $0 !~ /^127\./ { print; exit }' || true)"
  fi

  if [ -z "$ip" ] && command -v powershell.exe >/dev/null 2>&1; then
    ip="$(
      powershell.exe -NoProfile -Command 'Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } | Sort-Object InterfaceMetric | Select-Object -First 1 -ExpandProperty IPAddress' 2>/dev/null \
        | tr -d '\r' \
        | awk 'NF { print; exit }' \
        || true
    )"
  fi

  if [ -z "$ip" ]; then
    ip="localhost"
  fi

  printf '%s\n' "$ip"
}

prepare_lan_exports() {
  FRONTEND_PORT="${FRONTEND_PORT:-3000}"
  BACKEND_PORT="${BACKEND_PORT:-8000}"
  MINIO_PORT="${MINIO_PORT:-9000}"
  MINDMAP_GEN_PORT="${MINDMAP_GEN_PORT:-8008}"
  TEACHERLM_LAN_HOST_RESOLVED="$(detect_lan_host)"

  export FRONTEND_PORT BACKEND_PORT MINIO_PORT MINDMAP_GEN_PORT TEACHERLM_LAN_HOST_RESOLVED
  export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://$TEACHERLM_LAN_HOST_RESOLVED:$BACKEND_PORT}"
  export MINIO_PUBLIC_ENDPOINT="${MINIO_PUBLIC_ENDPOINT:-$TEACHERLM_LAN_HOST_RESOLVED:$MINIO_PORT}"
  export MINDMAP_GEN_PUBLIC_URL="${MINDMAP_GEN_PUBLIC_URL:-http://$TEACHERLM_LAN_HOST_RESOLVED:$MINDMAP_GEN_PORT}"

  echo "LAN host:      $TEACHERLM_LAN_HOST_RESOLVED"
  echo "Frontend API:  $NEXT_PUBLIC_API_BASE_URL"
  echo
}

print_stack_urls() {
  local host="${TEACHERLM_LAN_HOST_RESOLVED:-$(detect_lan_host)}"

  echo "Frontend:     http://localhost:${FRONTEND_PORT:-3000}"
  echo "Phone URL:    http://$host:${FRONTEND_PORT:-3000}"
  echo "Backend API:  http://$host:${BACKEND_PORT:-8000}/api/health"
  echo "teacher_gen:  http://$host:${TEACHER_GEN_PORT:-8001}/health"
  echo "quiz_gen:     http://$host:${QUIZ_GEN_PORT:-8002}/health"
  echo "podcast_gen:  http://$host:${PODCAST_GEN_PORT:-8007}/health"
  echo "mindmap_gen:  http://$host:${MINDMAP_GEN_PORT:-8008}/health"
  echo
  echo "Open the Phone URL on a device connected to the same Wi-Fi."
  echo "If it cannot connect, allow Docker/these ports through the firewall."
}

python_bin() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    printf '%s\n' "$PYTHON_BIN"
    return
  fi
  if command -v python >/dev/null 2>&1; then
    printf '%s\n' "python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
    return
  fi
  echo "python was not found. Install Python or set PYTHON_BIN=/path/to/python." >&2
  exit 1
}

run_report_charts() {
  local py
  py="$(python_bin)"
  local eval_dir="$ROOT/platform/backend/evals"

  echo "Starting backend dependencies if needed..."
  $COMPOSE up -d postgres qdrant backend

  echo "Copying report benchmark inputs into the backend container..."
  $COMPOSE cp "$ROOT/platform/backend/scripts/compare_retrieval_variants.py" \
    backend:/app/platform/backend/scripts/compare_retrieval_variants.py
  $COMPOSE cp "$eval_dir/current_mobile_rag_eval.json" \
    backend:/app/platform/backend/evals/current_mobile_rag_eval.json
  $COMPOSE cp "$eval_dir/current_mobile_exact_rag_eval.json" \
    backend:/app/platform/backend/evals/current_mobile_exact_rag_eval.json

  echo "Running mixed retrieval benchmark..."
  if ! $COMPOSE exec -T backend python scripts/compare_retrieval_variants.py \
      evals/current_mobile_rag_eval.json \
      --k-values 5 \
      --out evals/retrieval_variant_comparison.json \
      --csv-out evals/retrieval_variant_comparison.csv \
      --mermaid-out evals/retrieval_variant_chart.mmd \
      > "$eval_dir/retrieval_variant_comparison.log" 2>&1; then
    tail -n 80 "$eval_dir/retrieval_variant_comparison.log" >&2 || true
    return 1
  fi

  echo "Running exact-term retrieval benchmark..."
  if ! $COMPOSE exec -T backend python scripts/compare_retrieval_variants.py \
      evals/current_mobile_exact_rag_eval.json \
      --k-values 5 \
      --chart-title "TeacherLM Exact-Term Retrieval Comparison" \
      --out evals/retrieval_variant_exact_comparison.json \
      --csv-out evals/retrieval_variant_exact_comparison.csv \
      --mermaid-out evals/retrieval_variant_exact_chart.mmd \
      > "$eval_dir/retrieval_variant_exact_comparison.log" 2>&1; then
    tail -n 80 "$eval_dir/retrieval_variant_exact_comparison.log" >&2 || true
    return 1
  fi

  echo "Copying benchmark outputs back into the workspace..."
  $COMPOSE cp backend:/app/platform/backend/evals/retrieval_variant_comparison.json \
    "$eval_dir/retrieval_variant_comparison.json"
  $COMPOSE cp backend:/app/platform/backend/evals/retrieval_variant_comparison.csv \
    "$eval_dir/retrieval_variant_comparison.csv"
  $COMPOSE cp backend:/app/platform/backend/evals/retrieval_variant_chart.mmd \
    "$eval_dir/retrieval_variant_chart.mmd"
  $COMPOSE cp backend:/app/platform/backend/evals/retrieval_variant_exact_comparison.json \
    "$eval_dir/retrieval_variant_exact_comparison.json"
  $COMPOSE cp backend:/app/platform/backend/evals/retrieval_variant_exact_comparison.csv \
    "$eval_dir/retrieval_variant_exact_comparison.csv"
  $COMPOSE cp backend:/app/platform/backend/evals/retrieval_variant_exact_chart.mmd \
    "$eval_dir/retrieval_variant_exact_chart.mmd"

  echo "Rebuilding the report-charts notebook..."
  "$py" "$eval_dir/build_teacherlm_report_charts_notebook.py"

  echo "Executing notebook chart cells to generate SVG and PNG files..."
  (cd "$ROOT" && "$py" - <<'PY'
import json
from pathlib import Path

notebook_path = Path("platform/backend/evals/teacherlm_report_charts.ipynb")
nb = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace = {}

for index, cell in enumerate(nb["cells"], start=1):
    if cell.get("cell_type") != "code":
        continue
    source = "".join(cell.get("source", []))
    exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)

print("Notebook chart execution complete.")
PY
  )

  echo
  echo "Report charts generated:"
  echo "  $eval_dir/report_charts"
  echo
  ls -1 "$eval_dir/report_charts"
}

cmd="${1:-up}"
shift || true

case "$cmd" in
  up)
    prepare_lan_exports
    $COMPOSE build frontend
    $COMPOSE up -d --remove-orphans
    $COMPOSE ps
    echo
    print_stack_urls
    ;;
  build)
    prepare_lan_exports
    $COMPOSE up -d --build --remove-orphans
    $COMPOSE ps
    echo
    print_stack_urls
    ;;
  rebuild)
    prepare_lan_exports
    $COMPOSE build --no-cache "$@"
    $COMPOSE up -d
    $COMPOSE ps
    echo
    print_stack_urls
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
  report-charts|charts)
    run_report_charts
    ;;
  *)
    echo "unknown command: $cmd" >&2
    echo "usage: ./run.sh [up|build|rebuild|stop|down|logs|ps|shell|report-charts|charts]" >&2
    exit 1
    ;;
esac
