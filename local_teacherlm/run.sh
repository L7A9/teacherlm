#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$ROOT_DIR/python/local_api"
UI_DIR="$ROOT_DIR/apps/desktop"
VENV_DIR="$ROOT_DIR/.venv"
RUN_DIR="$ROOT_DIR/.teacherlm-run"
PID_DIR="$RUN_DIR/pids"
LOG_DIR="$RUN_DIR/logs"

HOST="127.0.0.1"
API_PORT="8765"
UI_PORT="1420"
API_HEALTH_URL="http://$HOST:$API_PORT/api/health"
UI_URL="http://$HOST:$UI_PORT/"
API_PID_FILE="$PID_DIR/api.pid"
UI_PID_FILE="$PID_DIR/ui.pid"
API_LOG="$LOG_DIR/api.log"
UI_LOG="$LOG_DIR/ui.log"

if [[ "${OS:-}" == "Windows_NT" || "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]; then
  VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
else
  VENV_PYTHON="$VENV_DIR/bin/python"
fi

is_windows() {
  [[ "${OS:-}" == "Windows_NT" || "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ]]
}

to_windows_path() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -w "$1"
  else
    printf '%s\n' "$1"
  fi
}

ps_quote() {
  local value="${1//\'/\'\'}"
  printf "'%s'" "$value"
}

usage() {
  cat <<EOF
TeacherLM local runner

Usage:
  ./run.sh                 Start the API + UI and open the app
  ./run.sh start           Start the API + UI
  ./run.sh stop            Stop processes started by this script
  ./run.sh restart         Stop, then start
  ./run.sh status          Show process and URL status
  ./run.sh logs [api|ui]   Show recent logs
  ./run.sh logs -f         Follow API + UI logs
  ./run.sh install         Install Python and Node dependencies
  ./run.sh build           Typecheck/build the UI and run API tests
  ./run.sh update          Pull latest git changes if possible, install, build, restart
  ./run.sh open            Open $UI_URL

Runtime files:
  Logs: $LOG_DIR
  PIDs: $PID_DIR
EOF
}

info() {
  printf '\033[1;34m==>\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2
}

die() {
  printf '\033[1;31merror:\033[0m %s\n' "$*" >&2
  exit 1
}

ensure_dirs() {
  mkdir -p "$PID_DIR" "$LOG_DIR"
}

find_system_python() {
  local cmd
  for cmd in python3.14 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1 && "$cmd" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 14) else 1)
PY
    then
      printf '%s\n' "$cmd"
      return 0
    fi
  done

  if command -v py >/dev/null 2>&1 && py -3.14 - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 14) else 1)
PY
  then
    printf '%s\n' "py -3.14"
    return 0
  fi

  return 1
}

run_python_cmd() {
  local cmd="$1"
  shift
  if [[ "$cmd" == "py -3.14" ]]; then
    py -3.14 "$@"
  else
    "$cmd" "$@"
  fi
}

ensure_venv() {
  if [[ ! -x "$VENV_PYTHON" ]]; then
    local system_python
    system_python="$(find_system_python)" || die "Python 3.14+ was not found. Install Python 3.14, then run ./run.sh install again."
    info "Creating local Python environment at .venv"
    run_python_cmd "$system_python" -m venv "$VENV_DIR"
  fi

  "$VENV_PYTHON" - <<'PY' >/dev/null
import sys
raise SystemExit(0 if sys.version_info >= (3, 14) else 1)
PY
}

find_npm() {
  if command -v npm.cmd >/dev/null 2>&1; then
    printf '%s\n' "npm.cmd"
  elif command -v npm >/dev/null 2>&1; then
    printf '%s\n' "npm"
  else
    return 1
  fi
}

npm_cmd() {
  find_npm || die "Node.js/npm was not found. Install Node.js 22+ or current LTS, then run ./run.sh install again."
}

install_deps() {
  ensure_dirs
  ensure_venv
  local npm
  npm="$(npm_cmd)"

  info "Installing Python dependencies"
  (
    cd "$API_DIR"
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r requirements.txt
  )

  info "Installing frontend dependencies"
  (
    cd "$UI_DIR"
    "$npm" install
  )
}

ensure_deps_for_start() {
  local needs_install=0
  [[ -x "$VENV_PYTHON" ]] || needs_install=1
  [[ -d "$UI_DIR/node_modules" ]] || needs_install=1

  if [[ "$needs_install" -eq 1 ]]; then
    install_deps
  else
    ensure_venv
    npm_cmd >/dev/null
  fi
}

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  if is_windows; then
    windows_pid_running "$pid"
    return $?
  fi
  kill -0 "$pid" >/dev/null 2>&1
}

windows_pid_running() {
  local pid="$1"
  powershell.exe -NoProfile -Command "if (Get-Process -Id $pid -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1
}

windows_pids_for_port() {
  local port="$1"
  powershell.exe -NoProfile -Command "Get-NetTCPConnection -LocalAddress '$HOST' -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique" 2>/dev/null |
    tr -d '\r' |
    awk 'NF && $1 != "0"'
}

stop_windows_pid_tree() {
  local pid="$1"
  MSYS2_ARG_CONV_EXCL='*' taskkill.exe /PID "$pid" /T >/dev/null 2>&1 || true
  for _ in $(seq 1 2); do
    if ! windows_pid_running "$pid"; then
      return 0
    fi
    sleep 1
  done
  MSYS2_ARG_CONV_EXCL='*' taskkill.exe /PID "$pid" /T /F >/dev/null 2>&1 || true
}

url_ok() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 2 "$url" >/dev/null 2>&1
  else
    "$VENV_PYTHON" - "$url" <<'PY' >/dev/null 2>&1
import sys
from urllib.request import urlopen

try:
    with urlopen(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if 200 <= response.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
  fi
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local log_file="$3"

  for _ in $(seq 1 60); do
    if url_ok "$url"; then
      info "$name is ready: $url"
      return 0
    fi
    sleep 1
  done

  warn "$name did not become ready in time. Recent log:"
  tail -n 80 "$log_file" 2>/dev/null || true
  return 1
}

vite_bin() {
  local unix_bin="$UI_DIR/node_modules/.bin/vite"
  local cmd_bin="$UI_DIR/node_modules/.bin/vite.cmd"
  if [[ -x "$unix_bin" ]]; then
    printf '%s\n' "$unix_bin"
  elif [[ -f "$cmd_bin" ]]; then
    printf '%s\n' "$cmd_bin"
  else
    return 1
  fi
}

start_detached_windows() {
  local cwd="$1"
  local log_file="$2"
  local pid_file="$3"
  local command_line="$4"

  local win_cwd win_log win_pid full_command
  win_cwd="$(to_windows_path "$cwd")"
  win_log="$(to_windows_path "$log_file")"
  win_pid="$(to_windows_path "$pid_file")"
  full_command="cd /d \"$win_cwd\" && $command_line >> \"$win_log\" 2>&1"

  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "\$p = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/d','/s','/c',$(ps_quote "$full_command")) -WindowStyle Hidden -PassThru; Set-Content -Path $(ps_quote "$win_pid") -Value \$p.Id" >/dev/null
}

start_api() {
  ensure_dirs
  if is_running "$API_PID_FILE"; then
    info "API already running (pid $(cat "$API_PID_FILE"))"
    return 0
  fi
  if url_ok "$API_HEALTH_URL"; then
    warn "API already responds on $API_HEALTH_URL, but it was not started by this script."
    return 0
  fi

  info "Starting API on $API_HEALTH_URL"
  : > "$API_LOG"
  if is_windows; then
    local win_python
    win_python="$(to_windows_path "$VENV_PYTHON")"
    start_detached_windows "$API_DIR" "$API_LOG" "$API_PID_FILE" "\"$win_python\" -m uvicorn local_api.main:app --host \"$HOST\" --port \"$API_PORT\""
  else
    (
      cd "$API_DIR"
      nohup "$VENV_PYTHON" -m uvicorn local_api.main:app --host "$HOST" --port "$API_PORT" >> "$API_LOG" 2>&1 &
      echo $! > "$API_PID_FILE"
    )
  fi
  wait_for_url "API" "$API_HEALTH_URL" "$API_LOG"
}

start_ui() {
  ensure_dirs
  if is_running "$UI_PID_FILE"; then
    info "UI already running (pid $(cat "$UI_PID_FILE"))"
    return 0
  fi
  if url_ok "$UI_URL"; then
    warn "UI already responds on $UI_URL, but it was not started by this script."
    return 0
  fi

  local vite
  vite="$(vite_bin)" || die "Vite was not installed. Run ./run.sh install first."

  info "Starting UI on $UI_URL"
  : > "$UI_LOG"
  if is_windows; then
    local win_vite
    win_vite="$(to_windows_path "$vite")"
    start_detached_windows "$UI_DIR" "$UI_LOG" "$UI_PID_FILE" "\"$win_vite\" --host \"$HOST\" --port \"$UI_PORT\""
  else
    (
      cd "$UI_DIR"
      nohup "$vite" --host "$HOST" --port "$UI_PORT" >> "$UI_LOG" 2>&1 &
      echo $! > "$UI_PID_FILE"
    )
  fi
  wait_for_url "UI" "$UI_URL" "$UI_LOG"
}

open_app() {
  info "Opening $UI_URL"
  case "${OSTYPE:-}" in
    darwin*) open "$UI_URL" >/dev/null 2>&1 || true ;;
    msys*|cygwin*) cmd.exe /c start "" "$UI_URL" >/dev/null 2>&1 || true ;;
    *) xdg-open "$UI_URL" >/dev/null 2>&1 || true ;;
  esac
}

start_all() {
  ensure_deps_for_start
  start_api
  start_ui
  open_app
  status
  info "TeacherLM is running in the background. You can close this Bash window and later run ./run.sh stop."
}

stop_one() {
  local name="$1"
  local pid_file="$2"
  local port="$3"

  if is_windows; then
    local stopped_any=0
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(cat "$pid_file" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && windows_pid_running "$pid"; then
        info "Stopping $name (pid $pid)"
        stop_windows_pid_tree "$pid"
        stopped_any=1
      fi
      rm -f "$pid_file"
    fi

    local port_pid
    while IFS= read -r port_pid; do
      [[ -n "$port_pid" ]] || continue
      info "Stopping $name on port $port (pid $port_pid)"
      stop_windows_pid_tree "$port_pid"
      stopped_any=1
    done < <(windows_pids_for_port "$port")

    if [[ "$stopped_any" -eq 0 ]]; then
      info "$name was not running"
    fi
    return 0
  fi

  if ! [[ -f "$pid_file" ]]; then
    info "$name is not managed by this script"
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" >/dev/null 2>&1; then
    rm -f "$pid_file"
    info "$name was not running"
    return 0
  fi

  info "Stopping $name (pid $pid)"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 15); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      rm -f "$pid_file"
      return 0
    fi
    sleep 1
  done

  warn "$name did not stop gracefully; forcing it down"
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pid_file"
}

stop_all() {
  stop_one "UI" "$UI_PID_FILE" "$UI_PORT"
  stop_one "API" "$API_PID_FILE" "$API_PORT"
}

status_line() {
  local name="$1"
  local pid_file="$2"
  local url="$3"
  local port="$4"

  if is_running "$pid_file"; then
    printf '%-4s running  pid=%s' "$name" "$(cat "$pid_file")"
  elif is_windows && [[ -n "$(windows_pids_for_port "$port")" ]]; then
    printf '%-4s running  pid=%s' "$name" "$(windows_pids_for_port "$port" | paste -sd, -)"
  else
    printf '%-4s stopped ' "$name"
  fi

  if url_ok "$url"; then
    printf '  url=ready (%s)\n' "$url"
  else
    printf '  url=not ready (%s)\n' "$url"
  fi
}

status() {
  ensure_dirs
  status_line "API" "$API_PID_FILE" "$API_HEALTH_URL" "$API_PORT"
  status_line "UI" "$UI_PID_FILE" "$UI_URL" "$UI_PORT"
}

show_logs() {
  ensure_dirs
  local target="${1:-all}"
  local follow=0
  if [[ "$target" == "-f" || "$target" == "--follow" ]]; then
    target="all"
    follow=1
  fi

  local files=()
  case "$target" in
    api) files+=("$API_LOG") ;;
    ui) files+=("$UI_LOG") ;;
    all) files+=("$API_LOG" "$UI_LOG") ;;
    *) die "Unknown log target '$target'. Use api, ui, or all." ;;
  esac

  touch "${files[@]}"
  if [[ "$follow" -eq 1 ]]; then
    tail -n 80 -f "${files[@]}"
  else
    tail -n 80 "${files[@]}"
  fi
}

build_all() {
  ensure_deps_for_start
  local npm
  npm="$(npm_cmd)"

  info "Running backend tests"
  (
    cd "$ROOT_DIR"
    "$VENV_PYTHON" -m pytest python/local_api/tests
  )

  info "Building frontend"
  (
    cd "$UI_DIR"
    "$npm" run build
  )
}

maybe_git_pull() {
  if ! git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    warn "Not inside a git worktree; skipping git pull."
    return 0
  fi

  local top
  top="$(git -C "$ROOT_DIR" rev-parse --show-toplevel)"
  if ! git -C "$top" rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    warn "No upstream branch configured; skipping git pull."
    return 0
  fi

  info "Pulling latest git changes"
  if ! git -C "$top" pull --ff-only; then
    warn "git pull failed. Continuing with local files."
  fi
}

update_all() {
  stop_all
  maybe_git_pull
  install_deps
  build_all
  start_all
}

cmd="${1:-start}"
shift || true

case "$cmd" in
  start) start_all ;;
  stop|down) stop_all ;;
  restart) stop_all; start_all ;;
  status) status ;;
  logs) show_logs "${1:-all}" ;;
  install) install_deps ;;
  build|check) build_all ;;
  update|upgrade) update_all ;;
  open) open_app ;;
  help|-h|--help) usage ;;
  *) usage; die "Unknown command '$cmd'" ;;
esac
