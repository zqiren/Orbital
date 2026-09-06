#!/usr/bin/env bash
# i18n-editor.sh — boot the translation editors.
#
# Starts an isolated dev daemon (own PID file, scratch data dir outside ~/Desktop,
# in-memory keyring) and the Vite dev server proxied to it, then prints the URLs:
#
#   bash scripts/i18n-editor.sh           # reuse the scratch data dir
#   bash scripts/i18n-editor.sh --fresh   # wipe it first: start at onboarding
#   bash scripts/i18n-editor.sh --stop    # stop what this script started
#
# Env overrides: PORT (daemon, default 8331), VITE_PORT (default 5173),
# DATA_DIR (default ~/.orbital-i18n-editor/data).
#
# Edits land as plain files under docs/i18n/pending/ (see CLAUDE.md → i18n);
# nothing is applied to strings.ts or the READMEs until someone folds them in.
# API keys pasted into the UI live only for this daemon's lifetime.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-8331}"
VITE_PORT="${VITE_PORT:-5173}"
STATE="$HOME/.orbital-i18n-editor"
DATA="${DATA_DIR:-$STATE/data}"
PY="$ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY=python3
mkdir -p "$STATE"

stop_pidfile() {
  local f="$1"
  if [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null; then
    kill "$(cat "$f")" 2>/dev/null || true
  fi
  rm -f "$f"
}

wait_for() { # url seconds
  local i
  for i in $(seq 1 $(( $2 * 2 ))); do
    if [[ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' "$1")" == "200" ]]; then return 0; fi
    sleep 0.5
  done
  return 1
}

if [[ "${1:-}" == "--stop" ]]; then
  stop_pidfile "$STATE/vite.pid"
  stop_pidfile "$STATE/daemon.pid"
  echo "stopped"
  exit 0
fi

if [[ "${1:-}" == "--fresh" ]]; then
  echo "Wiping $DATA"
  rm -rf "$DATA"
fi
mkdir -p "$DATA"

# Restart whatever this script started last time; leave other daemons alone.
stop_pidfile "$STATE/vite.pid"
stop_pidfile "$STATE/daemon.pid"
sleep 1
for p in "$PORT" "$VITE_PORT"; do
  if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $p is in use by another process:"; lsof -nP -iTCP:"$p" -sTCP:LISTEN | tail -n +2
    echo "Pick another with PORT=... / VITE_PORT=... or stop it first."
    exit 1
  fi
done

echo "[1/2] Daemon on :$PORT (data: $DATA)"
AGENT_OS_TELEMETRY_DISABLED=1 nohup "$PY" "$ROOT/scripts/dev_daemon_isolated.py" \
  --port "$PORT" --data-dir "$DATA" > "$STATE/daemon.log" 2>&1 &
echo $! > "$STATE/daemon.pid"
wait_for "http://127.0.0.1:$PORT/api/v2/settings" 30 || { echo "daemon did not come up — see $STATE/daemon.log"; tail -20 "$STATE/daemon.log"; exit 1; }

echo "[2/2] Vite on :$VITE_PORT (proxy → :$PORT)"
( cd "$ROOT/web" && exec env ORBITAL_DEV_API_PORT="$PORT" ./node_modules/.bin/vite --host 0.0.0.0 --port "$VITE_PORT" ) > "$STATE/vite.log" 2>&1 &
echo $! > "$STATE/vite.pid"
wait_for "http://127.0.0.1:$VITE_PORT/__i18n/overrides" 30 || { echo "vite did not come up — see $STATE/vite.log"; tail -20 "$STATE/vite.log"; exit 1; }

cat <<MSG

  UI translation editor : http://127.0.0.1:$VITE_PORT/?i18n=edit
  README proposal editor: http://127.0.0.1:$VITE_PORT/__i18n/readme

  Edits land in docs/i18n/pending/ (ui-overrides.json, README.md, README.en.md).
  Logs: $STATE/daemon.log, $STATE/vite.log.   Stop: bash scripts/i18n-editor.sh --stop
MSG
