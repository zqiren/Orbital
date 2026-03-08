#!/usr/bin/env bash
# restart-daemon.sh — Kill existing daemon, start fresh with new code, verify it's up.
# Usage: bash scripts/restart-daemon.sh [port]
#   port defaults to 8000

set -e

PORT="${1:-8000}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Restarting AgentOS daemon on port $PORT ==="

# 1. Kill any existing uvicorn/daemon processes on this port
echo "[1/4] Stopping existing daemon..."
if command -v tasklist &>/dev/null; then
    # Windows: find and kill python processes running uvicorn on our port
    for pid in $(tasklist //FO CSV 2>/dev/null | grep -i python | cut -d'"' -f2 | grep -o '[0-9]*'); do
        # Check if this PID is our daemon (listening on the port)
        if netstat -ano 2>/dev/null | grep ":$PORT " | grep "$pid" &>/dev/null; then
            echo "  Killing PID $pid"
            taskkill //F //PID "$pid" 2>/dev/null || true
        fi
    done
else
    # Unix-like
    pkill -f "uvicorn.*$PORT" 2>/dev/null || true
fi
sleep 2

# 2. Verify port is free
if curl -s --connect-timeout 2 "http://127.0.0.1:$PORT/api/v2/settings" &>/dev/null; then
    echo "  WARNING: Port $PORT still in use, force-killing all uvicorn..."
    if command -v taskkill &>/dev/null; then
        taskkill //F //IM python.exe 2>/dev/null || true
    else
        pkill -9 -f uvicorn 2>/dev/null || true
    fi
    sleep 3
fi

# 3. Start new daemon
echo "[2/4] Starting daemon from $PROJECT_ROOT ..."
cd "$PROJECT_ROOT"
python -m uvicorn agent_os.api.app:create_app --factory --port "$PORT" --host 0.0.0.0 &
DAEMON_PID=$!
echo "  Daemon PID: $DAEMON_PID"

# 4. Wait for startup
echo "[3/4] Waiting for daemon to be ready..."
for i in $(seq 1 15); do
    if curl -s --connect-timeout 1 "http://127.0.0.1:$PORT/api/v2/settings" &>/dev/null; then
        echo "  Daemon ready after ${i}s"
        break
    fi
    if [ "$i" -eq 15 ]; then
        echo "  ERROR: Daemon failed to start within 15s"
        exit 1
    fi
    sleep 1
done

# 5. Verify response
echo "[4/4] Verifying..."
RESPONSE=$(curl -s "http://127.0.0.1:$PORT/api/v2/settings" 2>&1)
if echo "$RESPONSE" | grep -q '"llm"'; then
    echo "  OK: Settings endpoint responding"
else
    echo "  ERROR: Unexpected response: $RESPONSE"
    exit 1
fi

echo ""
echo "=== Daemon running on http://127.0.0.1:$PORT (PID $DAEMON_PID) ==="
echo "    Frontend: http://127.0.0.1:5173 (start with: cd web && npx vite --host 127.0.0.1 --port 5173)"
