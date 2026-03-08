#!/usr/bin/env bash
# health-check.sh — SessionStart hook: output environment health summary.
#
# Called automatically by Claude Code when a session starts.
# SessionStart hooks do NOT receive stdin — this script only produces stdout.
# The output is injected as context into the Claude session.
#
# ALWAYS exits 0 — must never block session start.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"

# Fallback if PROJECT_ROOT resolution fails
if [ -z "$PROJECT_ROOT" ] || [ ! -d "$PROJECT_ROOT" ]; then
    PROJECT_ROOT="D:/AgentOS"
fi

STALE_FILE="$PROJECT_ROOT/.claude/.daemon-stale"
TEST_STATUS_FILE="$PROJECT_ROOT/.claude/.test-status"

# Collect issues to report at the end
ISSUES=""

# --- 1. Check daemon on port 8000 ---
if curl -s --connect-timeout 2 "http://127.0.0.1:8000/api/v2/settings" > /dev/null 2>&1; then
    DAEMON_STATUS="running"
else
    DAEMON_STATUS="not running"
    ISSUES="${ISSUES}\n- Daemon is not running. Start with: bash scripts/restart-daemon.sh"
fi

# --- 2. Check daemon code freshness ---
if [ -f "$STALE_FILE" ]; then
    DAEMON_CODE="STALE"
    STALE_INFO="$(cat "$STALE_FILE" 2>/dev/null)"
    ISSUES="${ISSUES}\n- Daemon code is stale (${STALE_INFO}). Restart with: bash scripts/restart-daemon.sh"
else
    DAEMON_CODE="fresh"
fi

# --- 3. Check frontend on port 5173 ---
if curl -s --connect-timeout 2 "http://127.0.0.1:5173" > /dev/null 2>&1; then
    FRONTEND_STATUS="running"
else
    FRONTEND_STATUS="not running"
    ISSUES="${ISSUES}\n- Frontend is not running. Start with: cd web && npx vite --host 127.0.0.1 --port 5173"
fi

# --- 4. Check last test status ---
if [ -f "$TEST_STATUS_FILE" ]; then
    # Read the last line (most recent test result)
    LAST_LINE="$(tail -1 "$TEST_STATUS_FILE" 2>/dev/null)"
    if [ -n "$LAST_LINE" ]; then
        # Format: <pass|fail> <timestamp> <test_type>
        TEST_RESULT="$(echo "$LAST_LINE" | awk '{print $1}')"
        TEST_TIMESTAMP="$(echo "$LAST_LINE" | awk '{print $2}')"
        TEST_TYPE="$(echo "$LAST_LINE" | awk '{print $3}')"

        if [ "$TEST_RESULT" = "pass" ]; then
            LAST_TEST="pass ${TEST_TIMESTAMP} (${TEST_TYPE})"
        elif [ "$TEST_RESULT" = "fail" ]; then
            LAST_TEST="fail ${TEST_TIMESTAMP} (${TEST_TYPE})"
            ISSUES="${ISSUES}\n- Last test run FAILED (${TEST_TYPE} at ${TEST_TIMESTAMP}). Fix before committing."
        else
            LAST_TEST="${TEST_RESULT} ${TEST_TIMESTAMP} (${TEST_TYPE})"
        fi
    else
        LAST_TEST="no results"
    fi
else
    LAST_TEST="no results"
fi

# --- Output summary ---
echo "=== AgentOS Environment Health ==="
echo "Daemon (port 8000): ${DAEMON_STATUS}"
echo "Daemon code: ${DAEMON_CODE}"
echo "Frontend (port 5173): ${FRONTEND_STATUS}"
echo "Last test run: ${LAST_TEST}"

# --- Output issues if any ---
if [ -n "$ISSUES" ]; then
    echo ""
    echo "Issues:"
    echo -e "$ISSUES"
fi

# ALWAYS exit 0 — never block session start
exit 0
