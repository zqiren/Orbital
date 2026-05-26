# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Regression tests for /new session feedback.
Requires live daemon: bash scripts/restart-daemon.sh

The third test ``test_new_session_with_active_agent_broadcasts_new_session_then_idle``
opens a WebSocket against the live daemon and depends on the daemon accepting
an unauthenticated handshake from the test runner. On dev machines where the
daemon is configured for normal use (auth required, restricted origin), the
handshake returns 403 and the test fails. Marked to require an explicit live
daemon test setup — opt in by setting ORBITAL_LIVE_DAEMON_TESTS=1.
"""
import os
import requests
import pytest

BASE = "http://localhost:8000/api/v2"

_requires_test_daemon = pytest.mark.skipif(
    os.environ.get("ORBITAL_LIVE_DAEMON_TESTS") != "1",
    reason="needs a live daemon configured for tests; set ORBITAL_LIVE_DAEMON_TESTS=1",
)

@pytest.fixture
def project(tmp_path):
    """Create a real project via API."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    resp = requests.post(f"{BASE}/projects", json={
        "name": "feedback-test",
        "workspace": str(ws),
        "model": "claude-sonnet-4-20250514",
        "api_key": "test-key",
    })
    assert resp.status_code == 201
    pid = resp.json()["project_id"]
    yield pid
    requests.delete(f"{BASE}/projects/{pid}")


@_requires_test_daemon
def test_new_session_returns_fresh_id_even_with_no_handle(project):
    """new_session is pure-create: it mints and returns a fresh session_id
    even when no agent has ever been started (no handle required — the session
    materializes on the first message)."""
    resp = requests.post(f"{BASE}/agents/{project}/new-session")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["session_id"].startswith("sess_")
    assert body["session_uuid"]


@_requires_test_daemon
def test_new_session_no_handle_does_not_broadcast_ws_event(project):
    """No WS event should be broadcast when there's no active session —
    frontend feedback must come from API response, not WS."""
    import websocket, threading, time, json
    events = []
    def collect(ws, msg): events.append(json.loads(msg))
    ws = websocket.WebSocketApp(
        f"ws://localhost:8000/api/v2/ws/{project}",
        on_message=collect
    )
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    requests.post(f"{BASE}/agents/{project}/new-session")
    time.sleep(1.0)
    ws.close()
    status_events = [e for e in events if e.get("type") == "agent.status"]
    assert not any(e.get("status") == "new_session" for e in status_events)


@_requires_test_daemon
def test_new_session_with_active_agent_is_pure_create(project):
    """Even while an agent is running, new_session is pure-create: it returns
    a fresh session_id and takes NO action on the running session — so no
    rotation status (no 'new_session' broadcast) fires for it. The running
    session keeps its slot; the new session materializes on its first
    message."""
    import websocket, threading, time, json

    # Start the agent
    requests.post(f"{BASE}/agents/start", json={"project_id": project})
    time.sleep(1.0)

    events = []
    def collect(ws, msg): events.append(json.loads(msg))
    ws = websocket.WebSocketApp(
        f"ws://localhost:8000/api/v2/ws/{project}",
        on_message=collect
    )
    t = threading.Thread(target=ws.run_forever, daemon=True)
    t.start()
    time.sleep(0.2)

    resp = requests.post(f"{BASE}/agents/{project}/new-session")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["session_id"].startswith("sess_")
    time.sleep(2.0)
    ws.close()

    statuses = [e.get("status") for e in events if e.get("type") == "agent.status"]
    # Pure-create does not rotate, so no 'new_session' status is broadcast.
    assert "new_session" not in statuses
