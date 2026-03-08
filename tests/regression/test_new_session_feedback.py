# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Regression tests for /new session feedback.
Requires live daemon: bash scripts/restart-daemon.sh
"""
import requests
import pytest

BASE = "http://localhost:8000/api/v2"

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


def test_new_session_no_handle_returns_no_active_session(project):
    """Backend returns no_active_session when agent has never been started."""
    resp = requests.post(f"{BASE}/agents/{project}/new-session")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_active_session"


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


def test_new_session_with_active_agent_broadcasts_new_session_then_idle(project):
    """When agent IS running, WS must broadcast new_session then idle in order."""
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
    assert resp.json()["status"] == "ok"
    time.sleep(2.0)
    ws.close()

    statuses = [e.get("status") for e in events if e.get("type") == "agent.status"]
    assert "new_session" in statuses
    assert "idle" in statuses
    assert statuses.index("idle") > statuses.index("new_session")
