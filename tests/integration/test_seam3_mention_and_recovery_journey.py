# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration (seam 3, Roots A + B): real FastAPI TestClient over an isolated
in-process daemon, exercising the cross-boundary route → sub_agent_manager →
session path.

A. @mention journey: POST /inject with a target reaches a (simulated) running
   sub-agent and returns success — not the seam-3 404 "No active session".
B. Session-less recovery poll: GET /pending-approval with no session_id returns
   200 {"pending": false}, not the seam-3 500 from the over-migrated resolver.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
os.environ.setdefault("AGENT_OS_API_KEY", "test-key")

from agent_os.api.app import create_app
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.models import make_session_key


class _FakeTransport:
    """Minimal SDK-like transport: supports non-blocking dispatch."""
    def __init__(self):
        self.dispatched = None

    async def dispatch(self, message):
        self.dispatched = message


class _FakeAdapter:
    """A 'running' sub-agent adapter without a real subprocess."""
    def __init__(self):
        self._transport = _FakeTransport()
        self._idle = False

    def is_alive(self):
        return True

    def is_idle(self):
        return False


@pytest.fixture
def client_and_project():
    data_dir = tempfile.mkdtemp(prefix="seam3_it_data_")
    workspace = tempfile.mkdtemp(prefix="seam3_it_ws_")
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        r = client.post("/api/v2/projects", json={
            "name": "seam3-it", "workspace": workspace,
            "model": "gpt-4o-mini", "api_key": "test-key",
        })
        assert r.status_code == 201, r.text
        pid = r.json()["project_id"]
        yield client, pid


def test_mention_journey_reaches_running_sub_agent(client_and_project):
    client, pid = client_and_project
    sid = "seam3_it_sess_1"
    # Simulate a running sub-agent attached to session `sid`.
    sam = agents_v2._sub_agent_manager
    sam._adapters[make_session_key(pid, sid)] = {"researcher": _FakeAdapter()}

    resp = client.post(
        f"/api/v2/agents/{pid}/inject",
        json={"content": "hello", "target": "researcher", "session_id": sid},
    )

    assert resp.status_code == 200, resp.text
    # The route reached send() (not the seam-3 resolver crash) and dispatched.
    assert "Message sent to researcher" in resp.json()["status"]
    assert sam._adapters[make_session_key(pid, sid)]["researcher"]._transport.dispatched == "hello"


def test_session_less_pending_approval_poll_returns_200_not_500(client_and_project):
    client, pid = client_and_project

    # No session_id query param: the over-migrated resolver used to 500 here.
    resp = client.get(f"/api/v2/agents/{pid}/pending-approval")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"pending": False}
