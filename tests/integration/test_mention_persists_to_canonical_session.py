# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration (seam 3 / D1): a real in-process daemon @mention journey.

POST /inject with a target must (1) dispatch to the (simulated) running
sub-agent AND (2) persist the authored user message into the SAME concrete chat
session — never a fabricated ``subagent_<hex>`` log — so the turn is in the
session's history on reload via GET /chat.

Red on current main: the route persists via ``_get_or_create_session`` ->
fabricated ``subagent_<hex>`` session, so the authored turn is missing from the
real session and a stray ``subagent_*.jsonl`` appears.
"""

import glob
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
os.environ.setdefault("AGENT_OS_API_KEY", "test-key")

from agent_os.api.app import create_app
from agent_os.api.routes import agents_v2
from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.models import make_session_key


class _FakeTransport:
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
def client_project_workspace():
    data_dir = tempfile.mkdtemp(prefix="mention_it_data_")
    workspace = tempfile.mkdtemp(prefix="mention_it_ws_")
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        r = client.post("/api/v2/projects", json={
            "name": "mention-it", "workspace": workspace,
            "model": "gpt-4o-mini", "api_key": "test-key",
        })
        assert r.status_code == 201, r.text
        pid = r.json()["project_id"]
        yield client, pid, workspace


def test_mention_persists_to_dispatched_session_and_survives_reload(client_project_workspace):
    client, pid, workspace = client_project_workspace
    sid = "mention_it_sess_1"
    # Simulate a running sub-agent attached to session `sid`.
    sam = agents_v2._sub_agent_manager
    sam._adapters[make_session_key(pid, sid)] = {"claude-code": _FakeAdapter()}

    content = "how does it enforce memory use?"
    resp = client.post(
        f"/api/v2/agents/{pid}/inject",
        json={"content": content, "target": "claude-code", "session_id": sid},
    )
    assert resp.status_code == 200, resp.text
    # Dispatch reached the running sub-agent.
    assert sam._adapters[make_session_key(pid, sid)]["claude-code"]._transport.dispatched == content

    sessions_dir = ProjectPaths(workspace).sessions_dir
    # No fabricated subagent_<hex> session was minted.
    assert glob.glob(os.path.join(sessions_dir, "subagent_*.jsonl")) == []
    # The authored turn landed in the dispatched session's own JSONL.
    sid_path = os.path.join(sessions_dir, f"{sid}.jsonl")
    assert os.path.isfile(sid_path), f"authored turn missing from {sid}.jsonl"

    # On reload, the authored turn is in that session's history.
    reload = client.get(f"/api/v2/agents/{pid}/chat", params={"session_id": sid})
    assert reload.status_code == 200, reload.text
    assert content in reload.text
