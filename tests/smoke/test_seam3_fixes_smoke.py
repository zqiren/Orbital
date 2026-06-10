# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke tests (seam 3, Roots A–D) against an isolated in-process daemon.

Each is a single end-to-end assertion on the live served API (or live manager
for the one path with no HTTP surface). Throwaway temp data/workspace dirs —
never the user's real Orbital data, no port bound.
"""

import os
import tempfile

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

os.environ.setdefault("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
os.environ.setdefault("AGENT_OS_API_KEY", "test-key")

from agent_os.api.app import create_app
from agent_os.api.routes import agents_v2
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


def _running_handle():
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = False  # live loop → holds the slot
    return h


@pytest.fixture
def daemon():
    data_dir = tempfile.mkdtemp(prefix="seam3_smoke_data_")
    workspace = tempfile.mkdtemp(prefix="seam3_smoke_ws_")
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        r = client.post("/api/v2/projects", json={
            "name": "seam3-smoke", "workspace": workspace,
            "model": "gpt-4o-mini", "api_key": "test-key",
        })
        assert r.status_code == 201, r.text
        yield client, r.json()["project_id"]


# Root A — @mention inject to a running sub-agent returns 2xx (not the 404).
def test_smoke_a_mention_inject_returns_2xx(daemon):
    client, pid = daemon
    sid = "smoke_a_sess"
    agents_v2._sub_agent_manager._adapters[make_session_key(pid, sid)] = {
        "researcher": _FakeAdapter(),
    }
    resp = client.post(
        f"/api/v2/agents/{pid}/inject",
        json={"content": "hi", "target": "researcher", "session_id": sid},
    )
    assert resp.status_code == 200, resp.text


# Root B — session-less pending-approval poll returns 200 {"pending": false} (not 500).
def test_smoke_b_pending_approval_session_less_200(daemon):
    client, pid = daemon
    resp = client.get(f"/api/v2/agents/{pid}/pending-approval")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"pending": False}


# Root C — deleting a running project returns clean 2xx and the loop is stopped.
def test_smoke_c_delete_running_project_clean_2xx(daemon, monkeypatch):
    client, pid = daemon
    am = agents_v2._agent_manager
    sid = pid + "_smoke_c"
    am._handles[make_session_key(pid, sid)] = _running_handle()
    assert am.is_running(pid) is True

    stopped = []

    async def spy_stop(project_id, *, session_id=None):
        stopped.append(session_id)
        am._handles.pop(make_session_key(project_id, session_id), None)

    # Avoid real loop teardown of the mock handle; assert the route forwards the
    # holder (the actual Root C fix) and the project ends up not running.
    monkeypatch.setattr(am, "stop_agent", spy_stop)

    resp = client.delete(f"/api/v2/projects/{pid}")
    assert resp.status_code == 200, resp.text
    assert stopped == [sid]              # holder forwarded, not a (pid, None) miss
    assert am.is_running(pid) is False   # no orphaned loop


# Root D — a corrective-turn inject WITH the session id is delivered (the
# dispatcher now forwards it); WITHOUT it the message is dropped ("no_session").
# (The dispatcher's rotation path has no HTTP surface, so this asserts the
# delivery semantics on the live manager; the dispatcher forwarding itself is
# covered by tests/regression/test_dispatcher_corrective_turn_session_id.py.)
async def test_smoke_d_corrective_inject_delivers_with_session():
    data_dir = tempfile.mkdtemp(prefix="seam3_smoke_d_")
    create_app(data_dir=data_dir)  # wires the live managers into the route module
    am = agents_v2._agent_manager
    pid, sid = "proj_smoke_d", "proj_smoke_d_sess"
    am._handles[make_session_key(pid, sid)] = _running_handle()

    delivered = await am.inject_system_message(pid, "[corrective]", session_id=sid)
    assert delivered == "deferred"  # routed to the session, not dropped
    am._handles[make_session_key(pid, sid)].session.defer_message.assert_called_once()

    dropped = await am.inject_system_message(pid, "[corrective2]", session_id=None)
    assert dropped == "no_session"  # the bug shape: session-less → silently dropped
