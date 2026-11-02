# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A failed pinned dispatch reaches the open chat live (Windows bench
2026-10-19).

The inject route persists a ``sub_agent_dispatch_failed`` system row and
answers 404. The 404 only carries the composer's generic "failed to send", so
the translated failure row (e.g. "claude-code is too old…") appeared only after
a reload. The route now broadcasts the session-scoped ``sub_agent.failed``
after the row is on disk — the same persist-then-broadcast pattern
LifecycleObserver uses for terminal rows — so ChatView's terminal handler
refetches the history. The 404 contract is unchanged.

Setup mirrors tests/unit/test_spec074_routes.py (real app via create_app,
sub-agent manager stubbed at send()).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agent_os.agent.session import Session


FAIL_TEXT = ("Error: claude-code 2.0.1 is older than the 2.1.0 Orbital needs. "
             "Update it and try again.")


class _FailingSubAgentManager:
    async def send(self, project_id, handle, message, **kwargs):
        return FAIL_TEXT

    def last_start_failure_meta(self, project_id, handle):
        return {"reason": "update_needed", "min_version": "2.1.0",
                "version": "2.0.1"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)
    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


def _routes_mod():
    from agent_os.api.routes import agents_v2 as routes_mod
    return routes_mod


def _make_project(client, tmp_path, name):
    ws = str(tmp_path / f"ws_{name}")
    os.makedirs(ws, exist_ok=True)
    resp = client.post("/api/v2/projects", json={
        "name": name, "workspace": ws, "model": "gpt-4", "api_key": "test-key",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"], ws


def _make_session(ws, stem="pinfail_sess_0001"):
    s = Session.new(stem, ws)
    s.append({"role": "user", "content": "earlier", "source": "user"})
    return stem


def _session_rows(ws, stem):
    p = os.path.join(ws, "orbital", "sessions", f"{stem}.jsonl")
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def env(client, tmp_path, monkeypatch):
    routes_mod = _routes_mod()
    monkeypatch.setattr(routes_mod, "_sub_agent_manager", _FailingSubAgentManager())
    monkeypatch.setattr(routes_mod, "_pinned_consolidation", MagicMock())
    pid, ws = _make_project(client, tmp_path, "pinfail")
    sid = _make_session(ws)

    sent: list[tuple[str, dict, list[dict]]] = []

    def _capture(project_id, event):
        # Snapshot the session log at broadcast time: the row must already
        # be on disk when the client refetches in response.
        sent.append((project_id, event, _session_rows(ws, sid)))

    monkeypatch.setattr(routes_mod._ws_manager, "broadcast", _capture)
    return client, pid, ws, sid, sent


def test_failed_pinned_send_broadcasts_sub_agent_failed(env):
    client, pid, ws, sid, sent = env

    resp = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "hi", "target": "claude-code", "session_id": sid,
    })

    # The 404 contract is unchanged.
    assert resp.status_code == 404
    assert "older than" in resp.json()["detail"]

    failed = [(p, e, rows) for p, e, rows in sent if e.get("type") == "sub_agent.failed"]
    assert len(failed) == 1, sent
    project_id, event, rows_at_broadcast = failed[0]
    assert project_id == pid
    assert event["project_id"] == pid
    assert event["session_id"] == sid
    assert event["handle"] == "claude-code"
    assert event["reason"] == FAIL_TEXT

    # Persist-then-broadcast: the failure row is already in the session log.
    last = rows_at_broadcast[-1]
    assert last["role"] == "system"
    assert last["_meta"]["event"] == "sub_agent_dispatch_failed"
    assert last["_meta"]["reason"] == "update_needed"
    # And no success ack went out for a failed dispatch.
    assert not [e for _, e, _ in sent if e.get("type") == "chat.sub_agent_message"]


def test_a_sent_message_is_not_announced_as_the_workers_own_words(env, monkeypatch):
    """``send()``'s "Message sent to …" is the route's answer. Broadcast as
    ``chat.sub_agent_message`` it rendered as a worker bubble, transcript path
    and all, for as long as the turn ran."""
    client, pid, ws, sid, sent = env

    class _Sent:
        async def send(self, project_id, handle, message, **kwargs):
            return "Message sent to claude-code. Transcript: C:/t.jsonl"

    monkeypatch.setattr(_routes_mod(), "_sub_agent_manager", _Sent())
    resp = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "hi", "target": "claude-code", "session_id": sid,
    })

    assert resp.status_code == 200
    assert resp.json()["status"].startswith("Message sent to claude-code")
    assert not [e for _, e, _ in sent if e.get("type") == "chat.sub_agent_message"]


def test_no_broadcast_when_the_failure_row_could_not_be_persisted(env, monkeypatch):
    client, pid, ws, sid, sent = env
    routes_mod = _routes_mod()
    real_persist = routes_mod._agent_manager.persist_mention_message

    async def _persist(project_id, session_id, row):
        if row.get("role") == "system":
            raise RuntimeError("disk full")
        return await real_persist(project_id, session_id, row)

    monkeypatch.setattr(routes_mod._agent_manager, "persist_mention_message", _persist)

    resp = client.post(f"/api/v2/agents/{pid}/inject", json={
        "content": "hi", "target": "claude-code", "session_id": sid,
    })
    assert resp.status_code == 404
    # Nothing new to refetch, so nothing is announced.
    assert not [e for _, e, _ in sent if e.get("type") == "sub_agent.failed"]
