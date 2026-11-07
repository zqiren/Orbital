# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 155 (replacing spec 081's ``queued`` rows): sessions of a project run
in parallel, so no session's first message waits behind another and the
session list never shows a ``queued`` row. Two running sessions are both
listed as running; a session's own queued follow-up adds no row.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle

PROJECT = "proj"


def _make_manager(tmp_path, project_store):
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=None,
        credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    return mgr


def _setup(tmp_path):
    ws = tmp_path / "ws"
    sessions = ws / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    ps = MagicMock()
    ps.get_project.return_value = {"project_id": PROJECT, "workspace": str(ws)}
    mgr = _make_manager(tmp_path, ps)
    return mgr, sessions


def _write_session(sessions_dir, sid, ts="2026-05-25T08:00:00+00:00", content="hi"):
    p = sessions_dir / f"{sid}.jsonl"
    rows = [
        {"role": "user", "content": content, "session_id": sid, "timestamp": ts},
        {"role": "assistant", "content": "yo", "session_id": sid, "timestamp": ts},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _plant_running(mgr, sid, *, queued=None):
    """A live handle whose main task is alive — the slot holder."""
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.session_uuid = sid
    session._messages = [{"timestamp": "2026-05-25T09:00:00+00:00"}]
    session.list_queued = MagicMock(return_value=list(queued or []))
    task = MagicMock()
    task.done.return_value = False
    mgr._handles[(PROJECT, sid)] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=task,
        config_snapshot={}, started_at="2026-01-01T00:00:00+00:00",
    )


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _by_id(entries):
    return {e["session_id"]: e for e in entries}


def _by_id(entries):
    return {e["session_id"]: e for e in entries}


def test_two_running_sessions_are_both_listed_as_running(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    _plant_running(mgr, "sess-B")

    out = _by_id(mgr.list_sessions(PROJECT))

    assert out["sess-A"]["status"] == "running"
    assert out["sess-B"]["status"] == "running"
    assert all(e["status"] != "queued" for e in out.values())


def test_same_session_queued_message_adds_no_entry(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A", queued=[("more for A", "n-2")])

    out = mgr.list_sessions(PROJECT)

    assert [e["session_id"] for e in out] == ["sess-A"]
    assert out[0]["status"] == "running"


def test_an_idle_session_on_disk_beside_a_running_one_is_listed_once(tmp_path):
    mgr, sessions = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    _write_session(sessions, "sess-B")

    out = mgr.list_sessions(PROJECT)

    b_entries = [e for e in out if e["session_id"] == "sess-B"]
    assert len(b_entries) == 1
    assert b_entries[0]["status"] == "idle"
