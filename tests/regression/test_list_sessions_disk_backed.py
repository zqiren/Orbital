# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Step 2 — list_sessions is disk-backed.

The sidebar must show every persisted session (each ↔ a session-log JSONL), not
just the one currently hydrated in memory. list_sessions now enumerates the
project's session JSONLs and overlays in-memory live status (running/paused) for
any session that has a live handle, deduping by session_uuid (the filename stem).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


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


def _write_session(sessions_dir, uuid, session_id, ts="2026-05-25T08:00:00+00:00"):
    p = sessions_dir / f"{uuid}.jsonl"
    rows = [
        {"role": "user", "content": "hi", "session_id": session_id, "timestamp": ts},
        {"role": "assistant", "content": "yo", "session_id": session_id, "timestamp": ts},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _setup(tmp_path):
    ws = tmp_path / "ws"
    sessions = ws / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    ps = MagicMock()
    ps.get_project.return_value = {"workspace": str(ws)}
    return ws, sessions, ps


def test_list_sessions_lists_archived_disk_sessions(tmp_path):
    ws, sessions, ps = _setup(tmp_path)
    _write_session(sessions, "proj_aaaaaaaa", "default")
    _write_session(sessions, "proj_bbbbbbbb", "sess_b")
    mgr = _make_manager(tmp_path, ps)

    out = mgr.list_sessions("proj")
    uuids = {s["session_uuid"] for s in out}
    assert {"proj_aaaaaaaa", "proj_bbbbbbbb"} <= uuids, out
    # No live handle → archived → idle, with last_activity populated.
    for s in out:
        if s["session_uuid"] in {"proj_aaaaaaaa", "proj_bbbbbbbb"}:
            assert s["status"] == "idle"
            assert s["last_activity_at"]


def test_in_memory_session_overlays_disk_and_not_duplicated(tmp_path):
    ws, sessions, ps = _setup(tmp_path)
    _write_session(sessions, "proj_live1234", "default")
    mgr = _make_manager(tmp_path, ps)

    # A live handle whose session corresponds to the on-disk uuid, running.
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.session_uuid = "proj_live1234"
    session._messages = [{"timestamp": "2026-05-25T09:00:00+00:00"}]
    task = MagicMock()
    task.done.return_value = False
    mgr._handles[("proj", "default")] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=task,
        config_snapshot={"workspace": str(ws)}, started_at="2026-01-01T00:00:00+00:00",
    )

    out = mgr.list_sessions("proj")
    entries = [s for s in out if s["session_uuid"] == "proj_live1234"]
    assert len(entries) == 1, f"must not duplicate in-memory + disk: {out}"
    assert entries[0]["status"] == "running", "in-memory live status overlays disk"
