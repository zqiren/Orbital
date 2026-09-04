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


# ---------------------------------------------------------------------------
# Spec 081 — pending-only entries vs the disk scan
# ---------------------------------------------------------------------------


def test_pending_entry_is_deduped_once_the_file_exists(tmp_path):
    # A queued message for a session that already has a log on disk (an idle
    # session the user wrote to while another held the slot) must not add a
    # second row: the disk entry — with its stored name and history — is the
    # row, exactly as it will be after dispatch.
    ws, sessions, ps = _setup(tmp_path)
    _write_session(sessions, "proj_ondisk01", "proj_ondisk01")
    mgr = _make_manager(tmp_path, ps)
    mgr.enqueue_pending_inject("proj", "proj_ondisk01", "another message", nonce="n1")

    out = mgr.list_sessions("proj")
    entries = [s for s in out if s["session_id"] == "proj_ondisk01"]
    assert len(entries) == 1, f"pending entry must dedupe against the file: {out}"
    assert entries[0]["status"] == "idle"
    assert entries[0]["name"] == "hi", "the stored/derived disk name wins"


def test_file_appearing_after_the_pending_snapshot_is_deduped_by_seen_uuids(tmp_path):
    # The race: the pending helper saw no file, then dispatch wrote the row
    # before the disk scan ran. The disk scan must skip the id the pending
    # helper already emitted (the existing seen_uuids step), not list it twice.
    ws, sessions, ps = _setup(tmp_path)
    _write_session(sessions, "proj_raced0001", "proj_raced0001")
    mgr = _make_manager(tmp_path, ps)

    queued_entry = {
        "session_id": "proj_raced0001",
        "status": "queued",
        "session_uuid": "proj_raced0001",
        "origin": "chat",
        "name": "raced",
        "pinned": False,
        "pinned_target": None,
        "last_terminal_event": None,
        "last_activity_at": "2026-05-25T08:30:00+00:00",
        "scope": {"mode": "off", "selected_project_ids": []},
    }
    mgr._pending_session_entries = lambda project_id, seen_ids: [dict(queued_entry)]

    out = mgr.list_sessions("proj")
    entries = [s for s in out if s["session_id"] == "proj_raced0001"]
    assert len(entries) == 1, f"seen_uuids must dedupe the raced file: {out}"
    assert entries[0]["status"] == "queued"


def test_meta_only_log_and_disk_cache_unchanged_with_pending_merge(tmp_path):
    # The meta-only skip rule and the mtime cache are untouched by the merge.
    ws, sessions, ps = _setup(tmp_path)
    (sessions / "proj_metaonly01.jsonl").write_text(
        json.dumps({"role": "meta", "event": "session_start", "name": "x"}) + "\n",
        encoding="utf-8",
    )
    _write_session(sessions, "proj_cached0001", "proj_cached0001")
    mgr = _make_manager(tmp_path, ps)
    mgr.enqueue_pending_inject("proj", "proj_queued0001", "queued one", nonce="n1")

    first = mgr.list_sessions("proj")
    ids = [s["session_id"] for s in first]
    assert "proj_metaonly01" not in ids
    assert ids.count("proj_cached0001") == 1
    assert ids.count("proj_queued0001") == 1
    # Second call: the cached disk verdicts are reused and the result is stable.
    cache_key = (str(sessions.resolve()), "proj_cached0001.jsonl")
    assert cache_key in mgr._disk_entry_cache
    second = mgr.list_sessions("proj")
    assert [s["session_id"] for s in second] == ids
