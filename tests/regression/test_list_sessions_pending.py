# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 081 — a session whose first message is queued behind the slot is
listed with status ``queued``.

``list_sessions`` knew only live handles and on-disk JSONLs. A session minted
by "+ new session" whose first message was enqueued while another session held
the project slot has neither (persist-at-dispatch, spec 006), so it was
invisible in the sidebar until the slot freed. The list now merges the
project's pending-inject registry between the handle pass and the disk scan:
one ``queued`` entry per session that has no handle and no file.

Backend cases from spec 081 §5.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from agent_os.daemon_v2.agent_manager import AgentManager, PendingInject, ProjectHandle

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


# ---------------------------------------------------------------------------
# §5 — slot held by A; enqueue for B → B listed as queued
# ---------------------------------------------------------------------------


def test_queued_session_is_listed_with_status_queued(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "Say hello to the world",
                               nonce="n-b")
    enqueued_at = mgr._pending_inject[PROJECT][0].enqueued_at

    out = _by_id(mgr.list_sessions(PROJECT))

    assert set(out) == {"sess-A", "sess-B"}, out
    assert out["sess-A"]["status"] == "running"
    b = out["sess-B"]
    assert b["status"] == "queued"
    assert b["session_uuid"] == "sess-B", "uuid-only identity (seam 3)"
    # D3: named by the same helper that names a materialized session.
    assert b["name"] == "Say hello to the world"
    assert b["origin"] == "chat"
    assert b["pinned"] is False
    assert b["pinned_target"] is None
    assert b["last_terminal_event"] is None
    # D4: sorts by its enqueue time (ISO 8601, UTC).
    assert b["last_activity_at"] == _iso(enqueued_at)
    assert datetime.fromisoformat(b["last_activity_at"]).tzinfo is not None
    assert b["scope"] == {"mode": "off", "selected_project_ids": []}


def test_queued_entry_carries_no_holder_or_position(tmp_path):
    # D7: the row says "Queued" and nothing more.
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "hello", nonce="n-b")

    b = _by_id(mgr.list_sessions(PROJECT))["sess-B"]

    assert "holder" not in b
    assert "position" not in b
    assert "nonce" not in b
    assert "content" not in b


def test_queued_name_uses_derive_name_rules(tmp_path):
    # The queued text can carry the attachment block / queue wrapper the
    # inject route prepends; the name must be the user's own words, exactly
    # as it will be once the session materializes (D3).
    from agent_os.agent.session import _derive_name

    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    long_text = "word " * 40
    mgr.enqueue_pending_inject(PROJECT, "sess-B", long_text, nonce="n-b")

    b = _by_id(mgr.list_sessions(PROJECT))["sess-B"]

    assert b["name"] == _derive_name(long_text)
    assert b["name"].endswith("…")


# ---------------------------------------------------------------------------
# §5 — grouping
# ---------------------------------------------------------------------------


def test_two_queued_messages_for_b_make_one_entry_with_first_enqueue_time(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "first message", nonce="n-1")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "second message", nonce="n-2")
    first, second = mgr._pending_inject[PROJECT]
    # Make the enqueue times unambiguous.
    first.enqueued_at = 1_800_000_000.0
    second.enqueued_at = 1_800_000_060.0

    out = mgr.list_sessions(PROJECT)

    b_entries = [e for e in out if e["session_id"] == "sess-B"]
    assert len(b_entries) == 1, out
    assert b_entries[0]["name"] == "first message"
    assert b_entries[0]["last_activity_at"] == _iso(first.enqueued_at)


def test_queued_for_b_and_c_give_two_entries_ordered_by_id(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    # Enqueue C before B: the list is ordered by id, not by enqueue order
    # (the sidebar re-sorts by activity).
    mgr.enqueue_pending_inject(PROJECT, "sess-C", "for C", nonce="n-c")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "for B", nonce="n-b")

    out = mgr.list_sessions(PROJECT)

    assert [e["session_id"] for e in out] == ["sess-A", "sess-B", "sess-C"]
    assert [e["status"] for e in out] == ["running", "queued", "queued"]
    assert _by_id(out)["sess-B"]["name"] == "for B"
    assert _by_id(out)["sess-C"]["name"] == "for C"


def test_same_session_queued_message_adds_no_entry(tmp_path):
    # kind:"same" — the holder's own queue. That session has a handle and is
    # already listed; nothing to add.
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A", queued=[("more for A", "n-2")])

    out = mgr.list_sessions(PROJECT)

    assert [e["session_id"] for e in out] == ["sess-A"]
    assert out[0]["status"] == "running"


# ---------------------------------------------------------------------------
# §5 — cancel / dispatch / materialization
# ---------------------------------------------------------------------------


def test_cancel_removes_the_queued_entry(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "hello", nonce="n-b")
    assert "sess-B" in _by_id(mgr.list_sessions(PROJECT))

    result = mgr.cancel_pending_inject(PROJECT, "sess-B", nonce="n-b")

    assert result["removed"] is True
    assert "sess-B" not in _by_id(mgr.list_sessions(PROJECT))


def test_cancel_one_of_two_keeps_the_entry(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "first", nonce="n-1")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "second", nonce="n-2")

    mgr.cancel_pending_inject(PROJECT, "sess-B", nonce="n-1")

    b = _by_id(mgr.list_sessions(PROJECT))["sess-B"]
    assert b["status"] == "queued"
    assert b["name"] == "second", "the surviving message names the row"


def test_dispatched_session_with_handle_is_listed_once_as_running(tmp_path):
    # After dispatch the handle exists and the registry entry is gone: the
    # handle pass owns the row. Model the overlap window too (registry entry
    # still present while the handle is already up): still exactly one row,
    # and the live handle wins.
    mgr, _ = _setup(tmp_path)
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "run B", nonce="n-b")
    _plant_running(mgr, "sess-B")

    out = mgr.list_sessions(PROJECT)

    b_entries = [e for e in out if e["session_id"] == "sess-B"]
    assert len(b_entries) == 1, out
    assert b_entries[0]["status"] == "running"


def test_in_flight_batch_still_lists_the_session_as_queued(tmp_path):
    # _maybe_dispatch_pending pops the batch into _pending_inflight before
    # _fire has written the row or registered the handle. A refresh in that
    # window (A's own agent.status idle triggers one) must not drop the row.
    mgr, _ = _setup(tmp_path)
    entry = PendingInject(id="p1", session_id="sess-B", content="run B",
                          nonce="n-b", attachments=None,
                          enqueued_at=1_800_000_000.0)
    mgr._pending_inflight[PROJECT] = [entry]

    b = _by_id(mgr.list_sessions(PROJECT))["sess-B"]

    assert b["status"] == "queued"
    assert b["last_activity_at"] == _iso(entry.enqueued_at)


def test_tombstoned_in_flight_entry_is_not_listed(tmp_path):
    # Cancelled mid-flight: chat.pending_cancelled already dropped the bubble;
    # the list must not resurrect the session.
    mgr, _ = _setup(tmp_path)
    entry = PendingInject(id="p1", session_id="sess-B", content="run B",
                          nonce="n-b", attachments=None,
                          enqueued_at=1_800_000_000.0)
    mgr._pending_inflight[PROJECT] = [entry]
    mgr._pending_tombstoned.add("p1")

    assert "sess-B" not in _by_id(mgr.list_sessions(PROJECT))


def test_materialized_session_on_disk_is_listed_once_from_disk(tmp_path):
    # After the loop ends the file exists and the handle is gone (evicted):
    # the disk entry is the row. A stale registry entry for the same id must
    # not produce a second row.
    mgr, sessions = _setup(tmp_path)
    _write_session(sessions, "sess-B", content="run B")
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "run B again", nonce="n-b")

    out = mgr.list_sessions(PROJECT)

    b_entries = [e for e in out if e["session_id"] == "sess-B"]
    assert len(b_entries) == 1, out
    assert b_entries[0]["status"] == "idle"
    assert b_entries[0]["last_activity_at"] == "2026-05-25T08:00:00+00:00"


def test_pending_registry_of_another_project_is_ignored(tmp_path):
    mgr, _ = _setup(tmp_path)
    _plant_running(mgr, "sess-A")
    mgr.enqueue_pending_inject("other-project", "sess-X", "elsewhere", nonce="n-x")

    assert [e["session_id"] for e in mgr.list_sessions(PROJECT)] == ["sess-A"]


def test_no_workspace_still_lists_queued_sessions(tmp_path):
    # The disk scan bails without a workspace; the pending merge must not.
    ps = MagicMock()
    ps.get_project.return_value = {"project_id": PROJECT}
    mgr = _make_manager(tmp_path, ps)
    mgr.enqueue_pending_inject(PROJECT, "sess-B", "hello", nonce="n-b")

    out = mgr.list_sessions(PROJECT)

    assert [e["session_id"] for e in out] == ["sess-B"]
    assert out[0]["status"] == "queued"


# ---------------------------------------------------------------------------
# §5 — property: ids are unique for every registry/handles/disk combination
# ---------------------------------------------------------------------------


def test_session_ids_are_unique_for_every_combination(tmp_path):
    ids = ["sess-1", "sess-2"]
    # Each session independently: in the registry? in flight? has a handle?
    # has a file? 16 states per id, 2 ids → 256 combinations. Two ids is
    # enough: the merge decides each id on its own state plus the shared
    # seen-ids set, so a pair exercises every cross-session interaction (a
    # third id took 42 s of temp-dir churn for no extra coverage).
    states = list(itertools.product([False, True], repeat=4))
    combos = itertools.product(states, repeat=len(ids))
    for n, combo in enumerate(combos):
        ws = tmp_path / f"ws-{n}"
        sessions = ws / "orbital" / "sessions"
        sessions.mkdir(parents=True)
        ps = MagicMock()
        ps.get_project.return_value = {"project_id": PROJECT, "workspace": str(ws)}
        mgr = _make_manager(tmp_path, ps)
        expected_present = set()
        for sid, (in_registry, in_flight, has_handle, has_file) in zip(ids, combo):
            if in_registry:
                mgr.enqueue_pending_inject(PROJECT, sid, f"msg for {sid}", nonce=f"n-{sid}")
                expected_present.add(sid)
            if in_flight:
                mgr._pending_inflight.setdefault(PROJECT, []).append(PendingInject(
                    id=f"p-{sid}", session_id=sid, content=f"flight {sid}",
                    nonce=None, attachments=None, enqueued_at=1_800_000_000.0,
                ))
                expected_present.add(sid)
            if has_handle:
                _plant_running(mgr, sid)
                expected_present.add(sid)
            if has_file:
                _write_session(sessions, sid)
                expected_present.add(sid)

        out = mgr.list_sessions(PROJECT)
        listed = [e["session_id"] for e in out]
        assert len(listed) == len(set(listed)), (combo, out)
        assert set(listed) == expected_present, (combo, out)
        # Precedence: handle > file > queued.
        for sid, (in_registry, in_flight, has_handle, has_file) in zip(ids, combo):
            if sid not in expected_present:
                continue
            status = _by_id(out)[sid]["status"]
            if has_handle:
                assert status == "running", (combo, out)
            elif has_file:
                assert status == "idle", (combo, out)
            else:
                assert status == "queued", (combo, out)
