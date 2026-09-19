# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 066 phase 2 — the rebuildable session index.

The session list used to open and parse every idle session log after each
daemon restart (the in-memory cache is cold on start). The index is a SQLite
file next to the logs holding each file's list row, keyed by the file's
mtime + size. JSONL stays the file of record: the index can be deleted or
corrupted at any time and is rebuilt from the logs, producing the identical
list. Until the (background) build finishes, the list falls back to the scan.
"""

from __future__ import annotations

import builtins
import json
import os
import time
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2 import session_index as si
from agent_os.daemon_v2.agent_manager import AgentManager


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


def _write(sessions_dir, uuid, rows):
    p = sessions_dir / f"{uuid}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                 encoding="utf-8")
    return p


def _chat(uuid, text, ts, **meta):
    return [
        {"role": "meta", "event": "session_start", "session_id": uuid,
         "session_uuid": uuid, "origin": "chat", **meta},
        {"role": "user", "content": text, "timestamp": ts},
        {"role": "assistant", "content": "ok", "timestamp": ts},
    ]


@pytest.fixture
def project(tmp_path):
    ws = tmp_path / "ws"
    sessions = ws / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    _write(sessions, "p_aaaaaaaa", _chat("p_aaaaaaaa", "first chat", "2026-09-01T00:00:00+00:00"))
    _write(sessions, "p_bbbbbbbb", _chat("p_bbbbbbbb", "[Triggered by schedule 'Daily' (9am)]\n\nrun",
                                         "2026-09-02T00:00:00+00:00"))
    _write(sessions, "p_cccccccc", _chat("p_cccccccc", "renamed one", "2026-09-03T00:00:00+00:00",
                                         name="My name", pinned=True))
    _write(sessions, "p_dddddddd", [
        {"role": "meta", "event": "session_start", "origin": "queue"},
        {"role": "user", "content": "[QUEUE ITEM | id=x | attempt=1]\nYou are working on a queue item.",
         "timestamp": "2026-09-04T00:00:00+00:00"},
    ])
    _write(sessions, "p_eeeeeeee", [{"role": "meta", "event": "session_start"}])  # meta-only
    _write(sessions, "p_ffffffff", [
        {"role": "meta", "event": "session_start"},
        {"role": "meta", "event": "session_kind", "kind": "worker"},
        {"role": "user", "content": "worker task", "timestamp": "2026-09-05T00:00:00+00:00"},
    ])
    (sessions / "p_gggggggg.jsonl").write_text('{"role": "user", "content": "x", "timestamp": "t"}\n{broken\n',
                                              encoding="utf-8")
    ps = MagicMock()
    ps.get_project.return_value = {"project_id": "proj", "workspace": str(ws)}
    ps.list_projects.return_value = [{"project_id": "proj", "workspace": str(ws)}]
    return ws, sessions, ps


class _JsonlOpenCounter:
    def __init__(self, monkeypatch, root):
        self.root = os.path.realpath(str(root))
        self.paths: list[str] = []
        real_open = builtins.open

        def counting_open(file, *args, **kwargs):
            try:
                p = os.path.realpath(str(file))
                if p.startswith(self.root) and p.endswith(".jsonl"):
                    self.paths.append(os.path.basename(p))
            except (TypeError, ValueError):
                pass
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)


def _dump(entries):
    return json.dumps(entries, sort_keys=False)


def test_index_list_is_identical_to_the_scan(project, tmp_path):
    ws, sessions, ps = project
    scanned = _make_manager(tmp_path, ps).list_sessions("proj")

    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    assert mgr.session_index_ready(str(ws))
    assert _dump(mgr.list_sessions("proj")) == _dump(scanned)


def test_after_restart_the_list_opens_no_session_log(project, tmp_path, monkeypatch):
    """Spec 066 §9 criterion 1: the list renders from the index alone."""
    ws, sessions, ps = project
    first = _make_manager(tmp_path, ps)
    first.build_session_index(str(ws))
    before = first.list_sessions("proj")

    counter = _JsonlOpenCounter(monkeypatch, sessions)
    restarted = _make_manager(tmp_path, ps)  # cold in-memory state
    restarted.build_session_index(str(ws))
    after = restarted.list_sessions("proj")

    assert counter.paths == [], f"session logs opened after restart: {counter.paths}"
    assert _dump(after) == _dump(before)


def test_deleting_the_index_rebuilds_an_identical_list(project, tmp_path):
    """Spec 066 §9 criterion 3."""
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    before = mgr.list_sessions("proj")

    os.remove(si.index_path(str(sessions)))
    restarted = _make_manager(tmp_path, ps)
    restarted.build_session_index(str(ws))
    assert os.path.isfile(si.index_path(str(sessions)))
    assert _dump(restarted.list_sessions("proj")) == _dump(before)


def test_corrupt_index_is_rebuilt_silently(project, tmp_path):
    ws, sessions, ps = project
    expected = _make_manager(tmp_path, ps).list_sessions("proj")
    with open(si.index_path(str(sessions)), "wb") as f:
        f.write(b"this is not a sqlite database" * 100)

    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    assert mgr.session_index_ready(str(ws))
    assert _dump(mgr.list_sessions("proj")) == _dump(expected)


def test_changed_log_is_reparsed_and_persisted(project, tmp_path, monkeypatch):
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    mgr.list_sessions("proj")

    path = sessions / "p_aaaaaaaa.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": "assistant", "content": "later",
                            "timestamp": "2026-09-10T00:00:00+00:00"}) + "\n")

    counter = _JsonlOpenCounter(monkeypatch, sessions)
    entries = {e["session_id"]: e for e in mgr.list_sessions("proj")}
    assert counter.paths == ["p_aaaaaaaa.jsonl"]
    assert entries["p_aaaaaaaa"]["last_activity_at"] == "2026-09-10T00:00:00+00:00"

    counter.paths.clear()
    restarted = _make_manager(tmp_path, ps)
    restarted.build_session_index(str(ws))
    entries = {e["session_id"]: e for e in restarted.list_sessions("proj")}
    assert counter.paths == [], "the reparsed row was written through to the index"
    assert entries["p_aaaaaaaa"]["last_activity_at"] == "2026-09-10T00:00:00+00:00"


def test_deleted_log_leaves_the_list_and_the_index(project, tmp_path):
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    os.remove(sessions / "p_aaaaaaaa.jsonl")
    ids = {e["session_id"] for e in mgr.list_sessions("proj")}
    assert "p_aaaaaaaa" not in ids
    assert "p_aaaaaaaa.jsonl" not in si.SessionIndex(str(sessions)).load_rows()


def test_version_mismatch_rebuilds(project, tmp_path, monkeypatch):
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    monkeypatch.setattr(si, "INDEX_VERSION", si.INDEX_VERSION + "-next")

    counter = _JsonlOpenCounter(monkeypatch, sessions)
    restarted = _make_manager(tmp_path, ps)
    restarted.build_session_index(str(ws))
    assert counter.paths, "a different deriver version must reparse the logs"


def test_list_falls_back_to_the_scan_until_ready(project, tmp_path):
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    assert not mgr.session_index_ready(str(ws))
    entries = mgr.list_sessions("proj")
    assert {e["session_id"] for e in entries} == {
        "p_aaaaaaaa", "p_bbbbbbbb", "p_cccccccc", "p_dddddddd", "p_gggggggg",
    }
    assert not os.path.exists(si.index_path(str(sessions))), (
        "without a started build the list never creates the index"
    )


def test_background_build_does_not_block_and_becomes_ready(project, tmp_path):
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    t0 = time.monotonic()
    mgr.start_session_index_builds()
    assert time.monotonic() - t0 < 0.5
    deadline = time.monotonic() + 10
    while not mgr.session_index_ready(str(ws)) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert mgr.session_index_ready(str(ws))


def test_trigger_type_is_listed(project, tmp_path):
    ws, sessions, ps = project
    entries = {e["session_id"]: e for e in _make_manager(tmp_path, ps).list_sessions("proj")}
    assert entries["p_bbbbbbbb"]["trigger_type"] == "schedule"
    assert entries["p_aaaaaaaa"]["trigger_type"] is None
    assert entries["p_dddddddd"]["origin"] == "queue"


def test_derive_disk_entry_skips_meta_only_and_worker_logs(project):
    ws, sessions, ps = project
    assert si.derive_disk_entry(str(sessions / "p_eeeeeeee.jsonl"), "p_eeeeeeee") is None
    assert si.derive_disk_entry(str(sessions / "p_ffffffff.jsonl"), "p_ffffffff") is None
    entry = si.derive_disk_entry(str(sessions / "p_cccccccc.jsonl"), "p_cccccccc")
    assert entry["name"] == "My name" and entry["pinned"] is True


def test_index_deleted_while_running_is_recreated_on_the_next_write(project, tmp_path):
    """The user deletes the index while the daemon runs: the list keeps
    working from memory and the next write-through recreates a sound file."""
    import sqlite3
    ws, sessions, ps = project
    mgr = _make_manager(tmp_path, ps)
    mgr.build_session_index(str(ws))
    before = mgr.list_sessions("proj")
    os.remove(si.index_path(str(sessions)))

    path = sessions / "p_aaaaaaaa.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"role": "assistant", "content": "more",
                            "timestamp": "2026-09-11T00:00:00+00:00"}) + "\n")
    after = {e["session_id"]: e for e in mgr.list_sessions("proj")}

    assert len(after) == len(before)
    assert after["p_aaaaaaaa"]["last_activity_at"] == "2026-09-11T00:00:00+00:00"
    conn = sqlite3.connect(si.index_path(str(sessions)))
    try:
        (n,) = conn.execute("SELECT count(*) FROM sessions WHERE fname = 'p_aaaaaaaa.jsonl'").fetchone()
    finally:
        conn.close()
    assert n == 1
