# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bug #48 — ``_disk_session_entries`` memoizes each session log per mtime.

The sidebar scan used to re-open and re-parse every idle session's full JSONL
on every call — once per session switch, and once per ``agent.status`` WS
broadcast. Idle logs don't change, so an unchanged file must be served from
the cache without touching it, and a changed file must invalidate on its own.
"""

from __future__ import annotations

import builtins
import json
import os
from unittest.mock import MagicMock

import pytest

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


def _write_session(sessions_dir, uuid, session_id, *, name=None,
                   ts="2026-05-25T08:00:00+00:00"):
    rows = []
    if name is not None:
        rows.append({"role": "meta", "event": "session_start", "name": name})
    rows.extend([
        {"role": "user", "content": "hi", "session_id": session_id, "timestamp": ts},
        {"role": "assistant", "content": "yo", "session_id": session_id, "timestamp": ts},
    ])
    p = sessions_dir / f"{uuid}.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def project(tmp_path):
    ws = tmp_path / "ws"
    sessions = ws / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    ps = MagicMock()
    ps.get_project.return_value = {"workspace": str(ws)}
    return ws, sessions, ps


class _OpenCounter:
    """Counts ``open()`` calls against files inside ``root``."""

    def __init__(self, monkeypatch, root):
        self.root = os.path.realpath(str(root))
        self.paths: list[str] = []
        real_open = builtins.open

        def counting_open(file, *args, **kwargs):
            try:
                if os.path.realpath(str(file)).startswith(self.root):
                    self.paths.append(os.path.basename(str(file)))
            except (TypeError, ValueError):
                pass
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)

    def reset(self):
        self.paths.clear()


def _bump_mtime(path, seconds=120):
    """Move a file's mtime forward past any filesystem-granularity slop."""
    st = os.stat(path)
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


def test_second_scan_opens_no_files(project, tmp_path, monkeypatch):
    ws, sessions, ps = project
    _write_session(sessions, "sess_aaaaaaaa", "default")
    _write_session(sessions, "sess_bbbbbbbb", "sess_b")
    mgr = _make_manager(tmp_path, ps)

    first = mgr.list_sessions("proj")
    counter = _OpenCounter(monkeypatch, sessions)
    second = mgr.list_sessions("proj")

    assert counter.paths == [], (
        f"unchanged session logs must not be re-opened, opened: {counter.paths}"
    )
    assert second == first, "cached scan must return the same entries"


def test_changed_file_reparses_only_that_file(project, tmp_path, monkeypatch):
    ws, sessions, ps = project
    _write_session(sessions, "sess_aaaaaaaa", "default")
    changed = _write_session(sessions, "sess_bbbbbbbb", "sess_b")
    mgr = _make_manager(tmp_path, ps)

    mgr.list_sessions("proj")
    _bump_mtime(changed)

    counter = _OpenCounter(monkeypatch, sessions)
    mgr.list_sessions("proj")

    assert counter.paths == ["sess_bbbbbbbb.jsonl"], (
        f"only the touched log may be re-parsed, opened: {counter.paths}"
    )


def test_rename_invalidates_cached_name(project, tmp_path):
    """A rename appends a new session_start meta line — a content+mtime change
    the cache must pick up (spec 048 §7)."""
    ws, sessions, ps = project
    path = _write_session(sessions, "sess_aaaaaaaa", "default", name="old name")
    mgr = _make_manager(tmp_path, ps)

    before = mgr.list_sessions("proj")
    assert before[0]["name"] == "old name"

    rows = [
        {"role": "meta", "event": "session_start", "name": "new name"},
        {"role": "user", "content": "hi", "session_id": "default",
         "timestamp": "2026-05-25T08:00:00+00:00"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _bump_mtime(path)

    after = mgr.list_sessions("proj")
    assert after[0]["name"] == "new name", "renamed session must not serve a stale name"


def test_meta_only_log_skip_is_cached(project, tmp_path, monkeypatch):
    """A meta-only log is not a session. The skip verdict is cached too, so the
    second scan neither re-opens it nor starts listing it."""
    ws, sessions, ps = project
    (sessions / "sess_metaonly.jsonl").write_text(
        json.dumps({"role": "meta", "event": "session_start"}) + "\n",
        encoding="utf-8",
    )
    _write_session(sessions, "sess_aaaaaaaa", "default")
    mgr = _make_manager(tmp_path, ps)

    first = mgr.list_sessions("proj")
    counter = _OpenCounter(monkeypatch, sessions)
    second = mgr.list_sessions("proj")

    assert counter.paths == [], f"skip verdicts must be cached, opened: {counter.paths}"
    assert [s["session_uuid"] for s in second] == ["sess_aaaaaaaa"]
    assert second == first


def test_cached_entry_is_not_shared_with_callers(project, tmp_path):
    """Callers get a fresh dict, so mutating one can't poison the cache."""
    ws, sessions, ps = project
    _write_session(sessions, "sess_aaaaaaaa", "default", name="keep me")
    mgr = _make_manager(tmp_path, ps)

    first = mgr.list_sessions("proj")
    first[0]["name"] = "clobbered"
    first[0]["status"] = "running"

    second = mgr.list_sessions("proj")
    assert second[0]["name"] == "keep me"
    assert second[0]["status"] == "idle"


def test_scope_is_recomputed_on_cache_hit(project, tmp_path):
    """``scope`` comes from live in-memory state, not the file — a scope change
    on an unchanged log must still show up."""
    ws, sessions, ps = project
    _write_session(sessions, "sess_aaaaaaaa", "default")
    ps.get_project.return_value = {"workspace": str(ws), "is_scratch": True}
    mgr = _make_manager(tmp_path, ps)

    first = mgr.list_sessions("proj")
    assert first[0]["scope"]["mode"] == "all"  # scratch default

    mgr.set_session_scope("proj", "sess_aaaaaaaa", "off", [])
    second = mgr.list_sessions("proj")
    assert second[0]["scope"]["mode"] == "off", "cache must not freeze session scope"
