# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hydrate-on-inject: a session that exists on disk is continued, not forked.

The cold-resume experiment (TASK/EXPERIMENT-cold-resume-behavior.md) proved
that injecting to a session with no live handle forked a fresh empty session
(start_agent -> Session.new) and lost all history. This fix makes inject load
the existing JSONL (Session.load) and continue it, preserving the original F1
identity from the session's meta record. Addressable by either F1 or F2.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import AgentConfig


def _make_manager(tmp_path, workspace):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    mgr._project_store.get_project.return_value = {"workspace": str(workspace), "name": "Proj"}
    return mgr


def _write_session(workspace, uuid, f1, msgs=(("user", "Remember the number 7."),)):
    sdir = ProjectPaths(str(workspace)).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    rows = [{"role": "meta", "event": "session_start", "session_id": f1,
             "session_uuid": uuid, "provider": "deepseek", "model": "deepseek-chat",
             "sdk": "openai", "fallback_models": [], "timestamp": "2026-05-26T00:00:00+00:00"}]
    for role, content in msgs:
        rows.append({"role": role, "content": content, "session_id": f1,
                     "session_uuid": uuid, "timestamp": "2026-05-26T00:00:01+00:00"})
    p = os.path.join(sdir, f"{uuid}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


# ── Step 1: _load_session_from_disk ───────────────────────────────────────

def test_load_session_from_disk_by_f1(tmp_path):
    ws = tmp_path / "ws"
    _write_session(ws, "proj_aaaa1111", "sess_abc")
    mgr = _make_manager(tmp_path, ws)
    s = mgr._load_session_from_disk("p1", "sess_abc")
    assert s is not None
    assert s.session_id == "sess_abc", "original F1 preserved from meta"
    assert s.session_uuid == "proj_aaaa1111", "F2 is the filename stem"
    assert any(m.get("content") == "Remember the number 7." for m in s.get_messages())


def test_load_session_from_disk_by_uuid_preserves_f1(tmp_path):
    ws = tmp_path / "ws"
    _write_session(ws, "proj_bbbb2222", "sess_xyz")
    mgr = _make_manager(tmp_path, ws)
    # Address by the F2 uuid (what the sidebar passes for disk-only sessions).
    s = mgr._load_session_from_disk("p1", "proj_bbbb2222")
    assert s is not None
    assert s.session_id == "sess_xyz", "F1 recovered from meta even when addressed by uuid"
    assert s.session_uuid == "proj_bbbb2222"


def test_load_session_from_disk_missing_returns_none(tmp_path):
    ws = tmp_path / "ws"
    (ws / "orbital" / "sessions").mkdir(parents=True)
    mgr = _make_manager(tmp_path, ws)
    assert mgr._load_session_from_disk("p1", "does-not-exist") is None


# ── Step 2: inject hydrates instead of forking ────────────────────────────

@pytest.mark.asyncio
async def test_inject_hydrates_existing_session(tmp_path):
    ws = tmp_path / "ws"
    _write_session(ws, "proj_cccc3333", "sess_live")
    mgr = _make_manager(tmp_path, ws)
    mgr.start_agent = AsyncMock()
    mgr._build_agent_config_from_project = MagicMock(return_value=MagicMock())

    # Address by the uuid (disk-only addressing).
    await mgr.inject_message("p1", "What number?", session_id="proj_cccc3333")

    assert mgr.start_agent.await_count == 1
    kwargs = mgr.start_agent.await_args.kwargs
    assert kwargs.get("session") is not None, "must pass the hydrated session (not fork fresh)"
    # The hydrated Session object still preserves the original F1 from meta
    # (for display / back-compat) — hydration does not rewrite it.
    assert kwargs["session"].session_id == "sess_live", "F1 preserved on the Session object"
    # Seam 3 / Phase 1: the routing identity adopted is the session's UUID
    # (the id the frontend addresses by), NOT the meta F1. This is what makes
    # viewed == holder. See test_hydrate_adopts_uuid_not_f1.py.
    assert kwargs.get("session_id") == "proj_cccc3333", "routes under the uuid, not the meta F1"
    # Bug #59: the message no longer travels as an argument — it is already
    # appended to the hydrated session (and to its JSONL) before start_agent
    # is entered. Assert it where it now lives, after the prior turn.
    assert "initial_message" not in kwargs
    contents = [m.get("content") for m in kwargs["session"].get_messages()
                if m.get("role") == "user"]
    assert contents == ["Remember the number 7.", "What number?"]
    on_disk = (ws / "orbital" / "sessions" / "proj_cccc3333.jsonl").read_text(
        encoding="utf-8")
    assert "What number?" in on_disk


@pytest.mark.asyncio
async def test_inject_forks_fresh_when_no_file(tmp_path):
    """No disk file → a fresh session is minted.

    Bug #59 moved the mint from inside ``start_agent`` to the inject
    write-ahead (the row must exist before the start window), so the fresh
    session now arrives AS the ``session=`` argument rather than being built
    downstream. ``_build_agent_config_from_project`` therefore has to return a
    real AgentConfig — a MagicMock's provider/model land in the session_start
    meta and are not JSON-serializable.
    """
    ws = tmp_path / "ws"
    (ws / "orbital" / "sessions").mkdir(parents=True)
    mgr = _make_manager(tmp_path, ws)
    mgr.start_agent = AsyncMock()
    mgr._build_agent_config_from_project = MagicMock(return_value=AgentConfig(
        workspace=str(ws), model="deepseek-chat", api_key="k",
        provider="deepseek", sdk="openai",
    ))

    await mgr.inject_message("p1", "hello", session_id="sess_brand_new")

    assert mgr.start_agent.await_count == 1
    kwargs = mgr.start_agent.await_args.kwargs
    fresh = kwargs.get("session")
    assert fresh is not None, "the write-ahead mints and passes the fresh session"
    assert fresh.session_uuid == "sess_brand_new"
    assert [m.get("content") for m in fresh.get_messages()] == ["hello"]
    # Freshly created on disk — no prior history was forked in.
    on_disk = (ws / "orbital" / "sessions" / "sess_brand_new.jsonl").read_text(
        encoding="utf-8")
    assert "hello" in on_disk


# ── Step 3: chat read-path resolves F1 and F2 to the same file ────────────

def test_find_session_uuid_on_disk_accepts_f1_and_f2(tmp_path):
    from agent_os.api.routes.agents_v2 import _find_session_uuid_on_disk
    ws = tmp_path / "ws"
    _write_session(ws, "proj_dddd4444", "sess_q")
    sdir = ProjectPaths(str(ws)).sessions_dir
    # F1 input → scans records → F2 stem
    assert _find_session_uuid_on_disk(sdir, "sess_q") == "proj_dddd4444"
    # F2 input → direct file-stem match → same stem
    assert _find_session_uuid_on_disk(sdir, "proj_dddd4444") == "proj_dddd4444"
    # unknown → None
    assert _find_session_uuid_on_disk(sdir, "nope") is None


# ── Spec 107: F1 resolution reads only the HEAD of each session log ──────
#
# "+ new session" lands the pane on a freshly minted id that no file carries.
# The chat route's fallback used to open every session log and json.loads
# every line looking for it (O(total bytes): 4.2 s on a 225 MB project). The
# hydrate resolver already stopped at the first session_id-bearing record of
# each file; both paths now share one head-only resolver that prefers the
# newest mtime when a legacy F1 ("default") is carried by several logs.

def _write_rows(ws, uuid, lines):
    """Write raw lines (already-serialised or deliberately torn) to a log."""
    sdir = ProjectPaths(str(ws)).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    p = os.path.join(sdir, f"{uuid}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def _meta(f1, uuid):
    return json.dumps({"role": "meta", "event": "session_start", "session_id": f1,
                       "session_uuid": uuid, "timestamp": "2026-05-26T00:00:00+00:00"})


def _row(role, content, f1, uuid):
    return json.dumps({"role": role, "content": content, "session_id": f1,
                       "session_uuid": uuid, "timestamp": "2026-05-26T00:00:01+00:00"})


def test_find_session_uuid_on_disk_reads_only_the_head(tmp_path):
    """The chat-route fallback stops at the first session_id-bearing record of
    each file (the contract ``_load_session_from_disk`` already had). A decoy
    F1 deeper in a file is never seen, and a torn line after the head is never
    parsed."""
    from agent_os.api.routes.agents_v2 import _find_session_uuid_on_disk
    ws = tmp_path / "ws"
    _write_rows(ws, "proj_eeee5555", [
        _meta("sess_head", "proj_eeee5555"),
        '{"role": "user", "content": "torn line, never valid',
        _row("user", "hi", "sess_decoy", "proj_eeee5555"),
    ])
    sdir = ProjectPaths(str(ws)).sessions_dir
    assert _find_session_uuid_on_disk(sdir, "sess_head") == "proj_eeee5555"
    assert _find_session_uuid_on_disk(sdir, "sess_decoy") is None


def test_find_session_uuid_on_disk_unknown_id_costs_one_record_per_file(tmp_path, monkeypatch):
    """A minted id matches nothing, so the scan must visit every file — but
    only its head: at most one JSON decode per file, not one per line."""
    import time
    from agent_os.api.routes import agents_v2
    from agent_os.daemon_v2 import session_index

    ws = tmp_path / "ws"
    n_files, n_lines = 200, 500
    for i in range(n_files):
        uuid = f"proj_{i:08x}"
        lines = [_meta(f"sess_{i}", uuid)]
        lines += [_row("user" if k % 2 else "assistant", f"line {k}", f"sess_{i}", uuid)
                  for k in range(n_lines - 1)]
        _write_rows(ws, uuid, lines)
    sdir = ProjectPaths(str(ws)).sessions_dir

    decodes = {"n": 0}
    real_loads = json.loads

    def counting_loads(s, *a, **kw):
        decodes["n"] += 1
        return real_loads(s, *a, **kw)

    # Count decodes wherever the resolver lives (route module or shared helper).
    monkeypatch.setattr(agents_v2.json, "loads", counting_loads)
    monkeypatch.setattr(session_index.json, "loads", counting_loads)

    t0 = time.perf_counter()
    assert agents_v2._find_session_uuid_on_disk(sdir, "proj_unknown_fresh") is None
    elapsed = time.perf_counter() - t0

    assert decodes["n"] <= n_files, (
        f"decoded {decodes['n']} records for {n_files} files — the fallback is "
        f"parsing whole files, not heads")
    # 200 × 500 lines = 100k records: a full parse takes ~1 s here; heads only
    # take tens of ms. Generous bound so CI noise never flakes it.
    assert elapsed < 0.75, f"unknown-id resolution took {elapsed:.2f}s"


def test_duplicate_f1_newest_mtime_wins_in_both_resolvers(tmp_path, monkeypatch):
    """Legacy logs share the F1 "default". The chat route used to take the
    first match in listdir order while the hydrate resolver preferred the
    newest mtime; both must agree on the newest."""
    import time
    from agent_os.api.routes.agents_v2 import _find_session_uuid_on_disk
    ws = tmp_path / "ws"
    old = _write_rows(ws, "aaaa_old", [_meta("default", "aaaa_old"),
                                       _row("user", "old", "default", "aaaa_old")])
    new = _write_rows(ws, "zzzz_new", [_meta("default", "zzzz_new"),
                                       _row("user", "new", "default", "zzzz_new")])
    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(new, (now, now))
    # Pin listdir to alphabetical so the old first-match behaviour would return
    # the OLD file deterministically.
    real_listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda p=".": sorted(real_listdir(p)))

    sdir = ProjectPaths(str(ws)).sessions_dir
    assert _find_session_uuid_on_disk(sdir, "default") == "zzzz_new"

    mgr = _make_manager(tmp_path, ws)
    s = mgr._load_session_from_disk("p1", "default")
    assert s is not None and s.session_uuid == "zzzz_new"
    assert s.session_id == "default"
