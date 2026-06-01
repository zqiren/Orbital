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
    assert kwargs.get("initial_message") == "What number?"


@pytest.mark.asyncio
async def test_inject_forks_fresh_when_no_file(tmp_path):
    ws = tmp_path / "ws"
    (ws / "orbital" / "sessions").mkdir(parents=True)
    mgr = _make_manager(tmp_path, ws)
    mgr.start_agent = AsyncMock()
    mgr._build_agent_config_from_project = MagicMock(return_value=MagicMock())

    await mgr.inject_message("p1", "hello", session_id="sess_brand_new")

    assert mgr.start_agent.await_count == 1
    kwargs = mgr.start_agent.await_args.kwargs
    assert kwargs.get("session") is None, "no disk file -> fresh session (Session.new path)"


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
