# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Auto-resume is removed: daemon startup creates no running agents.

Under the supervisor model, sessions are durable files on disk and "running at
shutdown" is not a directive to resume. A fresh AgentManager (a daemon restart)
creates no handles, reads no daemon-state.json, and injects no "interrupted"
warning. The user recovers by opening a session in the sidebar and sending a
message (which hydrates it — see test_hydrate_on_inject.py).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager


def _make_manager(workspace=None):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._project_store.get_project.return_value = (
        {"workspace": str(workspace)} if workspace else None
    )
    return mgr


def test_auto_resume_machinery_is_gone():
    """The entire auto-resume / daemon-state / heartbeat surface is removed."""
    for attr in (
        "auto_resume_agents", "_write_state", "_read_state",
        "mark_shutdown_clean", "_start_heartbeat",
        "_ensure_heartbeat_running", "_stop_heartbeat_if_idle",
    ):
        assert not hasattr(AgentManager, attr), f"{attr} should be removed"


def test_fresh_manager_creates_no_handles_or_state():
    """Constructing the manager (a daemon start) auto-starts nothing and holds
    no state-persistence machinery."""
    mgr = _make_manager()
    assert mgr._handles == {}
    assert not hasattr(mgr, "_state_file")
    assert not hasattr(mgr, "_heartbeat_task")


def test_restart_does_not_resume_or_warn_disk_session(tmp_path):
    """A session that was running before a restart is NOT resumed: a fresh
    manager has no handle for it, list_sessions reports it idle, and its JSONL
    is untouched (no 'interrupted' warning injected)."""
    ws = tmp_path / "ws"
    sdir = ProjectPaths(str(ws)).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    uuid = "proj_run00000"
    rows = [
        {"role": "meta", "event": "session_start", "session_id": "default",
         "session_uuid": uuid, "timestamp": "2026-05-26T00:00:00+00:00"},
        {"role": "user", "content": "do a long task", "session_id": "default",
         "timestamp": "2026-05-26T00:00:01+00:00"},
        {"role": "assistant", "content": "working on it", "session_id": "default",
         "timestamp": "2026-05-26T00:00:02+00:00"},
    ]
    path = os.path.join(sdir, f"{uuid}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
    before = open(path, encoding="utf-8").read()

    # Simulate a daemon restart: a brand-new manager observing the disk session.
    mgr = _make_manager(workspace=ws)

    # No handle was created for it — nothing auto-resumed.
    assert mgr._handles == {}

    # It shows up in the sidebar as idle (disk-backed listing), addressable.
    entries = [s for s in mgr.list_sessions("proj") if s["session_uuid"] == uuid]
    assert len(entries) == 1
    assert entries[0]["status"] == "idle"

    # The JSONL is byte-for-byte unchanged — no interrupted-warning injection.
    assert open(path, encoding="utf-8").read() == before


@pytest.mark.asyncio
async def test_shutdown_writes_no_state_file(tmp_path):
    """Graceful shutdown persists nothing (no daemon-state.json anywhere)."""
    mgr = _make_manager()
    await mgr.shutdown()
    assert list(tmp_path.glob("**/daemon-state.json")) == []
