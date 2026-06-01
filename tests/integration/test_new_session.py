# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for new_session — the CURRENT pure-create contract.

new_session is "pure-create" (seam 3 / Phase 4, decision D2): it mints a fresh
session IDENTITY only — the canonical uuid (session_id == session_uuid, the
'sess_' F1 mint retired). It writes no JSONL, registers no handle, runs no
turn, and takes NO action on any other session or on Layer-1 files. The session
materializes (handle, loop, file) on the first inject under the returned id.

(The previous behavior — archive old JSONL / create a (project,"default")
handle / broadcast new_session→idle / retry an LLM turn — was removed; these
tests assert the pure-create contract that replaced it.)
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager


def _make_agent_manager(workspace: str = "/tmp/ws", name: str = "Test Project"):
    project_store = MagicMock()
    ws = MagicMock(); ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock(); sub_agent_mgr.stop_all = AsyncMock()
    mgr = AgentManager(
        project_store=project_store, ws_manager=ws,
        sub_agent_manager=sub_agent_mgr, activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    project_store.get_project.return_value = {"workspace": workspace, "name": name}
    return mgr, ws, project_store


def _write_session_file(workspace, session_uuid, messages):
    pp = ProjectPaths(workspace)
    os.makedirs(pp.sessions_dir, exist_ok=True)
    filepath = pp.session_file(session_uuid)
    with open(filepath, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return filepath


class TestNewSessionPureCreate:
    @pytest.mark.asyncio
    async def test_returns_ok_with_uuid_only_ids(self):
        mgr, _, _ = _make_agent_manager()
        result = await mgr.new_session("proj_1")
        assert result["status"] == "ok"
        assert result["session_id"] == result["session_uuid"]  # D2: uuid-only
        assert not result["session_id"].startswith("sess_")

    @pytest.mark.asyncio
    async def test_registers_no_handle_and_writes_no_file(self):
        with tempfile.TemporaryDirectory() as ws:
            mgr, _, _ = _make_agent_manager(workspace=ws)
            result = await mgr.new_session("proj_1")
            assert (("proj_1", result["session_id"]) not in mgr._handles)
            path = ProjectPaths(ws).session_file(result["session_uuid"])
            assert not os.path.exists(path), "pure-create must not write a JSONL"

    @pytest.mark.asyncio
    async def test_broadcasts_nothing(self):
        mgr, ws, _ = _make_agent_manager()
        await mgr.new_session("proj_1")
        assert ws.broadcast.call_count == 0, "pure-create must not broadcast"

    @pytest.mark.asyncio
    async def test_does_not_touch_existing_session_files(self):
        with tempfile.TemporaryDirectory() as ws:
            mgr, _, _ = _make_agent_manager(workspace=ws)
            old = _write_session_file(ws, "proj_old_aaaa", [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ])
            before = open(old).read()
            await mgr.new_session("proj_1")
            assert open(old).read() == before, "must not archive/rewrite other sessions"

    @pytest.mark.asyncio
    async def test_does_not_touch_a_running_session_handle(self):
        mgr, _, _ = _make_agent_manager()
        running = MagicMock()
        running.task = MagicMock(); running.task.done.return_value = False
        mgr._handles[("proj_1", "proj_existing_bbbb")] = running
        await mgr.new_session("proj_1")
        # The running handle is untouched; no new handle was registered.
        assert mgr._handles[("proj_1", "proj_existing_bbbb")] is running
        assert len([k for k in mgr._handles if k[0] == "proj_1"]) == 1
