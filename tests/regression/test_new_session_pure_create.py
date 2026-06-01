# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Steps 2 — new_session is pure-create.

Creating a new session mints a fresh id (and uuid) and returns it. It writes
no file (creation is deferred to the first message), registers no handle, and
takes NO action on any running session — no terminate, no pre-flush, no
sub-agent stop, no session swap. The session becomes real on the first inject
under the returned session_id (inject_message Case 3 auto-starts it).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_manager(tmp_path):
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
        registry=MagicMock(), setup_engine=MagicMock(),
        settings_store=None, credential_store=None,
    )
    mgr._state_file = tmp_path / "daemon-state.json"
    mgr._sub_agent_manager.stop_all = AsyncMock()
    return mgr


@pytest.mark.asyncio
async def test_new_session_mints_ids_without_file_or_handle(tmp_path):
    ws = tmp_path / "ws"
    (ws / "orbital" / "sessions").mkdir(parents=True)
    mgr = _make_manager(tmp_path)
    mgr._project_store.get_project.return_value = {
        "name": "Orbital Marketing", "workspace": str(ws),
    }

    result = await mgr.new_session("proj")

    assert result["status"] == "ok"
    # D2: uuid-only — session_id IS the uuid (the 'sess_' F1 mint is retired).
    assert result["session_uuid"], result
    assert result["session_id"] == result["session_uuid"], result
    assert not result["session_id"].startswith("sess_"), result
    # No handle registered — pure create does not hydrate.
    assert ("proj", result["session_id"]) not in mgr._handles
    # No file on disk — deferred to first message.
    path = os.path.join(
        ProjectPaths(str(ws)).sessions_dir, f"{result['session_uuid']}.jsonl",
    )
    assert not os.path.exists(path), "new_session must not write a file"


@pytest.mark.asyncio
async def test_new_session_does_not_touch_running_session(tmp_path):
    ws = tmp_path / "ws"
    (ws / "orbital" / "sessions").mkdir(parents=True)
    mgr = _make_manager(tmp_path)
    mgr._project_store.get_project.return_value = {
        "name": "Proj", "workspace": str(ws),
    }

    # A live, running default session.
    session = MagicMock()
    running_task = MagicMock()
    running_task.done.return_value = False
    loop = MagicMock()
    loop.terminate = AsyncMock()
    handle = ProjectHandle(
        session=session, loop=loop, provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=running_task,
        config_snapshot={"workspace": str(ws)}, started_at="2026-01-01T00:00:00+00:00",
    )
    mgr._handles[("proj", "default")] = handle

    result = await mgr.new_session("proj")

    assert result["status"] == "ok"
    # The running session's handle is untouched: not terminated, not swapped.
    assert mgr._handles[("proj", "default")] is handle
    assert mgr._handles[("proj", "default")].session is session
    loop.terminate.assert_not_called()
    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    # The minted session is distinct from the running one.
    assert result["session_id"] != "default"
