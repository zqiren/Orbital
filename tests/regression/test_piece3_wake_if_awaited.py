# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part C: terminal sub-agent events are never silently dropped.

Pre-Piece-3, ``inject_system_message`` returned "no_session" and dropped the
event ENTIRELY (not even to disk) when the management handle was gone —
unlike user-facing ``inject_message``, it did not hydrate. An
evicted-while-straggling completion vanished without trace. New invariant:
hydrate-or-append — the event lands on disk first (durable), then the loop is
started so an awaiting session is actually woken.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.agent.project_paths import ProjectPaths

SID = "sess-wake-0001"
EVENT = "[Sub-agent] claude-code completed. Summary: straggler done."


def _manager(workspace: str) -> AgentManager:
    store = MagicMock()
    store.get_project.return_value = {"workspace": workspace}
    mgr = AgentManager(
        project_store=store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return mgr


def _write_session_jsonl(workspace: str, session_uuid: str) -> str:
    sessions_dir = ProjectPaths(workspace).sessions_dir
    os.makedirs(sessions_dir, exist_ok=True)
    path = os.path.join(sessions_dir, f"{session_uuid}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "type": "session_start", "session_id": SID,
            "session_uuid": session_uuid,
            "timestamp": "2026-06-06T00:00:00+00:00",
        }) + "\n")
        fh.write(json.dumps({
            "role": "user", "content": "dispatch something",
            "source": "user", "session_id": SID,
            "session_uuid": session_uuid,
            "timestamp": "2026-06-06T00:00:01+00:00",
        }) + "\n")
    return path


@pytest.mark.asyncio
async def test_handle_gone_hydrates_appends_and_wakes(tmp_path):
    """Handle evicted + session on disk: the event is appended to the JSONL
    and the loop is started (the wake)."""
    workspace = str(tmp_path)
    mgr = _manager(workspace)
    path = _write_session_jsonl(workspace, SID)

    mgr._build_agent_config_from_project = MagicMock(return_value=MagicMock())
    mgr.start_agent = AsyncMock()

    result = await mgr.inject_system_message("proj_w", EVENT, session_id=SID)

    assert result == "delivered"
    # Durable on disk:
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    sys_msgs = [l for l in lines
                if l.get("role") == "system" and l.get("content") == EVENT]
    assert sys_msgs, "terminal event must be appended to the session JSONL"
    assert sys_msgs[0].get("source") == "daemon"
    # Woken:
    mgr.start_agent.assert_awaited_once()
    kwargs = mgr.start_agent.await_args.kwargs
    assert kwargs.get("session") is not None
    assert kwargs.get("initial_message") is None


@pytest.mark.asyncio
async def test_wake_failure_still_persists_event(tmp_path):
    """If the loop start fails, the event must already be durable on disk."""
    workspace = str(tmp_path)
    mgr = _manager(workspace)
    path = _write_session_jsonl(workspace, SID)

    mgr._build_agent_config_from_project = MagicMock(return_value=MagicMock())
    mgr.start_agent = AsyncMock(side_effect=RuntimeError("wake blew up"))

    result = await mgr.inject_system_message("proj_w", EVENT, session_id=SID)

    assert result == "persisted"
    lines = [json.loads(l) for l in open(path, encoding="utf-8")]
    assert any(l.get("content") == EVENT for l in lines)


@pytest.mark.asyncio
async def test_no_session_anywhere_logs_and_returns(tmp_path, caplog):
    """No handle AND no on-disk session: returns no_session with a WARNING —
    never a silent drop."""
    import logging
    mgr = _manager(str(tmp_path))
    mgr.start_agent = AsyncMock()

    with caplog.at_level(logging.WARNING,
                         logger="agent_os.daemon_v2.agent_manager"):
        result = await mgr.inject_system_message(
            "proj_w", EVENT, session_id="sess-missing")

    assert result == "no_session"
    mgr.start_agent.assert_not_awaited()
    assert any("could not be delivered" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_live_idle_handle_still_appends_and_wakes(tmp_path):
    """Regression guard: the existing live-handle paths are unchanged —
    idle loop appends + hot-resumes."""
    mgr = _manager(str(tmp_path))
    handle = MagicMock()
    handle.task = None
    mgr._handles[("proj_w", SID)] = handle
    mgr._start_loop = AsyncMock()

    result = await mgr.inject_system_message("proj_w", EVENT, session_id=SID)

    assert result == "delivered"
    handle.session.append.assert_called_once()
    mgr._start_loop.assert_awaited_once_with("proj_w", session_id=SID)
