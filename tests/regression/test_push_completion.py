# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Push path: a completed sub-agent's result is injected and the loop resumes.

`lifecycle_observer.on_completed` formats a "[Sub-agent] … completed. Summary:
…" message and calls `AgentManager.inject_system_message` — the push target.
When the management loop is idle, the message is appended and `_start_loop` is
called (resume). When the loop is still running, the message is deferred (drained
on the loop's next exit) — no crash, no lost message.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


def _manager() -> AgentManager:
    mgr = AgentManager(
        project_store=MagicMock(), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._ws.broadcast = MagicMock()
    mgr._start_loop = AsyncMock()
    return mgr


def _handle(task_done: bool) -> MagicMock:
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.task = MagicMock()
    h.task.done.return_value = task_done
    return h


@pytest.mark.asyncio
async def test_completion_injects_system_message_and_resumes():
    """Idle loop → completion appended as a system message + loop restarted."""
    mgr = _manager()
    sk = ("proj", "default")
    handle = _handle(task_done=True)
    mgr._handles[sk] = handle

    content = "[Sub-agent] claude-code completed. Summary: Found 4 products that listen live."
    result = await mgr.inject_system_message("proj", content, session_id="default")

    assert handle.session.append.called
    appended = handle.session.append.call_args[0][0]
    assert appended["role"] == "system"
    assert "claude-code completed" in appended["content"]
    assert "Found 4 products" in appended["content"]
    mgr._start_loop.assert_awaited_once()
    assert result == "delivered"


@pytest.mark.asyncio
async def test_completion_while_running_is_deferred_no_crash():
    """Running loop → completion is deferred (not lost), no restart, no crash."""
    mgr = _manager()
    sk = ("proj", "default")
    handle = _handle(task_done=False)  # loop still running
    mgr._handles[sk] = handle

    content = "[Sub-agent] codex completed. Summary: refactor done."
    result = await mgr.inject_system_message("proj", content, session_id="default")

    # Deferred for safe insertion after the current tool batch; not appended now.
    assert handle.session.defer_message.called
    deferred_content = handle.session.defer_message.call_args[0][0]
    assert "codex completed" in deferred_content
    mgr._start_loop.assert_not_awaited()
    assert result == "deferred"


@pytest.mark.asyncio
async def test_completion_with_no_handle_is_no_op():
    """No handle for the session → no crash, returns no_session (message dropped)."""
    mgr = _manager()
    # No handle registered for ("proj", "default").
    result = await mgr.inject_system_message("proj", "[Sub-agent] x completed.", session_id="default")
    assert result == "no_session"
    mgr._start_loop.assert_not_awaited()
