# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: AgentManager.cancel_message() (T05).

Verifies that cancel_message():
  1. Calls AgentLoop.cancel_turn() exactly once when a turn is in flight.
  2. Returns {"status": "no_agent"} when no handle exists.
  3. Returns {"status": "idle"} when task.done() == True.
  4. Does NOT stop the session, pop the handle, or call stop_all.
  5. Is idempotent — second call returns {"status": "idle"} without double broadcast.

Each test is designed to FAIL before the fix (no cancel_message method) and
PASS after.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager():
    """Return a minimal AgentManager with mocked collaborators."""
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    process_manager.set_session = MagicMock()

    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, ws, sub_agent_mgr


def _make_active_handle(mgr, project_id: str):
    """Inject a fake in-flight handle into _handles."""
    from agent_os.daemon_v2.agent_manager import ProjectHandle

    loop = MagicMock()
    loop.cancel_turn = AsyncMock()
    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session.stop = MagicMock()

    # Create a real asyncio.Task that's not done yet.
    async def _never_done():
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_done())

    handle = ProjectHandle(
        session=session,
        loop=loop,
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
    )
    mgr._handles[project_id] = handle
    return handle, task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_message_calls_cancel_turn():
    """cancel_message() must call loop.cancel_turn() exactly once and broadcast idle."""
    mgr, ws, _ = _make_manager()
    pid = "proj-active"
    handle, task = _make_active_handle(mgr, pid)

    result = await mgr.cancel_message(pid)

    assert result == {"status": "cancelled"}
    handle.loop.cancel_turn.assert_awaited_once()
    ws.broadcast.assert_called_once_with(pid, {
        "type": "agent.status",
        "project_id": pid,
        "status": "idle",
    })

    # Cleanup task
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_cancel_message_no_agent():
    """cancel_message() on unknown project returns no_agent without error or broadcast."""
    mgr, ws, _ = _make_manager()

    result = await mgr.cancel_message("nonexistent-project")

    assert result == {"status": "no_agent"}
    ws.broadcast.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_message_idle_loop():
    """cancel_message() when task is already done returns idle, skip cancel_turn."""
    mgr, ws, _ = _make_manager()
    pid = "proj-idle"
    handle, task = _make_active_handle(mgr, pid)

    # Mark the task as done
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    assert task.done()

    result = await mgr.cancel_message(pid)

    assert result == {"status": "idle"}
    handle.loop.cancel_turn.assert_not_awaited()
    ws.broadcast.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_message_does_not_stop_session():
    """cancel_message() must NOT stop session, pop handle, or call stop_all."""
    mgr, ws, sub_agent_mgr = _make_manager()
    pid = "proj-session-alive"
    handle, task = _make_active_handle(mgr, pid)

    await mgr.cancel_message(pid)

    # Session must not be stopped
    handle.session.stop.assert_not_called()

    # Handle must still be in _handles
    assert pid in mgr._handles

    # Sub-agent stop_all must not be called
    sub_agent_mgr.stop_all.assert_not_awaited()

    # Cleanup
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_cancel_message_idempotent():
    """Second call after loop is idle returns 'idle' without double broadcast."""
    mgr, ws, _ = _make_manager()
    pid = "proj-idempotent"
    handle, task = _make_active_handle(mgr, pid)

    # First call — in-flight, should cancel
    result1 = await mgr.cancel_message(pid)
    assert result1 == {"status": "cancelled"}
    assert ws.broadcast.call_count == 1

    # task cancelled externally to simulate post-cancel state
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass

    # Second call — loop is done
    result2 = await mgr.cancel_message(pid)
    assert result2 == {"status": "idle"}

    # No additional broadcast on second call
    assert ws.broadcast.call_count == 1
