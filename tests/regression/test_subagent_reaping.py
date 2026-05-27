# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent processes are reaped at the turn boundary, not leaked.

Per REPORT-subagent-leak-and-slot-gap.md + REPORT-is-idle-and-adapter-lifecycle.md:
a sub-agent's `is_idle()==True` means its turn completed (the ACP `send()`
returned, `_pending_response=False`) — it is safe to reap. The idle-poll
(`_check_sub_agents_done`) and the loop-done callback (`_on_loop_done`) must
call `stop_all` when no busy sub-agent remains, instead of leaving the process
alive-but-idle. Busy sub-agents are never reaped.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


async def _noop_sleep(*_a, **_k):
    return None


def _manager() -> AgentManager:
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    mgr._ws.broadcast = MagicMock()
    mgr._sub_agent_manager.stop_all = AsyncMock()
    return mgr


def _idle_handle() -> MagicMock:
    h = MagicMock()
    h.session.is_stopped.return_value = False
    h.session._paused_for_approval = False
    h.session.pop_deferred_messages.return_value = []
    h.session.pop_queued_messages.return_value = []
    h.task = MagicMock()
    h.task.done.return_value = True  # loop finished
    return h


@pytest.mark.asyncio
async def test_idle_poll_reaps_on_no_busy(monkeypatch):
    """_check_sub_agents_done: when list_active shows only idle adapters, call stop_all."""
    monkeypatch.setattr("agent_os.daemon_v2.agent_manager.asyncio.sleep", _noop_sleep)
    mgr = _manager()
    sk = ("proj", "default")
    mgr._handles[sk] = _idle_handle()
    # Alive-but-idle sub-agent → list_active reports it as "idle" (not busy).
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "codex", "status": "idle"}]
    )

    await mgr._check_sub_agents_done("proj", session_id="default")

    mgr._sub_agent_manager.stop_all.assert_awaited_once_with("proj", session_id="default")


@pytest.mark.asyncio
async def test_idle_poll_reaps_when_handle_gone(monkeypatch):
    """_check_sub_agents_done: if the handle was removed while sub-agents were
    alive, reap before returning rather than orphaning the process."""
    monkeypatch.setattr("agent_os.daemon_v2.agent_manager.asyncio.sleep", _noop_sleep)
    mgr = _manager()
    # No handle in _handles for ("proj", "default").
    mgr._sub_agent_manager.list_active = MagicMock(return_value=[])

    await mgr._check_sub_agents_done("proj", session_id="default")

    mgr._sub_agent_manager.stop_all.assert_awaited_once_with("proj", session_id="default")


@pytest.mark.asyncio
async def test_on_loop_done_reaps_when_no_busy():
    """_on_loop_done's no-busy branch reaps alive-but-idle sub-agents."""
    mgr = _manager()
    sk = ("proj", "default")
    mgr._handles[sk] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "codex", "status": "idle"}]
    )

    # Build a genuinely-done task with no exception.
    async def _done():
        return None
    task = asyncio.ensure_future(_done())
    await task

    cb = mgr._on_loop_done("proj", session_id="default")
    cb(task)
    # Reaping is scheduled via ensure_future inside the sync callback.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    mgr._sub_agent_manager.stop_all.assert_awaited_once_with("proj", session_id="default")


@pytest.mark.asyncio
async def test_on_loop_done_does_not_reap_when_busy():
    """_on_loop_done with a busy sub-agent registers the idle-poll and does NOT reap."""
    mgr = _manager()
    sk = ("proj", "default")
    mgr._handles[sk] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "codex", "status": "running"}]
    )

    async def _done():
        return None
    task = asyncio.ensure_future(_done())
    await task

    cb = mgr._on_loop_done("proj", session_id="default")
    cb(task)
    await asyncio.sleep(0)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    poll = mgr._idle_poll_tasks.get("proj")
    assert poll is not None
    # Clean up the registered poll task (it would otherwise sleep then reap).
    poll.cancel()
    try:
        await poll
    except asyncio.CancelledError:
        pass
