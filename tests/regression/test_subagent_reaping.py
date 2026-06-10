# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part B: sub-agents are NOT reaped at turn boundaries.

REWRITTEN from the pre-Piece-3 contract (which asserted reap-on-turn-boundary,
per REPORT-subagent-leak-and-slot-gap.md). That policy destroyed live
background work (the orbital-marketing live bug). The new invariant:

- Turn close / loop-done NEVER reaps. Alive-but-idle agents wait for reuse
  (live adapters are reused cleanly — REPORT-piece3-prerequisites.md Phase A);
  inactivity eviction is the only non-teardown reaper.
- When the owning session disappears mid-poll, ONLY idle adapters are reaped;
  working and background-running agents are left alive.
- "background-running" does not count as busy for the awaiting poll (the turn
  is done; on_completed already fired).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager

SID = "sess-reap-0001"


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
    mgr._sub_agent_manager.stop = AsyncMock()
    mgr._sub_agent_manager.ADAPTER_STOP_TIMEOUT = 6.0
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
async def test_idle_poll_no_reap_on_turn_close(monkeypatch):
    """Turn closes (no busy sub-agent) → broadcast idle, NO reap of any kind."""
    monkeypatch.setattr(
        "agent_os.daemon_v2.agent_manager.asyncio.sleep", _noop_sleep)
    mgr = _manager()
    mgr._handles[("proj", SID)] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "claude-code", "status": "idle"}]
    )

    await mgr._check_sub_agents_done("proj", session_id=SID)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    mgr._sub_agent_manager.stop.assert_not_awaited()
    payloads = [c.args[1] for c in mgr._ws.broadcast.call_args_list]
    assert any(p.get("status") == "idle" for p in payloads)


@pytest.mark.asyncio
async def test_background_running_is_not_busy_for_the_poll(monkeypatch):
    """background-running = turn done → the poll exits idle without reaping."""
    monkeypatch.setattr(
        "agent_os.daemon_v2.agent_manager.asyncio.sleep", _noop_sleep)
    mgr = _manager()
    mgr._handles[("proj", SID)] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "claude-code", "status": "background-running"}]
    )

    await mgr._check_sub_agents_done("proj", session_id=SID)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    mgr._sub_agent_manager.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_gone_reaps_only_idle(monkeypatch):
    """Session gone mid-poll: idle adapters reaped (lossless via resume);
    working/background-running adapters left alive."""
    monkeypatch.setattr(
        "agent_os.daemon_v2.agent_manager.asyncio.sleep", _noop_sleep)
    mgr = _manager()
    # No handle registered for ("proj", SID).
    mgr._sub_agent_manager.list_active = MagicMock(return_value=[
        {"handle": "idle-one", "status": "idle"},
        {"handle": "bg-one", "status": "background-running"},
        {"handle": "busy-one", "status": "running"},
    ])

    await mgr._check_sub_agents_done("proj", session_id=SID)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    stopped = [c.args[1] for c in mgr._sub_agent_manager.stop.await_args_list]
    assert stopped == ["idle-one"]


@pytest.mark.asyncio
async def test_on_loop_done_no_reap_when_no_busy():
    """loop-done with only idle sub-agents: idle broadcast, NO reap."""
    mgr = _manager()
    mgr._handles[("proj", SID)] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "claude-code", "status": "idle"}]
    )

    async def _done():
        return None
    task = asyncio.ensure_future(_done())
    await task

    cb = mgr._on_loop_done("proj", session_id=SID)
    cb(task)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    mgr._sub_agent_manager.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_loop_done_background_running_not_busy_no_poll_no_reap():
    """loop-done with a background-running sub-agent: not busy (turn done) →
    no waiting-poll registered, no reap, idle broadcast."""
    mgr = _manager()
    mgr._handles[("proj", SID)] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "claude-code", "status": "background-running"}]
    )

    async def _done():
        return None
    task = asyncio.ensure_future(_done())
    await task

    cb = mgr._on_loop_done("proj", session_id=SID)
    cb(task)
    await asyncio.sleep(0)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    assert ("proj", SID) not in mgr._idle_poll_tasks
    payloads = [c.args[1] for c in mgr._ws.broadcast.call_args_list]
    assert any(p.get("status") == "idle" for p in payloads)


@pytest.mark.asyncio
async def test_on_loop_done_registers_poll_when_running():
    """loop-done with a RUNNING sub-agent registers the awaiting poll and
    does not reap (unchanged behavior, re-keyed post-'default'-retirement)."""
    mgr = _manager()
    mgr._handles[("proj", SID)] = _idle_handle()
    mgr._sub_agent_manager.list_active = MagicMock(
        return_value=[{"handle": "claude-code", "status": "running"}]
    )

    async def _done():
        return None
    task = asyncio.ensure_future(_done())
    await task

    cb = mgr._on_loop_done("proj", session_id=SID)
    cb(task)
    await asyncio.sleep(0)

    mgr._sub_agent_manager.stop_all.assert_not_awaited()
    poll = mgr._idle_poll_tasks.get(("proj", SID))
    assert poll is not None
    poll.cancel()
    try:
        await poll
    except asyncio.CancelledError:
        pass
