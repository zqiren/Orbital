# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part B: the 300s busy-watchdog kill is REMOVED.

REWRITTEN from the pre-Piece-3 contract (TASK-cancel-arch-03), which asserted
that exceeding _MAX_IDLE_POLLS calls stop_all. That watchdog killed a
correctly-busy sub-agent mid-work (the orbital-marketing live bug: forced stop
at exactly 300s, then a force-drop leak, then silence). The new invariant:

    NO fixed clock may kill a working or background-running sub-agent.

A sub-agent that previously died at 150 polls (300s) now survives arbitrarily
many polls; the awaiting poll is status-only and exits without any stop call
when the turn closes.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle

PROJECT_ID = "proj-watchdog-test"
SID = "sess-wd-0001"

# Well past the old kill point (150 polls == 300s) — survival here is the
# proof the watchdog is gone.
POLLS_PAST_OLD_KILL = 400


def _make_handle(task_done: bool = True) -> ProjectHandle:
    mock_session = MagicMock()
    mock_session.is_stopped.return_value = False

    mock_task = MagicMock(spec=asyncio.Task)
    mock_task.done.return_value = task_done

    return ProjectHandle(
        session=mock_session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=mock_task,
    )


def _make_manager(sub_agent_status: str = "running"):
    mock_ws = MagicMock()
    mock_ws.broadcast = MagicMock()

    mock_sam = MagicMock()
    mock_sam.stop_all = AsyncMock()
    mock_sam.stop = AsyncMock()
    mock_sam.ADAPTER_STOP_TIMEOUT = 6.0
    mock_sam.list_active = MagicMock(
        return_value=[{"handle": "h1", "display_name": "helper",
                       "status": sub_agent_status}]
    )

    manager = AgentManager(
        project_store=MagicMock(),
        ws_manager=mock_ws,
        sub_agent_manager=mock_sam,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    manager._handles[(PROJECT_ID, SID)] = _make_handle(task_done=True)
    return manager, mock_ws, mock_sam


@pytest.mark.asyncio
async def test_busy_subagent_survives_past_old_watchdog_horizon(monkeypatch):
    """A sub-agent that stays busy for >150 polls is NEVER stopped — it
    previously died here ('Sub-agent poll timeout ... forcing stop'). The
    poll keeps waiting; when the turn finally closes it exits idle with no
    stop call of any kind."""
    manager, mock_ws, mock_sam = _make_manager(sub_agent_status="running")

    real_sleep = asyncio.sleep
    polls = {"n": 0}

    async def counting_sleep(_secs, *a, **k):
        polls["n"] += 1
        if polls["n"] >= POLLS_PAST_OLD_KILL:
            # the turn finally closes, far beyond the old kill horizon
            mock_sam.list_active = MagicMock(
                return_value=[{"handle": "h1", "status": "idle"}])
        await real_sleep(0)

    monkeypatch.setattr(
        "agent_os.daemon_v2.agent_manager.asyncio.sleep", counting_sleep)

    await manager._check_sub_agents_done(PROJECT_ID, session_id=SID)

    assert polls["n"] >= POLLS_PAST_OLD_KILL, "poll must outlive the old 150-poll kill point"
    mock_sam.stop_all.assert_not_awaited()
    mock_sam.stop.assert_not_awaited()
    payloads = [c.args[1] for c in mock_ws.broadcast.call_args_list]
    assert any(p.get("status") == "idle" for p in payloads)


@pytest.mark.asyncio
async def test_background_running_exits_poll_without_kill(monkeypatch):
    """background-running (turn done, work underneath) is not busy: the poll
    exits idle on its first look, with no stop call."""
    manager, mock_ws, mock_sam = _make_manager(
        sub_agent_status="background-running")

    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(
        "agent_os.daemon_v2.agent_manager.asyncio.sleep", _noop)

    await manager._check_sub_agents_done(PROJECT_ID, session_id=SID)

    mock_sam.stop_all.assert_not_awaited()
    mock_sam.stop.assert_not_awaited()
    payloads = [c.args[1] for c in mock_ws.broadcast.call_args_list]
    assert any(p.get("status") == "idle" for p in payloads)


def test_max_idle_polls_constant_removed():
    """The kill-horizon constant itself is gone — nothing can quietly
    reintroduce a poll-bounded force stop."""
    assert not hasattr(AgentManager, "_MAX_IDLE_POLLS")
