# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: ``/cancel`` propagates to sub-agents (TASK-cancel-propagates-to-subagents).

Before this fix, ``AgentManager.cancel_message()`` had zero sub-agent
awareness — it never touched ``sub_agent_manager`` or ``_idle_poll_tasks`` in
any branch. When the management loop dispatched a sub-agent and exited, the
daemon entered the "waiting" state (idle-poll alive, sub-agent running) and
``/cancel`` hit branch B2 (``handle.task.done()`` True, ``_paused_for_approval``
False), returning ``{"status": "idle"}`` — a pure no-op. The sub-agent kept
running, bounded only by the 5-minute idle-poll watchdog.

After this fix, ``/cancel`` means "stop all work in this project":

- Waiting state  → cancel the idle-poll, ``stop_all``, write a cancellation
  marker, broadcast idle, return ``cancelled``.
- Running loop   → ``cancel_turn()`` (as before) PLUS ``stop_all`` for any
  sub-agents spawned earlier in the turn.
- Approval paused → dismiss approvals (as before) PLUS ``stop_all``.
- Truly idle     → unchanged no-op, returns ``idle``.

``stop_agent`` is refactored to share the same ``_stop_sub_agents`` helper.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager():
    """Minimal AgentManager with mocked collaborators. stop_all is async."""
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.stop_all = AsyncMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    mgr = AgentManager(
        project_store=MagicMock(),
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
    )
    return mgr, ws, sub_agent_mgr


def _running_handle():
    """Handle whose loop task is alive (running state)."""
    loop = MagicMock()
    loop.cancel_turn = AsyncMock()
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False

    async def _never_done():
        await asyncio.sleep(9999)

    task = asyncio.get_event_loop().create_task(_never_done())
    handle = ProjectHandle(
        session=session, loop=loop, provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=task,
    )
    return handle, task


def _done_handle(*, paused: bool = False):
    """Handle whose loop task is already finished (idle / waiting / paused)."""
    loop = MagicMock()
    loop.cancel_turn = AsyncMock()
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = paused
    session.has_result_for = MagicMock(return_value=False)

    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = True
    task.exception.return_value = None

    handle = ProjectHandle(
        session=session, loop=loop, provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=MagicMock(), task=task,
    )
    return handle, session


def _alive_poll_task():
    """A real, not-done asyncio task to stand in for an idle-poll."""
    async def _poll():
        await asyncio.sleep(9999)
    return asyncio.get_event_loop().create_task(_poll())


_MARKER = {
    "role": "system",
    "content": "[cancelled by user]",
    "source": "management",
    "cancelled_by_user": True,
}


# ---------------------------------------------------------------------------
# Test 1 — Waiting-state cancel stops sub-agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_waiting_state_cancel_stops_subagents():
    mgr, ws, sub_agent_mgr = _make_manager()
    pid = "proj-waiting"
    handle, session = _done_handle()
    mgr._handles[(pid, "default")] = handle
    poll = _alive_poll_task()
    mgr._idle_poll_tasks[pid] = poll

    result = await mgr.cancel_message(pid)

    # stop_all called for the right project + session
    sub_agent_mgr.stop_all.assert_awaited_once_with(pid, session_id="default")

    # Poll task removed from the registry and cancellation requested
    assert pid not in mgr._idle_poll_tasks
    with pytest.raises(asyncio.CancelledError):
        await poll
    assert poll.cancelled()

    # Cancellation marker appended to the session
    session.append.assert_called_once_with(_MARKER)

    # Response contract: cancelled, NOT idle
    assert result == {"status": "cancelled"}

    # idle broadcast emitted
    idle_broadcasts = [
        c for c in ws.broadcast.call_args_list
        if isinstance(c.args[1], dict) and c.args[1].get("status") == "idle"
    ]
    assert idle_broadcasts, f"expected an idle broadcast, got {ws.broadcast.call_args_list}"


# ---------------------------------------------------------------------------
# Test 2 — Running loop + sub-agents: both cancel_turn and stop_all fire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_running_loop_cancels_turn_and_subagents():
    mgr, ws, sub_agent_mgr = _make_manager()
    pid = "proj-running"
    handle, task = _running_handle()
    mgr._handles[(pid, "default")] = handle

    result = await mgr.cancel_message(pid)

    handle.loop.cancel_turn.assert_awaited_once()
    sub_agent_mgr.stop_all.assert_awaited_once_with(pid, session_id="default")
    assert result == {"status": "cancelled"}

    task.cancel()
    with pytest.raises((asyncio.CancelledError, Exception)):
        await task


# ---------------------------------------------------------------------------
# Test 3 — Truly idle (no loop, no poll, no sub-agents)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_truly_idle_cancel_is_noop():
    mgr, ws, sub_agent_mgr = _make_manager()
    pid = "proj-idle"
    handle, session = _done_handle()
    mgr._handles[(pid, "default")] = handle
    # No poll task registered, not paused.

    result = await mgr.cancel_message(pid)

    assert result == {"status": "idle"}
    sub_agent_mgr.stop_all.assert_not_awaited()
    session.append.assert_not_called()
    ws.broadcast.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4 — Approval paused + sub-agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_paused_dismisses_and_stops_subagents():
    mgr, ws, sub_agent_mgr = _make_manager()
    mgr._record_approval_decision = MagicMock()
    pid = "proj-paused"
    handle, session = _done_handle(paused=True)

    tc_id = "tc_7"
    handle.interceptor._pending_approvals = {
        tc_id: {"tool_name": "write_file", "tool_args": {"path": "x"}},
    }
    handle.interceptor.get_pending = MagicMock(
        side_effect=lambda t: handle.interceptor._pending_approvals.get(t)
    )
    handle.interceptor.remove_pending = MagicMock(
        side_effect=lambda t: handle.interceptor._pending_approvals.pop(t, None)
    )
    mgr._handles[(pid, "default")] = handle

    result = await mgr.cancel_message(pid)

    # Approval dismissed
    session.append_tool_result.assert_called_once()
    handle.interceptor.remove_pending.assert_called_once_with(tc_id)
    assert session._paused_for_approval is False
    session.resume.assert_called_once()

    # AND sub-agents stopped
    sub_agent_mgr.stop_all.assert_awaited_once_with(pid, session_id="default")

    assert result == {"status": "cancelled"}


# ---------------------------------------------------------------------------
# Test 5 — Timeout resilience: stop_all hanging does not wedge the response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_all_timeout_still_returns_cancelled(caplog):
    mgr, ws, sub_agent_mgr = _make_manager()
    pid = "proj-timeout"
    handle, session = _done_handle()
    mgr._handles[(pid, "default")] = handle
    mgr._idle_poll_tasks[pid] = _alive_poll_task()

    async def _hang(*_a, **_kw):
        await asyncio.sleep(999)

    sub_agent_mgr.stop_all.side_effect = _hang

    # Enforce a tiny timeout in place of the real 5s budget so the test is
    # fast while still exercising the wait_for/TimeoutError path.
    real_wait_for = asyncio.wait_for

    async def _fast_wait_for(coro, timeout):  # noqa: ARG001
        return await real_wait_for(coro, timeout=0.05)

    with caplog.at_level(logging.WARNING, logger="agent_os.daemon_v2.agent_manager"):
        with patch("agent_os.daemon_v2.agent_manager.asyncio.wait_for", new=_fast_wait_for):
            result = await mgr.cancel_message(pid)

    assert result == {"status": "cancelled"}
    assert any(
        "timed out" in r.message.lower() or "time" in r.message.lower()
        for r in caplog.records if r.levelno >= logging.WARNING
    ), f"expected a timeout warning, got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 6 — stop_agent uses the shared _stop_sub_agents helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_agent_uses_shared_helper():
    mgr, ws, sub_agent_mgr = _make_manager()
    pid = "proj-stopagent"
    handle, session = _done_handle()
    mgr._handles[(pid, "default")] = handle

    mgr._stop_sub_agents = AsyncMock()

    await mgr.stop_agent(pid)

    mgr._stop_sub_agents.assert_awaited_once_with(pid, session_id="default")
    # Behavior preserved: terminal event recorded + idle broadcast.
    assert mgr.get_last_terminal_event(pid) is not None
    idle_broadcasts = [
        c for c in ws.broadcast.call_args_list
        if isinstance(c.args[1], dict) and c.args[1].get("status") == "idle"
    ]
    assert idle_broadcasts
