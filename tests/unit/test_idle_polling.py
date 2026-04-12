# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for sub-agent-aware idle status polling.

Verifies that:
- _on_loop_done broadcasts 'waiting' when sub-agents are active
- _on_loop_done broadcasts 'idle' when no sub-agents are active
- _check_sub_agents_done polls and transitions to 'idle' when sub-agents finish
- _check_sub_agents_done terminates when a new loop starts
- _check_sub_agents_done terminates when session is stopped
- _check_sub_agents_done forces idle after max polls
- stop_agent cancels the poll task
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_manager():
    """Create an AgentManager with mock dependencies."""
    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
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


def _make_handle(session_stopped=False, task_done=True):
    """Create a mock ProjectHandle."""
    session = MagicMock()
    session.is_stopped.return_value = session_stopped
    session.pop_queued_messages.return_value = []
    session._paused_for_approval = False
    task = MagicMock()
    task.done.return_value = task_done
    task.exception.return_value = None
    handle = ProjectHandle(
        session=session,
        loop=MagicMock(),
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
    )
    return handle


class TestOnLoopDoneWaiting:
    """_on_loop_done should broadcast 'waiting' when sub-agents are active."""

    def test_broadcasts_waiting_when_sub_agents_active(self):
        mgr, ws, sub_agent_mgr = _make_manager()
        sub_agent_mgr.list_active.return_value = [{"handle": "agent-a"}]

        handle = _make_handle()
        mgr._handles["proj"] = handle

        # Simulate task completing successfully
        mock_task = MagicMock()
        mock_task.exception.return_value = None

        callback = mgr._on_loop_done("proj")
        mock_future = MagicMock()
        with patch("asyncio.ensure_future", return_value=mock_future) as mock_ensure:
            callback(mock_task)
            # Close the coroutine passed to ensure_future to suppress warning
            coro = mock_ensure.call_args[0][0]
            coro.close()

        # Should broadcast waiting, not idle
        calls = ws.broadcast.call_args_list
        assert len(calls) == 1
        event = calls[0][0][1]
        assert event["status"] == "waiting"

        # Should have launched poll task
        mock_ensure.assert_called_once()
        # Task should be stored
        assert mgr._idle_poll_tasks.get("proj") is mock_future

    def test_broadcasts_idle_when_no_sub_agents(self):
        mgr, ws, sub_agent_mgr = _make_manager()
        sub_agent_mgr.list_active.return_value = []

        handle = _make_handle()
        mgr._handles["proj"] = handle

        mock_task = MagicMock()
        mock_task.exception.return_value = None

        callback = mgr._on_loop_done("proj")
        callback(mock_task)

        calls = ws.broadcast.call_args_list
        assert len(calls) == 1
        event = calls[0][0][1]
        assert event["status"] == "idle"

    def test_broadcasts_error_on_exception(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle()
        mgr._handles["proj"] = handle

        mock_task = MagicMock()
        mock_task.exception.return_value = RuntimeError("boom")

        callback = mgr._on_loop_done("proj")
        callback(mock_task)

        calls = ws.broadcast.call_args_list
        assert len(calls) == 1
        event = calls[0][0][1]
        assert event["status"] == "error"
        # Sub-agents should not be checked on error path
        sub_agent_mgr.list_active.assert_not_called()

    def test_broadcast_stopped_on_cancelled_task(self):
        """Fix 3B: CancelledError should broadcast 'stopped' so the frontend
        doesn't get stuck showing a running indicator."""
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle()
        mgr._handles["proj"] = handle

        mock_task = MagicMock()
        mock_task.exception.side_effect = asyncio.CancelledError()

        callback = mgr._on_loop_done("proj")
        callback(mock_task)

        ws.broadcast.assert_called_once()
        payload = ws.broadcast.call_args[0][1]
        assert payload["type"] == "agent.status"
        assert payload["status"] == "stopped"

    def test_broadcasts_stopped_when_session_stopped(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle(session_stopped=True)
        mgr._handles["proj"] = handle

        mock_task = MagicMock()
        mock_task.exception.return_value = None

        callback = mgr._on_loop_done("proj")
        callback(mock_task)

        ws.broadcast.assert_called_once()
        event = ws.broadcast.call_args[0][1]
        assert event["status"] == "stopped"
        # Handle should be cleaned up
        assert "proj" not in mgr._handles


class TestCheckSubAgentsDone:
    """_check_sub_agents_done should poll and transition to idle."""

    @pytest.mark.asyncio
    async def test_transitions_to_idle_when_sub_agents_finish(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle()
        mgr._handles["proj"] = handle

        # First poll: still active. Second poll: done.
        call_count = [0]
        def list_active_side_effect(pid):
            call_count[0] += 1
            if call_count[0] <= 1:
                return [{"handle": "agent-a"}]
            return []
        sub_agent_mgr.list_active.side_effect = list_active_side_effect

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await mgr._check_sub_agents_done("proj")

        # Should broadcast idle
        calls = ws.broadcast.call_args_list
        assert len(calls) == 1
        event = calls[0][0][1]
        assert event["status"] == "idle"

    @pytest.mark.asyncio
    async def test_stops_polling_when_new_loop_starts(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle(task_done=False)  # task not done = loop running
        mgr._handles["proj"] = handle

        sub_agent_mgr.list_active.return_value = [{"handle": "agent-a"}]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await mgr._check_sub_agents_done("proj")

        # Should NOT broadcast — new loop handles its own status
        ws.broadcast.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_polling_when_session_stopped(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle(session_stopped=True)
        mgr._handles["proj"] = handle

        sub_agent_mgr.list_active.return_value = [{"handle": "agent-a"}]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await mgr._check_sub_agents_done("proj")

        ws.broadcast.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_polling_when_handle_removed(self):
        mgr, ws, sub_agent_mgr = _make_manager()
        # No handle registered
        sub_agent_mgr.list_active.return_value = [{"handle": "agent-a"}]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await mgr._check_sub_agents_done("proj")

        ws.broadcast.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_terminates_on_max_polls(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle()
        mgr._handles["proj"] = handle

        # Sub-agents always active (stuck)
        sub_agent_mgr.list_active.return_value = [{"handle": "agent-a"}]

        # Set low cap for testing
        mgr._MAX_IDLE_POLLS = 3

        sleep_count = [0]
        async def counting_sleep(duration):
            sleep_count[0] += 1

        with patch("asyncio.sleep", side_effect=counting_sleep):
            await mgr._check_sub_agents_done("proj")

        # Should have polled exactly 3 times, then forced idle
        assert sleep_count[0] == 3
        calls = ws.broadcast.call_args_list
        assert len(calls) == 1
        event = calls[0][0][1]
        assert event["status"] == "idle"


class TestStopAgentCancelsPolling:
    """stop_agent should cancel any active polling task."""

    @pytest.mark.asyncio
    async def test_stop_agent_cancels_poll_task(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle()
        mgr._handles["proj"] = handle

        # Simulate a running poll task
        mock_poll_task = MagicMock()
        mock_poll_task.done.return_value = False
        mock_poll_task.cancel = MagicMock()
        mgr._idle_poll_tasks["proj"] = mock_poll_task

        await mgr.stop_agent("proj")

        mock_poll_task.cancel.assert_called_once()
        assert "proj" not in mgr._idle_poll_tasks

    @pytest.mark.asyncio
    async def test_stop_agent_no_poll_task_ok(self):
        mgr, ws, sub_agent_mgr = _make_manager()

        handle = _make_handle()
        mgr._handles["proj"] = handle

        # No poll task — should not raise
        await mgr.stop_agent("proj")
        assert "proj" not in mgr._idle_poll_tasks
