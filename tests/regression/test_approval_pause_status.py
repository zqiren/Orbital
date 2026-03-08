# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: pending_approval status when loop exits for approval.

When the autonomy interceptor pauses the agent for user approval, the loop
exits normally. _on_loop_done was broadcasting 'idle', making the agent appear
dead in the UI. It should broadcast 'pending_approval' instead.

Similarly, get_run_status should return 'pending_approval' when the session
is paused for approval, so polling clients see the correct status.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


@pytest.fixture
def manager():
    """Create an AgentManager with minimal mocks."""
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr


def _make_handle(paused_for_approval: bool):
    """Create a mock ProjectHandle with session._paused_for_approval set."""
    session = MagicMock()
    session.is_stopped.return_value = False
    session.pop_queued_messages.return_value = []
    session._paused_for_approval = paused_for_approval

    handle = MagicMock()
    handle.session = session

    task_mock = MagicMock()
    task_mock.exception.return_value = None
    task_mock.done.return_value = True
    handle.task = task_mock

    return handle, task_mock


class TestOnLoopDonePendingApproval:
    """_on_loop_done must broadcast pending_approval when paused for approval."""

    def test_broadcasts_pending_approval_when_paused(self, manager):
        handle, task_mock = _make_handle(paused_for_approval=True)
        manager._handles["proj_test"] = handle

        callback = manager._on_loop_done("proj_test")
        callback(task_mock)

        manager._ws.broadcast.assert_called()
        call_args = manager._ws.broadcast.call_args[0]
        assert call_args[0] == "proj_test"
        payload = call_args[1]
        assert payload["type"] == "agent.status"
        assert payload["status"] == "pending_approval"

    def test_does_not_broadcast_idle_when_paused(self, manager):
        handle, task_mock = _make_handle(paused_for_approval=True)
        manager._handles["proj_test"] = handle

        callback = manager._on_loop_done("proj_test")
        callback(task_mock)

        # None of the broadcast calls should contain 'idle'
        for call in manager._ws.broadcast.call_args_list:
            payload = call[0][1]
            if payload.get("type") == "agent.status":
                assert payload["status"] != "idle"

    def test_broadcasts_idle_when_not_paused(self, manager):
        """Normal case: no approval pause -> broadcasts idle."""
        handle, task_mock = _make_handle(paused_for_approval=False)
        manager._handles["proj_test"] = handle

        callback = manager._on_loop_done("proj_test")
        callback(task_mock)

        manager._ws.broadcast.assert_called()
        call_args = manager._ws.broadcast.call_args[0]
        assert call_args[0] == "proj_test"
        assert call_args[1]["status"] == "idle"


class TestGetRunStatusPendingApproval:
    """get_run_status must return 'pending_approval' when paused for approval."""

    def test_returns_pending_approval_when_paused(self, manager):
        handle, _ = _make_handle(paused_for_approval=True)
        manager._handles["proj_test"] = handle

        status = manager.get_run_status("proj_test")
        assert status == "pending_approval"

    def test_returns_idle_when_not_paused(self, manager):
        handle, _ = _make_handle(paused_for_approval=False)
        manager._handles["proj_test"] = handle

        status = manager.get_run_status("proj_test")
        assert status == "idle"
