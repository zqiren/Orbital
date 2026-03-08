# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: messages queued during session-end routine must be drained.

When a user sends a message while the agent loop's session-end LLM call
is still running, the message is queued (Case 1 in inject_message).
But _on_loop_done never checked the queue, so the message was silently
lost — the agent went idle with an unprocessed user message.

This test verifies that _on_loop_done drains queued messages and
restarts the loop to process them.
"""

from unittest.mock import MagicMock, patch

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


def test_on_loop_done_checks_queued_messages(manager):
    """_on_loop_done must check for queued messages when loop finishes."""
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pop_queued_messages.return_value = []

    handle = MagicMock()
    handle.session = session

    task = MagicMock()
    task.exception.return_value = None
    handle.task = task

    manager._handles["proj_test"] = handle

    callback = manager._on_loop_done("proj_test")
    callback(task)

    # _on_loop_done must call pop_queued_messages to check for pending work
    session.pop_queued_messages.assert_called_once()


def test_on_loop_done_appends_queued_messages_to_session(manager):
    """_on_loop_done must append queued messages to the session."""
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = False
    session.pop_queued_messages.return_value = ["yes please proceed"]
    session.append = MagicMock()

    handle = MagicMock()
    handle.session = session

    task = MagicMock()
    task.exception.return_value = None
    handle.task = task

    manager._handles["proj_test"] = handle

    callback = manager._on_loop_done("proj_test")

    # Patch ensure_future since we're not in an async context
    mock_future = MagicMock()
    with patch("asyncio.ensure_future", return_value=mock_future) as mock_ensure:
        callback(task)
        # Close the coroutine to suppress warning
        if mock_ensure.call_args:
            coro = mock_ensure.call_args[0][0]
            coro.close()

    # The queued message must be appended to the session
    session.append.assert_called()
    appended = session.append.call_args[0][0]
    assert appended["role"] == "user"
    assert appended["content"] == "yes please proceed"


def test_on_loop_done_broadcasts_idle_when_no_queued_messages(manager):
    """_on_loop_done broadcasts idle when queue is empty (no regression)."""
    session = MagicMock()
    session.is_stopped.return_value = False
    session.pop_queued_messages.return_value = []
    session._paused_for_approval = False

    handle = MagicMock()
    handle.session = session

    task = MagicMock()
    task.exception.return_value = None
    handle.task = task

    manager._handles["proj_test"] = handle

    callback = manager._on_loop_done("proj_test")
    callback(task)

    # Should broadcast idle (no queued messages to process)
    manager._ws.broadcast.assert_called()
    call_args = manager._ws.broadcast.call_args[0]
    assert call_args[0] == "proj_test"
    assert call_args[1]["status"] == "idle"
