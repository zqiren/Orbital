# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: CancelledError in loop task broadcasts 'stopped' status.

Fix 3B — when a loop task is cancelled, _on_loop_done previously returned
silently without broadcasting any status, leaving the frontend stuck on
'running'. Now it broadcasts 'stopped'.
"""

import asyncio
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


def test_cancelled_error_broadcasts_stopped(manager):
    """When a loop task is cancelled, _on_loop_done should broadcast 'stopped'."""
    project_id = "proj_test"
    callback = manager._on_loop_done(project_id)

    # Create a task that we'll cancel
    async def dummy():
        await asyncio.sleep(100)

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(dummy())
        task.cancel()
        # Let the cancellation propagate
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

        # Call the done-callback
        callback(task)

        # Verify that 'stopped' status was broadcast
        calls = manager._ws.broadcast.call_args_list
        assert len(calls) == 1
        call_args = calls[0]
        assert call_args[0][0] == project_id
        payload = call_args[0][1]
        assert payload["type"] == "agent.status"
        assert payload["status"] == "stopped"
    finally:
        loop.close()
