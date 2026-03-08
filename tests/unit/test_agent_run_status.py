# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for AgentManager.get_run_status().

Fix 1B prerequisite — the new get_run_status() method returns the current
runtime status for a project, used by the status-on-subscribe feature.
"""

import asyncio
from unittest.mock import MagicMock, PropertyMock

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


class TestGetRunStatus:
    def test_returns_idle_when_no_handle(self, manager):
        """Projects with no handle should report idle."""
        assert manager.get_run_status("nonexistent") == "idle"

    def test_returns_running_when_task_active(self, manager):
        """Projects with an active loop task should report running."""
        handle = MagicMock()
        handle.session.is_stopped.return_value = False
        handle.session._paused_for_approval = False
        handle.task = MagicMock()
        handle.task.done.return_value = False
        manager._handles["proj_1"] = handle

        assert manager.get_run_status("proj_1") == "running"

    def test_returns_idle_when_task_done(self, manager):
        """Projects with a done task and no sub-agents should report idle."""
        handle = MagicMock()
        handle.session.is_stopped.return_value = False
        handle.session._paused_for_approval = False
        handle.task = MagicMock()
        handle.task.done.return_value = True
        manager._handles["proj_1"] = handle

        assert manager.get_run_status("proj_1") == "idle"

    def test_returns_stopped_when_session_stopped(self, manager):
        """Projects with a stopped session should report stopped."""
        handle = MagicMock()
        handle.session.is_stopped.return_value = True
        handle.task = None
        manager._handles["proj_1"] = handle

        assert manager.get_run_status("proj_1") == "stopped"

    def test_returns_waiting_when_poll_task_active(self, manager):
        """Projects with active sub-agent polling should report waiting."""
        handle = MagicMock()
        handle.session.is_stopped.return_value = False
        handle.session._paused_for_approval = False
        handle.task = MagicMock()
        handle.task.done.return_value = True
        manager._handles["proj_1"] = handle

        poll_task = MagicMock()
        poll_task.done.return_value = False
        manager._idle_poll_tasks["proj_1"] = poll_task

        assert manager.get_run_status("proj_1") == "waiting"
