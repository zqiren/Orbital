# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: REST recovery path for approval cards.

When a mobile client misses the WebSocket approval.request event (late connect,
tunnel drop, reconnect), it can fetch the pending approval via
GET /api/v2/agents/{project_id}/pending-approval.

Tests:
1. Full payload is stored in _pending_approvals (not just tool_name/tool_args)
2. get_pending_approval() returns the payload when paused
3. get_pending_approval() returns None when no approval pending
4. Payload is cleared after approve/deny
"""

from unittest.mock import MagicMock

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.agent_manager import AgentManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ws():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    return ws


@pytest.fixture
def interceptor(ws):
    return AutonomyInterceptor(
        preset=Autonomy.SUPERVISED,
        ws_manager=ws,
        project_id="proj_test",
    )


@pytest.fixture
def manager():
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_manager = MagicMock()
    sub_agent_manager.list_active = MagicMock(return_value=[])
    activity_translator = MagicMock()
    process_manager = MagicMock()
    return AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )


def _make_handle_with_interceptor(paused: bool, interceptor):
    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = paused

    handle = MagicMock()
    handle.session = session
    handle.interceptor = interceptor

    task_mock = MagicMock()
    task_mock.done.return_value = True
    handle.task = task_mock

    return handle


# ---------------------------------------------------------------------------
# AutonomyInterceptor stores full payload
# ---------------------------------------------------------------------------

class TestInterceptorStoresFullPayload:

    def test_on_intercept_stores_full_payload(self, interceptor, ws):
        tool_call = {"id": "call_123", "name": "shell", "arguments": {"command": "ls"}}
        recent = [{"role": "assistant", "content": "Let me list files"}]

        interceptor.on_intercept(tool_call, recent, reasoning="Need to see directory")

        stored = interceptor.get_pending("call_123")
        assert stored is not None
        assert stored["tool_name"] == "shell"
        assert stored["tool_args"] == {"command": "ls"}
        assert stored["what"]  # human-readable description should be present
        assert stored["tool_call_id"] == "call_123"
        assert stored["recent_activity"] == recent
        assert stored["reasoning"] == "Need to see directory"

    def test_on_intercept_stores_payload_without_reasoning(self, interceptor, ws):
        tool_call = {"id": "call_456", "name": "read", "arguments": {"file_path": "/tmp/a"}}
        recent = []

        interceptor.on_intercept(tool_call, recent)

        stored = interceptor.get_pending("call_456")
        assert stored is not None
        assert stored["tool_name"] == "read"
        assert "reasoning" not in stored

    def test_remove_pending_clears_payload(self, interceptor, ws):
        tool_call = {"id": "call_789", "name": "shell", "arguments": {"command": "echo hi"}}
        interceptor.on_intercept(tool_call, [])

        assert interceptor.get_pending("call_789") is not None
        interceptor.remove_pending("call_789")
        assert interceptor.get_pending("call_789") is None


# ---------------------------------------------------------------------------
# AgentManager.get_pending_approval()
# ---------------------------------------------------------------------------

class TestGetPendingApproval:

    def test_returns_payload_when_paused(self, manager, ws):
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_test",
        )
        tool_call = {"id": "call_abc", "name": "shell", "arguments": {"command": "rm -rf /"}}
        interceptor.on_intercept(tool_call, [{"role": "assistant", "content": "danger"}], reasoning="cleanup")

        handle = _make_handle_with_interceptor(paused=True, interceptor=interceptor)
        manager._handles["proj_test"] = handle

        result = manager.get_pending_approval("proj_test")
        assert result is not None
        assert result["tool_name"] == "shell"
        assert result["tool_call_id"] == "call_abc"
        assert result["tool_args"] == {"command": "rm -rf /"}
        assert result["what"]
        assert result["reasoning"] == "cleanup"

    def test_returns_none_when_not_paused(self, manager, ws):
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_test",
        )
        handle = _make_handle_with_interceptor(paused=False, interceptor=interceptor)
        manager._handles["proj_test"] = handle

        result = manager.get_pending_approval("proj_test")
        assert result is None

    def test_returns_none_for_unknown_project(self, manager):
        result = manager.get_pending_approval("nonexistent")
        assert result is None

    def test_returns_none_when_paused_but_no_pending(self, manager, ws):
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_test",
        )
        handle = _make_handle_with_interceptor(paused=True, interceptor=interceptor)
        manager._handles["proj_test"] = handle

        result = manager.get_pending_approval("proj_test")
        assert result is None

    def test_returned_dict_is_a_copy(self, manager, ws):
        """Modifying the returned dict must not mutate internal state."""
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_test",
        )
        tool_call = {"id": "call_copy", "name": "shell", "arguments": {"command": "echo"}}
        interceptor.on_intercept(tool_call, [])

        handle = _make_handle_with_interceptor(paused=True, interceptor=interceptor)
        manager._handles["proj_test"] = handle

        result = manager.get_pending_approval("proj_test")
        result["tool_name"] = "MUTATED"

        internal = manager.get_pending_approval("proj_test")
        assert internal["tool_name"] == "shell"
