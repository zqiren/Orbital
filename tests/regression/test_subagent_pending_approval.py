# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: REST recovery for sub-agent pending approvals.

GET /api/v2/agents/{project_id}/pending-approval must include sub-agent
approvals when the main agent has none pending. This ensures mobile clients
can recover missed sub-agent approval WebSocket events on page refresh.

Tests:
1. get_pending_sub_agent_approval() returns approval data with source field
2. get_pending_sub_agent_approval() returns None when no sub-agent has pending
3. REST endpoint falls back to sub-agent approval when main agent has none
4. Main agent approval takes priority over sub-agent approval
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_transport(pending_request_id: str | None = None,
                         tool_name: str = "",
                         tool_input: dict | None = None):
    """Create a mock transport mimicking SDKTransport approval state."""
    transport = MagicMock()
    transport.respond_to_permission = AsyncMock()

    if pending_request_id:
        # Use a MagicMock as a presence indicator. asyncio.Future() requires a
        # running event loop in modern Python and trips the "no current event
        # loop" deprecation when constructed in sync test code from inside a
        # full pytest run; the future is never awaited here, so a stand-in is
        # sufficient.
        future = MagicMock()
        transport._pending_approvals = {pending_request_id: future}
        transport._pending_approval_data = {
            pending_request_id: {
                "request_id": pending_request_id,
                "tool_name": tool_name,
                "tool_input": tool_input or {},
            }
        }
    else:
        transport._pending_approvals = {}
        transport._pending_approval_data = {}

    return transport


def _make_mock_adapter(transport=None):
    """Create a mock CLIAdapter with optional transport."""
    adapter = MagicMock()
    adapter._transport = transport
    adapter.is_alive.return_value = True
    adapter.is_idle.return_value = False
    return adapter


def _make_sub_agent_manager():
    """Create a SubAgentManager with mocked dependencies."""
    process_manager = MagicMock()
    return SubAgentManager(process_manager=process_manager)


def _make_agent_manager():
    """Create an AgentManager with mocked dependencies."""
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
# SubAgentManager.get_pending_sub_agent_approval()
# ---------------------------------------------------------------------------

class TestGetPendingSubAgentApproval:

    def test_returns_approval_with_source(self):
        sam = _make_sub_agent_manager()
        transport = _make_mock_transport(
            pending_request_id="req_001",
            tool_name="Edit",
            tool_input={"file_path": "/tmp/foo.py", "old_string": "a", "new_string": "b"},
        )
        adapter = _make_mock_adapter(transport=transport)
        sam._adapters["proj_x"] = {"claude-code": adapter}

        result = sam.get_pending_sub_agent_approval("proj_x")
        assert result is not None
        assert result["tool_call_id"] == "req_001"
        assert result["tool_name"] == "Edit"
        assert result["tool_args"] == {"file_path": "/tmp/foo.py", "old_string": "a", "new_string": "b"}
        assert result["source"] == "claude-code"
        assert "what" in result

    def test_returns_none_when_no_pending(self):
        sam = _make_sub_agent_manager()
        transport = _make_mock_transport()  # no pending approvals
        adapter = _make_mock_adapter(transport=transport)
        sam._adapters["proj_x"] = {"claude-code": adapter}

        result = sam.get_pending_sub_agent_approval("proj_x")
        assert result is None

    def test_returns_none_for_unknown_project(self):
        sam = _make_sub_agent_manager()
        result = sam.get_pending_sub_agent_approval("nonexistent")
        assert result is None

    def test_returns_none_when_adapter_has_no_transport(self):
        sam = _make_sub_agent_manager()
        adapter = _make_mock_adapter(transport=None)
        sam._adapters["proj_x"] = {"legacy-agent": adapter}

        result = sam.get_pending_sub_agent_approval("proj_x")
        assert result is None

    def test_skips_adapters_without_pending(self):
        """When multiple sub-agents exist, only the one with pending is returned."""
        sam = _make_sub_agent_manager()
        transport_clean = _make_mock_transport()  # no pending
        transport_pending = _make_mock_transport(
            pending_request_id="req_002",
            tool_name="Bash",
            tool_input={"command": "ls"},
        )
        adapter_clean = _make_mock_adapter(transport=transport_clean)
        adapter_pending = _make_mock_adapter(transport=transport_pending)
        sam._adapters["proj_x"] = {
            "agent-a": adapter_clean,
            "agent-b": adapter_pending,
        }

        result = sam.get_pending_sub_agent_approval("proj_x")
        assert result is not None
        assert result["source"] == "agent-b"
        assert result["tool_name"] == "Bash"

    def test_handles_transport_without_metadata_dict(self):
        """If transport has _pending_approvals but no _pending_approval_data,
        still returns a result with empty tool info."""
        sam = _make_sub_agent_manager()
        transport = MagicMock()
        transport._pending_approvals = {"req_003": MagicMock()}  # presence indicator (see _make_mock_transport note)
        # Deliberately no _pending_approval_data attribute
        del transport._pending_approval_data
        adapter = _make_mock_adapter(transport=transport)
        sam._adapters["proj_x"] = {"agent-c": adapter}

        result = sam.get_pending_sub_agent_approval("proj_x")
        assert result is not None
        assert result["tool_call_id"] == "req_003"
        assert result["tool_name"] == ""
        assert result["source"] == "agent-c"


# ---------------------------------------------------------------------------
# Main agent approval still takes priority in REST endpoint logic
# ---------------------------------------------------------------------------

class TestMainAgentPriority:

    def test_main_agent_approval_returned_when_present(self):
        """get_pending_approval() on AgentManager returns main agent's approval."""
        manager = _make_agent_manager()
        ws = MagicMock()
        ws.broadcast = MagicMock()
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_test",
        )
        tool_call = {"id": "call_main", "name": "shell", "arguments": {"command": "ls"}}
        interceptor.on_intercept(tool_call, [])

        handle = _make_handle_with_interceptor(paused=True, interceptor=interceptor)
        manager._handles["proj_test"] = handle

        result = manager.get_pending_approval("proj_test")
        assert result is not None
        assert result["tool_call_id"] == "call_main"
        assert result["tool_name"] == "shell"
        # No "source" field for main agent approvals
        assert "source" not in result

    def test_main_agent_returns_none_falls_through(self):
        """When main agent has no pending, get_pending_approval returns None
        so the route can fall back to sub-agent check."""
        manager = _make_agent_manager()
        result = manager.get_pending_approval("proj_no_handle")
        assert result is None
