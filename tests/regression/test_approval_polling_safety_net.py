# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: Polling safety net for pending approvals during sub-agent runs.

When the WebSocket event for an approval is missed during a live session
(relay disconnect, transient drop), the frontend polls
GET /api/v2/agents/{project_id}/pending-approval every 5 seconds while
the agent status is 'running'. This test validates:

1. The poll response would add an approval to the map when not already present.
2. Dedup: if an approval already exists in the map, the poll does not duplicate it.
3. Polling is conditional — only active when agent status is 'running'.

Since the polling logic lives in a React useEffect, these tests validate the
backend endpoint behavior and the dedup/conditional logic conceptually by
exercising the same AgentManager.get_pending_approval() path.
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


def _make_handle(paused: bool, interceptor):
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
# 1. Poll finds a pending approval not yet in the map
# ---------------------------------------------------------------------------

class TestPollSurfacesNewApproval:
    """Simulates the scenario where the agent has a pending approval and the
    frontend's poll of /pending-approval would discover it."""

    def test_poll_returns_approval_while_agent_running(self, manager, ws):
        """When the agent is paused for approval (but WS event was missed and
        frontend still sees status as 'running'), the REST endpoint returns
        the approval payload that the frontend polling useEffect would consume."""
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_poll",
        )
        tool_call = {
            "id": "call_poll_1",
            "name": "shell",
            "arguments": {"command": "deploy --prod"},
        }
        interceptor.on_intercept(
            tool_call,
            [{"role": "assistant", "content": "Deploying to production"}],
            reasoning="User asked for deployment",
        )

        handle = _make_handle(paused=True, interceptor=interceptor)
        manager._handles["proj_poll"] = handle

        result = manager.get_pending_approval("proj_poll")
        assert result is not None
        # The API endpoint wraps this with {"pending": True, **result}
        # so here we validate the payload shape that get_pending_approval returns
        assert result["tool_call_id"] == "call_poll_1"
        assert result["tool_name"] == "shell"
        assert result["tool_args"] == {"command": "deploy --prod"}
        assert result["reasoning"] == "User asked for deployment"


# ---------------------------------------------------------------------------
# 2. Dedup: existing approval in map is not duplicated
# ---------------------------------------------------------------------------

class TestPollDedup:
    """Validates that the frontend dedup logic works: if the approval is already
    present, repeated polls returning the same tool_call_id would be skipped.

    We simulate this by checking that get_pending_approval returns the same
    tool_call_id on repeated calls — the frontend checks `prev.has(tool_call_id)`
    before adding."""

    def test_repeated_polls_return_same_id(self, manager, ws):
        """Multiple calls to get_pending_approval return the same tool_call_id,
        so the frontend Map.has() check would prevent duplicate entries."""
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_dedup",
        )
        tool_call = {
            "id": "call_dedup_1",
            "name": "write_file",
            "arguments": {"path": "/etc/config"},
        }
        interceptor.on_intercept(tool_call, [])

        handle = _make_handle(paused=True, interceptor=interceptor)
        manager._handles["proj_dedup"] = handle

        # Simulate multiple poll cycles
        result_1 = manager.get_pending_approval("proj_dedup")
        result_2 = manager.get_pending_approval("proj_dedup")

        assert result_1 is not None
        assert result_2 is not None
        assert result_1["tool_call_id"] == result_2["tool_call_id"]

        # Simulate the frontend dedup logic: a Map with existing entry
        approvals_map: dict[str, dict] = {}

        # First poll adds
        if result_1["tool_call_id"] not in approvals_map:
            approvals_map[result_1["tool_call_id"]] = result_1
        assert len(approvals_map) == 1

        # Second poll is deduped
        if result_2["tool_call_id"] not in approvals_map:
            approvals_map[result_2["tool_call_id"]] = result_2
        assert len(approvals_map) == 1  # still 1, no duplicate


# ---------------------------------------------------------------------------
# 3. Polling is conditional on agent activity
# ---------------------------------------------------------------------------

class TestPollConditional:
    """The polling useEffect only runs when agentStatus === 'running'.
    We validate the backend side: when the agent is NOT paused, the endpoint
    returns no approval, so even if the timer fires, no card is added."""

    def test_no_approval_when_agent_idle(self, manager, ws):
        """When there's no active handle (agent idle), poll returns None."""
        result = manager.get_pending_approval("proj_idle")
        assert result is None

    def test_no_approval_when_not_paused(self, manager, ws):
        """Agent is running but not paused for approval — poll returns None."""
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_active",
        )
        handle = _make_handle(paused=False, interceptor=interceptor)
        manager._handles["proj_active"] = handle

        result = manager.get_pending_approval("proj_active")
        assert result is None

    def test_approval_cleared_after_resolve(self, manager, ws):
        """After an approval is resolved (approved/denied), subsequent polls
        should not re-surface the same approval."""
        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=ws,
            project_id="proj_resolve",
        )
        tool_call = {
            "id": "call_resolve_1",
            "name": "shell",
            "arguments": {"command": "rm temp"},
        }
        interceptor.on_intercept(tool_call, [])

        handle = _make_handle(paused=True, interceptor=interceptor)
        manager._handles["proj_resolve"] = handle

        # Before resolve: approval is present
        assert manager.get_pending_approval("proj_resolve") is not None

        # Simulate resolve: remove_pending clears the stored approval
        interceptor.remove_pending("call_resolve_1")

        # After resolve: poll returns None (no stale card)
        result = manager.get_pending_approval("proj_resolve")
        assert result is None
