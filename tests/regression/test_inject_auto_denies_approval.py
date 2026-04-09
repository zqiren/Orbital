# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: inject_message must auto-deny pending approval instead of queuing.

Bug: When the agent is paused for approval and the user sends a new message,
inject_message() used to call session.queue_message() and return "queued".
Since the loop has exited, the queued message was never drained — silently
swallowed. The user thought their message was being processed; in reality
nothing happened.

Fix: inject_message() now auto-denies the pending approval, records the
denial in approval history, appends a visible system message explaining the
dismissal, delivers the new user message, and restarts the loop.

See DIAGNOSIS-request-credential-hang.md (RC3) and
TASK-inject-auto-deny-on-approval-pause.md for the full rationale.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

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


def _build_paused_handle(*, tool_call_id="tc_99", tool_name="write_file",
                         tool_args=None):
    """Create a mock handle paused for approval with one pending approval."""
    if tool_args is None:
        tool_args = {"path": "foo.txt", "content": "bar"}

    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = True
    session.has_result_for = MagicMock(return_value=False)
    session.append_tool_result = MagicMock()
    session.append = MagicMock()
    session.pop_queued_messages = MagicMock(return_value=[])
    session.queue_message = MagicMock()
    session.resume = MagicMock()

    interceptor = MagicMock()
    interceptor._pending_approvals = {
        tool_call_id: {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "what": f"{tool_name}({tool_args})",
            "tool_call_id": tool_call_id,
            "recent_activity": [],
        }
    }
    interceptor.get_pending = MagicMock(
        side_effect=lambda tc_id: interceptor._pending_approvals.get(tc_id)
    )
    interceptor.remove_pending = MagicMock(
        side_effect=lambda tc_id: interceptor._pending_approvals.pop(tc_id, None)
    )

    handle = MagicMock()
    handle.session = session
    handle.interceptor = interceptor

    task_mock = MagicMock()
    task_mock.done.return_value = True  # loop exited when it hit approval
    task_mock.exception.return_value = None
    handle.task = task_mock
    return handle, session, interceptor


class TestInjectAutoDeniesApproval:
    """inject_message must auto-deny pending approval and deliver the message."""

    @pytest.mark.asyncio
    async def test_inject_while_paused_auto_denies_and_delivers(self, manager):
        """When paused for approval, inject_message auto-denies the pending
        approval, appends the user message, and restarts the loop."""
        handle, session, interceptor = _build_paused_handle(
            tool_call_id="tc_99", tool_name="write_file",
        )
        manager._handles["proj_test"] = handle
        manager._start_loop = AsyncMock()
        manager._record_approval_decision = MagicMock()

        result = await manager.inject_message(
            "proj_test", "just do something else",
        )

        # Return value is a dict with status="delivered" and dismissal info
        assert isinstance(result, dict), f"expected dict, got {type(result)}"
        assert result["status"] == "delivered"
        assert result["approval_dismissed"] is True
        assert result["dismissed_tool_call_id"] == "tc_99"

        # Tool result written for the dismissed tool_call_id
        session.append_tool_result.assert_called_once()
        tool_call_args = session.append_tool_result.call_args
        assert tool_call_args[0][0] == "tc_99"
        assert "dismissed" in tool_call_args[0][1].lower()

        # Pending approval removed from interceptor
        interceptor.remove_pending.assert_called_once_with("tc_99")

        # Session resumed
        session.resume.assert_called_once()

        # User message appended
        append_calls = session.append.call_args_list
        user_msg_calls = [
            c for c in append_calls
            if c[0][0].get("role") == "user" and c[0][0].get("content") == "just do something else"
        ]
        assert len(user_msg_calls) == 1, (
            f"expected user message appended once, got {len(user_msg_calls)}"
        )

        # System message appended indicating dismissal
        system_msg_calls = [
            c for c in append_calls
            if c[0][0].get("role") == "system"
            and "dismissed" in str(c[0][0].get("content", "")).lower()
        ]
        assert len(system_msg_calls) >= 1, (
            "expected at least one system message about dismissal"
        )

        # Loop restarted
        manager._start_loop.assert_called_once_with("proj_test")

        # queue_message was NOT called (old behavior)
        session.queue_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_inject_while_paused_records_denial_in_history(self, manager):
        """Auto-denied approval must be recorded in approval history with
        decision='denied' and a deny_reason explaining the cause."""
        handle, session, interceptor = _build_paused_handle(
            tool_call_id="tc_99", tool_name="write_file",
            tool_args={"path": "secret.txt"},
        )
        manager._handles["proj_test"] = handle
        manager._start_loop = AsyncMock()
        manager._record_approval_decision = MagicMock()

        await manager.inject_message("proj_test", "never mind")

        manager._record_approval_decision.assert_called_once()
        call_args = manager._record_approval_decision.call_args
        # Positional args: project_id, tool_name, tool_args, decision
        assert call_args[0][0] == "proj_test"
        assert call_args[0][1] == "write_file"
        assert call_args[0][2] == {"path": "secret.txt"}
        assert call_args[0][3] == "denied"
        # deny_reason keyword contains the phrase "user sent"
        deny_reason = call_args[1].get("deny_reason", "")
        assert "user sent" in deny_reason.lower()

    @pytest.mark.asyncio
    async def test_inject_while_paused_nonce_preserved(self, manager):
        """The nonce passed to inject_message must be attached to the
        appended user message so the WS echo can be deduplicated."""
        handle, session, interceptor = _build_paused_handle()
        manager._handles["proj_test"] = handle
        manager._start_loop = AsyncMock()
        manager._record_approval_decision = MagicMock()

        await manager.inject_message(
            "proj_test", "hi there", nonce="nonce-xyz",
        )

        append_calls = session.append.call_args_list
        user_msg_calls = [
            c for c in append_calls
            if c[0][0].get("role") == "user" and c[0][0].get("content") == "hi there"
        ]
        assert len(user_msg_calls) == 1
        assert user_msg_calls[0][0][0].get("nonce") == "nonce-xyz"

    @pytest.mark.asyncio
    async def test_inject_while_paused_broadcasts_approval_resolved(self, manager):
        """Clients must be told the approval was resolved so the approval
        card can transition to the 'Denied' state."""
        handle, session, interceptor = _build_paused_handle(
            tool_call_id="tc_99",
        )
        manager._handles["proj_test"] = handle
        manager._start_loop = AsyncMock()
        manager._record_approval_decision = MagicMock()

        await manager.inject_message("proj_test", "hi")

        broadcasts = manager._ws.broadcast.call_args_list
        resolved_broadcasts = [
            c for c in broadcasts
            if c[0][1].get("type") == "approval.resolved"
            and c[0][1].get("tool_call_id") == "tc_99"
        ]
        assert len(resolved_broadcasts) == 1
        assert resolved_broadcasts[0][0][1].get("resolution") == "denied"


class TestInjectResponseShape:
    """API-level test: /inject endpoint returns the dismissal shape."""

    def test_inject_response_shape_includes_dismissed_info(self):
        """POST /api/v2/agents/{pid}/inject while paused returns a payload
        including approval_dismissed=True and dismissed_tool_call_id."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agent_os.api.routes import agents_v2

        # Stub project store: project exists
        project_store = MagicMock()
        project_store.get_project.return_value = {
            "workspace": "/tmp/ws",
            "name": "test",
        }
        agents_v2._project_store = project_store

        # Stub sub_agent_manager (no target in request)
        agents_v2._sub_agent_manager = None

        # Stub agent_manager.inject_message to return the dismissal dict
        agent_manager = MagicMock()

        async def _inject(project_id, content, *, nonce=None):
            return {
                "status": "delivered",
                "approval_dismissed": True,
                "dismissed_tool_call_id": "tc_99",
            }

        agent_manager.inject_message = _inject
        agents_v2._agent_manager = agent_manager

        app = FastAPI()
        app.include_router(agents_v2.router)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={"content": "never mind"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "delivered"
        assert body["approval_dismissed"] is True
        assert body["dismissed_tool_call_id"] == "tc_99"

    def test_inject_response_shape_string_status_wrapped(self):
        """When inject_message returns a plain string (no dismissal), the
        response still has the legacy shape {'status': <str>}."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from agent_os.api.routes import agents_v2

        project_store = MagicMock()
        project_store.get_project.return_value = {
            "workspace": "/tmp/ws",
            "name": "test",
        }
        agents_v2._project_store = project_store
        agents_v2._sub_agent_manager = None

        agent_manager = MagicMock()

        async def _inject(project_id, content, *, nonce=None):
            return "delivered"

        agent_manager.inject_message = _inject
        agents_v2._agent_manager = agent_manager

        app = FastAPI()
        app.include_router(agents_v2.router)
        client = TestClient(app)

        resp = client.post(
            "/api/v2/agents/proj_test/inject",
            json={"content": "hi"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"status": "delivered"}
