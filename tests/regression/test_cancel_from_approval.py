# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: ``/cancel`` must actually cancel a session paused for approval.

Before this fix: ``AgentManager.cancel_message()`` returned
``{"status": "idle"}`` when the loop task was done, without checking for
``_paused_for_approval``. A session in ``pending_approval`` would stay
stuck there: ``_paused_for_approval`` remained ``True``, the slot stayed
held (per the fix-2 slot-predicate update), and the user's cancel click
had no effect.

After this fix: ``cancel_message()`` checks for ``_paused_for_approval``
when the task is done. If set, it dismisses any pending approvals (mirrors
the inject-Case-1b path), clears the flag, resumes the session, broadcasts
``agent.status: idle``, and returns ``{"status": "cancelled"}``.

See: ``TASK/TASK-state-model-alignment-fixes.md`` §2.1.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


@pytest.fixture
def manager():
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


def _build_paused_handle(*, tool_call_id="tc_42", tool_name="write_file",
                         tool_args=None):
    """Mock handle in pending_approval state."""
    if tool_args is None:
        tool_args = {"path": "foo.txt", "content": "bar"}

    session = MagicMock()
    session.is_stopped.return_value = False
    session._paused_for_approval = True
    session.has_result_for = MagicMock(return_value=False)
    session.append_tool_result = MagicMock()
    session.append = MagicMock()
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


@pytest.mark.asyncio
async def test_cancel_from_pending_approval_dismisses_and_returns_cancelled(manager):
    """/cancel on a pending_approval session must clear the flag, dismiss
    the pending approval, broadcast idle, and return ``cancelled``."""
    handle, session, interceptor = _build_paused_handle(
        tool_call_id="tc_42", tool_name="write_file",
    )
    manager._handles[("proj_test", "default")] = handle
    manager._record_approval_decision = MagicMock()

    result = await manager.cancel_message("proj_test")

    # Contract: API must distinguish "cancelled by user" from "wasn't running."
    assert result == {"status": "cancelled"}, (
        f"expected {{status: cancelled}}, got {result}"
    )

    # Approval flag cleared
    assert session._paused_for_approval is False, (
        "_paused_for_approval must be cleared after /cancel"
    )

    # Session.resume() called (clears _paused flag)
    session.resume.assert_called_once()

    # Pending approval dismissed (tool_result written + interceptor entry removed)
    session.append_tool_result.assert_called_once()
    tc_args = session.append_tool_result.call_args[0]
    assert tc_args[0] == "tc_42"
    assert "cancel" in tc_args[1].lower() or "dismiss" in tc_args[1].lower()
    interceptor.remove_pending.assert_called_once_with("tc_42")

    # Approval recorded as denied with cancel-specific reason
    manager._record_approval_decision.assert_called_once()
    call_args = manager._record_approval_decision.call_args
    assert call_args[0][0] == "proj_test"  # project_id
    assert call_args[0][1] == "write_file"  # tool_name
    assert call_args[0][3] == "denied"
    deny_reason = call_args[1].get("deny_reason", "")
    assert "cancel" in deny_reason.lower()

    # WS broadcast: agent.status idle
    status_broadcasts = [
        c for c in manager._ws.broadcast.call_args_list
        if len(c.args) >= 2
        and isinstance(c.args[1], dict)
        and c.args[1].get("type") == "agent.status"
        and c.args[1].get("status") == "idle"
    ]
    assert len(status_broadcasts) >= 1, (
        f"expected at least one agent.status:idle broadcast, got: "
        f"{manager._ws.broadcast.call_args_list}"
    )


@pytest.mark.asyncio
async def test_cancel_with_no_pending_approval_still_returns_idle(manager):
    """When loop task is done AND not paused for approval, /cancel should
    return {"status": "idle"} (unchanged behavior — the existing legitimate
    early-return path).
    """
    handle, session, _interceptor = _build_paused_handle()
    # Override: not actually paused for approval
    session._paused_for_approval = False
    manager._handles[("proj_test", "default")] = handle

    result = await manager.cancel_message("proj_test")

    assert result == {"status": "idle"}, (
        f"non-paused done-task should still return idle, got {result}"
    )
    # No approval-dismissal side effects
    session.resume.assert_not_called()
    session.append_tool_result.assert_not_called()
