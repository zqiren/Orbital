# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: ``/stop`` must terminate a session paused for approval.

``AgentManager.stop_agent`` is state-agnostic: when the loop task is done
(as it is in ``pending_approval``), it calls ``handle.session.stop()`` and
pops the handle (see ``agent_manager.py:1759-1771``). The investigation
in ``TASK/INVESTIGATION-state-model-gaps-verification.md`` confirmed this
is correct by code reading, but no test exercised the path. This test
locks in the contract.

See: ``TASK/TASK-state-model-alignment-fixes.md`` §2.3.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle


def _make_manager_with_paused_handle(project_id="proj_stop_approval",
                                      session_id="default",
                                      tool_call_id="tc_77",
                                      tool_name="write_file"):
    """Build an AgentManager + ProjectHandle in pending_approval state."""
    ws = MagicMock()
    ws.broadcast = MagicMock()
    project_store = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )

    session = MagicMock()
    session.is_stopped = MagicMock(return_value=False)
    session._paused_for_approval = True
    session.stop = MagicMock(
        side_effect=lambda: setattr(session, "is_stopped",
                                      MagicMock(return_value=True))
    )
    session.session_id = session_id

    interceptor = MagicMock()
    interceptor._pending_approvals = {
        tool_call_id: {
            "tool_name": tool_name,
            "tool_args": {"path": "x.txt"},
            "tool_call_id": tool_call_id,
            "what": f"{tool_name}(x.txt)",
            "recent_activity": [],
        }
    }

    # task is DONE in pending_approval (loop exited at approval boundary).
    async def _done_immediately():
        return None
    task = asyncio.get_event_loop().create_task(_done_immediately())

    loop = MagicMock()
    loop.terminate = AsyncMock()

    handle = ProjectHandle(
        session=session,
        loop=loop,
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=interceptor,
        task=task,
    )
    mgr._handles[(project_id, session_id)] = handle
    return mgr, ws, sub_agent_mgr, handle, session, task


@pytest.mark.asyncio
async def test_stop_from_pending_approval_pops_handle_and_broadcasts():
    """/stop on a pending_approval session must:
      - call session.stop() (legacy stop flag)
      - pop the handle from _handles (so it disappears from list_sessions)
      - broadcast agent.status: stopped
      - call sub_agent_manager.stop_all
    """
    project_id = "proj_stop_approval"
    mgr, ws, sub_agent_mgr, handle, session, task = (
        _make_manager_with_paused_handle(project_id=project_id)
    )
    # Let the done-task settle so handle.task.done() is True.
    await task

    # Pre-conditions
    assert (project_id, "default") in mgr._handles, "Sanity: handle planted"
    assert handle.task.done(), "Sanity: task is done (approval-paused)"
    assert session._paused_for_approval is True, (
        "Sanity: handle is in pending_approval"
    )

    await mgr.stop_agent(project_id)

    # Handle popped — session no longer appears in list_sessions.
    assert (project_id, "default") not in mgr._handles, (
        "stop_agent must pop the handle from _handles"
    )
    assert mgr.list_sessions(project_id) == [], (
        "list_sessions must not include the stopped session"
    )

    # session.stop() called (handle.task already done → else branch)
    session.stop.assert_called_once()

    # Sub-agent cleanup happened.
    sub_agent_mgr.stop_all.assert_awaited_once()

    # WS broadcast: agent.status idle (stop folds into idle; the 'stopped'
    # fact is recorded separately in last_terminal_event).
    idle_broadcasts = [
        c for c in ws.broadcast.call_args_list
        if len(c.args) >= 2
        and isinstance(c.args[1], dict)
        and c.args[1].get("type") == "agent.status"
        and c.args[1].get("status") == "idle"
    ]
    assert len(idle_broadcasts) >= 1, (
        f"expected agent.status:idle broadcast, got: "
        f"{ws.broadcast.call_args_list}"
    )
