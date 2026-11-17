# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A worker's connector write asks the user through the existing sub-agent
approval card, whatever the worker's own permission settings.

The card is the same ``approval.request`` event a worker's own permission
prompt raises (``source`` = the worker), answered through the same
``resolve_sub_agent_approval`` the approve / deny routes already call, and
recoverable through ``get_pending_sub_agent_approval``. The worker's call
blocks until the user answers. "Approve all" (Orbital's ten-minute bypass for
that worker) counts as an answer. Stopping the worker denies what it was
waiting on.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.dispatch_tools import DispatchContext
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

CTX = DispatchContext("p1", "s1", "codex", can_dispatch=False, connectors=("gmail",))


def _mgr():
    mgr = SubAgentManager(process_manager=MagicMock(), registry=None,
                          setup_engine=None, project_store=None)
    mgr._ws_manager = MagicMock()
    return mgr


def _events(mgr, kind):
    return [c.args[1] for c in mgr._ws_manager.broadcast.call_args_list
            if c.args[1].get("type") == kind]


async def _ask(mgr, ctx=CTX):
    task = asyncio.create_task(mgr.request_connector_approval(
        ctx, "gmail.send_message", {"to": "a@b.c", "body": "hi"}))
    for _ in range(5):
        await asyncio.sleep(0)
    return task


async def test_a_write_raises_the_sub_agent_approval_card():
    mgr = _mgr()
    task = await _ask(mgr)
    assert not task.done()
    [card] = _events(mgr, "approval.request")
    assert card["project_id"] == "p1" and card["session_id"] == "s1"
    assert card["source"] == "codex"
    assert card["tool_name"] == "gmail.send_message"
    assert card["tool_args"] == {"to": "a@b.c", "body": "hi"}
    assert card["what"] == "Sub-agent codex requests approval"
    assert card["tool_call_id"]
    task.cancel()


@pytest.mark.parametrize("approved", [True, False])
async def test_the_answer_reaches_the_waiting_call(approved):
    mgr = _mgr()
    task = await _ask(mgr)
    tcid = _events(mgr, "approval.request")[0]["tool_call_id"]
    assert await mgr.resolve_sub_agent_approval("p1", tcid, approved=approved,
                                                session_id="s1") is True
    assert await task is approved
    # Answered once; a second answer finds nothing.
    assert await mgr.resolve_sub_agent_approval("p1", tcid, approved=True) is False


async def test_a_session_less_lookup_finds_it_and_another_project_does_not():
    mgr = _mgr()
    task = await _ask(mgr)
    tcid = _events(mgr, "approval.request")[0]["tool_call_id"]
    assert await mgr.resolve_sub_agent_approval("p2", tcid, approved=True) is False
    assert await mgr.resolve_sub_agent_approval("p1", tcid, approved=True,
                                                session_id="other") is False
    assert await mgr.resolve_sub_agent_approval("p1", tcid, approved=True) is True
    assert await task is True


async def test_the_pending_card_is_recoverable_over_rest():
    mgr = _mgr()
    task = await _ask(mgr)
    tcid = _events(mgr, "approval.request")[0]["tool_call_id"]
    for sid in (None, "s1"):
        pending = mgr.get_pending_sub_agent_approval("p1", session_id=sid)
        assert pending == {
            "tool_call_id": tcid, "tool_name": "gmail.send_message",
            "tool_args": {"to": "a@b.c", "body": "hi"},
            "what": "Sub-agent codex requests approval: gmail.send_message",
            "source": "codex",
        }
    assert mgr.get_pending_sub_agent_approval("p1", session_id="other") is None
    await mgr.resolve_sub_agent_approval("p1", tcid, approved=False)
    await task
    assert mgr.get_pending_sub_agent_approval("p1") is None


async def test_the_worker_reads_as_waiting_on_the_user():
    mgr = _mgr()
    adapter = MagicMock()
    adapter._transport = None
    assert mgr._awaiting_approval(adapter, "p1", "s1", "codex") is False
    task = await _ask(mgr)
    assert mgr._awaiting_approval(adapter, "p1", "s1", "codex") is True
    assert mgr._awaiting_approval(adapter, "p1", "s1", "claude-code") is False
    tcid = _events(mgr, "approval.request")[0]["tool_call_id"]
    await mgr.resolve_sub_agent_approval("p1", tcid, approved=True)
    await task
    assert mgr._awaiting_approval(adapter, "p1", "s1", "codex") is False


async def test_approve_all_covers_the_next_writes_of_that_worker_only():
    mgr = _mgr()
    task = await _ask(mgr)
    tcid = _events(mgr, "approval.request")[0]["tool_call_id"]
    await mgr.resolve_sub_agent_approval("p1", tcid, approved=True, approve_all=True)
    assert await task is True
    assert await mgr.request_connector_approval(CTX, "gmail.send_message", {}) is True
    assert len(_events(mgr, "approval.request")) == 1
    other = DispatchContext("p1", "s1", "claude-code", connectors=("gmail",))
    task2 = await _ask(mgr, other)
    assert not task2.done()
    task2.cancel()


async def test_stopping_the_worker_denies_what_it_waited_on():
    mgr = _mgr()
    mgr._dispatch_tokens["tok"] = CTX
    task = await _ask(mgr)
    adapter = MagicMock()
    adapter._dispatch_token = "tok"
    mgr._revoke_dispatch_token(adapter)
    assert await task is False
    resolved = _events(mgr, "approval.resolved")
    assert resolved and resolved[-1]["resolution"] == "denied"


async def test_a_call_abandoned_by_the_worker_takes_the_card_down():
    mgr = _mgr()
    task = await _ask(mgr)
    tcid = _events(mgr, "approval.request")[0]["tool_call_id"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert mgr.get_pending_sub_agent_approval("p1") is None
    resolved = _events(mgr, "approval.resolved")
    assert resolved[-1]["tool_call_id"] == tcid
