# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.4: a seat-holder with a RUNNING child is not quiescent.

The memory reminder (spec 112 A6 / 129 §5D) fires at the first quiet
terminal of the pinned worker's run. A worker that only dispatched has not
finished its run: the reminder waits for the follow-up turn after the child
is done. Warm idle children do not count — their routing entry is never
cleared, so counting every adapter would make the parent never quiet.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.loop import MEMORY_RECORD_KIND
from agent_os.agent.session import Session, persist_user_row
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key

P, S, PARENT, CHILD = "p1", "s1", "claude-code", "codex"


class SDKTransport:  # memory-reminder eligibility is by transport class name
    pass


def _manager(tmp_path, child_status="running"):
    sub = MagicMock()
    sub.send = AsyncMock(return_value="Message sent. Transcript: t.jsonl")
    sub._prompt_queues = {}
    parent = MagicMock()
    parent._transport = SDKTransport()
    parent._transport.dispatch_pending = False
    child = MagicMock()
    sub._adapters = {make_session_key(P, S): {PARENT: parent, CHILD: child}}
    status = {"value": child_status}
    sub.status = MagicMock(side_effect=lambda pid, handle, session_id=None:
                           status["value"] if handle == CHILD else "idle")
    transcript_path = os.path.join(str(tmp_path), "cc.jsonl")
    open(transcript_path, "w").close()
    tr = MagicMock()
    tr.filepath = transcript_path
    sub._transcripts = {(P, S, PARENT): tr}
    store = MagicMock()
    store.get_project = MagicMock(return_value={"workspace": str(tmp_path)})
    mgr = AgentManager(project_store=store, ws_manager=MagicMock(), sub_agent_manager=sub,
                       activity_translator=MagicMock(), process_manager=MagicMock())
    obs = LifecycleObserver(mgr, MagicMock())
    sub._lifecycle_observer = obs
    obs.set_dispatch_initiator(P, PARENT, "user_pinned", session_id=S)
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    return mgr, sub, status


@pytest.mark.parametrize("st", ["running", "background-running"])
def test_running_child_means_not_quiescent(tmp_path, st):
    mgr, _, _ = _manager(tmp_path, st)
    assert mgr.has_running_children(P, S, PARENT) is True
    assert mgr._worker_quiescent(P, S, PARENT) is False


@pytest.mark.parametrize("st", ["idle", "unknown", "stopped"])
def test_idle_warm_child_does_not_count(tmp_path, st):
    mgr, _, _ = _manager(tmp_path, st)
    assert mgr.has_running_children(P, S, PARENT) is False
    assert mgr._worker_quiescent(P, S, PARENT) is True


def test_a_child_of_another_parent_does_not_count(tmp_path):
    mgr, sub, _ = _manager(tmp_path)
    sub._lifecycle_observer.set_dispatch_initiator(P, CHILD, "worker:pi", session_id=S)
    assert mgr._worker_quiescent(P, S, PARENT) is True


def _plant(mgr, tmp_path):
    session = Session.new(S, str(tmp_path))
    session.pinned_target = PARENT
    persist_user_row(session, "have codex write hello.txt")
    mgr._handles[make_session_key(P, S)] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=None, task=None)
    return session


def _completed(handle, reply):
    return {"event": "sub_agent_terminal", "kind": "completed",
            "display_content": f"[Sub-agent] {handle} completed. Summary: {reply}. Transcript: t.",
            "reply": reply, "suppress_wake": True}


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_reminder_is_sent_exactly_once_after_child_and_follow_up(tmp_path):
    mgr, sub, status = _manager(tmp_path)
    _plant(mgr, tmp_path)

    def reminders():
        return [c for c in sub.send.call_args_list
                if c.args[1] == PARENT and c.args[2].startswith("[Orbital memory]")]

    # 1. The parent's dispatch turn ends while the child runs: no reminder.
    await mgr.inject_system_message(
        P, f"[Sub-agent] {PARENT} completed. Summary: Dispatched.", session_id=S,
        meta=_completed(PARENT, "Dispatched to codex."))
    await _settle()
    assert reminders() == []

    # 2. The child finishes (its terminal is not the parent's): no reminder.
    status["value"] = "idle"
    await mgr.inject_system_message(
        P, f"[Sub-agent] {CHILD} completed. Summary: Wrote it.", session_id=S,
        meta=_completed(CHILD, "Wrote it."))
    await _settle()
    assert reminders() == []

    # 3. The parent's follow-up ends with nothing running under it: one reminder.
    await mgr.inject_system_message(
        P, f"[Sub-agent] {PARENT} completed. Summary: Verified.", session_id=S,
        meta=_completed(PARENT, "Verified: hello.txt says PONG."))
    await _settle()
    assert len(reminders()) == 1
    assert sub.send.call_args.kwargs["initiator"] == "user_pinned"
