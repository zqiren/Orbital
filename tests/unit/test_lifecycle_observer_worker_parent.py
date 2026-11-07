# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.3: a child's marker and terminal route to the worker that
dispatched it, never to the Orbital loop.

Who gets the result is fixed at dispatch time (spec 074's push model): the
``worker:<parent>`` initiator is recorded per (project, session, child). The
child's dispatch marker and every terminal row stamp ``suppress_wake`` (two
seats must never talk), and each terminal is handed to
``worker_terminal_hook`` with the text the parent will read. A turn the child
started on its own (spec 129 ``continued``) is not a reply to the parent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver

P, S, PARENT, CHILD = "p1", "s1", "claude-code", "codex"


def _observer(*, running_children=False):
    am = MagicMock()
    am.inject_system_message = AsyncMock(
        side_effect=lambda pid, content, **kw:
        "suppressed" if (kw.get("meta") or {}).get("suppress_wake") else "delivered")
    am.has_running_children = MagicMock(return_value=running_children)
    obs = LifecycleObserver(am, MagicMock())
    obs.worker_terminal_hook = MagicMock()
    return obs, am


def _meta_of(am):
    return am.inject_system_message.call_args.kwargs.get("meta") or {}


def test_worker_initiator_is_recorded_and_a_later_dispatch_clears_it():
    obs, _ = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    assert obs.parent_of(P, CHILD, S) == PARENT
    assert obs.children_of(P, S, PARENT) == [CHILD]
    obs.set_dispatch_initiator(P, CHILD, "management_agent", session_id=S)
    assert obs.parent_of(P, CHILD, S) is None
    assert obs.children_of(P, S, PARENT) == []


def test_a_worker_dispatch_is_neither_pinned_nor_a_queue_dispatch():
    obs, _ = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    assert not obs._is_pinned_dispatch(P, CHILD, S)
    assert (P, S, CHILD) not in obs._queue_dispatches


@pytest.mark.parametrize("own,expected", [
    ("user_pinned", "user_pinned"),
    ("queue_item", "queue_item"),
    ("management_agent", "management_agent"),
])
def test_initiator_of_reports_the_parents_own_last_initiator(own, expected):
    obs, _ = _observer()
    obs.set_dispatch_initiator(P, PARENT, own, session_id=S)
    assert obs.initiator_of(P, PARENT, S) == expected


@pytest.mark.asyncio
async def test_the_childs_dispatch_marker_does_not_wake_the_manager():
    obs, am = _observer()
    await obs.on_message_routed(P, CHILD, initiator=f"worker:{PARENT}",
                                message_preview="write hello.txt",
                                transcript_path="t.jsonl", session_id=S,
                                dispatch_id="s1:abc")
    meta = _meta_of(am)
    assert meta["suppress_wake"] is True
    assert meta["dispatch_id"] == "s1:abc" and meta["handle"] == CHILD
    # No queue-item guidance, no rewritten content.
    assert "display_content" not in meta


async def _fire(obs, kind):
    if kind == "completed":
        await obs.on_completed(P, CHILD, "Wrote hello.txt.", "t.jsonl", session_id=S)
    elif kind == "error":
        await obs.on_error(P, CHILD, "rate limited", "t.jsonl", session_id=S)
    elif kind == "failed":
        await obs.on_failed(P, CHILD, "transport died", session_id=S)
    elif kind == "queue_dropped":
        await obs.on_queue_dropped(P, CHILD, why="agent stopped before dispatch", session_id=S)
    elif kind == "stopped":
        await obs.on_user_stopped(P, CHILD, session_id=S)
    elif kind == "interrupted":
        await obs.on_turn_interrupted(P, CHILD, "t.jsonl", session_id=S)


EXPECTED = {
    "completed": ("Your dispatched worker codex finished. Its reply is in the conversation "
                  "above (labelled assistant). The user already sees it; do not repeat it. "
                  "Verify or continue, then reply only with what is new."),
    "error": "Your dispatched worker codex did not finish: rate limited. Decide what to do next.",
    "failed": "Your dispatched worker codex did not finish: transport died. Decide what to do next.",
    "queue_dropped": ("Your dispatched worker codex did not finish: queued message dropped: "
                      "agent stopped before dispatch. Decide what to do next."),
    "stopped": "Your dispatched worker codex did not finish: stopped by user. Decide what to do next.",
    "interrupted": ("Your dispatched worker codex did not finish: turn interrupted before "
                    "completion. Decide what to do next."),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(EXPECTED))
async def test_every_terminal_suppresses_the_wake_and_reaches_the_parent(kind):
    obs, am = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await _fire(obs, kind)
    assert _meta_of(am)["suppress_wake"] is True
    obs.worker_terminal_hook.assert_called_once_with(
        P, S, PARENT, child=CHILD, kind=kind, content=EXPECTED[kind])


@pytest.mark.asyncio
async def test_a_completion_without_a_reply_says_so():
    obs, _ = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_completed(P, CHILD, "", "t.jsonl", session_id=S)
    content = obs.worker_terminal_hook.call_args.kwargs["content"]
    assert content == ("Your dispatched worker codex finished without a reply. "
                       "Check its work, then tell the user the outcome.")


@pytest.mark.asyncio
async def test_a_dropped_prompt_routes_by_its_own_initiator():
    obs, am = _observer()
    # The registry says the child's live turn came from the manager, but the
    # dropped prompt itself was the worker's dispatch.
    obs.set_dispatch_initiator(P, CHILD, "management_agent", session_id=S)
    await obs.on_queue_dropped(P, CHILD, why="agent stopped before dispatch",
                               session_id=S, initiator=f"worker:{PARENT}")
    assert _meta_of(am)["suppress_wake"] is True
    assert obs.worker_terminal_hook.call_args.args[2] == PARENT


@pytest.mark.asyncio
async def test_a_self_initiated_turn_is_not_delivered():
    obs, _ = _observer()
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_continued(P, CHILD, "background build done", "t.jsonl", session_id=S)
    obs.worker_terminal_hook.assert_not_called()


@pytest.mark.asyncio
async def test_a_manager_dispatch_still_wakes_and_is_not_delivered():
    obs, am = _observer()
    obs.set_dispatch_initiator(P, CHILD, "management_agent", session_id=S)
    await obs.on_completed(P, CHILD, "done", "t.jsonl", session_id=S)
    assert "suppress_wake" not in _meta_of(am)
    obs.worker_terminal_hook.assert_not_called()


@pytest.mark.asyncio
async def test_a_hook_that_raises_never_breaks_the_terminal():
    obs, am = _observer()
    obs.worker_terminal_hook.side_effect = RuntimeError("boom")
    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_completed(P, CHILD, "done", "t.jsonl", session_id=S)
    am.inject_system_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Through the real AgentManager: both rows land and neither starts a loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inject_returns_suppressed_for_the_childs_marker_and_terminal(tmp_path):
    from agent_os.agent.session import Session, persist_user_row
    from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
    from agent_os.daemon_v2.models import make_session_key

    sub = MagicMock()
    sub.send = AsyncMock(return_value="Message sent. Transcript: t.jsonl")
    sub._adapters = {}
    sub._prompt_queues = {}
    store = MagicMock()
    store.get_project = MagicMock(return_value={"workspace": str(tmp_path)})
    mgr = AgentManager(project_store=store, ws_manager=MagicMock(), sub_agent_manager=sub,
                       activity_translator=MagicMock(), process_manager=MagicMock())
    session = Session.new(S, str(tmp_path))
    session.pinned_target = PARENT
    persist_user_row(session, "have codex write hello.txt")
    mgr._handles[make_session_key(P, S)] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=None, task=None)
    mgr._start_loop = AsyncMock()

    returns = []
    real = mgr.inject_system_message

    async def spy(*a, **kw):
        out = await real(*a, **kw)
        returns.append(out)
        return out

    mgr.inject_system_message = spy
    obs = LifecycleObserver(mgr, MagicMock())
    obs.worker_terminal_hook = MagicMock()
    sub._lifecycle_observer = obs

    obs.set_dispatch_initiator(P, CHILD, f"worker:{PARENT}", session_id=S)
    await obs.on_message_routed(P, CHILD, initiator=f"worker:{PARENT}",
                                message_preview="write hello.txt",
                                transcript_path="t.jsonl", session_id=S,
                                dispatch_id="s1:abc")
    await obs.on_completed(P, CHILD, "Wrote hello.txt.", "t.jsonl", session_id=S)

    assert returns == ["suppressed", "suppressed"]
    mgr._start_loop.assert_not_awaited()
