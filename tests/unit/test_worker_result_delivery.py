# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.3: the child's result reaches the dispatcher as its next turn.

The delivery rides the daemon's existing follow-up turn to a pinned worker
(the memory reminder, spec 112 A6), parameterised rather than copied: a
hidden ``role: user, source: daemon, _kind: worker_result`` row, then
``send(parent, initiator=<the parent's own last initiator>)``. Unlike the
reminder it is NOT tagged into ``_worker_memory_records``, so the parent's
follow-up marker, tool rows and reply bubble stay ordinary rows that render
and reach the recap. A failed send is a warning: the child's completion row
is on disk and the next message's recap carries it.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.context import WORKER_RESULT_KIND
from agent_os.agent.session import Session, persist_user_row
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.recap import build_recap
from agent_os.utils.file_lock import FileLockError

P, S, PARENT, CHILD = "p1", "s1", "claude-code", "codex"
TEXT = ("Your dispatched worker codex finished. Its reply is in the conversation above "
        "(labelled assistant). The user already sees it; do not repeat it. Verify or "
        "continue, then reply only with what is new.")


def _manager(tmp_path, *, parent_initiator="user_pinned"):
    sub = MagicMock()
    sub.send = AsyncMock(return_value=f"Message sent to {PARENT}. Transcript: t.jsonl")
    sub._adapters = {}
    sub._prompt_queues = {}
    store = MagicMock()
    store.get_project = MagicMock(return_value={"workspace": str(tmp_path)})
    mgr = AgentManager(project_store=store, ws_manager=MagicMock(), sub_agent_manager=sub,
                       activity_translator=MagicMock(), process_manager=MagicMock())
    obs = LifecycleObserver(mgr, MagicMock())
    obs.set_dispatch_initiator(P, PARENT, parent_initiator, session_id=S)
    sub._lifecycle_observer = obs
    session = Session.new(S, str(tmp_path))
    session.pinned_target = PARENT
    persist_user_row(session, "have codex write hello.txt")
    mgr._handles[make_session_key(P, S)] = ProjectHandle(
        session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
        context_manager=MagicMock(), interceptor=None, task=None)
    return mgr, sub, session


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("own", ["user_pinned", "queue_item"])
async def test_delivery_persists_a_hidden_row_and_sends_with_the_parents_own_initiator(tmp_path, own):
    mgr, sub, session = _manager(tmp_path, parent_initiator=own)
    mgr._deliver_worker_result(P, S, PARENT, child=CHILD, kind="completed", content=TEXT)
    await _settle()

    row = session.get_messages()[-1]
    assert row["role"] == "user" and row["source"] == "daemon"
    assert row["target"] == PARENT and row["content"] == TEXT
    assert row["_kind"] == WORKER_RESULT_KIND
    sub.send.assert_awaited_once()
    args, kwargs = sub.send.call_args
    assert args == (P, PARENT, TEXT)
    assert kwargs["session_id"] == S and kwargs["initiator"] == own
    assert kwargs["dispatch_id"].startswith(f"{S}:")
    # The follow-up is an ordinary turn: nothing marks its rows as bookkeeping.
    assert mgr._worker_memory_records == set()


@pytest.mark.asyncio
async def test_the_follow_ups_marker_and_terminal_stay_untagged(tmp_path):
    mgr, sub, session = _manager(tmp_path)
    mgr._deliver_worker_result(P, S, PARENT, child=CHILD, kind="completed", content=TEXT)
    await _settle()
    dispatch_id = sub.send.call_args.kwargs["dispatch_id"]
    await mgr.inject_system_message(
        P, f'[Sub-agent] Message sent to {PARENT}: "Your dispatched…". Transcript: t.jsonl',
        session_id=S, meta={"dispatch_id": dispatch_id, "handle": PARENT,
                            "transcript_path": "t.jsonl", "suppress_wake": True})
    await mgr.inject_system_message(
        P, f"[Sub-agent] {PARENT} completed. Summary: Verified. Transcript: t.jsonl.",
        session_id=S, meta={"event": "sub_agent_terminal", "kind": "completed",
                            "display_content": "x", "reply": "Verified.",
                            "suppress_wake": True})
    rows = session.get_messages()
    assert rows[-2].get("_kind") is None and rows[-1].get("_kind") is None


@pytest.mark.asyncio
async def test_the_hidden_row_is_not_broadcast_live(tmp_path):
    mgr, _, _ = _manager(tmp_path)
    callback = mgr._on_message(P, S)
    callback({"role": "user", "content": TEXT, "source": "daemon",
              "target": PARENT, "_kind": WORKER_RESULT_KIND})
    mgr._activity_translator.on_message.assert_not_called()


@pytest.mark.asyncio
async def test_a_send_error_is_one_warning_and_no_raise(tmp_path, caplog):
    mgr, sub, _ = _manager(tmp_path)
    sub.send.return_value = "Error: concurrent sub-agent limit reached"
    with caplog.at_level(logging.WARNING):
        mgr._deliver_worker_result(P, S, PARENT, child=CHILD, kind="completed", content=TEXT)
        await _settle()
    assert [r for r in caplog.records if r.levelno == logging.WARNING]
    sub.send.assert_awaited_once()          # no retry


@pytest.mark.asyncio
async def test_a_lock_contended_session_is_one_warning_and_no_send(tmp_path, caplog):
    mgr, sub, _ = _manager(tmp_path)
    mgr.persist_mention_message = AsyncMock(side_effect=FileLockError("held"))
    with caplog.at_level(logging.WARNING):
        mgr._deliver_worker_result(P, S, PARENT, child=CHILD, kind="error", content="x")
        await _settle()
    assert [r for r in caplog.records if r.levelno == logging.WARNING]
    sub.send.assert_not_awaited()


def test_the_hook_never_raises_without_a_loop_target(tmp_path):
    mgr, sub, _ = _manager(tmp_path)
    mgr._sub_agent_manager = None
    mgr._deliver_worker_result(P, S, PARENT, child=CHILD, kind="completed", content=TEXT)


# ---------------------------------------------------------------------------
# Recap: the hidden row is bookkeeping; the child's reply is conversation
# ---------------------------------------------------------------------------


def _rows():
    return [
        {"role": "user", "content": "have codex write hello.txt", "target": PARENT},
        {"role": "system", "content": "[Sub-agent] Message sent to claude-code: …",
         "_meta": {"dispatch_id": "s1:a", "handle": PARENT, "suppress_wake": True}},
        {"role": "system", "content": "[Sub-agent] Message sent to codex: …",
         "_meta": {"dispatch_id": "s1:b", "handle": CHILD, "suppress_wake": True}},
        {"role": "system", "content": "[Sub-agent] codex completed. Summary: Wrote it.",
         "_meta": {"event": "sub_agent_terminal", "kind": "completed",
                   "display_content": "[Sub-agent] codex completed. Summary: Wrote it.",
                   "reply": "Wrote hello.txt with PONG."}},
        {"role": "user", "content": TEXT, "source": "daemon", "target": PARENT,
         "_kind": WORKER_RESULT_KIND},
    ]


@pytest.mark.parametrize("fresh", [False, True])
def test_recap_skips_the_row_and_carries_the_childs_reply(fresh):
    block = build_recap(_rows(), PARENT, fresh=fresh)
    assert "assistant: Wrote hello.txt with PONG." in block
    assert "Your dispatched worker" not in block


def test_the_managers_context_never_sees_the_hidden_row():
    """After a pin switch the Orbital loop reads this session: the
    instruction addressed to the worker that dispatched is not its message."""
    from agent_os.agent.context import WINDOW_EXCLUDED_KINDS
    assert WORKER_RESULT_KIND in WINDOW_EXCLUDED_KINDS
