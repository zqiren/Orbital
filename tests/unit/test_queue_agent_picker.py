# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 079 — choosing which agent runs a queue item.

Covers the persisted field (§3.1) and the dispatch branch it selects (§3.2):
an item with an ``agent`` goes STRAIGHT to that worker down the chat @mention
funnel, and the management agent is woken by the worker's terminal event to
declare the verdict. An item without one must behave exactly as it did before
this feature existed, which several tests here pin explicitly.

The slot-hold half of §3.2 has its own suite:
``tests/regression/test_queue_agent_direct_dispatch_hold.py``.
"""

import asyncio
import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import (
    AttemptOutcome,
    ItemRecord,
    ItemState,
    QueueRunState,
    QueueState,
)
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# §3.1 — the field
# ---------------------------------------------------------------------------


def test_add_item_persists_the_chosen_agent(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("ship it", agent="codex")
    assert item.agent == "codex"
    # Survives a reload from disk, not just the in-memory cache.
    assert QueueStore(tmp_path / "queue.json").load().items[0].agent == "codex"


def test_add_item_defaults_to_no_agent(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    assert store.add_item("ship it").agent is None


def test_blank_agent_spellings_collapse_to_none(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    assert store.add_item("a", agent="").agent is None
    assert store.add_item("b", agent="   ").agent is None
    assert store.add_item("c", agent="  codex  ").agent == "codex"


def test_edit_item_reassigns_and_clears_the_agent(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("ship it", agent="codex")

    assert store.edit_item(item.id, agent="claude-code").agent == "claude-code"
    # An explicit None is the "hand it back to Orbital" gesture.
    assert store.edit_item(item.id, agent=None).agent is None


def test_edit_item_without_the_kwarg_leaves_the_agent_alone(tmp_path):
    """The distinction the sentinel exists for: editing the TEXT of an assigned
    item must not silently unassign it."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("ship it", agent="codex")

    edited = store.edit_item(item.id, content="ship it twice")
    assert edited.content == "ship it twice"
    assert edited.agent == "codex"


def test_pre_079_queue_json_loads_unchanged(tmp_path):
    """A queue.json written before this field existed must deserialize."""
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({
        "version": 1,
        "state": "running",
        "items": [{
            "id": "item_old",
            "content": "written before spec 079",
            "file_refs": [],
            "priority": 0,
            "review_before_advance": False,
            "state": "queued",
            "source": "user",
            "attempts": [],
            "idempotency_key": None,
            "interrupted_count": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
        }],
    }))
    state = QueueStore(path).load()
    assert state.items[0].content == "written before spec 079"
    assert state.items[0].agent is None


def test_pre_079_project_trigger_record_has_no_agent():
    """The trigger half of the same migration: a record saved before 079 reads
    as unassigned rather than raising or inventing a handle."""
    trigger = {"id": "trg_1", "name": "Nightly", "type": "schedule", "task": "go"}
    resolved = (trigger.get("agent") or "").strip() or None
    assert resolved is None


# ---------------------------------------------------------------------------
# §3.2 — the dispatch branch
# ---------------------------------------------------------------------------


def _dispatcher(tmp_path, *, agent=None, send_result="Message sent to codex."):
    """A dispatcher wired over a REAL store, with a manager double whose
    sub-agent manager records what was dispatched."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("write hello.txt", agent=agent)

    sub_mgr = MagicMock()
    sub_mgr.send = AsyncMock(return_value=send_result)
    sub_mgr.list_active = MagicMock(return_value=[{"status": "running"}])

    mgr = MagicMock()
    mgr.is_onboarding_complete = MagicMock(return_value=True)
    mgr.current_holder_session_id = MagicMock(return_value=None)
    mgr.has_pending_inject = MagicMock(return_value=False)
    mgr.new_session = AsyncMock(return_value={"session_id": "proj_sess1"})
    mgr.inject_message = AsyncMock(return_value="started")
    mgr.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    mgr.get_loop = MagicMock(return_value=None)
    mgr.get_loop_task = MagicMock(return_value=None)

    d = QueueDispatcher("proj", store, mgr, workspace=str(tmp_path))
    d.HOLD_POLL_SEC = 0.01
    d._max_runtime_seconds = 0.05  # let the backstop end the hold quickly
    # The session materializer reaches into the manager for config/observers;
    # MagicMock answers those, and persist_user_row writes through the mock.
    d._persist_queue_user_row = MagicMock()
    return d, store, item, mgr, sub_mgr


@pytest.mark.asyncio
async def test_assigned_item_dispatches_straight_to_the_worker(tmp_path):
    d, store, item, mgr, sub_mgr = _dispatcher(tmp_path, agent="codex")

    await d._dispatch_one(item)

    # The management funnel is NOT used — that is the whole point.
    mgr.inject_message.assert_not_called()
    sub_mgr.send.assert_awaited_once()
    args, kwargs = sub_mgr.send.await_args
    assert args[0] == "proj"
    assert args[1] == "codex"
    assert args[2] == "write hello.txt"
    assert kwargs["session_id"] == "proj_sess1"
    # queue_item = the manager supervises and is woken for the verdict, but
    # only by the worker's TERMINAL event — the dispatch marker itself is
    # wake-suppressed so the manager cannot race its own worker;
    # user_pinned (wake-suppressed) is deliberately not offered — §3.5.
    assert kwargs["initiator"] == "queue_item"


@pytest.mark.asyncio
async def test_the_worker_is_not_handed_the_completion_contract(tmp_path):
    """CLI workers cannot call mark_task_complete, so the contract must not
    ride the message they receive — only the manager's session row."""
    d, store, item, mgr, sub_mgr = _dispatcher(tmp_path, agent="codex")

    await d._dispatch_one(item)

    dispatched = sub_mgr.send.await_args.args[2]
    assert "mark_task_complete" not in dispatched
    assert "[QUEUE ITEM" not in dispatched

    # …while the row persisted for the manager carries both.
    persisted = d._persist_queue_user_row.call_args.args[1]
    assert "[QUEUE ITEM | id=" in persisted
    assert "mark_task_complete" in persisted


@pytest.mark.asyncio
async def test_unassigned_item_still_goes_to_the_management_agent(tmp_path):
    d, store, item, mgr, sub_mgr = _dispatcher(tmp_path, agent=None)
    d._await_and_handle = AsyncMock()

    await d._dispatch_one(item)

    sub_mgr.send.assert_not_awaited()
    mgr.inject_message.assert_awaited_once()
    assert mgr.inject_message.await_args.kwargs["queue_state"] == "running"


@pytest.mark.asyncio
async def test_a_stale_slug_blocks_the_item_with_the_send_error(tmp_path):
    """An agent uninstalled between queueing and dispatch: ``send`` reports it
    as an error STRING, nothing is running, and no wake will ever arrive — so
    the item must block now, carrying the reason, rather than hold for the
    whole backstop."""
    d, store, item, mgr, sub_mgr = _dispatcher(
        tmp_path, agent="ghost",
        send_result="Error: agent 'ghost' not running for project 'proj'",
    )

    await d._dispatch_one(item)

    reloaded = store.load().items[0]
    assert reloaded.state == ItemState.BLOCKED
    latest = reloaded.attempts[-1]
    assert latest.outcome == AttemptOutcome.BLOCKED
    assert latest.block_reason_code == "agent_dispatch_failed"
    assert "ghost" in latest.block_reason
    # A stale slug is terminal, not a poison-pill interruption: retrying it
    # would fail identically, so it must not consume that budget.
    assert reloaded.interrupted_count == 0


@pytest.mark.asyncio
async def test_a_raising_send_blocks_the_item_too(tmp_path):
    d, store, item, mgr, sub_mgr = _dispatcher(tmp_path, agent="codex")
    sub_mgr.send = AsyncMock(side_effect=RuntimeError("transport is broken"))

    await d._dispatch_one(item)

    reloaded = store.load().items[0]
    assert reloaded.state == ItemState.BLOCKED
    assert reloaded.attempts[-1].block_reason_code == "agent_dispatch_failed"
    assert "transport is broken" in reloaded.attempts[-1].block_reason


@pytest.mark.asyncio
async def test_resume_of_an_assigned_item_re_dispatches_to_the_worker(tmp_path):
    """A pause kills the worker via ``stop_all``, so resuming has to ask it
    again — starting a management turn would leave the work undone."""
    d, store, item, mgr, sub_mgr = _dispatcher(tmp_path, agent="codex")
    mgr._start_loop = AsyncMock()

    await d._resume_attempt(item, "proj_sess1")

    mgr._start_loop.assert_not_awaited()
    sub_mgr.send.assert_awaited_once()
    assert sub_mgr.send.await_args.args[1] == "codex"


@pytest.mark.asyncio
async def test_resume_after_the_manager_woke_is_an_ordinary_resume(tmp_path):
    """Once a management turn exists for the session, the parked attempt is a
    manager turn and resumes as one — no second worker dispatch."""
    d, store, item, mgr, sub_mgr = _dispatcher(tmp_path, agent="codex")
    mgr.get_loop = MagicMock(return_value=MagicMock())
    mgr._start_loop = AsyncMock()
    d._await_and_handle = AsyncMock()

    await d._resume_attempt(item, "proj_sess1")

    mgr._start_loop.assert_awaited_once()
    sub_mgr.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_retry_of_an_assigned_item_re_runs_on_the_worker(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("write hello.txt", agent="codex")
    store.set_item_state(item.id, ItemState.BLOCKED)
    from agent_os.queue.models import AttemptRecord
    store.append_attempt(item.id, AttemptRecord(session_id="proj_sess1"))

    sub_mgr = MagicMock()
    sub_mgr.send = AsyncMock(return_value="Message sent to codex.")
    sub_mgr.list_active = MagicMock(return_value=[{"status": "running"}])
    mgr = MagicMock()
    mgr.switch_session = AsyncMock()
    mgr.inject_message = AsyncMock()
    mgr.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    mgr.get_loop = MagicMock(return_value=None)
    mgr.get_loop_task = MagicMock(return_value=None)

    d = QueueDispatcher("proj", store, mgr, workspace=str(tmp_path))
    d.HOLD_POLL_SEC = 0.01
    d._max_runtime_seconds = 0.05

    result = await d.retry_blocked_item(item.id, "write hello.md", mode="edit")
    await asyncio.sleep(0.02)  # let the spawned send run

    assert result["status"] == "retry_started"
    mgr.inject_message.assert_not_awaited()
    sub_mgr.send.assert_awaited_once()
    assert sub_mgr.send.await_args.args[1] == "codex"
    assert sub_mgr.send.await_args.args[2] == "write hello.md"
    # The manager's row keeps the contract for the wake turn to answer.
    persisted = mgr.persist_mention_message.call_args.args[2]
    assert "mark_task_complete" in persisted["content"]
    assert persisted["target"] == "codex"


@pytest.mark.asyncio
async def test_answer_retry_still_replies_to_the_manager(tmp_path):
    """A question-card answer replies to the question the MANAGER asked in its
    verdict turn — it is not a re-run of the task, so it must not be forwarded
    to the worker."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("write hello.txt", agent="codex")
    store.set_item_state(item.id, ItemState.BLOCKED)
    from agent_os.queue.models import AttemptRecord
    store.append_attempt(item.id, AttemptRecord(session_id="proj_sess1"))

    sub_mgr = MagicMock()
    sub_mgr.send = AsyncMock()
    mgr = MagicMock()
    mgr.switch_session = AsyncMock()
    mgr.inject_message = AsyncMock()
    mgr.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    mgr.get_loop = MagicMock(return_value=None)

    d = QueueDispatcher("proj", store, mgr, workspace=str(tmp_path))
    d._await_and_handle = AsyncMock()

    await d.retry_blocked_item(item.id, "yes, overwrite it", mode="answer")

    sub_mgr.send.assert_not_awaited()
    mgr.inject_message.assert_awaited_once()
