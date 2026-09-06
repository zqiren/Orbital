# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 079 amendment (2026-09-05) — an assigned item closes on its worker's
terminal event.

The user picked the runner, so what the runner finished with IS the item's
outcome: finished → DONE with the worker's own final message as the summary,
anything else → BLOCKED with the worker's own reason. The management agent is
no longer woken to reach the same conclusion — unless the item asked to be
reviewed, which is the one case that still buys a verdict turn.

Why it changed: waiting for a verdict turn made the item's fate depend on the
manager being able to run at all. With no (or a rejected) API key the wake
never started, nothing was reported, and the item sat RUNNING until the
30-minute backstop blocked it with a timeout reason instead of the truth.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState
from agent_os.queue.store import QueueStore


# ---------------------------------------------------------------------------
# Fixtures — the shape test_queue_agent_picker.py uses, plus a dispatched item
# ---------------------------------------------------------------------------


def _dispatcher(tmp_path, *, agent="codex", review=False):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item(
        "write hello.txt", agent=agent, review_before_advance=review,
    )

    sub_mgr = MagicMock()
    sub_mgr.send = AsyncMock(return_value="Message sent to codex.")

    mgr = MagicMock()
    mgr.is_onboarding_complete = MagicMock(return_value=True)
    mgr.current_holder_session_id = MagicMock(return_value=None)
    mgr.has_pending_inject = MagicMock(return_value=False)
    mgr.new_session = AsyncMock(return_value={"session_id": "sess_1"})
    mgr.inject_message = AsyncMock(return_value="started")
    mgr.get_sub_agent_manager = MagicMock(return_value=sub_mgr)
    mgr.get_loop = MagicMock(return_value=None)
    mgr.get_loop_task = MagicMock(return_value=None)

    d = QueueDispatcher("proj", store, mgr, workspace=str(tmp_path))
    d.HOLD_POLL_SEC = 0.01
    d._persist_queue_user_row = MagicMock()
    return d, store, item, mgr


def _running(d, store, item, *, session_id="sess_1"):
    """Put the item in the state a dispatch leaves it in."""
    from agent_os.queue.models import AttemptRecord

    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(item.id, AttemptRecord(session_id=session_id))
    return store.load().items[0]


# ---------------------------------------------------------------------------
# The settle itself
# ---------------------------------------------------------------------------


def test_worker_completion_closes_the_item_with_its_own_summary(tmp_path):
    d, store, item, mgr = _dispatcher(tmp_path)
    _running(d, store, item)

    assert d.on_worker_terminal(
        "sess_1", "codex", kind="completed", summary="wrote hello.txt",
    ) is True

    closed = store.load().items[0]
    assert closed.state == ItemState.DONE
    assert closed.attempts[-1].outcome == AttemptOutcome.COMPLETED
    # The worker's own words, not a manager's paraphrase of them.
    assert closed.attempts[-1].summary == "wrote hello.txt"
    # No management turn was asked for anywhere in this path.
    mgr.inject_message.assert_not_awaited()


def test_worker_error_blocks_the_item_with_its_own_reason(tmp_path):
    d, store, item, mgr = _dispatcher(tmp_path)
    _running(d, store, item)

    assert d.on_worker_terminal(
        "sess_1", "codex", kind="error", summary="model refused the request",
    ) is True

    closed = store.load().items[0]
    assert closed.state == ItemState.BLOCKED
    assert closed.attempts[-1].outcome == AttemptOutcome.BLOCKED
    assert closed.attempts[-1].block_reason == "model refused the request"
    assert closed.attempts[-1].block_reason_code == "worker_terminal"


def test_a_worker_that_failed_without_a_reason_still_says_what_happened(tmp_path):
    d, store, item, _mgr = _dispatcher(tmp_path)
    _running(d, store, item)

    assert d.on_worker_terminal("sess_1", "codex", kind="failed") is True
    closed = store.load().items[0]
    assert closed.state == ItemState.BLOCKED
    assert "codex" in closed.attempts[-1].block_reason


def test_review_before_advance_still_buys_a_verdict_turn(tmp_path):
    """The one case where the manager is still worth waking: the user asked
    for the work to be reviewed before the queue advances."""
    d, store, item, _mgr = _dispatcher(tmp_path, review=True)
    _running(d, store, item)

    assert d.on_worker_terminal(
        "sess_1", "codex", kind="completed", summary="done",
    ) is False
    # Untouched — the wake path owns it, exactly as before.
    assert store.load().items[0].state == ItemState.RUNNING


def test_a_terminal_from_another_session_is_not_ours(tmp_path):
    d, store, item, _mgr = _dispatcher(tmp_path)
    _running(d, store, item, session_id="sess_1")

    assert d.on_worker_terminal("sess_other", "codex", kind="completed") is False
    assert store.load().items[0].state == ItemState.RUNNING


def test_a_terminal_from_another_worker_in_our_session_is_not_ours(tmp_path):
    """A worker the manager dispatched itself, inside the item's session, must
    not close the item the user assigned to somebody else."""
    d, store, item, _mgr = _dispatcher(tmp_path, agent="codex")
    _running(d, store, item)

    assert d.on_worker_terminal("sess_1", "claude-code", kind="completed") is False
    assert store.load().items[0].state == ItemState.RUNNING


def test_an_unassigned_item_is_never_settled_this_way(tmp_path):
    """No agent means the management agent is doing the work; its own verdict
    is the outcome, and a stray worker terminal must not pre-empt it."""
    d, store, item, _mgr = _dispatcher(tmp_path, agent=None)
    _running(d, store, item)

    assert d.on_worker_terminal("sess_1", "codex", kind="completed") is False
    assert store.load().items[0].state == ItemState.RUNNING


# ---------------------------------------------------------------------------
# The hold: it must notice and stop, without disposing of the item twice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_slot_hold_releases_as_soon_as_the_item_is_settled(tmp_path):
    d, store, item, _mgr = _dispatcher(tmp_path)
    running = _running(d, store, item)
    d._max_runtime_seconds = 30  # the backstop must NOT be what ends this

    async def settle_soon():
        await asyncio.sleep(0.05)
        d.on_worker_terminal("sess_1", "codex", kind="completed", summary="ok")

    task = asyncio.ensure_future(settle_soon())
    hold = await asyncio.wait_for(
        d._hold_slot_for_continuation(running, "sess_1", d._stop_generation),
        timeout=5,
    )
    await task

    assert hold == "settled"
    # And settling the released hold leaves the closed item exactly as it was.
    d._settle_released_hold(running, hold, corrective_turn_used=False)
    closed = store.load().items[0]
    assert closed.state == ItemState.DONE
    assert len(closed.attempts) == 1
    assert closed.attempts[-1].outcome == AttemptOutcome.COMPLETED


# ---------------------------------------------------------------------------
# The observer half: who gets offered the event, and what suppresses the wake
# ---------------------------------------------------------------------------


class _AgentManager:
    def __init__(self):
        self.injections = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((project_id, content, kwargs))


class _WS:
    def broadcast(self, project_id, payload):
        pass


def _observer():
    mgr = _AgentManager()
    return LifecycleObserver(mgr, _WS()), mgr


@pytest.mark.asyncio
async def test_a_settled_queue_dispatch_lands_its_row_without_waking(tmp_path):
    observer, mgr = _observer()
    seen = []

    def hook(project_id, session_id, handle, *, kind, summary, transcript_path):
        seen.append((project_id, session_id, handle, kind, summary))
        return True

    observer.queue_terminal_hook = hook
    observer.set_dispatch_initiator("proj", "codex", "queue_item", session_id="s1")

    await observer.on_completed(
        "proj", "codex", summary="1", transcript_path="/t.jsonl", session_id="s1",
    )

    assert seen == [("proj", "s1", "codex", "completed", "1")]
    meta = mgr.injections[-1][2]["meta"]
    # The row still lands — the session must read honestly — but the manager
    # is not woken to judge what the worker already reported.
    assert meta["suppress_wake"] is True
    assert meta["kind"] == "completed"


@pytest.mark.asyncio
async def test_an_unsettled_queue_dispatch_wakes_the_manager_as_before(tmp_path):
    observer, mgr = _observer()
    observer.queue_terminal_hook = lambda *a, **k: False
    observer.set_dispatch_initiator("proj", "codex", "queue_item", session_id="s1")

    await observer.on_completed(
        "proj", "codex", summary="1", transcript_path="/t.jsonl", session_id="s1",
    )

    assert "suppress_wake" not in mgr.injections[-1][2]["meta"]


@pytest.mark.asyncio
async def test_a_plain_mention_is_never_offered_to_the_queue(tmp_path):
    observer, mgr = _observer()
    called = []
    observer.queue_terminal_hook = lambda *a, **k: called.append(a) or True
    observer.set_dispatch_initiator("proj", "codex", "user_mention", session_id="s1")

    await observer.on_completed(
        "proj", "codex", summary="1", transcript_path="/t.jsonl", session_id="s1",
    )

    assert called == []
    assert "suppress_wake" not in mgr.injections[-1][2]["meta"]


@pytest.mark.asyncio
async def test_a_later_mention_clears_the_queue_registration(tmp_path):
    """The registry tracks the CURRENT dispatch: a mention after a queue
    dispatch for the same worker must not be settled as a queue item."""
    observer, mgr = _observer()
    called = []
    observer.queue_terminal_hook = lambda *a, **k: called.append(a) or True

    observer.set_dispatch_initiator("proj", "codex", "queue_item", session_id="s1")
    observer.set_dispatch_initiator("proj", "codex", "user_mention", session_id="s1")
    await observer.on_completed(
        "proj", "codex", summary="1", transcript_path="/t.jsonl", session_id="s1",
    )

    assert called == []


@pytest.mark.asyncio
async def test_a_hook_that_raises_falls_back_to_waking(tmp_path):
    observer, mgr = _observer()

    def boom(*a, **k):
        raise RuntimeError("hook is broken")

    observer.queue_terminal_hook = boom
    observer.set_dispatch_initiator("proj", "codex", "queue_item", session_id="s1")

    await observer.on_error(
        "proj", "codex", error="bad", transcript_path="/t.jsonl", session_id="s1",
    )

    # The event still reached the session, and the manager still wakes for it.
    assert "suppress_wake" not in mgr.injections[-1][2]["meta"]
