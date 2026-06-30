# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase-1 integration tests for the queue dispatcher.

Phase 1 scope: dispatcher pulls items, marks them running, and (post
Concern 4) routes a text-only loop exit through the contract-violation
path — closing the attempt as BLOCKED. These tests exercise the
QueueDispatcher directly against a mocked AgentManager that simulates the
inject_message + loop-task contract.

History: pre-Concern-4 the dispatcher stalled on text-only exits; the
phase-1 assertion below was 'item ends RUNNING with 1 attempt and no
outcome'. After Concern 4 the same scenario must end with the item
BLOCKED and the attempt closed with a contract-violation reason.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, session_id: str = "sess_phase1"):
        self.session_id = session_id


class _TextOnlyLoop:
    """Loop double whose run always exits text-only (no completion tool).

    A plain object (not MagicMock) so it has no fabricated
    get_completion_state — the dispatcher reads its real _exit_* attrs via the
    fallback path.
    """
    _exit_reason = "text"
    _exit_summary = None
    _exit_block_reason = None
    _queue_state = "chat"


def _make_mock_manager(session_id: str = "sess_phase1"):
    """Build a mocked AgentManager that simulates a text-only loop run."""
    mgr = MagicMock()
    mgr.get_session = MagicMock(return_value=_FakeSession(session_id))
    mgr.get_loop = MagicMock(return_value=_TextOnlyLoop())
    # Busy-slot guard: the dispatcher checks current_holder_session_id before
    # minting a session. A bare MagicMock would auto-fabricate a TRUTHY return
    # here, which the guard reads as "another session holds the slot" → the
    # dispatcher defers forever and never records an attempt. Model a free slot.
    mgr.current_holder_session_id = MagicMock(return_value=None)
    # Pending-input defer-check (spec 006 §3e): same MagicMock hazard — a bare
    # mock auto-fabricates a TRUTHY return, which the dispatcher reads as
    # "interactive input is waiting" → defers forever. Model an empty queue.
    mgr.has_pending_inject = MagicMock(return_value=False)
    mgr.inject_message = AsyncMock(return_value="delivered")
    # Corrective-turn path awaits inject_system_message; mock it as async.
    mgr.inject_system_message = AsyncMock(return_value=None)
    # Concern 4: dispatcher rotates the session on every terminal exit
    # (including the text-only contract-violation path). MagicMock would
    # return a non-awaitable for new_session, so we patch it with an
    # AsyncMock that mints a fresh session id on each call.
    counter = {"n": 0}

    async def _new_session(_pid):
        counter["n"] += 1
        new_sid = f"{session_id}_r{counter['n']}"
        mgr.get_session.return_value = _FakeSession(new_sid)
        # Current new_session contract (D2): uuid-only — session_id IS the uuid.
        # The dispatcher reads result["session_id"]; the old {"status":
        # "new_session"} shape predated that and KeyError'd.
        return {"status": "ok", "session_id": new_sid, "session_uuid": new_sid}

    mgr.new_session = _new_session

    # The "loop task" — completes immediately, simulating a text-only exit.
    async def _instant_loop():
        return None

    task_holder = {"task": None}

    def _get_loop_task(_pid, *args, **kwargs):
        # Accept the session_id the dispatcher now threads (project, session_id).
        # Lazily create the task on first call so the asyncio loop is alive.
        if task_holder["task"] is None or task_holder["task"].done():
            task_holder["task"] = asyncio.create_task(_instant_loop())
        return task_holder["task"]

    mgr.get_loop_task = MagicMock(side_effect=_get_loop_task)
    return mgr


@pytest.fixture
def store(tmp_path):
    return QueueStore(tmp_path / "queue.json")


@pytest.fixture
async def dispatcher(store):
    mgr = _make_mock_manager()
    d = QueueDispatcher(
        project_id="proj_phase1",
        store=store,
        agent_manager=mgr,
    )
    yield d, mgr, store
    await d.shutdown()


@pytest.mark.asyncio
async def test_adding_items_persists_to_disk(store):
    store.add_item("first")
    store.add_item("second")
    store.add_item("third")
    state = store.load()
    assert [it.content for it in state.items] == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_dispatcher_picks_first_item_and_routes_text_to_block(dispatcher):
    """Post-Concern-4: the dispatcher injects item 1, the loop exits
    text-only (mock default), and the contract-violation path closes the
    attempt as BLOCKED + rotates. With multiple items queued, subsequent
    items get the same treatment (the mock always returns text-only) until
    all reach BLOCKED."""
    d, mgr, store = dispatcher
    item1 = store.add_item("first")
    store.add_item("second")
    store.add_item("third")

    await d.start()
    d.notify_new_item()

    # Wait for the dispatcher to drain all three items into terminal state.
    for _ in range(100):
        await asyncio.sleep(0.1)
        state = store.load()
        if all(
            it.state == ItemState.BLOCKED for it in state.items
        ):
            break
    else:
        state = store.load()
        pytest.fail(
            "Dispatcher did not block all items within 10s. Final: "
            + ", ".join(f"{it.id}={it.state.value}" for it in state.items)
        )

    state = store.load()
    for it in state.items:
        assert it.state == ItemState.BLOCKED
        assert len(it.attempts) == 1
        assert it.attempts[0].outcome == AttemptOutcome.BLOCKED
        assert "contract violation" in (it.attempts[0].block_reason or "")

    # First inject_message call carries the metadata header AND the queue
    # contract block (H1 verification: contract moved from system prompt to
    # the per-item header for better signal compliance).
    from agent_os.queue.dispatcher import QueueDispatcher
    first_call_args, _ = mgr.inject_message.await_args_list[0]
    assert first_call_args[1] == (
        f"[QUEUE ITEM | id={item1.id} | attempt=1]\n"
        + QueueDispatcher.HEADER_CONTRACT
        + "first"
    )


@pytest.mark.asyncio
async def test_dispatcher_records_attempt_with_session_id(dispatcher):
    d, mgr, store = dispatcher
    item = store.add_item("only one")

    await d.start()
    d.notify_new_item()

    for _ in range(50):
        await asyncio.sleep(0.1)
        state = store.load()
        if state.items[0].attempts:
            break
    else:
        pytest.fail("Dispatcher did not record an attempt within 5s")

    state = store.load()
    assert len(state.items[0].attempts) == 1
    # Each queue item runs in its OWN freshly-minted session (the dispatcher
    # calls new_session per attempt); the fixture mints "{base}_r{n}". The
    # recorded attempt id is that minted session — the canonical id D3 remaps.
    assert state.items[0].attempts[0].session_id == "sess_phase1_r1"


@pytest.mark.asyncio
async def test_disk_state_matches_in_memory_after_each_mutation(store, tmp_path):
    store.add_item("a")
    on_disk_state_1 = QueueStore(tmp_path / "queue.json").load()
    assert on_disk_state_1.items[0].content == "a"

    store.add_item("b", priority=1)
    on_disk_state_2 = QueueStore(tmp_path / "queue.json").load()
    assert on_disk_state_2.items[0].content == "b"
    assert on_disk_state_2.items[1].content == "a"

    item_b_id = on_disk_state_2.items[0].id
    store.remove_item(item_b_id)
    on_disk_state_3 = QueueStore(tmp_path / "queue.json").load()
    assert len(on_disk_state_3.items) == 1
    assert on_disk_state_3.items[0].content == "a"


@pytest.mark.asyncio
async def test_dispatcher_idles_when_queue_paused(store):
    store.set_queue_state(QueueRunState.PAUSED)
    store.add_item("first")
    mgr = _make_mock_manager()
    d = QueueDispatcher(
        project_id="proj_paused",
        store=store,
        agent_manager=mgr,
    )
    await d.start()
    d.notify_new_item()
    await asyncio.sleep(0.5)
    state = store.load()
    assert state.items[0].state == ItemState.QUEUED
    mgr.inject_message.assert_not_called()
    await d.shutdown()
