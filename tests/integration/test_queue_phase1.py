# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase-1 integration tests for the queue dispatcher.

Phase 1 scope: dispatcher pulls items, marks them running, and stalls (no
signal detection yet). These tests exercise the QueueDispatcher directly
against a mocked AgentManager that simulates the inject_message + loop-task
contract. The dispatcher itself is unmodified across the phases — its
Phase 2+ behaviour is added by reading loop._exit_reason after the task
completes.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, session_id: str = "sess_phase1"):
        self.session_id = session_id


def _make_mock_manager(session_id: str = "sess_phase1"):
    """Build a mocked AgentManager that simulates a text-only loop run."""
    mgr = MagicMock()
    mgr.get_session = MagicMock(return_value=_FakeSession(session_id))
    mgr.inject_message = AsyncMock(return_value="delivered")

    # The "loop task" — completes immediately, simulating a text-only exit.
    async def _instant_loop():
        return None

    task_holder = {"task": None}

    def _get_loop_task(_pid):
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
async def test_dispatcher_marks_first_item_running_then_stalls(dispatcher):
    d, mgr, store = dispatcher
    item1 = store.add_item("first")
    item2 = store.add_item("second")
    item3 = store.add_item("third")

    await d.start()
    d.notify_new_item()

    # Wait up to 5s for the dispatcher to pick up item1 and stall on it.
    for _ in range(50):
        await asyncio.sleep(0.1)
        state = store.load()
        if (
            state.items[0].state == ItemState.RUNNING
            and len(state.items[0].attempts) == 1
        ):
            break
    else:
        pytest.fail("Dispatcher did not advance item to running state within 5s")

    # Phase 1 contract: only item 1 has moved. Items 2 and 3 are still queued.
    state = store.load()
    assert state.items[0].state == ItemState.RUNNING
    assert state.items[1].state == ItemState.QUEUED
    assert state.items[2].state == ItemState.QUEUED

    # The dispatcher should have invoked inject_message exactly once,
    # with the item content wrapped in the [QUEUE ITEM | id | attempt]
    # header introduced by Concern 3 of the queue-architecture amendments.
    mgr.inject_message.assert_awaited_once()
    args, _ = mgr.inject_message.call_args
    assert args[1] == f"[QUEUE ITEM | id={item1.id} | attempt=1]\nfirst"

    # Give the stall some breathing room and assert item 2 STILL hasn't started.
    await asyncio.sleep(0.5)
    state = store.load()
    assert state.items[1].state == ItemState.QUEUED


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
    assert state.items[0].attempts[0].session_id == "sess_phase1"


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
