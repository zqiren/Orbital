# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a running queue auto-transitions to IDLE after the last
queueable item is exhausted.

Concern 1 of TASK-queue-architecture-amendments.md: QueueStore exposes an
auto_idle_if_empty() helper the dispatcher calls after each advance. Once
no QUEUED / RUNNING / BLOCKED items remain, the helper flips state to IDLE.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore
from tests.integration.test_queue_phase2 import _ScriptedAgentManager


async def _wait(predicate, timeout=10.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


def test_helper_transitions_to_idle_when_no_queueable_items(tmp_path):
    """Direct unit-level test of the store helper. Empty items list → IDLE."""
    store = QueueStore(tmp_path / "queue.json")
    state = store.load()
    assert state.state == QueueRunState.RUNNING
    assert state.items == []

    changed = store.auto_idle_if_empty()
    assert changed is True
    assert store.load().state == QueueRunState.IDLE

    # Idempotent: a second call returns False (already IDLE)
    changed2 = store.auto_idle_if_empty()
    assert changed2 is False


def test_helper_leaves_state_alone_when_queueable_items_remain(tmp_path):
    """QUEUED items keep the queue out of IDLE."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("still here")
    changed = store.auto_idle_if_empty()
    assert changed is False
    assert store.load().state == QueueRunState.RUNNING


def test_helper_treats_blocked_items_as_queueable(tmp_path):
    """A blocked item still occupies the tray; the queue should not idle."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("will block")
    store.set_item_state(item.id, ItemState.BLOCKED)
    changed = store.auto_idle_if_empty()
    assert changed is False
    assert store.load().state == QueueRunState.RUNNING


def test_helper_ignores_done_items(tmp_path):
    """DONE items are terminal — they do NOT keep the queue out of IDLE."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("will finish")
    store.set_item_state(item.id, ItemState.DONE)
    changed = store.auto_idle_if_empty()
    assert changed is True
    assert store.load().state == QueueRunState.IDLE


@pytest.mark.asyncio
async def test_dispatcher_auto_idles_after_last_item_completes(tmp_path):
    """End-to-end: a running queue draining its last item transitions to
    IDLE via the dispatcher's post-advance hook."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("only one")

    mgr = _ScriptedAgentManager([
        {"reason": "complete", "summary": "done"},
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_auto_idle",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: (
        store.load().items
        and store.load().items[0].state == ItemState.DONE
        and store.load().state == QueueRunState.IDLE
    ), timeout=10.0)

    if not ok:
        s = store.load()
        await dispatcher.shutdown()
        pytest.fail(
            "Queue did not auto-idle. Final state: "
            f"queue={s.state.value}, items="
            + ", ".join(f"{it.id}={it.state.value}" for it in s.items)
        )

    await dispatcher.shutdown()
