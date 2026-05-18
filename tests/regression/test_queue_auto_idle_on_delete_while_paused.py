# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a paused queue auto-transitions to IDLE when its last item
is removed.

Concern 1 of TASK-queue-architecture-amendments.md: auto_idle_if_empty
triggers regardless of current state, so PAUSED → IDLE on deletion of the
last queueable item. The dispatcher's main loop calls the helper on every
tick, so even without an advance event the empty queue collapses to IDLE.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import QueueRunState
from agent_os.queue.store import QueueStore
from tests.integration.test_queue_phase2 import _ScriptedAgentManager


async def _wait(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


def test_helper_transitions_paused_queue_to_idle_when_empty(tmp_path):
    """Direct store-level test: paused empty queue → IDLE on helper call."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("only one")
    store.set_queue_state(QueueRunState.PAUSED)
    assert store.load().state == QueueRunState.PAUSED

    # Remove the lone item; helper should transition state to IDLE.
    assert store.remove_item(item.id) is True
    changed = store.auto_idle_if_empty()
    assert changed is True
    assert store.load().state == QueueRunState.IDLE


def test_helper_keeps_paused_state_when_items_remain(tmp_path):
    """Paused queue with items still queued must stay PAUSED."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("a")
    store.add_item("b")
    store.set_queue_state(QueueRunState.PAUSED)

    changed = store.auto_idle_if_empty()
    assert changed is False
    assert store.load().state == QueueRunState.PAUSED


@pytest.mark.asyncio
async def test_dispatcher_auto_idles_paused_queue_on_delete(tmp_path):
    """End-to-end: pause the queue, delete all items, watch the dispatcher
    main loop collapse the queue to IDLE on its next tick."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("will be deleted")
    store.set_queue_state(QueueRunState.PAUSED)

    # Trim the IDLE_WAIT timeout so the helper-call cadence is fast in test.
    mgr = _ScriptedAgentManager([])
    dispatcher = QueueDispatcher(
        project_id="proj_delete_while_paused",
        store=store,
        agent_manager=mgr,
    )
    dispatcher.IDLE_WAIT_TIMEOUT_SEC = 0.1
    await dispatcher.start()

    # Delete the lone item while paused, then nudge the dispatcher.
    assert store.remove_item(item.id) is True
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().state == QueueRunState.IDLE,
        timeout=5.0,
    )
    await dispatcher.shutdown()

    if not ok:
        pytest.fail(
            f"Expected queue to auto-idle after deletion. "
            f"Final state: {store.load().state.value}"
        )
