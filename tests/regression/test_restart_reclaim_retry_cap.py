# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 5: an item that has already been interrupted twice gets BLOCKED
instead of being reclaimed for a third try. Poison-pill protection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemState,
    QueueRunState,
)
from agent_os.queue.store import QueueStore


def test_repeated_interruption_results_in_blocked(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("repeatedly interrupted")
    # Simulate one prior reclaim cycle that already happened.
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(item.id, AttemptRecord(session_id="prev_sess"))
    # Pre-load interrupted_count to 1 (already reclaimed once before)
    store.increment_interrupted(item.id)

    mgr = MagicMock()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_cap",
        store=store,
        agent_manager=mgr,
    )

    summary = dispatcher.reclaim_on_startup()
    assert summary["queue_state"] == "draining"
    # This time it gets blocked, not reclaimed
    assert summary["reclaimed_items"] == []
    assert summary["blocked_items"] == [item.id]

    state = store.load()
    assert state.items[0].state == ItemState.BLOCKED
    assert state.items[0].interrupted_count == 2
    # The latest attempt was closed as INTERRUPTED on this cycle
    assert state.items[0].attempts[-1].outcome == AttemptOutcome.INTERRUPTED
