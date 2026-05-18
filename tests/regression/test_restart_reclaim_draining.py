# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 5: a RUNNING queue with a running item at daemon startup means
the previous attempt was interrupted. Close it, requeue at head, bump
interrupted_count.
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


def test_running_queue_running_item_is_reclaimed_at_head(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    # Three items: the second one was running when the daemon died.
    a = store.add_item("first queued")
    b = store.add_item("second was running")
    c = store.add_item("third queued")
    store.set_item_state(b.id, ItemState.RUNNING)
    store.append_attempt(b.id, AttemptRecord(session_id="b_sess"))
    # Queue is still RUNNING — user did NOT pause before the crash.

    mgr = MagicMock()
    dispatcher = QueueDispatcher(
        project_id="proj_running_reclaim",
        store=store,
        agent_manager=mgr,
    )

    summary = dispatcher.reclaim_on_startup()
    assert summary["queue_state"] == "running"
    assert summary["reclaimed_items"] == [b.id]
    assert summary["blocked_items"] == []

    state = store.load()
    # b should now be QUEUED at the head, with interrupted_count == 1,
    # and its previous attempt closed as INTERRUPTED.
    assert state.items[0].id == b.id
    assert state.items[0].state == ItemState.QUEUED
    assert state.items[0].interrupted_count == 1
    assert state.items[0].attempts[0].outcome == AttemptOutcome.INTERRUPTED
    assert (
        state.items[0].attempts[0].block_reason
        == "interrupted by daemon restart"
    )
    # a and c still queued in their original order behind b
    assert state.items[1].id == a.id
    assert state.items[2].id == c.id
    # Queue state itself is unchanged
    assert state.state == QueueRunState.RUNNING
