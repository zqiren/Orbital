# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 5: at daemon startup, a queue in STOPPED state must NOT be
reclaimed. The user paused on purpose; the parked attempt stays parked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import (
    AttemptRecord,
    ItemState,
    QueueRunState,
)
from agent_os.queue.store import QueueStore


def test_stopped_queue_preserves_running_item(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("parked task")
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(
        item.id, AttemptRecord(session_id="parked_sess"),
    )
    store.set_queue_state(QueueRunState.STOPPED)

    # Spin up a fresh dispatcher (mimicking daemon restart)
    mgr = MagicMock()
    dispatcher = QueueDispatcher(
        project_id="proj_stopped_reclaim",
        store=store,
        agent_manager=mgr,
    )

    summary = dispatcher.reclaim_on_startup()
    assert summary["queue_state"] == "stopped"
    assert summary["reclaimed_items"] == []
    assert summary["blocked_items"] == []

    state = store.load()
    # Item is STILL running; queue is STILL stopped; attempt NOT closed
    assert state.state == QueueRunState.STOPPED
    assert state.items[0].state == ItemState.RUNNING
    assert state.items[0].attempts[0].outcome is None
    assert state.items[0].interrupted_count == 0
