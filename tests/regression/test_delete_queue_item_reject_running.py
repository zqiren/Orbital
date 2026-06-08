# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: DELETE queue item is IDLE-ONLY — a RUNNING item is rejected.

Confirmed bug: the old ``delete_queue_item`` route → ``QueueStore.remove_item``
deleted a RUNNING queue item with no liveness check. It stamped the latest
attempt ``outcome=CANCELLED`` and popped the record, but never stopped the bound
session or released the slot — orphaning a running session that keeps holding
the slot and spending.

Locked model: DELETE IS IDLE-ONLY. The route must REJECT a running item with
HTTP 409 and perform ZERO mutation (no ``remove_item``, no CANCELLED stamp, no
session teardown). The liveness gate is ``current_holder_session_id`` — NOT the
stored ``item.state`` flag (which can be stale).

RED-before evidence: against the pre-fix route this test FAILS — the delete
returns 200, the record is gone, and the latest attempt is stamped CANCELLED.
GREEN-after: 409, record intact, attempt outcome unchanged, no session torn
down.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import agents_v2
from agent_os.queue.models import AttemptRecord, ItemState
from agent_os.queue.store import QueueStore


PID = "proj_reject_running"
WORKSPACE = "/tmp/does-not-matter-reject-running"
HOLDER_SID = "uuid-holder-session"


def _make_app(store: QueueStore, *, holder_sid: str | None):
    """Build a minimal app whose agent_manager reports ``holder_sid`` as the
    current slot holder and exposes the real ``store``.

    ``delete_session`` is an AsyncMock so the test can assert it is NEVER
    called on the reject path (zero session teardown)."""
    agent_manager = MagicMock()
    agent_manager.get_queue_store = MagicMock(return_value=store)
    agent_manager.current_holder_session_id = MagicMock(return_value=holder_sid)
    agent_manager.delete_session = AsyncMock()

    project_store = MagicMock()
    project_store.get_project = MagicMock(
        return_value={"workspace": WORKSPACE}
    )

    app = FastAPI()
    agents_v2.configure(
        project_store=project_store,
        agent_manager=agent_manager,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
    )
    app.include_router(agents_v2.router)
    return app, agent_manager


def _seed_running_item(store: QueueStore):
    """Add an item and bind its latest attempt to the holder session, mirroring
    a dispatched-and-running item. Returns the item id."""
    item = store.add_item("do the running thing")
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(
        item.id,
        AttemptRecord(session_id=HOLDER_SID),  # outcome=None → open attempt
    )
    return item.id


def test_delete_running_queue_item_rejected_409(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item_id = _seed_running_item(store)

    # The item's latest-attempt session IS the live slot holder → running.
    app, agent_manager = _make_app(store, holder_sid=HOLDER_SID)
    client = TestClient(app)

    resp = client.delete(f"/api/v2/projects/{PID}/queue/items/{item_id}")

    # 409 reject.
    assert resp.status_code == 409, resp.text
    assert "running queue item" in resp.json()["detail"].lower()

    # ZERO mutation: record still present.
    fresh = QueueStore(tmp_path / "queue.json").load()
    surviving = [it for it in fresh.items if it.id == item_id]
    assert len(surviving) == 1, "running item must NOT be removed"

    # Latest attempt outcome NOT stamped CANCELLED — still open (None).
    assert surviving[0].attempts[-1].outcome is None, (
        "running item's latest attempt must stay open, not be stamped CANCELLED"
    )

    # No session torn down.
    agent_manager.delete_session.assert_not_called()


def test_delete_idle_item_with_stale_running_flag_proceeds(tmp_path):
    """Liveness gate is the holder, NOT item.state. An item whose state flag is
    RUNNING but whose bound session does NOT hold the slot is treated as idle
    and deleted (its session JSONL is cleaned up)."""
    store = QueueStore(tmp_path / "queue.json")
    item_id = _seed_running_item(store)  # state=RUNNING, attempt sid=HOLDER_SID

    # Holder is someone else (or nobody) → the item is NOT live → idle delete.
    app, agent_manager = _make_app(store, holder_sid="some-other-session")
    client = TestClient(app)

    resp = client.delete(f"/api/v2/projects/{PID}/queue/items/{item_id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "removed"

    # Record removed.
    fresh = QueueStore(tmp_path / "queue.json").load()
    assert all(it.id != item_id for it in fresh.items)

    # Idle delete cleaned up the bound session.
    agent_manager.delete_session.assert_awaited_once_with(PID, HOLDER_SID)
