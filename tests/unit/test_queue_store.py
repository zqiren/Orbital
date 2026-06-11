# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for QueueStore — atomic write, dedupe, ordering, round-trip."""

import json
import tempfile
from pathlib import Path

import pytest

from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemState,
    QueueRunState,
    QueueState,
)
from agent_os.queue.store import QueueStore


def _store(tmp_path: Path) -> QueueStore:
    return QueueStore(tmp_path / "queue.json")


def test_load_creates_empty_on_first_access(tmp_path):
    store = _store(tmp_path)
    state = store.load()
    assert isinstance(state, QueueState)
    assert state.state == QueueRunState.RUNNING
    assert state.items == []
    assert (tmp_path / "queue.json").exists()


def test_atomic_write_no_stray_tmp_file(tmp_path):
    store = _store(tmp_path)
    store.add_item("hello")
    # After save, no .tmp sibling should remain
    tmps = list(tmp_path.glob("queue.json.tmp"))
    assert tmps == []
    assert (tmp_path / "queue.json").exists()


def test_round_trip_persists_full_state(tmp_path):
    store = _store(tmp_path)
    item = store.add_item("task one")
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(
        item.id, AttemptRecord(session_id="sess_abc"),
    )
    store.close_latest_attempt(
        item.id, outcome=AttemptOutcome.COMPLETED, summary="done",
    )

    # Re-open as a fresh store and verify
    store2 = QueueStore(tmp_path / "queue.json")
    state = store2.load()
    assert len(state.items) == 1
    reloaded = state.items[0]
    assert reloaded.content == "task one"
    assert reloaded.state == ItemState.RUNNING
    assert len(reloaded.attempts) == 1
    assert reloaded.attempts[0].outcome == AttemptOutcome.COMPLETED
    assert reloaded.attempts[0].summary == "done"


def test_idempotency_key_dedupes(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("first", idempotency_key="abc")
    b = store.add_item("second", idempotency_key="abc")
    assert a.id == b.id  # same record returned
    state = store.load()
    assert len(state.items) == 1
    assert state.items[0].content == "first"  # original content preserved


def test_priority_zero_appends_to_tail(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("first")
    b = store.add_item("second")
    c = store.add_item("third")
    state = store.load()
    assert [it.id for it in state.items] == [a.id, b.id, c.id]


def test_priority_one_inserts_at_head_of_queued(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("first")
    b = store.add_item("second")
    c = store.add_item("urgent", priority=1)
    state = store.load()
    assert state.items[0].id == c.id
    assert [it.id for it in state.items[1:]] == [a.id, b.id]


def test_priority_one_respects_running_items(tmp_path):
    store = _store(tmp_path)
    running = store.add_item("first")
    store.set_item_state(running.id, ItemState.RUNNING)
    store.add_item("queued_first")
    urgent = store.add_item("urgent", priority=1)
    state = store.load()
    # urgent should be after the running one but before queued_first
    assert state.items[0].id == running.id
    assert state.items[1].id == urgent.id


def test_edit_item_only_when_queued(tmp_path):
    store = _store(tmp_path)
    item = store.add_item("original")
    edited = store.edit_item(item.id, content="new content")
    assert edited is not None
    assert edited.content == "new content"

    store.set_item_state(item.id, ItemState.RUNNING)
    blocked = store.edit_item(item.id, content="should not apply")
    assert blocked is None
    state = store.load()
    assert state.items[0].content == "new content"


def test_remove_running_item_cancels_attempt(tmp_path):
    store = _store(tmp_path)
    item = store.add_item("ongoing")
    store.set_item_state(item.id, ItemState.RUNNING)
    store.append_attempt(item.id, AttemptRecord(session_id="sess_x"))
    assert store.remove_item(item.id) is True

    # Item itself is gone, but verifying via raw file that the attempt was
    # closed prior to removal is not possible (item is removed). The contract
    # is "cancelled attempt then remove" — confirm at least that remove
    # returned True and the file no longer has this item.
    state = store.load()
    assert state.items == []


def test_reorder_respects_request(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("a")
    b = store.add_item("b")
    c = store.add_item("c")
    store.reorder([c.id, a.id, b.id])
    state = store.load()
    assert [it.id for it in state.items] == [c.id, a.id, b.id]


def test_reorder_appends_unmentioned_items(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("a")
    b = store.add_item("b")
    c = store.add_item("c")
    store.reorder([c.id])
    state = store.load()
    # c first, then a and b in original order
    assert state.items[0].id == c.id
    remaining_ids = {state.items[1].id, state.items[2].id}
    assert remaining_ids == {a.id, b.id}


def test_set_queue_state_persists(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED)
    state = QueueStore(tmp_path / "queue.json").load()
    assert state.state == QueueRunState.PAUSED


def test_corrupt_file_falls_back_to_empty(tmp_path):
    (tmp_path / "queue.json").write_text("{not valid json", encoding="utf-8")
    store = QueueStore(tmp_path / "queue.json")
    state = store.load()
    assert state.items == []
    assert state.state == QueueRunState.RUNNING


def test_head_returns_running_or_queued(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("first")
    b = store.add_item("second")
    assert store.head().id == a.id
    store.set_item_state(a.id, ItemState.DONE)
    assert store.head().id == b.id


def test_next_queued_skips_running(tmp_path):
    store = _store(tmp_path)
    a = store.add_item("first")
    b = store.add_item("second")
    store.set_item_state(a.id, ItemState.RUNNING)
    nxt = store.next_queued()
    assert nxt is not None
    assert nxt.id == b.id


def test_close_latest_attempt_is_idempotent(tmp_path):
    store = _store(tmp_path)
    item = store.add_item("x")
    store.append_attempt(item.id, AttemptRecord(session_id="s1"))
    store.close_latest_attempt(item.id, outcome=AttemptOutcome.COMPLETED, summary="ok")
    store.close_latest_attempt(item.id, outcome=AttemptOutcome.BLOCKED, summary="ignored")
    state = store.load()
    # First close wins; second is a no-op when outcome is already set
    assert state.items[0].attempts[0].outcome == AttemptOutcome.COMPLETED
    assert state.items[0].attempts[0].summary == "ok"


def test_increment_interrupted(tmp_path):
    store = _store(tmp_path)
    item = store.add_item("x")
    assert store.increment_interrupted(item.id) == 1
    assert store.increment_interrupted(item.id) == 2
    state = store.load()
    assert state.items[0].interrupted_count == 2


def test_snapshot_serializes_enums(tmp_path):
    store = _store(tmp_path)
    item = store.add_item("a")
    snap = store.snapshot()
    # Round-trip through JSON to mimic an HTTP response
    serialized = json.dumps(snap)
    parsed = json.loads(serialized)
    assert parsed["state"] == "running"
    assert parsed["items"][0]["state"] == "queued"
    assert parsed["items"][0]["source"] == "user"


def test_paused_until_round_trip(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, paused_until="2026-06-11T09:00:00+00:00")
    store2 = QueueStore(tmp_path / "queue.json")
    state = store2.load()
    assert state.state == QueueRunState.PAUSED
    assert state.paused_until == "2026-06-11T09:00:00+00:00"


def test_paused_until_cleared_on_leaving_paused(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, paused_until="2026-06-11T09:00:00+00:00")
    store.set_queue_state(QueueRunState.RUNNING)
    assert store.load().paused_until is None


def test_pause_without_duration_has_no_deadline(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED)
    assert store.load().paused_until is None


def test_auto_idle_clears_paused_until(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, paused_until="2026-06-11T09:00:00+00:00")
    changed = store.auto_idle_if_empty()
    assert changed is True
    assert store.load().paused_until is None


# ---------------------------------------------------------------------------
# pause_reason (Budget Piece 2, Task P2-B / C9)
# ---------------------------------------------------------------------------


def test_old_queue_json_without_pause_reason_loads(tmp_path):
    """Back-compat: a persisted queue.json predating the pause_reason field
    must deserialize fine, defaulting pause_reason to None."""
    legacy = {
        "version": 1,
        "state": "paused",
        "items": [],
        "paused_until": None,
    }
    (tmp_path / "queue.json").write_text(json.dumps(legacy), encoding="utf-8")
    state = QueueStore(tmp_path / "queue.json").load()
    assert state.state == QueueRunState.PAUSED
    assert state.pause_reason is None


def test_pause_reason_defaults_to_user(tmp_path):
    """Existing callers that omit reason get the user code — every legacy
    pause is user-initiated."""
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED)
    assert store.load().pause_reason == "user"


def test_pause_reason_stored_explicitly(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    assert store.load().pause_reason == "budget"


def test_pause_reason_round_trips(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    state = QueueStore(tmp_path / "queue.json").load()
    assert state.pause_reason == "budget"


def test_pause_reason_cleared_on_running(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    store.set_queue_state(QueueRunState.RUNNING)
    assert store.load().pause_reason is None


def test_pause_reason_cleared_on_idle(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    store.set_queue_state(QueueRunState.IDLE)
    assert store.load().pause_reason is None


def test_pause_reason_cleared_on_auto_idle(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    assert store.auto_idle_if_empty() is True
    assert store.load().pause_reason is None


def test_precedence_budget_does_not_downgrade_user(tmp_path):
    """A user pause is sticky: a later budget pause must NOT overwrite it."""
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="user")
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    assert store.load().pause_reason == "user"


def test_precedence_user_overrides_budget(tmp_path):
    """An explicit user action always wins: a manual pause over a budget
    pause sets the reason to user."""
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    store.set_queue_state(QueueRunState.PAUSED, reason="user")
    assert store.load().pause_reason == "user"


def test_precedence_user_resets_after_clearing(tmp_path):
    """The sticky-user rule only applies while already PAUSED: once the queue
    leaves PAUSED, a fresh budget pause is honored."""
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="user")
    store.set_queue_state(QueueRunState.RUNNING)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    assert store.load().pause_reason == "budget"


def test_budget_pause_can_be_re_set_to_budget(tmp_path):
    """Budget-over-budget is fine (no user pause to protect)."""
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    assert store.load().pause_reason == "budget"


def test_pause_reason_in_snapshot(tmp_path):
    store = _store(tmp_path)
    store.set_queue_state(QueueRunState.PAUSED, reason="budget")
    snap = store.snapshot()
    parsed = json.loads(json.dumps(snap))
    assert parsed["pause_reason"] == "budget"
