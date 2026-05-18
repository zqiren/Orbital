# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concern 5: retry with mode='edit' wraps the input with the queue header.

The exact wire format under test:
    [QUEUE ITEM | id={item_id} | attempt={N+1}]\\n{edited_content}

Header parameters match what _dispatch_one emits for first-time dispatch
(Concern 3) — the attempt number is computed as len(item.attempts) + 1.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemState,
)
from agent_os.queue.store import QueueStore
from tests.regression.test_retry_hot_resumes_session import (
    _RetryAgentManager,
)


@pytest.mark.asyncio
async def test_retry_edit_wraps_with_attempt_2_header(tmp_path):
    """Edit-mode retry wraps with [QUEUE ITEM | id=... | attempt=2]\\n{content}."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("original content that turned out wrong")

    # Pre-existing blocked attempt on session "sess_A".
    store.append_attempt(item.id, AttemptRecord(
        session_id="sess_A",
        outcome=AttemptOutcome.BLOCKED,
        ended_at="2026-05-18T00:00:00+00:00",
        block_reason="instructions were ambiguous",
    ))
    store.set_item_state(item.id, ItemState.BLOCKED)

    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_edit_hdr",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()

    # Hang the retry so we can assert pre-exit state cleanly.
    mgr.queue_next_exit("blocked", block_reason="still need more")

    result = await dispatcher.retry_blocked_item(
        item.id, "edited content", mode="edit",
    )

    assert result["mode"] == "edit"
    assert result["attempt_number"] == 2
    assert result["session_id"] == "sess_A"

    # Exact wire format.
    expected = f"[QUEUE ITEM | id={item.id} | attempt=2]\nedited content"
    assert mgr.injected_messages == [expected], (
        f"edit-mode retry must wrap with header\n"
        f"  expected: {expected!r}\n  actual:   {mgr.injected_messages!r}"
    )

    # Switched to the parked session, not a fresh one.
    assert ("sess_A", False) in mgr.switch_calls
    assert mgr._current_sid == "sess_A"

    # New attempt sharing sess_A is on the item.
    item_after = store.load().items[0]
    assert len(item_after.attempts) == 2
    assert item_after.attempts[1].session_id == "sess_A"

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_retry_edit_header_uses_correct_attempt_number_after_many(
    tmp_path,
):
    """If the item already has 4 closed attempts, edit-mode retry header
    reports attempt=5 — i.e. the counter is len(attempts)+1, not a constant."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("stubborn task")

    for i in range(4):
        store.append_attempt(item.id, AttemptRecord(
            session_id=f"sess_old_{i}",
            outcome=AttemptOutcome.BLOCKED,
            ended_at="2026-05-18T00:00:00+00:00",
            block_reason="nope",
        ))
    store.set_item_state(item.id, ItemState.BLOCKED)

    mgr = _RetryAgentManager()
    # Most recent attempt's session is the one retry continues.
    # (We don't pre-create the session — switch_session creates it lazily.)
    dispatcher = QueueDispatcher(
        project_id="proj_retry_edit_attempt5",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    mgr.queue_next_exit("blocked", block_reason="still")

    result = await dispatcher.retry_blocked_item(
        item.id, "try this instead", mode="edit",
    )

    assert result["attempt_number"] == 5
    expected = f"[QUEUE ITEM | id={item.id} | attempt=5]\ntry this instead"
    assert mgr.injected_messages == [expected]

    await dispatcher.shutdown()
