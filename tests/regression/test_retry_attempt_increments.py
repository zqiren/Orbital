# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concern 5: each retry appends an attempt sharing the prior session_id.

Three blocked cycles in a row:
  attempt 1 → blocked → retry → attempt 2 → blocked → retry → attempt 3
All three attempts must share session_id == "sess_A" because retries continue
the same JSONL — they do NOT open a new session per retry.
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


async def _wait(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_three_retries_preserve_same_session_id(tmp_path):
    """Block → retry → block → retry → block → retry: three attempts after
    the original one, all sharing the original session_id."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("the impossible task")

    # Seed: one prior blocked attempt on sess_A.
    store.append_attempt(item.id, AttemptRecord(
        session_id="sess_A",
        outcome=AttemptOutcome.BLOCKED,
        ended_at="2026-05-18T00:00:00+00:00",
        block_reason="initial block",
    ))
    store.set_item_state(item.id, ItemState.BLOCKED)

    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_increment",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()

    # --- Retry 1 → attempt 2 → blocked ---
    mgr.queue_next_exit("blocked", block_reason="still wrong")
    r1 = await dispatcher.retry_blocked_item(
        item.id, "attempt 2 input", mode="edit",
    )
    assert r1["attempt_number"] == 2
    assert r1["session_id"] == "sess_A"

    # Wait for the handler to record the BLOCKED outcome and flip state back.
    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.BLOCKED
        and store.load().items[0].attempts[-1].outcome == AttemptOutcome.BLOCKED
    ), timeout=5.0)
    assert ok, (
        "attempt 2 should reach BLOCKED outcome before retry 2; "
        f"state={store.load().items[0].state.value} "
        f"last_outcome={store.load().items[0].attempts[-1].outcome}"
    )

    # --- Retry 2 → attempt 3 → blocked ---
    mgr.queue_next_exit("blocked", block_reason="still wrong x2")
    r2 = await dispatcher.retry_blocked_item(
        item.id, "attempt 3 input", mode="edit",
    )
    assert r2["attempt_number"] == 3
    assert r2["session_id"] == "sess_A"

    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.BLOCKED
        and len(store.load().items[0].attempts) == 3
        and store.load().items[0].attempts[-1].outcome == AttemptOutcome.BLOCKED
    ), timeout=5.0)
    assert ok

    # --- Retry 3 → attempt 4 → blocked ---
    mgr.queue_next_exit("blocked", block_reason="still wrong x3")
    r3 = await dispatcher.retry_blocked_item(
        item.id, "attempt 4 input", mode="edit",
    )
    assert r3["attempt_number"] == 4
    assert r3["session_id"] == "sess_A"

    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.BLOCKED
        and len(store.load().items[0].attempts) == 4
    ), timeout=5.0)
    assert ok

    # ---- Final invariants ----
    final = store.load().items[0]
    assert len(final.attempts) == 4, (
        f"expected 4 attempts (1 original + 3 retries); "
        f"got {len(final.attempts)}"
    )
    # All four attempts share sess_A.
    session_ids = [a.session_id for a in final.attempts]
    assert session_ids == ["sess_A"] * 4, (
        f"all attempts must share session_id 'sess_A'; got {session_ids!r}"
    )
    # Attempt numbers are monotonic: 1, 2, 3, 4 implied by len-based counter.
    # Verify the retry results reported them in order.
    assert (r1["attempt_number"], r2["attempt_number"], r3["attempt_number"]) == (
        2, 3, 4,
    )

    # Each retry's session rotation: NO new_session call between retries,
    # because rotation only happens on the standard advance path AFTER the
    # exit handler runs. Each blocked exit DOES rotate, so by the end the
    # session manager has rotated 3 times (once per retry's blocked exit).
    # We don't assert an exact count to avoid coupling to rotation timing —
    # just verify all four attempts kept sess_A as their recorded id.

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_retry_then_complete_yields_done_with_same_session(tmp_path):
    """Cycle: block → retry → block → retry → complete. Final attempt is
    COMPLETED with session_id == sess_A, item state == DONE."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("finally works")

    store.append_attempt(item.id, AttemptRecord(
        session_id="sess_A",
        outcome=AttemptOutcome.BLOCKED,
        ended_at="2026-05-18T00:00:00+00:00",
        block_reason="first block",
    ))
    store.set_item_state(item.id, ItemState.BLOCKED)

    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_then_complete",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()

    # Retry 1: still blocked.
    mgr.queue_next_exit("blocked", block_reason="nope")
    await dispatcher.retry_blocked_item(item.id, "v2", mode="edit")
    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.BLOCKED
        and len(store.load().items[0].attempts) == 2
        and store.load().items[0].attempts[-1].outcome == AttemptOutcome.BLOCKED
    ), timeout=5.0)
    assert ok

    # Retry 2: completes.
    mgr.queue_next_exit("complete", summary="got it")
    await dispatcher.retry_blocked_item(item.id, "v3", mode="edit")
    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.DONE
    ), timeout=5.0)
    assert ok, (
        f"item should be DONE after completing retry; "
        f"actual={store.load().items[0].state.value}"
    )

    final = store.load().items[0]
    assert len(final.attempts) == 3
    assert all(a.session_id == "sess_A" for a in final.attempts)
    assert final.attempts[-1].outcome == AttemptOutcome.COMPLETED
    assert final.attempts[-1].summary == "got it"

    await dispatcher.shutdown()
