# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: when an item is dispatched a second time (e.g. via Concern-5
retry), the injected header carries attempt=2.

Concern 3 of TASK-queue-architecture-amendments.md: attempt_number is derived
as len(item.attempts) + 1 at dispatch time. This is a white-box test of that
derivation — Concern 5 lands the actual retry endpoint and will add an
end-to-end test.

The retry endpoint doesn't exist yet, so we mimic the post-block state by
hand: append a closed attempt to the item, reset its state to QUEUED, and
let the dispatcher pick it up again.
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
from tests.integration.test_queue_phase2 import _ScriptedAgentManager


class _CapturingAgentManager(_ScriptedAgentManager):
    """Scripted manager that also records every inject_message argument."""

    def __init__(self, script):
        super().__init__(script)
        self.injected: list[tuple[str, str]] = []

    async def inject_message(self, project_id, content, *, nonce=None):
        self.injected.append((project_id, content))
        return await super().inject_message(project_id, content, nonce=nonce)


async def _wait(predicate, timeout=10.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_second_dispatch_of_same_item_uses_attempt_two(tmp_path):
    """An item with one closed attempt in its history is dispatched again
    and the header shows attempt=2."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("redo me")

    # Simulate the post-block state Concern 5's retry will produce: one
    # closed attempt with outcome=BLOCKED, item state reset to QUEUED so
    # the dispatcher will pick it back up.
    prior_attempt = AttemptRecord(
        session_id="sess_prior",
        outcome=AttemptOutcome.BLOCKED,
        ended_at="2026-05-18T00:00:00+00:00",
        block_reason="needs more info",
    )
    store.append_attempt(item.id, prior_attempt)
    store.set_item_state(item.id, ItemState.QUEUED)

    # Sanity: one prior attempt is on disk.
    assert len(store.load().items[0].attempts) == 1

    mgr = _CapturingAgentManager([{"reason": "complete", "summary": "done"}])
    dispatcher = QueueDispatcher(
        project_id="proj_retry_header",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: len(mgr.injected) >= 1, timeout=5.0)
    if not ok:
        await dispatcher.shutdown()
        pytest.fail("dispatcher never re-injected the item")

    _, content = mgr.injected[0]
    expected = (
        f"[QUEUE ITEM | id={item.id} | attempt=2]\n"
        + QueueDispatcher.HEADER_CONTRACT
        + "redo me"
    )
    assert content == expected, (
        f"expected attempt=2 wrapping on the re-dispatch\n"
        f"  expected: {expected!r}\n  actual:   {content!r}"
    )

    # And the new attempt was appended, so the item now has 2 total attempts.
    final_attempts = store.load().items[0].attempts
    assert len(final_attempts) == 2
    assert final_attempts[0].session_id == "sess_prior"
    assert final_attempts[1].session_id != "sess_prior"

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_attempt_counter_continues_past_two(tmp_path):
    """Three pre-existing closed attempts → next dispatch carries
    attempt=4. Verifies the counter is genuinely
    `len(attempts) + 1`, not a hard-coded `2`."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("stubborn item")

    for i in range(3):
        store.append_attempt(item.id, AttemptRecord(
            session_id=f"sess_{i}",
            outcome=AttemptOutcome.BLOCKED,
            ended_at="2026-05-18T00:00:00+00:00",
            block_reason="still blocked",
        ))
    store.set_item_state(item.id, ItemState.QUEUED)

    mgr = _CapturingAgentManager([{"reason": "complete", "summary": "ok"}])
    dispatcher = QueueDispatcher(
        project_id="proj_retry_n",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: len(mgr.injected) >= 1, timeout=5.0)
    assert ok, "dispatcher never re-injected the item"

    _, content = mgr.injected[0]
    assert content.startswith(
        f"[QUEUE ITEM | id={item.id} | attempt=4]\n"
        + QueueDispatcher.HEADER_CONTRACT
    )
    assert content.endswith("stubborn item")

    await dispatcher.shutdown()
