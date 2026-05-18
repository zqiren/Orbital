# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: text-only exit on a queue item is a contract violation
(Concern 4 of TASK-queue-architecture-amendments.md).

The dispatcher must close the attempt as BLOCKED with the contract-violation
reason, rotate the session, broadcast advance, and proceed to the next
queued item. Pre-Concern-4 behavior was 'stall and idle' — explicitly
forbidden here.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState
from agent_os.queue.store import QueueStore
from tests.integration.test_queue_phase2 import (
    _ScriptedAgentManager,
    _wait_until,
)


@pytest.mark.asyncio
async def test_text_only_exit_blocks_item_with_contract_violation_reason(tmp_path):
    """A queue item whose loop exits text-only ends up BLOCKED with the
    contract-violation reason, rotates the session, and the dispatcher
    advances to the next queued item."""
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("first task")
    item2 = store.add_item("second task")

    mgr = _ScriptedAgentManager([
        {"reason": "text"},                       # contract violation
        {"reason": "complete", "summary": "ok"},  # next item completes cleanly
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_contract_regression",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    # Both items reach terminal states.
    ok = await _wait_until(
        lambda: (
            store.load().items[0].state == ItemState.BLOCKED
            and store.load().items[1].state == ItemState.DONE
        ),
        timeout=10.0,
    )
    if not ok:
        state = store.load()
        await dispatcher.shutdown()
        pytest.fail(
            "Items did not reach expected terminal states. Final: "
            + ", ".join(f"{it.id}={it.state.value}" for it in state.items)
        )

    state = store.load()

    # Item 1: blocked with contract violation
    assert state.items[0].id == item1.id
    assert state.items[0].state == ItemState.BLOCKED
    assert len(state.items[0].attempts) == 1
    attempt1 = state.items[0].attempts[0]
    assert attempt1.outcome == AttemptOutcome.BLOCKED
    assert "contract violation" in (attempt1.block_reason or ""), (
        f"expected 'contract violation' in block_reason, "
        f"got: {attempt1.block_reason!r}"
    )

    # Item 2: advanced past the blocked one and completed
    assert state.items[1].id == item2.id
    assert state.items[1].state == ItemState.DONE
    assert state.items[1].attempts[0].outcome == AttemptOutcome.COMPLETED

    # Session rotated for BOTH terminal exits (no longer 1 — Concern 4
    # makes the violation path rotate just like the explicit blocked path).
    assert mgr.new_session_calls >= 1, (
        f"expected at least one new_session call after contract violation, "
        f"got {mgr.new_session_calls}"
    )

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_text_only_exit_records_exact_block_reason(tmp_path):
    """Pin the block_reason string so a future refactor that softens the
    wording (e.g. drops the em-dash, lowercases differently) is caught."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("single item")

    mgr = _ScriptedAgentManager([{"reason": "text"}])

    dispatcher = QueueDispatcher(
        project_id="proj_contract_pin",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait_until(
        lambda: store.load().items[0].state == ItemState.BLOCKED,
        timeout=5.0,
    )
    assert ok, "item never reached BLOCKED"

    state = store.load()
    attempt = state.items[0].attempts[0]
    assert attempt.block_reason == (
        "agent exited without declaring outcome — contract violation"
    ), f"block_reason mismatch: {attempt.block_reason!r}"

    await dispatcher.shutdown()
