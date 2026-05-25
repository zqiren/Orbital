# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: text-only exit on queue item gets ONE corrective turn
before being recorded as a contract violation (CHANGE 2 of the queue
architecture amendments).

The dispatcher's _await_and_handle, on first text-only exit during a
queue dispatch, injects a stern system message reminding the agent of
the signal contract and restarts the loop on the same session. The
agent gets one more chance to call mark_task_complete or
mark_task_blocked. Only if THAT turn also exits text-only does the
attempt close as a contract violation — with reason text noting that
the corrective turn was ignored.

A retry of a blocked item (Concern 5 of the prior amendments) starts a
fresh _await_and_handle call with its own corrective_turn_used flag, so
each attempt gets its own corrective turn budget.
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemState,
)
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid):
        self.session_id = sid


class _FakeLoop:
    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "running"


class _MultiTurnAgentManager:
    """Mock manager that scripts a SEQUENCE of exit_reasons per attempt.

    Each call to inject_message (initial dispatch) or inject_system_message
    (corrective turn) consumes the next script entry and "runs the loop"
    to that exit. The loop task completes immediately so the dispatcher
    proceeds to read the exit reason.
    """

    def __init__(self, script: list[dict]):
        self._script = list(script)
        self._loop = _FakeLoop()
        self._session_counter = 0
        self._session = _FakeSession(f"sess_{self._session_counter}")
        self._task: Optional[asyncio.Task] = None
        self.new_session_calls = 0
        self.inject_system_calls: list[str] = []
        self.inject_message_calls: list[str] = []

    def is_onboarding_complete(self, project_id):
        return True

    def get_session(self, project_id):
        return self._session

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def get_sub_agent_manager(self):
        return None

    def _consume_and_run(self) -> None:
        """Set the loop's exit fields from the next script entry and
        spawn an immediately-completing task that the dispatcher will await."""
        entry = self._script.pop(0) if self._script else {"reason": "text"}
        self._loop._exit_reason = entry["reason"]
        self._loop._exit_summary = entry.get("summary")
        self._loop._exit_block_reason = entry.get("block_reason")

        async def _instant():
            return None

        self._task = asyncio.create_task(_instant())

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        self.inject_message_calls.append(content)
        self._consume_and_run()
        return "delivered"

    async def inject_system_message(self, project_id, content):
        self.inject_system_calls.append(content)
        self._consume_and_run()
        return "delivered"

    async def new_session(self, project_id, *, session_id=None):
        self.new_session_calls += 1
        self._session_counter += 1
        sid = f"sess_{self._session_counter}"
        self._session = _FakeSession(sid)
        # Mirror the real loop.run() reset
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        return {
            "status": "ok",
            "session_id": sid,
            "session_uuid": f"proj_{self._session_counter:08d}",
        }

    async def switch_session(self, project_id, session_id, *, start_loop=False):
        # Used by retry. Park current task if any, then reuse the named
        # session as the current pointer. No loop start here.
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        self._session = _FakeSession(session_id)
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        return self._session


async def _wait(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# Test 1: corrective turn lets a fumbled attempt recover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_only_exit_gets_one_corrective_turn(tmp_path):
    """First text-only exit triggers corrective injection. Agent's
    corrective turn calls mark_task_blocked with a real reason. The
    attempt is recorded with the agent's reason — NOT contract violation."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("ambiguous task")

    mgr = _MultiTurnAgentManager([
        {"reason": "text"},                              # first turn
        {"reason": "blocked", "block_reason": "needs API key"},  # corrective
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_corrective",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.BLOCKED,
        timeout=5.0,
    )
    assert ok, "Item should reach BLOCKED state"

    state = store.load()
    item = state.items[0]
    assert len(item.attempts) == 1, "Single attempt, just two loop runs inside it"
    att = item.attempts[0]
    assert att.outcome == AttemptOutcome.BLOCKED
    assert att.block_reason == "needs API key", (
        "block_reason should be the agent's stated reason from the corrective "
        f"turn, not a contract-violation message. Got: {att.block_reason!r}"
    )
    assert "contract violation" not in (att.block_reason or "")

    # Verify the corrective system message was actually injected once
    assert len(mgr.inject_system_calls) == 1
    assert "did not declare" in mgr.inject_system_calls[0].lower() or "exited without declaring" in mgr.inject_system_calls[0].lower()

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# Test 2: second text-only exit is a violation with the "ignored" reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_text_only_exits_is_violation(tmp_path):
    """Agent exits text-only on both first and corrective turns. The
    attempt closes as BLOCKED with the contract-violation reason that
    explicitly notes the corrective turn was ignored."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("the agent will fumble twice")

    mgr = _MultiTurnAgentManager([
        {"reason": "text"},  # first turn — text-only
        {"reason": "text"},  # corrective turn — also text-only
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_double_text",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.BLOCKED,
        timeout=5.0,
    )
    assert ok

    state = store.load()
    att = state.items[0].attempts[0]
    assert att.outcome == AttemptOutcome.BLOCKED
    assert "contract violation" in (att.block_reason or "").lower()
    assert "corrective turn ignored" in (att.block_reason or "").lower(), (
        f"block_reason should note the corrective turn was used and ignored. "
        f"Got: {att.block_reason!r}"
    )

    # Corrective injection happened exactly once before giving up
    assert len(mgr.inject_system_calls) == 1

    await dispatcher.shutdown()


# ---------------------------------------------------------------------------
# Test 3: each retry gets a fresh corrective turn budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corrective_turn_resets_per_attempt(tmp_path):
    """First attempt: text → text (uses + burns corrective, blocks as
    violation). Retry: text → blocked-with-real-reason (the retry's
    corrective turn is fresh and the agent recovers on it). The retry
    attempt's block_reason must be the agent's reason, NOT a violation,
    proving the flag did NOT carry over."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("requires two tries")

    mgr = _MultiTurnAgentManager([
        # First attempt: two text-only exits → contract violation
        {"reason": "text"},
        {"reason": "text"},
        # Retry: text on first turn (uses corrective), then agent recovers
        {"reason": "text"},
        {"reason": "blocked", "block_reason": "still missing context"},
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_retry_fresh_corrective",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    # Wait for first attempt to land as BLOCKED with violation message
    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.BLOCKED
        and store.load().items[0].attempts
        and "corrective turn ignored" in (store.load().items[0].attempts[-1].block_reason or "").lower(),
        timeout=5.0,
    )
    assert ok, (
        "First attempt should end BLOCKED with corrective-ignored message. "
        f"Final state: {store.load().items[0].state}, "
        f"reason: {store.load().items[0].attempts[-1].block_reason if store.load().items[0].attempts else None!r}"
    )
    assert len(mgr.inject_system_calls) == 1

    # Now retry the blocked item with mode=answer — fresh corrective budget
    await dispatcher.retry_blocked_item(
        item.id, "here is more context", mode="answer",
    )

    # Wait for the retry to close
    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.BLOCKED
        and len(store.load().items[0].attempts) == 2
        and store.load().items[0].attempts[-1].outcome is not None,
        timeout=5.0,
    )
    assert ok, (
        "Retry attempt should reach a terminal state. "
        f"attempts: {[(a.outcome.value if a.outcome else None, (a.block_reason or '')[:40]) for a in store.load().items[0].attempts]}"
    )

    retry_att = store.load().items[0].attempts[-1]
    assert retry_att.outcome == AttemptOutcome.BLOCKED
    assert retry_att.block_reason == "still missing context", (
        f"Retry's corrective turn should have been honored — agent's reason "
        f"should be recorded, not contract violation. Got: {retry_att.block_reason!r}"
    )
    assert "contract violation" not in (retry_att.block_reason or "").lower()

    # Two corrective injections total: one in attempt 1, one in retry attempt 2
    assert len(mgr.inject_system_calls) == 2, (
        f"Each attempt should consume its own corrective turn. "
        f"inject_system_calls={len(mgr.inject_system_calls)}"
    )

    await dispatcher.shutdown()
