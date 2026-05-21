# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 2 integration: dispatcher advances on complete/blocked, stalls on text.

This test exercises the QueueDispatcher against a mocked AgentManager whose
loop_object exposes a controllable _exit_reason. It verifies the dispatcher's
routing logic for the three exit kinds and the per-item session rotation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid: str):
        self.session_id = sid


class _FakeLoop:
    """Mutable per-attempt exit state."""

    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "chat"


class _ScriptedAgentManager:
    """Mocked AgentManager that scripts exit_reasons across attempts.

    Each call to inject_message consumes the next script entry and sets
    _exit_reason on the (single, shared) loop object. The loop "task" then
    completes immediately so the dispatcher proceeds to read the reason.
    """

    def __init__(self, script: list[dict]):
        # script entries: {"reason": "complete"|"blocked"|"text",
        #                  "summary"?: str, "block_reason"?: str}
        self._script = list(script)
        self._loop = _FakeLoop()
        self._session_counter = 0
        self._session = _FakeSession(f"sess_{self._session_counter}")
        self._task = None
        self.new_session_calls = 0

    def get_session(self, project_id):
        return self._session

    def get_loop(self, project_id):
        return self._loop

    def get_loop_task(self, project_id):
        return self._task

    async def inject_message(self, project_id, content, *, nonce=None, session_id=None):
        # Set up the scripted exit and complete the "loop task" immediately.
        entry = self._script.pop(0) if self._script else {"reason": "text"}
        self._loop._exit_reason = entry["reason"]
        self._loop._exit_summary = entry.get("summary")
        self._loop._exit_block_reason = entry.get("block_reason")

        async def _instant():
            return None

        self._task = asyncio.create_task(_instant())
        return "delivered"

    async def new_session(self, project_id):
        # Mint a fresh session id, mimicking AgentManager.new_session.
        self.new_session_calls += 1
        self._session_counter += 1
        self._session = _FakeSession(f"sess_{self._session_counter}")
        # Reset the loop's exit_reason like the real loop.run() does
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        return {"status": "new_session"}


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_three_item_pipeline_complete_blocked_text(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("write the code")
    item2 = store.add_item("deploy it")
    item3 = store.add_item("write the docs")

    mgr = _ScriptedAgentManager([
        {"reason": "complete", "summary": "wrote the code"},
        {"reason": "blocked", "block_reason": "no prod creds"},
        {"reason": "text"},  # contract violation — Concern 4
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_phase2",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    # Concern 4: all three items reach a terminal state. Item 3's text-only
    # exit is now a contract violation and routes through the blocked path.
    ok = await _wait_until(lambda: (
        store.load().items[0].state == ItemState.DONE
        and store.load().items[1].state == ItemState.BLOCKED
        and store.load().items[2].state == ItemState.BLOCKED
    ), timeout=10.0)

    if not ok:
        state = store.load()
        await dispatcher.shutdown()
        pytest.fail(
            "Items did not reach expected states. Final: "
            + ", ".join(f"{it.id}={it.state.value}" for it in state.items)
        )

    state = store.load()

    # Item 1: completed
    assert state.items[0].id == item1.id
    assert state.items[0].state == ItemState.DONE
    assert state.items[0].attempts[0].outcome == AttemptOutcome.COMPLETED
    assert state.items[0].attempts[0].summary == "wrote the code"

    # Item 2: explicitly blocked
    assert state.items[1].id == item2.id
    assert state.items[1].state == ItemState.BLOCKED
    assert state.items[1].attempts[0].outcome == AttemptOutcome.BLOCKED
    assert state.items[1].attempts[0].block_reason == "no prod creds"

    # Item 3: blocked-by-contract-violation (Concern 4)
    assert state.items[2].id == item3.id
    assert state.items[2].state == ItemState.BLOCKED
    assert len(state.items[2].attempts) == 1
    assert state.items[2].attempts[0].outcome == AttemptOutcome.BLOCKED
    assert "contract violation" in (
        state.items[2].attempts[0].block_reason or ""
    )

    # Session rotation happened after each of the three terminal exits.
    assert mgr.new_session_calls == 3

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_each_attempt_records_its_own_session_id(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("a")
    store.add_item("b")

    mgr = _ScriptedAgentManager([
        {"reason": "complete", "summary": "did a"},
        {"reason": "complete", "summary": "did b"},
    ])

    dispatcher = QueueDispatcher(
        project_id="proj_session_ids",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait_until(lambda: all(
        it.state == ItemState.DONE for it in store.load().items
    ), timeout=10.0)
    assert ok, "Both items should be done"

    state = store.load()
    sid_a = state.items[0].attempts[0].session_id
    sid_b = state.items[1].attempts[0].session_id
    assert sid_a != sid_b, "Each attempt should record a distinct session id"

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_text_only_exit_is_contract_violation_and_rotates(tmp_path):
    """Concern 4 inversion: text-only exit on a queue item is now a contract
    violation, not a stall. The dispatcher MUST close the attempt as BLOCKED
    with a contract-violation reason and rotate the session — same shape as
    an explicit mark_task_blocked. The pre-Concern-4 behavior (stall, no
    rotation) is exactly what this assertion now forbids."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("just chat")

    mgr = _ScriptedAgentManager([{"reason": "text"}])

    dispatcher = QueueDispatcher(
        project_id="proj_contract_violation",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait_until(
        lambda: store.load().items[0].state == ItemState.BLOCKED,
        timeout=5.0,
    )
    if not ok:
        state = store.load()
        await dispatcher.shutdown()
        pytest.fail(
            f"item never reached BLOCKED. final state: "
            f"{state.items[0].state.value}, attempts={state.items[0].attempts}"
        )

    state = store.load()
    assert state.items[0].id == item.id
    assert state.items[0].state == ItemState.BLOCKED
    assert len(state.items[0].attempts) == 1
    attempt = state.items[0].attempts[0]
    assert attempt.outcome == AttemptOutcome.BLOCKED
    assert "contract violation" in (attempt.block_reason or ""), (
        f"expected 'contract violation' in block_reason, "
        f"got: {attempt.block_reason!r}"
    )
    # Rotation MUST have happened — pre-Concern-4 this was 0.
    assert mgr.new_session_calls == 1

    await dispatcher.shutdown()
