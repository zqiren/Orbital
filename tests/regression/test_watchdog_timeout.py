# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: per-attempt watchdog terminates a wedged loop and unblocks
the queue.

Verifies the QueueDispatcher's asyncio.wait_for wrapper: when an attempt
exceeds max_runtime_seconds, terminate() fires, the attempt closes as
interrupted with reason "exceeded runtime cap", and the dispatcher advances.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid):
        self.session_id = sid


class _SlowLoop:
    """A 'loop' whose task hangs until terminate() is called."""

    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "chat"
        self._terminated = asyncio.Event()
        self.terminate_called = False

    async def terminate(self):
        self.terminate_called = True
        self._terminated.set()


class _WedgedAgentManager:
    """Manager whose loop never completes until terminate() fires."""

    def __init__(self):
        self._loop = _SlowLoop()
        self._sessions = [_FakeSession("sess_w_0")]
        self.new_session_calls = 0
        self._task = None

    def get_session(self, project_id):
        return self._sessions[-1]

    def get_loop(self, project_id):
        return self._loop

    def get_loop_task(self, project_id):
        return self._task

    async def inject_message(self, project_id, content, *, nonce=None):
        # The loop "task" waits for terminate to fire, then completes.
        async def _hang():
            await self._loop._terminated.wait()

        self._task = asyncio.create_task(_hang())
        return "delivered"

    async def new_session(self, project_id):
        self.new_session_calls += 1
        self._sessions.append(_FakeSession(f"sess_w_{len(self._sessions)}"))
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        self._loop._terminated = asyncio.Event()
        return {"status": "new_session"}


@pytest.mark.asyncio
async def test_watchdog_fires_when_loop_exceeds_cap(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("hang forever")

    mgr = _WedgedAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_watchdog",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=2,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    # Wait up to 8s for the watchdog to fire and the item to be closed.
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        state = store.load()
        if state.items and state.items[0].state == ItemState.BLOCKED:
            break
        await asyncio.sleep(0.1)

    state = store.load()
    assert state.items[0].state == ItemState.BLOCKED
    assert state.items[0].attempts[0].outcome == AttemptOutcome.INTERRUPTED
    assert state.items[0].attempts[0].block_reason == "exceeded runtime cap"

    # The dispatcher should have invoked terminate on the loop
    assert mgr._loop.terminate_called

    # And rotated the session for the next attempt
    assert mgr.new_session_calls == 1

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_fast_attempt_not_killed_by_watchdog(tmp_path):
    """Sanity: an item that completes in <1s under a 60s cap is NOT killed."""
    from tests.integration.test_queue_phase2 import _ScriptedAgentManager

    store = QueueStore(tmp_path / "queue.json")
    store.add_item("be quick")

    mgr = _ScriptedAgentManager([{"reason": "complete", "summary": "fast"}])
    dispatcher = QueueDispatcher(
        project_id="proj_fast",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    deadline = asyncio.get_event_loop().time() + 3.0
    while asyncio.get_event_loop().time() < deadline:
        state = store.load()
        if state.items[0].state == ItemState.DONE:
            break
        await asyncio.sleep(0.05)

    state = store.load()
    assert state.items[0].state == ItemState.DONE
    assert state.items[0].attempts[0].outcome == AttemptOutcome.COMPLETED

    await dispatcher.shutdown()
