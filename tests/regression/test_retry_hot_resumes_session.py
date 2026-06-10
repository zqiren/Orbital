# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Concern 5: retry_blocked_item hot-resumes the parked attempt's session.

Contract being exercised:
  Item is BLOCKED with one closed attempt on session "sess_A". Caller invokes
  retry_blocked_item(item_id, "use postgres", mode="answer"). The dispatcher
  must:
    - Switch the active session back to "sess_A" (via switch_session with
      start_loop=False, so _cold_resume_injected and related ContextManager
      flags are reset).
    - Append a NEW attempt (attempt_number=2) sharing session_id="sess_A"
      because retries CONTINUE the JSONL, they don't open a new one.
    - Inject the new input RAW (no header) because mode=="answer".
    - Set the item state to RUNNING.
    - When the loop exits with reason=="complete", route through the standard
      handler: close the new attempt with COMPLETED, rotate the session, mark
      the item DONE.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemState,
    QueueRunState,
)
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid: str):
        self.session_id = sid
        self.messages: list[dict] = []

    def append(self, msg: dict) -> None:
        self.messages.append(msg)


class _FakeLoop:
    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "running"
        self._session = None


class _RetryAgentManager:
    """Scripted manager that captures injects and runs a loop body per inject."""

    def __init__(self):
        self._loop = _FakeLoop()
        self._sessions: dict[str, _FakeSession] = {}
        self._current_sid = "sess_A"
        self._sessions[self._current_sid] = _FakeSession(self._current_sid)
        self._loop._session = self._sessions[self._current_sid]
        self._task: asyncio.Task | None = None
        # Scripting
        self._next_exit_reason: str | None = None
        self._next_summary: str | None = None
        self._next_block_reason: str | None = None
        # Bookkeeping
        self.injected_messages: list[str] = []
        self.switch_calls: list[tuple[str, bool]] = []
        self.new_session_calls = 0

    # ---- API expected by QueueDispatcher ----

    def is_onboarding_complete(self, project_id):
        return True

    def get_session(self, project_id):
        return self._sessions[self._current_sid]

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def get_sub_agent_manager(self):
        return None

    # ---- Scripting helpers ----

    def queue_next_exit(
        self, reason: str, *, summary=None, block_reason=None,
    ):
        self._next_exit_reason = reason
        self._next_summary = summary
        self._next_block_reason = block_reason

    # ---- Lifecycle ----

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state="chat"):
        # Capture the injected text RAW for assertions.
        self.injected_messages.append(content)
        self._sessions[self._current_sid].append(
            {"role": "user", "content": content},
        )
        # Reset exit_reason as the real loop.run() does on entry.
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None

        next_reason = self._next_exit_reason
        next_summary = self._next_summary
        next_block_reason = self._next_block_reason
        self._next_exit_reason = None
        self._next_summary = None
        self._next_block_reason = None

        async def _loop_body():
            if next_reason is None:
                # Hang — never exits on its own.
                await asyncio.sleep(60)
            else:
                self._loop._exit_reason = next_reason
                self._loop._exit_summary = next_summary
                self._loop._exit_block_reason = next_block_reason
                await asyncio.sleep(0.01)

        self._task = asyncio.create_task(_loop_body())
        return "delivered"

    async def switch_session(self, project_id, session_id, *, start_loop=False):
        self.switch_calls.append((session_id, start_loop))
        # Terminate any in-flight task
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if session_id not in self._sessions:
            self._sessions[session_id] = _FakeSession(session_id)
        self._current_sid = session_id
        self._loop._session = self._sessions[session_id]
        return self._sessions[session_id]

    async def _start_loop(self, project_id, *, session_id=None):
        # Not used by retry path; included for completeness.
        async def _noop():
            return None
        self._task = asyncio.create_task(_noop())

    async def new_session(self, project_id, *, session_id=None):
        self.new_session_calls += 1
        sid = f"sess_rotated_{self.new_session_calls}"
        self._sessions[sid] = _FakeSession(sid)
        self._current_sid = sid
        self._loop._session = self._sessions[sid]
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        return {
            "status": "ok",
            "session_id": sid,
            "session_uuid": f"proj_{self.new_session_calls:08d}",
        }


async def _wait(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_retry_answer_mode_hot_resumes_prior_session(tmp_path):
    """retry with mode='answer' must reuse the prior session_id, inject the
    input raw (no header), and route the next exit through the standard
    advance path."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("set up the database")

    # Simulate a prior blocked attempt on session "sess_A": agent asked
    # "postgres or sqlite?" and exited blocked.
    store.append_attempt(item.id, AttemptRecord(
        session_id="sess_A",
        outcome=AttemptOutcome.BLOCKED,
        ended_at="2026-05-18T00:00:00+00:00",
        block_reason="postgres or sqlite?",
    ))
    store.set_item_state(item.id, ItemState.BLOCKED)

    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_answer",
        store=store,
        agent_manager=mgr,
    )
    await dispatcher.start()

    # Script the retry loop to complete immediately.
    mgr.queue_next_exit("complete", summary="postgres set up")

    result = await dispatcher.retry_blocked_item(
        item.id, "use postgres", mode="answer",
    )

    # ---- Immediate contract ----
    assert result["status"] == "retry_started"
    assert result["item_id"] == item.id
    assert result["attempt_number"] == 2
    assert result["session_id"] == "sess_A"
    assert result["mode"] == "answer"

    # switch_session was called with the parked session id and start_loop=False
    assert ("sess_A", False) in mgr.switch_calls, (
        f"expected switch_session('sess_A', start_loop=False); "
        f"got {mgr.switch_calls!r}"
    )

    # Active session is now "sess_A" (the parked one), not a fresh one.
    assert mgr._current_sid == "sess_A"

    # Injected text is RAW — no [QUEUE ITEM ...] header in answer mode.
    assert mgr.injected_messages == ["use postgres"], (
        f"expected raw 'use postgres' injection; "
        f"got {mgr.injected_messages!r}"
    )

    # Item state is RUNNING and a fresh attempt sharing sess_A was appended.
    state = store.load()
    item_after = state.items[0]
    assert item_after.state == ItemState.RUNNING
    assert len(item_after.attempts) == 2
    assert item_after.attempts[1].session_id == "sess_A", (
        "retry must reuse prior session_id"
    )
    assert item_after.attempts[1].outcome is None  # still open

    # ---- Eventual contract: loop exit routes to DONE ----
    ok = await _wait(
        lambda: store.load().items[0].state == ItemState.DONE, timeout=5.0,
    )
    assert ok, (
        "item should reach DONE after the retry's loop exits with complete; "
        f"final state={store.load().items[0].state.value}"
    )

    final = store.load().items[0]
    assert final.attempts[1].outcome == AttemptOutcome.COMPLETED
    assert final.attempts[1].summary == "postgres set up"

    # Queue auto-idled (only item is now DONE).
    assert store.load().state == QueueRunState.IDLE

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_retry_rejects_non_blocked_item(tmp_path):
    """Retrying an item that isn't BLOCKED raises ValueError (→ 409 in route)."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("not blocked yet")

    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_reject",
        store=store,
        agent_manager=mgr,
    )

    with pytest.raises(ValueError, match="not BLOCKED"):
        await dispatcher.retry_blocked_item(item.id, "anything", mode="answer")


@pytest.mark.asyncio
async def test_retry_rejects_unknown_item(tmp_path):
    """Unknown item id raises KeyError (→ 404 in route)."""
    store = QueueStore(tmp_path / "queue.json")
    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_404",
        store=store,
        agent_manager=mgr,
    )
    with pytest.raises(KeyError):
        await dispatcher.retry_blocked_item(
            "item_does_not_exist", "x", mode="answer",
        )


@pytest.mark.asyncio
async def test_retry_rejects_bad_mode(tmp_path):
    """mode must be 'answer' or 'edit'."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("x")
    store.append_attempt(item.id, AttemptRecord(
        session_id="sess_A", outcome=AttemptOutcome.BLOCKED,
        ended_at="2026-05-18T00:00:00+00:00", block_reason="why",
    ))
    store.set_item_state(item.id, ItemState.BLOCKED)
    mgr = _RetryAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_retry_badmode",
        store=store,
        agent_manager=mgr,
    )
    with pytest.raises(ValueError, match="mode must be"):
        await dispatcher.retry_blocked_item(item.id, "x", mode="bogus")
