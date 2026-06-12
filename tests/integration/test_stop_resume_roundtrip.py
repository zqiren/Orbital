# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pause / resume integration: parked-session continuity, no session swap.

Contract under test (revised after dropping the dedicated chat session):

  Start a 3-item queue. While item 1 is running (slow attempt), call pause.
  Verify: queue PAUSED, item 1 still RUNNING (attempt preserved, NOT closed),
  no rotation, no session swap. Send a chat message while paused — it must
  land in the parked attempt's session (NOT a separate chat session). Call
  resume. The agent's next loop run sees the clarification in conversation
  history. Mock LLM then returns mark_task_complete; item 1 completes and
  item 2 starts.

Regression: on `feature/render-chat-variant-a` HEAD the dispatcher minted
a separate chat_session_id and swapped to it on pause, so chats landed in
that dedicated session and the agent could not see them on resume. This
test fails on that branch and passes on `fix/queue-pause-drop-session-swap`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import AttemptOutcome, ItemState, QueueRunState
from agent_os.queue.store import QueueStore


class _FakeSession:
    def __init__(self, sid: str):
        self.session_id = sid
        self.messages: list[dict] = []

    def append(self, msg: dict) -> None:
        self.messages.append(msg)


class _FakeLoop:
    """Models the parts of AgentLoop the dispatcher reads."""

    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "chat"
        self._session = None
        self._terminate_event = asyncio.Event()
        self.terminate_called = False
        self.cancel_turn_called = 0

    async def cancel_turn(self):
        """Mirror loop.cancel_turn: writes cancellation marker + sets
        _exit_reason='cancelled' synchronously (the production change in
        _persist_cancellation_marker_inner). Does NOT end the loop task."""
        self.cancel_turn_called += 1
        # The real cancellation marker write would happen here. For this
        # test the dispatcher reads _exit_reason after the loop task ends,
        # so we set it now.
        self._exit_reason = "cancelled"
        if self._session is not None:
            self._session.append({
                "role": "system",
                "content": "[cancelled by user]",
                "source": "management",
                "cancelled_by_user": True,
            })

    async def terminate(self):
        """Mirror loop.terminate: cancel_turn + session.stop + task.cancel.
        The fake collapses that into 'signal the loop body to exit'."""
        self.terminate_called = True
        await self.cancel_turn()
        self._terminate_event.set()


class _PauseResumeAgentManager:
    """Sophisticated mock that tracks every cross-boundary call the
    dispatcher might make. Crucially: no implicit session swap on pause —
    inject_message during pause MUST land on the same parked session.
    """

    def __init__(self):
        self._loop = _FakeLoop()
        self._sessions: dict[str, _FakeSession] = {}
        self._current_sid = "sess_attempt_1"
        self._sessions[self._current_sid] = _FakeSession(self._current_sid)
        self._loop._session = self._sessions[self._current_sid]
        self._task: asyncio.Task | None = None
        self._next_exit_reason = None
        self._next_summary = None
        self._next_block_reason = None
        self._loop_started_event = asyncio.Event()
        self.new_session_calls = 0
        self.switch_calls: list[tuple[str, bool]] = []
        # Sub-agent manager mock so dispatcher.pause's stop_all path is exercised.
        self._sub_agent_manager_obj = AsyncMock()
        self._sub_agent_manager_obj.stop_all = AsyncMock(return_value=None)

    def get_session(self, project_id, *, session_id=None):
        return self._sessions[self._current_sid]

    def current_holder_session_id(self, project_id):
        # The dispatcher live-reads the slot holder since 8083e35 (pure-create
        # + materialize-on-first-message); without this accessor its run loop
        # died with AttributeError ("unhandled error, backing off") and items
        # never reached RUNNING.
        return self._current_sid if self._task and not self._task.done() else None

    def is_onboarding_complete(self, project_id):
        # Dispatcher gates auto-start on onboarding since a1cffaa; these
        # roundtrip tests model an onboarded project.
        return True

    def get_loop(self, project_id, *, session_id=None):
        return self._loop

    def get_loop_task(self, project_id, *, session_id=None):
        return self._task

    def get_sub_agent_manager(self):
        return self._sub_agent_manager_obj

    def queue_next_exit(self, reason: str, *, summary=None, block_reason=None):
        self._next_exit_reason = reason
        self._next_summary = summary
        self._next_block_reason = block_reason

    async def inject_message(self, project_id, content, *, nonce=None,
                             session_id=None, queue_state=None):
        # queue_state kwarg added with the dispatcher's first-turn contract inject
        # Append to the CURRENTLY ACTIVE session — no implicit swap. This is
        # the heart of the regression test: if pause swapped sessions, this
        # message would land in the chat session, not the parked attempt.
        self._sessions[self._current_sid].append(
            {"role": "user", "content": content},
        )
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        self._loop._terminate_event = asyncio.Event()
        next_reason = self._next_exit_reason
        next_summary = self._next_summary
        next_block_reason = self._next_block_reason
        self._next_exit_reason = None
        self._next_summary = None
        self._next_block_reason = None

        async def _loop_body():
            self._loop_started_event.set()
            if next_reason is None:
                await self._loop._terminate_event.wait()
            else:
                self._loop._exit_reason = next_reason
                self._loop._exit_summary = next_summary
                self._loop._exit_block_reason = next_block_reason
                await asyncio.sleep(0.01)

        self._task = asyncio.create_task(_loop_body())
        return "delivered"

    async def switch_session(self, project_id, session_id, *, start_loop=False):
        """Recorded so we can assert dispatcher does NOT call this on pause
        or resume in the new architecture. Retry paths still use it."""
        self.switch_calls.append((session_id, start_loop))
        if self._task is not None and not self._task.done():
            await self._loop.terminate()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                pass
        self._loop.terminate_called = False
        if session_id not in self._sessions:
            self._sessions[session_id] = _FakeSession(session_id)
        self._current_sid = session_id
        self._loop._session = self._sessions[session_id]
        return self._sessions[session_id]

    async def _start_loop(self, project_id, *, session_id=None):
        # Used by dispatcher._resume_attempt after switch_session(start_loop=False)
        # In the real AgentManager, _start_loop runs handle.loop.run(). Here we
        # just spawn a loop body that mirrors the inject path.
        next_reason = self._next_exit_reason
        next_summary = self._next_summary
        next_block_reason = self._next_block_reason
        self._next_exit_reason = None
        self._next_summary = None
        self._next_block_reason = None
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        self._loop._terminate_event = asyncio.Event()

        async def _loop_body():
            if next_reason is None:
                await self._loop._terminate_event.wait()
            else:
                self._loop._exit_reason = next_reason
                self._loop._exit_summary = next_summary
                self._loop._exit_block_reason = next_block_reason
                await asyncio.sleep(0.01)

        self._task = asyncio.create_task(_loop_body())

    async def new_session(self, project_id):
        self.new_session_calls += 1
        sid = f"sess_rotated_{self.new_session_calls}"
        self._sessions[sid] = _FakeSession(sid)
        self._current_sid = sid
        self._loop._session = self._sessions[sid]
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        # The dispatcher reads minted["session_id"] since the pure-create
        # rewrite (8083e35) — mirror the real new_session() return shape.
        return {"status": "new_session", "session_id": sid}


async def _wait(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_pause_does_not_swap_chat_lands_in_parked_session_resume_continues(tmp_path):
    """End-to-end: pause → chat → resume must keep the same session and
    surface the chat in conversation history on resume.

    This is the regression test referenced in TASK §2.1: on the pre-fix
    branch the dispatcher swapped to a dedicated chat session on pause, so
    the chat landed in `chat_<hex>` and the agent never saw it on resume.
    """
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("first thing")
    store.add_item("second thing")
    store.add_item("third thing")

    mgr = _PauseResumeAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_pause_resume",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.RUNNING
        and mgr._loop_started_event.is_set()
    ), timeout=5.0)
    assert ok, "Item 1 should be running with loop in-flight"

    parked_sid = store.load().items[0].attempts[0].session_id

    # === PAUSE ===
    pause_result = await dispatcher.stop()
    assert pause_result["status"] == "paused"
    assert "chat_session_id" not in pause_result, (
        "pause must NOT return a chat_session_id in the new architecture"
    )

    # Allow _await_and_handle to run its stop_generation early-return.
    await asyncio.sleep(0.1)

    state = store.load()
    assert state.state == QueueRunState.PAUSED
    assert state.items[0].state == ItemState.RUNNING
    assert state.items[0].attempts[-1].outcome is None, (
        "pause must NOT close the in-flight attempt (preserve for resume)"
    )

    mgr._sub_agent_manager_obj.stop_all.assert_awaited()
    assert mgr._loop.terminate_called, (
        "pause must terminate the active loop so _await_and_handle returns"
    )

    # No swap to a chat session, ever.
    assert mgr._current_sid == parked_sid, (
        f"current session must still be the parked attempt's session, "
        f"got {mgr._current_sid!r}, expected {parked_sid!r}"
    )
    assert not any(
        sid.startswith("chat_") for sid, _ in mgr.switch_calls
    ), f"pause must not switch to a chat_* session; saw {mgr.switch_calls!r}"

    # Items 2/3 untouched.
    assert state.items[1].state == ItemState.QUEUED
    assert state.items[2].state == ItemState.QUEUED

    # === CHAT WHILE PAUSED ===
    # User asks a clarifying question while the queue is paused. The chat
    # injection routes through inject_message — in the real system, Case 2
    # (idle session + alive handle) appends to the parked session.
    await mgr.inject_message(
        "proj_pause_resume", "wait, what does 'thing' mean here?",
    )
    parked_session = mgr._sessions[parked_sid]
    assert any(
        m["content"] == "wait, what does 'thing' mean here?"
        for m in parked_session.messages
    ), (
        "chat-while-paused must land in the parked attempt session — "
        "this is the headline fix"
    )
    # And critically, no chat_<hex> session exists at all.
    assert not any(
        sid.startswith("chat_") for sid in mgr._sessions
    ), f"no chat_* session should exist; saw {list(mgr._sessions)!r}"

    # The inject above spawned a fake loop task on the parked session; let
    # the dispatcher observe it then drain so resume operates on a clean
    # slate. We cancel the lingering task ourselves.
    if mgr._task is not None and not mgr._task.done():
        await mgr._loop.terminate()
        try:
            await asyncio.wait_for(mgr._task, timeout=1.0)
        except Exception:
            pass

    # === RESUME ===
    # Script the next loop run to mark complete.
    mgr.queue_next_exit("complete", summary="now I understand — thing means widget")
    resume_result = await dispatcher.resume()
    assert resume_result["status"] == "running"
    assert resume_result["resumed_item_id"] == item1.id

    # Wait for item 1 to complete and item 2 to start.
    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.DONE
        and store.load().items[1].state == ItemState.RUNNING
    ), timeout=10.0)
    assert ok, (
        "Item 1 should complete and item 2 start. "
        + ", ".join(f"{it.id}={it.state.value}" for it in store.load().items)
    )

    state = store.load()
    assert state.items[0].state == ItemState.DONE
    assert state.items[0].attempts[-1].outcome == AttemptOutcome.COMPLETED
    assert state.items[0].attempts[-1].summary == (
        "now I understand — thing means widget"
    )
    # Resume must NOT have called switch_session for the parked sid
    # (the parked session is already active after pause).
    parked_swap = [c for c in mgr.switch_calls if c[0] == parked_sid]
    assert len(parked_swap) == 0, (
        f"resume must not call switch_session on the already-active parked "
        f"session; saw {mgr.switch_calls!r}"
    )

    # Final attempt on the original session id (no rotation across pause/resume).
    assert state.items[0].attempts[-1].session_id == parked_sid

    assert state.state == QueueRunState.RUNNING

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_pause_does_not_mint_chat_session_id_state_field_removed(tmp_path):
    """QueueState.chat_session_id no longer exists; pause/resume don't
    write or read it. Old queue.json files that still carry the field are
    silently dropped (extra='ignore') on first load."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("the only thing")

    mgr = _PauseResumeAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_no_chat_field",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()
    ok = await _wait(lambda: store.load().items[0].state == ItemState.RUNNING)
    assert ok

    result1 = await dispatcher.stop()
    assert "chat_session_id" not in result1

    state_after_pause = store.load()
    # The QueueState model no longer has the field.
    assert not hasattr(state_after_pause, "chat_session_id"), (
        "QueueState must no longer expose chat_session_id"
    )

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_resume_no_parked_attempt_just_kicks_main_loop(tmp_path):
    """Edge case: resume when the parked attempt no longer exists (e.g.
    user removed the head item while paused) must still flip queue state
    and wake the main loop — no crash on missing head."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("a")

    mgr = _PauseResumeAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_no_parked",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()
    await _wait(lambda: store.load().items[0].state == ItemState.RUNNING)

    await dispatcher.stop()

    # Caller cancels the (still-RUNNING) item before resume.
    head_id = store.load().items[0].id
    store.remove_item(head_id)

    result = await dispatcher.resume()
    assert result["status"] == "running"
    assert result["resumed_item_id"] is None

    await dispatcher.shutdown()
