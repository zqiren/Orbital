# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 4 integration: stop/resume roundtrip + chat session continuity.

The contract being exercised:

  Start a 3-item queue. While item 1 is running (slow attempt), call stop.
  Verify: queue STOPPED, item 1 still RUNNING (no new attempt closed yet),
  no rotation. Send a chat message — it must route to the chat session,
  not item 1's session. Call resume. Verify item 1 resumes on the SAME
  session. Mock LLM now returns mark_task_complete. Item 1 completes and
  item 2 starts.

We exercise the QueueDispatcher against a hand-crafted mock AgentManager
that simulates the full inject/loop/switch-session lifecycle.
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
    def __init__(self):
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        self._queue_state = "chat"
        self._session = None
        self.terminate_called = False
        self._terminate_event = asyncio.Event()

    async def terminate(self):
        self.terminate_called = True
        self._terminate_event.set()


class _StopResumeAgentManager:
    """Sophisticated mock that tracks session swaps and loop lifecycle."""

    def __init__(self):
        self._loop = _FakeLoop()
        self._sessions: dict[str, _FakeSession] = {}
        self._current_sid = "sess_attempt_1"
        self._sessions[self._current_sid] = _FakeSession(self._current_sid)
        self._loop._session = self._sessions[self._current_sid]
        self._task: asyncio.Task | None = None
        self._next_exit_reason = None
        self._next_summary = None
        self._loop_started_event = asyncio.Event()
        self.new_session_calls = 0
        self.switch_calls: list[tuple[str, bool]] = []
        # Sub-agent manager mock
        self._sub_agent_manager_obj = AsyncMock()
        self._sub_agent_manager_obj.stop_all = AsyncMock(return_value=None)

    def get_session(self, project_id):
        return self._sessions[self._current_sid]

    def get_loop(self, project_id):
        return self._loop

    def get_loop_task(self, project_id):
        return self._task

    def get_sub_agent_manager(self):
        return self._sub_agent_manager_obj

    def queue_next_exit(self, reason: str, *, summary=None, block_reason=None):
        self._next_exit_reason = reason
        self._next_summary = summary
        self._next_block_reason = block_reason

    async def inject_message(self, project_id, content, *, nonce=None):
        # Record the message into the current session
        self._sessions[self._current_sid].append(
            {"role": "user", "content": content},
        )
        # Reset the loop's exit_reason (mimicking AgentLoop.run reset)
        self._loop._exit_reason = "text"
        self._loop._exit_summary = None
        self._loop._exit_block_reason = None
        self._loop._terminate_event = asyncio.Event()
        # Spawn a loop task: either it hangs (default — for stop test) or
        # immediately sets a scripted exit reason and completes.
        next_reason = self._next_exit_reason
        next_summary = self._next_summary
        next_block_reason = getattr(self, "_next_block_reason", None)
        self._next_exit_reason = None
        self._next_summary = None
        self._next_block_reason = None

        async def _loop_body():
            self._loop_started_event.set()
            if next_reason is None:
                # Hang until terminate fires
                await self._loop._terminate_event.wait()
            else:
                # Immediate exit with the scripted reason
                self._loop._exit_reason = next_reason
                self._loop._exit_summary = next_summary
                self._loop._exit_block_reason = next_block_reason
                await asyncio.sleep(0.01)

        self._task = asyncio.create_task(_loop_body())
        return "delivered"

    async def switch_session(self, project_id, session_id, *, start_loop=False):
        self.switch_calls.append((session_id, start_loop))
        # Terminate current loop if it's running
        if self._task is not None and not self._task.done():
            await self._loop.terminate()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except Exception:
                pass
        # Reset terminate marker on the loop (next attempt starts fresh)
        self._loop.terminate_called = False
        # Create the session if it doesn't exist
        if session_id not in self._sessions:
            self._sessions[session_id] = _FakeSession(session_id)
        self._current_sid = session_id
        self._loop._session = self._sessions[session_id]
        if start_loop:
            # Start a loop run on this existing session — same shape as inject
            # but without injecting a new user message.
            next_reason = self._next_exit_reason
            next_summary = self._next_summary
            next_block_reason = getattr(self, "_next_block_reason", None)
            self._next_exit_reason = None
            self._next_summary = None
            self._next_block_reason = None

            async def _loop_body():
                if next_reason is None:
                    await self._loop._terminate_event.wait()
                else:
                    self._loop._exit_reason = next_reason
                    self._loop._exit_summary = next_summary
                    self._loop._exit_block_reason = next_block_reason
                    await asyncio.sleep(0.01)

            self._task = asyncio.create_task(_loop_body())
        return self._sessions[session_id]

    async def _start_loop(self, project_id):
        # Used by dispatcher._resume_attempt after switch_session(start_loop=False)
        # In the real AgentManager, _start_loop runs handle.loop.run(). Here we
        # just spawn a loop body that mirrors the inject path.
        next_reason = self._next_exit_reason
        next_summary = self._next_summary
        next_block_reason = getattr(self, "_next_block_reason", None)
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
        return {"status": "new_session"}


async def _wait(predicate, timeout=5.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


@pytest.mark.asyncio
async def test_full_stop_resume_roundtrip(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    item1 = store.add_item("first thing")
    item2 = store.add_item("second thing")
    item3 = store.add_item("third thing")

    mgr = _StopResumeAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_stop_resume",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    # Wait until item 1 is running and the loop is in-flight (hanging)
    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.RUNNING
        and mgr._loop_started_event.is_set()
    ), timeout=5.0)
    assert ok, "Item 1 should be running with loop in-flight"

    # Capture the parked session id for later comparison
    parked_sid = store.load().items[0].attempts[0].session_id

    # === STOP ===
    stop_result = await dispatcher.stop()
    assert stop_result["status"] == "stopped"
    chat_sid = stop_result["chat_session_id"]
    assert chat_sid.startswith("chat_")

    # Queue state is STOPPED, item 1 still RUNNING (attempt preserved)
    state = store.load()
    assert state.state == QueueRunState.STOPPED
    assert state.items[0].state == ItemState.RUNNING
    assert state.items[0].attempts[-1].outcome is None  # NOT closed
    assert state.chat_session_id == chat_sid

    # Sub-agents were told to stop
    mgr._sub_agent_manager_obj.stop_all.assert_awaited()

    # Items 2 and 3 still queued; no new attempts
    assert state.items[1].state == ItemState.QUEUED
    assert state.items[2].state == ItemState.QUEUED

    # The active session was swapped to the chat session
    assert mgr._current_sid == chat_sid

    # === Chat message routes to chat session ===
    await mgr.inject_message("proj_stop_resume", "user chat while stopped")
    chat_session = mgr._sessions[chat_sid]
    parked_session = mgr._sessions[parked_sid]
    assert any(
        m["content"] == "user chat while stopped" for m in chat_session.messages
    )
    assert not any(
        m["content"] == "user chat while stopped" for m in parked_session.messages
    )

    # === RESUME ===
    # Script the next loop run to complete the item
    mgr.queue_next_exit("complete", summary="finally done")
    resume_result = await dispatcher.resume()
    assert resume_result["status"] == "draining"
    assert resume_result["resumed_item_id"] == item1.id

    # Wait for item 1 to complete and item 2 to start
    ok = await _wait(lambda: (
        store.load().items[0].state == ItemState.DONE
        and store.load().items[1].state == ItemState.RUNNING
    ), timeout=10.0)
    assert ok, (
        "Item 1 should complete and item 2 start. "
        + ", ".join(f"{it.id}={it.state.value}" for it in store.load().items)
    )

    state = store.load()
    # Item 1 done with the summary
    assert state.items[0].state == ItemState.DONE
    assert state.items[0].attempts[-1].outcome == AttemptOutcome.COMPLETED
    assert state.items[0].attempts[-1].summary == "finally done"
    # Item 1's resumed attempt used the SAME session id as the parked one
    assert state.items[0].attempts[-1].session_id == parked_sid

    # Queue back to draining
    assert state.state == QueueRunState.DRAINING

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_chat_session_id_persists_across_stop_resume(tmp_path):
    """Multiple stop/resume cycles must reuse the same chat_session_id."""
    store = QueueStore(tmp_path / "queue.json")
    item = store.add_item("the only thing")

    mgr = _StopResumeAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_chat_persist",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: store.load().items[0].state == ItemState.RUNNING)
    assert ok

    result1 = await dispatcher.stop()
    chat_sid = result1["chat_session_id"]

    await dispatcher.resume()
    # Wait for the loop to be back in-flight after resume
    await asyncio.sleep(0.1)

    result2 = await dispatcher.stop()
    assert result2["chat_session_id"] == chat_sid, (
        "Second stop must reuse the same chat session id"
    )

    assert store.load().chat_session_id == chat_sid

    await dispatcher.shutdown()


@pytest.mark.asyncio
async def test_resume_resets_cold_resume_flag_via_switch_session(tmp_path):
    """switch_session is the mechanism that resets _cold_resume_injected;
    this test verifies the dispatcher calls switch_session on resume."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("a")

    mgr = _StopResumeAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_cold_resume",
        store=store,
        agent_manager=mgr,
        max_runtime_seconds=60,
    )

    await dispatcher.start()
    dispatcher.notify_new_item()
    await _wait(lambda: store.load().items[0].state == ItemState.RUNNING)

    await dispatcher.stop()
    # switch_session was called with the chat session id
    chat_swap = [c for c in mgr.switch_calls if c[0].startswith("chat_")]
    assert len(chat_swap) == 1

    parked_sid = store.load().items[0].attempts[0].session_id

    mgr.queue_next_exit("complete", summary="ok")
    await dispatcher.resume()

    # switch_session must also have been called with the parked session id
    parked_swap = [c for c in mgr.switch_calls if c[0] == parked_sid]
    assert len(parked_swap) == 1

    await dispatcher.shutdown()
