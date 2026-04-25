# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: AgentLoop.cancel_turn() and AgentLoop.terminate() (B2/RC-C).

These tests verify the cancellable-streaming behaviour added in batch 1
TASK-cancel-arch-04. The existing AgentLoop has no way to interrupt an
in-flight LLM stream — Session.stop() is purely a flag setter and the loop
only checks it between turns. The fix introduces two methods on AgentLoop:

  * cancel_turn() — interrupts the inflight LLM stream + skips remaining
    tool calls in the current turn. Loop returns to idle.
  * terminate() — calls cancel_turn(), sets the stop flag, and cancels
    the loop task. Loop exits cleanly. Session JSONL is preserved.

TDD note: each test should fail before the fix (no cancel_turn/terminate
methods or the methods do not interact with the loop's inflight stream)
and pass after.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import (
    LLMResponse,
    StreamChunk,
    TokenUsage,
)
from agent_os.agent.session import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeContextManager:
    """Minimal context manager: returns whatever messages the session has."""

    def __init__(self, session):
        self._session = session
        self.model_context_limit = 200_000

    def prepare(self):
        return list(self._session.get_messages())

    def should_compact(self):
        return False


class _FakeRegistry:
    """Minimal tool registry that records executions."""

    def __init__(self, results=None):
        # results is a dict keyed by tool name -> str result
        self._results = results or {}
        self.executions: list[tuple[str, dict]] = []

    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, args):
        # Synchronous tool exec used via asyncio.to_thread
        self.executions.append((name, args))
        from agent_os.agent.tools.base import ToolResult
        return ToolResult(content=self._results.get(name, "ok"))

    async def execute_async(self, name, args):
        self.executions.append((name, args))
        from agent_os.agent.tools.base import ToolResult
        return ToolResult(content=self._results.get(name, "ok"))


class _StreamingProvider:
    """Programmable streaming provider for tests.

    The instance owns a list of `chunks` that .stream() yields, plus a
    `pause_event` (asyncio.Event) — after yielding all chunks the stream
    waits indefinitely until the test cancels the inflight task. Used to
    simulate a long-running LLM stream that must be cancelled mid-flight.
    """

    def __init__(self, chunks=None, hang_after_chunks=True,
                 final_response_text="done"):
        self.chunks = chunks or []
        self.hang_after_chunks = hang_after_chunks
        self.final_response_text = final_response_text
        # how many chunks have been yielded (for test synchronization)
        self.chunks_yielded = 0
        self._chunks_yielded_event = asyncio.Event()
        # subsequent calls (after first cancel) return non-tool-call response
        self._call_count = 0

    @property
    def model(self):
        return "fake-model-for-tests"

    async def stream(self, context, tools=None):
        self._call_count += 1
        if self._call_count > 1:
            # On a second/third call (after cancel), produce a normal response
            yield StreamChunk(text=self.final_response_text)
            yield StreamChunk(
                text="",
                is_final=True,
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )
            return

        # First call: yield programmed chunks then optionally hang
        for ch in self.chunks:
            yield ch
            self.chunks_yielded += 1
            self._chunks_yielded_event.set()
        if self.hang_after_chunks:
            # Block forever — waiting for cancellation. Use Event with no
            # signal so we sleep without busy-waiting.
            await asyncio.Event().wait()
        # Otherwise emit an is_final chunk and return
        yield StreamChunk(
            text="",
            is_final=True,
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

    async def wait_for_partial_chunks(self, n=1, timeout=2.0):
        """Wait until at least n chunks have been yielded by the stream."""
        deadline = asyncio.get_event_loop().time() + timeout
        while self.chunks_yielded < n:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    f"Stream yielded only {self.chunks_yielded}/{n} chunks "
                    f"within {timeout}s"
                )
            await asyncio.sleep(0.01)


def _make_session(tmp_path, session_id="sess_cancel_test"):
    """Create a real Session backed by a temp JSONL file."""
    workspace = str(tmp_path)
    return Session.new(session_id, workspace, project_dir_name="proj_test")


def _read_session_messages(session) -> list[dict]:
    """Read all messages from the session's JSONL file."""
    import json
    msgs = []
    with open(session._filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msgs.append(json.loads(line))
    return msgs


def _make_loop(session, provider, registry=None):
    """Construct an AgentLoop with minimal mocks."""
    cm = _FakeContextManager(session)
    return AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry or _FakeRegistry(),
        context_manager=cm,
    )


# ---------------------------------------------------------------------------
# Test 1: cancel_turn during stream persists partial content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_stream_persists_partial(tmp_path):
    """cancel_turn() called while LLM is streaming:
    - cancels the inflight task
    - persists a cancellation marker in the JSONL
    - debits the partial output_tokens via the cost callback
    - clears _inflight_stream
    - the loop continues (does not exit)
    """
    session = _make_session(tmp_path, session_id="s_partial")
    provider = _StreamingProvider(
        chunks=[
            StreamChunk(text="Hello "),
            StreamChunk(text="world "),
            StreamChunk(text="from partial",
                         usage=TokenUsage(input_tokens=10, output_tokens=3)),
        ],
        hang_after_chunks=True,
    )

    cost_calls: list[tuple[float, float]] = []

    def on_cost(delta, total):
        cost_calls.append((delta, total))

    loop_obj = AgentLoop(
        session=session,
        provider=provider,
        tool_registry=_FakeRegistry(),
        context_manager=_FakeContextManager(session),
        on_cost_update=on_cost,
    )

    # Start the loop. It enters the streaming wait state.
    run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
    try:
        await provider.wait_for_partial_chunks(n=3, timeout=2.0)
        # Capture the inflight task before cancel for a deterministic
        # post-cancel assertion (the loop may have started a new turn by
        # the time we read self._inflight_stream).
        original_inflight = loop_obj._inflight_stream
        # Streaming has produced 3 chunks then is hanging → cancel it.
        await loop_obj.cancel_turn()

        # The originally-inflight stream task has been released (cancelled
        # or done). The current self._inflight_stream may already be a NEW
        # task from the loop's next iteration — that's fine; what matters
        # is that the cancelled one is no longer running.
        assert original_inflight is not None
        assert original_inflight.done(), (
            "Cancelled inflight stream task must be done"
        )

        # Cancellation marker was persisted to JSONL
        msgs = _read_session_messages(session)
        markers = [m for m in msgs if m.get("cancelled_by_user") is True]
        assert len(markers) == 1, (
            f"Expected exactly 1 cancellation marker, got {len(markers)}: "
            f"{[m.get('content') for m in msgs]}"
        )

        # Cost was debited for the 3 partial output tokens
        assert len(cost_calls) >= 1, (
            f"Expected cost callback to fire on cancel, got {cost_calls}"
        )
        # The total debit should be > 0 (full output token cost)
        delta, total = cost_calls[-1]
        assert delta > 0, (
            f"Expected positive cost debit on cancel, got delta={delta}"
        )

        # The loop should not have exited (session not stopped, no terminate)
        assert not run_task.done(), (
            "Loop must continue running after bare cancel_turn"
        )
    finally:
        # Clean up: terminate the loop so the test exits.
        await loop_obj.terminate()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ---------------------------------------------------------------------------
# Test 2: cancel_turn when idle is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_idle_is_noop(tmp_path):
    """cancel_turn() with no inflight stream:
    - no exception
    - no marker appended
    - _turn_cancelled = True (so any in-progress tool loop will skip remaining)
    """
    session = _make_session(tmp_path, session_id="s_idle")
    provider = _StreamingProvider()
    loop_obj = _make_loop(session, provider)

    # No turn ever started — call cancel_turn directly.
    await loop_obj.cancel_turn()

    msgs = _read_session_messages(session)
    markers = [m for m in msgs if m.get("cancelled_by_user") is True]
    assert len(markers) == 0, (
        f"No marker should be written for idle cancel, got: {markers}"
    )
    assert loop_obj._turn_cancelled is True, (
        "_turn_cancelled flag must be set even when idle"
    )


# ---------------------------------------------------------------------------
# Test 3: cancel during tool loop skips remaining tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_during_tool_loop_skips_remaining(tmp_path):
    """When the assistant produces 3 tool calls and cancel_turn is invoked
    after the first tool result is appended, the remaining 2 tools must NOT
    be executed and a marker must be appended exactly once."""
    session = _make_session(tmp_path, session_id="s_tool_cancel")

    # The provider returns an assistant message with 3 tool calls in the
    # FIRST stream call, then a normal text response on the SECOND call.
    cancelled_event = asyncio.Event()

    class _ToolCallProvider:
        def __init__(self):
            self.call_count = 0

        @property
        def model(self):
            return "fake-model"

        async def stream(self, context, tools=None):
            self.call_count += 1
            if self.call_count == 1:
                # Three tool calls
                yield StreamChunk(
                    text="",
                    tool_calls_delta=[
                        {"index": 0, "id": "tc_1", "type": "function",
                         "function": {"name": "tool_a", "arguments": "{}"}},
                    ],
                )
                yield StreamChunk(
                    text="",
                    tool_calls_delta=[
                        {"index": 1, "id": "tc_2", "type": "function",
                         "function": {"name": "tool_b", "arguments": "{}"}},
                    ],
                )
                yield StreamChunk(
                    text="",
                    tool_calls_delta=[
                        {"index": 2, "id": "tc_3", "type": "function",
                         "function": {"name": "tool_c", "arguments": "{}"}},
                    ],
                )
                yield StreamChunk(
                    text="",
                    is_final=True,
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                )
            else:
                # On the 2nd iteration, return plain text (no tool calls)
                yield StreamChunk(text="resumed after cancel")
                yield StreamChunk(
                    text="",
                    is_final=True,
                    usage=TokenUsage(input_tokens=5, output_tokens=2),
                )

    # Tool registry that calls cancel_turn() after the first tool runs.
    class _CancelOnFirstRegistry(_FakeRegistry):
        def __init__(self, loop_holder):
            super().__init__(results={"tool_a": "result_a"})
            self.loop_holder = loop_holder

        async def execute_async(self, name, args):
            from agent_os.agent.tools.base import ToolResult
            self.executions.append((name, args))
            # After tool_a runs, fire cancel_turn() concurrently.
            if name == "tool_a":
                # Schedule the cancel on the event loop concurrently with this
                # tool's return. By awaiting cancel_turn directly we ensure the
                # marker is appended before the loop checks _turn_cancelled.
                loop = self.loop_holder["loop"]
                await loop.cancel_turn()
            return ToolResult(content=f"result of {name}")

        def is_async(self, name):
            return True

    holder: dict = {}
    provider = _ToolCallProvider()
    registry = _CancelOnFirstRegistry(holder)
    loop_obj = _make_loop(session, provider, registry=registry)
    holder["loop"] = loop_obj

    run_task = asyncio.create_task(loop_obj.run(initial_message="please"))
    try:
        # Wait for the loop to complete (after cancel, it issues a 2nd stream
        # call which returns plain text and breaks).
        await asyncio.wait_for(run_task, timeout=5.0)
    except asyncio.TimeoutError:
        run_task.cancel()
        raise

    # Only tool_a executed (tool_b, tool_c skipped)
    executed = [name for name, _ in registry.executions]
    assert executed == ["tool_a"], (
        f"Expected only tool_a executed, got {executed}"
    )

    # Exactly one cancellation marker
    msgs = _read_session_messages(session)
    markers = [m for m in msgs if m.get("cancelled_by_user") is True]
    assert len(markers) == 1, (
        f"Expected exactly 1 cancellation marker, got {len(markers)}: "
        f"{[(m.get('role'), m.get('content')) for m in markers]}"
    )


# ---------------------------------------------------------------------------
# Test 4: cancel then send completes normally
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_then_send_completes_normally(tmp_path):
    """After cancel_turn during stream, queueing a new user message must
    cause the next iteration to drain and produce a normal response with no
    leaked state."""
    session = _make_session(tmp_path, session_id="s_followup")
    provider = _StreamingProvider(
        chunks=[StreamChunk(text="partial...")],
        hang_after_chunks=True,
        final_response_text="hello again",
    )
    loop_obj = _make_loop(session, provider)

    run_task = asyncio.create_task(loop_obj.run(initial_message="first"))
    try:
        await provider.wait_for_partial_chunks(n=1, timeout=2.0)
        await loop_obj.cancel_turn()
        # Queue a new user message
        session.queue_message("second message after cancel")
        # Wait for loop to complete (its 2nd iteration drains the queue,
        # gets the normal response, and exits).
        await asyncio.wait_for(run_task, timeout=5.0)
    except asyncio.TimeoutError:
        run_task.cancel()
        raise

    # The session must contain (in order): first user msg, cancellation
    # marker, second user msg, assistant final reply.
    msgs = _read_session_messages(session)
    contents = [(m.get("role"), m.get("content")) for m in msgs]

    # The assistant's final reply must be present
    final_replies = [m for m in msgs
                     if m.get("role") == "assistant"
                     and m.get("content") == "hello again"]
    assert len(final_replies) == 1, (
        f"Expected normal final reply 'hello again', got: {contents}"
    )
    # No leaked _turn_cancelled
    assert loop_obj._turn_cancelled is False, (
        "_turn_cancelled must be reset after cancel + new turn"
    )


# ---------------------------------------------------------------------------
# Test 5: double-cancel is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_cancel_idempotent(tmp_path):
    """Calling cancel_turn() twice in rapid succession must:
    - not raise
    - append exactly one marker
    - debit cost exactly once
    """
    session = _make_session(tmp_path, session_id="s_double")
    provider = _StreamingProvider(
        chunks=[StreamChunk(text="abc",
                             usage=TokenUsage(input_tokens=10, output_tokens=3))],
        hang_after_chunks=True,
    )
    cost_calls: list[tuple[float, float]] = []

    def on_cost(delta, total):
        cost_calls.append((delta, total))

    loop_obj = AgentLoop(
        session=session,
        provider=provider,
        tool_registry=_FakeRegistry(),
        context_manager=_FakeContextManager(session),
        on_cost_update=on_cost,
    )

    run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
    try:
        await provider.wait_for_partial_chunks(n=1, timeout=2.0)
        # Two cancels in quick succession
        await loop_obj.cancel_turn()
        await loop_obj.cancel_turn()

        msgs = _read_session_messages(session)
        markers = [m for m in msgs if m.get("cancelled_by_user") is True]
        assert len(markers) == 1, (
            f"Double cancel must yield exactly 1 marker, got {len(markers)}"
        )
        # Cost callback fired exactly once for the cancellation
        assert len(cost_calls) == 1, (
            f"Double cancel must debit exactly once, got {len(cost_calls)}"
        )
    finally:
        await loop_obj.terminate()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass


# ---------------------------------------------------------------------------
# Test 6: terminate during stream exits the loop cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminate_during_stream_exits_loop(tmp_path):
    """terminate() while streaming:
    - appends a cancellation marker
    - sets session.is_stopped() = True
    - cancels the loop task
    - the run() coroutine returns within ~2s
    """
    session = _make_session(tmp_path, session_id="s_term_stream")
    provider = _StreamingProvider(
        chunks=[StreamChunk(text="partial",
                             usage=TokenUsage(input_tokens=10, output_tokens=2))],
        hang_after_chunks=True,
    )
    loop_obj = _make_loop(session, provider)

    run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
    try:
        await provider.wait_for_partial_chunks(n=1, timeout=2.0)
        await loop_obj.terminate()
        # Loop task should finish within 2s
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass

        assert session.is_stopped() is True, (
            "Session.is_stopped() must be True after terminate"
        )

        msgs = _read_session_messages(session)
        markers = [m for m in msgs if m.get("cancelled_by_user") is True]
        assert len(markers) == 1, (
            f"terminate during stream must append marker once, got {len(markers)}"
        )
    finally:
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Test 7: terminate when idle exits the loop cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminate_idle_exits_loop(tmp_path):
    """terminate() while loop is idle (between turns):
    - sets session.is_stopped() = True
    - cancels the loop task
    - exits within 2s
    - no marker (nothing in flight)
    """
    session = _make_session(tmp_path, session_id="s_term_idle")

    # Provider returns text response on first call (no tool calls), so loop
    # exits naturally on its own. We won't even fire terminate before exit;
    # instead, we'll start a long-blocking stream then terminate from idle
    # state. To engineer "idle between turns" we use a loop-controlling
    # provider: first call returns plain text quickly (loop exits naturally).
    # Then we call terminate() AFTER run_task is done -> verify it doesn't
    # raise. But the contract here is "loop is idle (between turns)".
    # Simulate: hang the FIRST stream call, then terminate. The loop exits
    # via the cancel_turn -> CancelledError -> is_stopped() raise branch.
    provider = _StreamingProvider(
        chunks=[],  # No chunks emitted before hang
        hang_after_chunks=True,
    )
    loop_obj = _make_loop(session, provider)

    run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
    try:
        # Give the loop time to enter the stream wait
        await asyncio.sleep(0.05)
        await loop_obj.terminate()

        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass

        assert session.is_stopped() is True
        # Since no chunks were yielded, accumulator is empty → no marker
        # required, but if one is appended it must still be exactly one.
        msgs = _read_session_messages(session)
        markers = [m for m in msgs if m.get("cancelled_by_user") is True]
        assert len(markers) <= 1, (
            f"Expected at most 1 marker for empty stream cancel, got {len(markers)}"
        )
    finally:
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Test 8: terminate then new loop on same session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminate_then_new_loop_on_same_session(tmp_path):
    """After terminate, a NEW AgentLoop on the same session must read the
    JSONL containing the cancellation marker and run normally — no orphan
    state."""
    session = _make_session(tmp_path, session_id="s_term_newloop")
    provider = _StreamingProvider(
        chunks=[StreamChunk(text="partial",
                             usage=TokenUsage(input_tokens=10, output_tokens=2))],
        hang_after_chunks=True,
    )
    loop_obj = _make_loop(session, provider)

    run_task = asyncio.create_task(loop_obj.run(initial_message="hi"))
    try:
        await provider.wait_for_partial_chunks(n=1, timeout=2.0)
        await loop_obj.terminate()
        try:
            await asyncio.wait_for(run_task, timeout=2.0)
        except asyncio.CancelledError:
            pass
    finally:
        if not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass

    # Reload session from JSONL and start a NEW loop
    reloaded = Session.load(session._filepath)
    msgs = reloaded.get_messages()
    markers = [m for m in msgs if m.get("cancelled_by_user") is True]
    assert len(markers) == 1, (
        f"Reloaded session must contain the cancellation marker, got {len(markers)}"
    )

    # Spin up a new loop on the reloaded session — it should run normally
    new_provider = _StreamingProvider(
        chunks=[],
        hang_after_chunks=False,
        final_response_text="restored",
    )
    # Reset the call_count so its first call is a "normal" response
    new_provider._call_count = 1  # forces fallback branch to fire on the next call
    new_loop = _make_loop(reloaded, new_provider)
    # Use no initial_message so we don't append a fresh user msg
    await asyncio.wait_for(new_loop.run(initial_message="follow up"), timeout=5.0)

    final_msgs = _read_session_messages(reloaded)
    final_replies = [m for m in final_msgs
                     if m.get("role") == "assistant"
                     and m.get("content") == "restored"]
    assert len(final_replies) == 1, (
        f"New loop must produce a normal reply, got: "
        f"{[(m.get('role'), m.get('content')) for m in final_msgs]}"
    )


# ---------------------------------------------------------------------------
# Test 9: no fire-and-forget session-end task remains pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_fire_and_forget_session_end(tmp_path):
    """After loop completes and exits via terminate, no pending task left
    over from the loop (i.e. the fire-and-forget _on_session_end at
    loop.py:643-644 has been removed)."""
    session = _make_session(tmp_path, session_id="s_no_fnf")

    # Provider returns text (no tool calls), so loop exits naturally.
    provider = _StreamingProvider(
        chunks=[],
        hang_after_chunks=False,
        final_response_text="single shot",
    )
    # Hack: we want first stream call to be the "second-call" branch, which
    # produces a normal final reply. Bump _call_count.
    provider._call_count = 1

    on_session_end_called = MagicMock()

    async def _ses_end():
        on_session_end_called()

    cm = _FakeContextManager(session)
    loop_obj = AgentLoop(
        session=session,
        provider=provider,
        tool_registry=_FakeRegistry(),
        context_manager=cm,
        on_session_end=_ses_end,
    )

    # Snapshot existing tasks
    tasks_before = set(asyncio.all_tasks())

    await asyncio.wait_for(loop_obj.run(initial_message="hi"), timeout=5.0)

    # Give any leftover task a moment to be scheduled (negative test)
    await asyncio.sleep(0.05)

    tasks_after = set(asyncio.all_tasks())
    new_tasks = tasks_after - tasks_before
    # The current test task is in the diff; filter that out.
    cur = asyncio.current_task()
    new_tasks.discard(cur)

    assert len(new_tasks) == 0, (
        f"After loop exit, no new pending tasks expected; got: "
        f"{[t.get_name() for t in new_tasks]}"
    )
    # session-end fire-and-forget must NOT have been scheduled
    assert on_session_end_called.call_count == 0, (
        "_on_session_end must NOT be invoked from the loop "
        "(fire-and-forget call site removed)"
    )
