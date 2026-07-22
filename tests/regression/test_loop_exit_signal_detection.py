# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: AgentLoop detects mark_task_complete / mark_task_blocked
and exits with the right exit_reason. Co-emitted sibling tool calls in the
same response are executed (not discarded) before the signal is honored.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session


class _FakeContextManager:
    def __init__(self, session):
        self._session = session
        self.model_context_limit = 200_000

    def prepare(self):
        return list(self._session.get_messages())

    def should_compact(self):
        return False


class _RecordingRegistry:
    """Records tool executions so we can verify short-circuit semantics."""

    def __init__(self):
        self.executions: list[tuple[str, dict]] = []

    def reset_run_state(self):
        pass

    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, args):
        from agent_os.agent.tools.base import ToolResult
        self.executions.append((name, args))
        return ToolResult(content=f"{name} executed")

    async def execute_async(self, name, args):
        from agent_os.agent.tools.base import ToolResult
        self.executions.append((name, args))
        return ToolResult(content=f"{name} executed")


class _OneShotProvider:
    """Yields a single response then a text-only "done" on subsequent calls."""

    def __init__(self, first_chunks: list[StreamChunk]):
        self._first_chunks = first_chunks
        self._call_count = 0

    @property
    def model(self):
        return "fake-model"

    async def stream(self, context, tools=None):
        self._call_count += 1
        if self._call_count == 1:
            for ch in self._first_chunks:
                yield ch
            return
        # Subsequent: produce a text-only response so the loop exits cleanly
        yield StreamChunk(text="ok")
        yield StreamChunk(is_final=True, usage=TokenUsage(1, 1))


def _make_session(tmp_path, sid="sess_signal_test"):
    return Session.new(sid, str(tmp_path))


def _make_loop(session, provider, registry=None):
    return AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry or _RecordingRegistry(),
        context_manager=_FakeContextManager(session),
    )


def _tool_call_chunk(name: str, args: dict, *, index: int = 0, call_id: str | None = None):
    return StreamChunk(
        tool_calls_delta=[{
            "index": index,
            "id": call_id or f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    )


@pytest.mark.asyncio
async def test_mark_task_complete_alone_sets_exit_reason(tmp_path):
    session = _make_session(tmp_path)
    provider = _OneShotProvider([
        _tool_call_chunk("mark_task_complete", {"summary": "all done"}),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider)
    await loop.run("please finish")
    assert loop._exit_reason == "complete"
    assert loop._exit_summary == "all done"
    # No write_file ever ran
    assert all(tc[0] != "write_file" for tc in loop._tool_registry.executions)


@pytest.mark.asyncio
async def test_mark_task_blocked_alone_sets_exit_reason(tmp_path):
    session = _make_session(tmp_path)
    provider = _OneShotProvider([
        _tool_call_chunk("mark_task_blocked", {"reason": "need API key"}),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider)
    await loop.run("attempt blocked task")
    assert loop._exit_reason == "blocked"
    assert loop._exit_block_reason == "need API key"


def _multi_tool_call_chunk(*calls: tuple[str, dict, str]):
    """Build one StreamChunk carrying several tool_calls_delta entries.

    Each entry in ``calls`` is (name, args, call_id); index is the position
    in ``calls``, matching emission order.
    """
    return StreamChunk(
        tool_calls_delta=[
            {
                "index": i,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for i, (name, args, call_id) in enumerate(calls)
        ],
    )


@pytest.mark.asyncio
async def test_signal_executes_coemitted_tools_then_exits(tmp_path):
    """If the model emits write_file AND mark_task_complete in the same
    response, write_file MUST still execute — through the normal tool path
    — before the signal is honored. This replaces the old discard contract:
    a batched response no longer silently drops sibling work."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _multi_tool_call_chunk(
            ("write_file", {"path": "x.txt", "content": "y"}, "call_write"),
            ("mark_task_complete", {"summary": "done"}, "call_complete"),
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    await loop.run("do work then complete")

    assert loop._exit_reason == "complete"
    assert loop._exit_summary == "done"
    # Both siblings ran, in emission order, through the real registry.
    assert registry.executions == [
        ("write_file", {"path": "x.txt", "content": "y"}),
        ("mark_task_complete", {"summary": "done"}),
    ]


@pytest.mark.asyncio
async def test_signal_batch_executes_siblings_no_cancelled(tmp_path):
    """Production repro: notify + checkpoint_state + mark_task_complete
    batched in one response. All three must execute in order, the run
    exits complete, and no CANCELLED row appears anywhere in the session —
    the original bug dropped the notify/checkpoint work silently."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _multi_tool_call_chunk(
            ("notify", {"message": "hi"}, "call_notify"),
            ("checkpoint_state", {"note": "chk"}, "call_checkpoint"),
            ("mark_task_complete", {"summary": "batch done"}, "call_complete"),
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    await loop.run("notify, checkpoint, then complete")

    assert loop._exit_reason == "complete"
    assert loop._exit_summary == "batch done"
    assert registry.executions == [
        ("notify", {"message": "hi"}),
        ("checkpoint_state", {"note": "chk"}),
        ("mark_task_complete", {"summary": "batch done"}),
    ]

    messages = list(session.get_messages())
    markers = [
        m for m in messages
        if m.get("source") == "queue_signal" and m.get("signal") == "complete"
    ]
    assert len(markers) == 1
    assert not any("CANCELLED" in str(m) for m in messages)


@pytest.mark.asyncio
async def test_signal_first_then_sibling_still_executes(tmp_path):
    """Signal emitted FIRST, sibling AFTER it: the sibling still executes
    (siblings positioned after the signal are not skipped), and the signal
    is only honored once it — and every other sibling — has run."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _multi_tool_call_chunk(
            ("mark_task_complete", {"summary": "done early"}, "call_complete"),
            ("write_file", {"path": "x.txt", "content": "y"}, "call_write"),
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    await loop.run("complete then do work")

    assert loop._exit_reason == "complete"
    assert loop._exit_summary == "done early"
    # write_file (the sibling) executes; the signal is deferred and
    # executed last regardless of its position in the emitted batch.
    assert registry.executions == [
        ("write_file", {"path": "x.txt", "content": "y"}),
        ("mark_task_complete", {"summary": "done early"}),
    ]


@pytest.mark.asyncio
async def test_signal_blocked_executes_coemitted_tools_then_exits(tmp_path):
    """Symmetric case for mark_task_blocked: write_file executes, then the
    run exits blocked with the registry having seen both calls."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _multi_tool_call_chunk(
            ("write_file", {"path": "x.txt", "content": "y"}, "call_write"),
            ("mark_task_blocked", {"reason": "need API key"}, "call_blocked"),
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    await loop.run("do work then block")

    assert loop._exit_reason == "blocked"
    assert loop._exit_block_reason == "need API key"
    assert registry.executions == [
        ("write_file", {"path": "x.txt", "content": "y"}),
        ("mark_task_blocked", {"reason": "need API key"}),
    ]


@pytest.mark.asyncio
async def test_extra_signal_call_left_pending_for_cancel_machinery(tmp_path):
    """Two signal calls in one response: the FIRST wins (emission order);
    the second is never executed by the registry — it's left pending and
    picked up by the existing resolve_pending_tool_calls CANCEL path."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _multi_tool_call_chunk(
            ("mark_task_complete", {"summary": "first wins"}, "call_first"),
            ("mark_task_blocked", {"reason": "should not run"}, "call_second"),
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    await loop.run("two signals in one batch")

    assert loop._exit_reason == "complete"
    assert loop._exit_summary == "first wins"
    # The second signal call was never dispatched to the registry.
    assert registry.executions == [("mark_task_complete", {"summary": "first wins"})]

    messages = list(session.get_messages())
    second_row = next(
        (m for m in messages if m.get("tool_call_id") == "call_second"), None,
    )
    assert second_row is not None
    assert "CANCELLED" in second_row["content"]


class _FakeInterceptor:
    """Minimal interceptor: intercepts a fixed set of tool names, records
    on_intercept calls, and otherwise does nothing (mirrors the shape of
    MockInterceptor in tests/unit/test_component_a.py)."""

    def __init__(self, intercept_names):
        self._intercept_names = set(intercept_names)
        self.intercepted_calls: list[dict] = []

    def should_intercept(self, tool_call: dict) -> bool:
        return tool_call.get("name", "") in self._intercept_names

    def on_intercept(self, tool_call: dict, recent_context: list[dict], reasoning=None) -> None:
        self.intercepted_calls.append(tool_call)


@pytest.mark.asyncio
async def test_signal_deferred_when_sibling_requires_approval(tmp_path):
    """If a sibling in the batch requires approval, the run must pause
    there — the signal is NOT honored this iteration (the work isn't done).
    The signal's tool call is left pending and gets CANCELLED by the
    existing resolve_pending_tool_calls machinery."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _multi_tool_call_chunk(
            ("write_file", {"path": "x.txt", "content": "y"}, "call_write"),
            ("mark_task_complete", {"summary": "done"}, "call_complete"),
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    loop._interceptor = _FakeInterceptor(intercept_names={"write_file"})
    await loop.run("do work then complete")

    # The run paused for approval — it did not complete.
    assert loop._exit_reason != "complete"
    messages = list(session.get_messages())
    assert not any(
        m.get("source") == "queue_signal" and m.get("signal") == "complete"
        for m in messages
    )
    # write_file was never executed (it's the intercepted call, awaiting approval).
    assert registry.executions == []
    # The signal call was left pending and got CANCELLED.
    signal_row = next(
        (m for m in messages if m.get("tool_call_id") == "call_complete"), None,
    )
    assert signal_row is not None
    assert "CANCELLED" in signal_row["content"]


@pytest.mark.asyncio
async def test_text_only_response_yields_default_exit_reason(tmp_path):
    session = _make_session(tmp_path)
    provider = _OneShotProvider([
        StreamChunk(text="I am just chatting"),
        StreamChunk(is_final=True, usage=TokenUsage(5, 3)),
    ])
    loop = _make_loop(session, provider)
    await loop.run("hello")
    assert loop._exit_reason == "text"
    assert loop._exit_summary is None
    assert loop._exit_block_reason is None


@pytest.mark.asyncio
async def test_signal_writes_marker_to_session(tmp_path):
    session = _make_session(tmp_path)
    provider = _OneShotProvider([
        _tool_call_chunk("mark_task_complete", {"summary": "rewrote auth"}),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider)
    await loop.run("the work")

    messages = list(session.get_messages())
    # Find the queue_signal marker
    markers = [
        m for m in messages
        if m.get("source") == "queue_signal" and m.get("signal") == "complete"
    ]
    assert len(markers) == 1
    assert markers[0]["payload"]["summary"] == "rewrote auth"


@pytest.mark.asyncio
async def test_exit_reason_resets_between_runs(tmp_path):
    """Run 1 fires complete; run 2 is text-only. exit_reason must be 'text'
    after run 2, not lingering as 'complete'. Prevents stale-state advance."""
    session = _make_session(tmp_path)
    provider = _OneShotProvider([
        _tool_call_chunk("mark_task_complete", {"summary": "first"}),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider)
    await loop.run("turn 1")
    assert loop._exit_reason == "complete"

    # Second run with a fresh provider that emits text-only
    loop._provider = _OneShotProvider([
        StreamChunk(text="I am thinking"),
        StreamChunk(is_final=True, usage=TokenUsage(5, 3)),
    ])
    await loop.run("turn 2")
    assert loop._exit_reason == "text"
    assert loop._exit_summary is None
