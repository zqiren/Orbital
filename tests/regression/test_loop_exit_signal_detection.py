# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: AgentLoop detects mark_task_complete / mark_task_blocked
and exits with the right exit_reason. Short-circuits any co-emitted tools.
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


@pytest.mark.asyncio
async def test_signal_short_circuits_coemitted_tools(tmp_path):
    """If the model emits write_file AND mark_task_complete in the same
    response, write_file MUST NOT execute. This is the contract that lets
    us promise the agent: "your other tools are discarded."""
    session = _make_session(tmp_path)
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        StreamChunk(
            tool_calls_delta=[
                {
                    "index": 0,
                    "id": "call_write",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": "x.txt", "content": "y"}),
                    },
                },
                {
                    "index": 1,
                    "id": "call_complete",
                    "type": "function",
                    "function": {
                        "name": "mark_task_complete",
                        "arguments": json.dumps({"summary": "done"}),
                    },
                },
            ],
        ),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = _make_loop(session, provider, registry=registry)
    await loop.run("do work then complete")

    assert loop._exit_reason == "complete"
    assert loop._exit_summary == "done"
    # Crucial: write_file was discarded — the registry never saw it
    assert registry.executions == []


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
