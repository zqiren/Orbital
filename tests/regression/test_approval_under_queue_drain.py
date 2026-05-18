# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: under queue-running state, an approval-required tool call
translates to a block exit rather than a pause.

This is the Phase 3 contract: a supervised-autonomy agent that hits an
approval boundary while a queued task is running must NOT halt the queue
indefinitely. Instead, it should record the block and let the dispatcher
move on.
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
        return ToolResult(content="ok")

    async def execute_async(self, name, args):
        from agent_os.agent.tools.base import ToolResult
        self.executions.append((name, args))
        return ToolResult(content="ok")


class _AlwaysInterceptInterceptor:
    """Interceptor that intercepts every tool call. Mirrors AutonomyInterceptor
    surface enough for AgentLoop."""

    def __init__(self):
        self.intercepted_calls: list[dict] = []

    def should_intercept(self, tc):
        return True

    def on_intercept(self, tc, recent_activity, reasoning=None):
        # In chat mode, this would broadcast an approval card via WS.
        # We record the call so we can assert it is NOT called under draining.
        self.intercepted_calls.append(tc)


class _OneShotProvider:
    def __init__(self, first_chunks):
        self._chunks = first_chunks
        self._count = 0

    @property
    def model(self):
        return "fake"

    async def stream(self, context, tools=None):
        self._count += 1
        if self._count == 1:
            for c in self._chunks:
                yield c
            return
        yield StreamChunk(text="follow-up")
        yield StreamChunk(is_final=True, usage=TokenUsage(1, 1))


def _tool_call_chunk(name: str, args: dict, *, call_id: str = "call_1"):
    return StreamChunk(
        tool_calls_delta=[{
            "index": 0,
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    )


@pytest.mark.asyncio
async def test_running_state_makes_approval_a_block(tmp_path):
    session = Session.new("sess_running_test", str(tmp_path))
    interceptor = _AlwaysInterceptInterceptor()
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _tool_call_chunk("write_file", {"path": "x.txt", "content": "y"}),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry,
        context_manager=_FakeContextManager(session),
        interceptor=interceptor,
    )
    loop._queue_state = "running"

    await loop.run("execute the write")

    assert loop._exit_reason == "blocked"
    assert "approval required" in (loop._exit_block_reason or "")
    assert "write_file" in (loop._exit_block_reason or "")
    # The interceptor's on_intercept should NOT have fired — no approval card
    assert interceptor.intercepted_calls == []
    # The tool itself should not have executed
    assert registry.executions == []
    # Session should NOT be paused (Phase 3 contract: dispatcher rotates,
    # we don't park)
    assert not session.is_paused()

    # Verify the structured marker was appended
    markers = [
        m for m in session.get_messages()
        if m.get("source") == "queue_signal" and m.get("signal") == "blocked"
    ]
    assert len(markers) == 1
    assert markers[0]["payload"]["tool"] == "write_file"


@pytest.mark.asyncio
async def test_chat_state_keeps_pause_behavior(tmp_path):
    """Negative control: with default queue_state ('chat'), the same setup
    must use the legacy pause path so chat-driven approval still works."""
    session = Session.new("sess_chat_test", str(tmp_path))
    interceptor = _AlwaysInterceptInterceptor()
    registry = _RecordingRegistry()
    provider = _OneShotProvider([
        _tool_call_chunk("write_file", {"path": "x.txt", "content": "y"}),
        StreamChunk(is_final=True, usage=TokenUsage(10, 5)),
    ])
    loop = AgentLoop(
        session=session,
        provider=provider,
        tool_registry=registry,
        context_manager=_FakeContextManager(session),
        interceptor=interceptor,
    )
    # Default _queue_state is "chat"; do NOT override it here
    await loop.run("execute the write")

    # Loop should NOT have set exit_reason=blocked; should still be "text" default
    assert loop._exit_reason == "text"
    # Approval card fired
    assert len(interceptor.intercepted_calls) == 1
    # Tool was not executed (held for approval)
    assert registry.executions == []
    # Session IS paused per chat-mode contract
    assert session.is_paused()
