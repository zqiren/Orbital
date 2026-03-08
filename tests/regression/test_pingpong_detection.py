# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: ping-pong loop detection for alternating tool call patterns.

When the agent alternates between two (or more) tool calls in a cycle
(A -> B -> A -> B -> ...) without making progress, the loop detects
consecutive pair repetition and injects a diagnostic system message.

Detection: track (hash_N-1, hash_N) pairs. If the same pair appears 3+
times within a sliding window of 20 pairs, trigger ping-pong detection.

Result-awareness: hash includes first 500 chars of tool result, so
changing results (progress) produce different hashes and don't trigger.
"""

import json

import pytest

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    TokenUsage,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read_file", "write_file", "search"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str]:
        return ("cached-system-prefix", "dynamic-suffix")


def _make_text_response(text: str) -> LLMResponse:
    return LLMResponse(
        raw_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=[],
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


def _make_tool_response(tool_calls: list[dict]) -> LLMResponse:
    raw = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    return LLMResponse(
        raw_message=raw,
        text=None,
        tool_calls=tool_calls,
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


class MockProvider:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self._call_idx = 0
        self.model = "test-model"

    async def stream(self, messages, tools=None):
        idx = self._call_idx
        self._call_idx += 1
        if idx < len(self._responses):
            resp = self._responses[idx]
            if resp.text:
                yield StreamChunk(text=resp.text)
            if resp.tool_calls:
                for i, tc in enumerate(resp.tool_calls):
                    tc_with_index = dict(tc)
                    tc_with_index["index"] = i
                    yield StreamChunk(tool_calls_delta=[tc_with_index])
            yield StreamChunk(is_final=True, usage=resp.usage)
        else:
            yield StreamChunk(text="default", is_final=True,
                              usage=TokenUsage(10, 5))


class SequentialToolRegistry:
    """Tool registry that returns a sequence of ToolResult objects per tool."""
    def __init__(self, result_sequences: dict[str, list[ToolResult]]):
        self._sequences = {k: list(v) for k, v in result_sequences.items()}
        self._call_idx: dict[str, int] = {k: 0 for k in result_sequences}

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n}} for n in self._sequences]

    def is_async(self, name: str) -> bool:
        return True

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        seq = self._sequences.get(name, [])
        idx = self._call_idx.get(name, 0)
        self._call_idx[name] = idx + 1
        if idx < len(seq):
            return seq[idx]
        return ToolResult(content=f"result of {name}")

    def execute(self, name: str, arguments: dict) -> ToolResult:
        raise RuntimeError("Should not be called when is_async returns True")

    def tool_names(self) -> list[str]:
        return list(self._sequences.keys())

    def reset_run_state(self) -> None:
        pass


def _tc(name: str, arguments: dict | None = None) -> list[dict]:
    """Build a single-element tool_calls list."""
    args = arguments or {}
    return [{
        "id": f"call_{id(object())}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPingPongDetection:

    @pytest.mark.asyncio
    async def test_two_tool_alternation_detected(self, tmp_path):
        """A -> B -> A -> B -> A -> B triggers ping-pong detection after 3 pair repeats."""
        session = Session.new("pp_2tool", str(tmp_path))

        # 8 alternating tool calls (4 pairs of A,B), then text
        responses = []
        for i in range(8):
            if i % 2 == 0:
                responses.append(_make_tool_response(_tc("read_file", {"path": "/a.txt"})))
            else:
                responses.append(_make_tool_response(_tc("write_file", {"path": "/b.txt"})))
        responses.append(_make_text_response("Done."))

        provider = MockProvider(responses)
        # Both tools return identical results each time (no progress)
        registry = SequentialToolRegistry(result_sequences={
            "read_file": [ToolResult(content="file content")] * 8,
            "write_file": [ToolResult(content="written ok")] * 8,
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="alternate between read and write")

        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "ping-pong" in m.get("content", "").lower()
        ]
        assert len(system_msgs) >= 1, (
            "Expected ping-pong detection message. System messages: "
            f"{[m.get('content', '')[:80] for m in msgs if m['role'] == 'system']}"
        )

    @pytest.mark.asyncio
    async def test_not_triggered_when_results_differ(self, tmp_path):
        """Same alternating tools but results change each time -> NOT triggered."""
        session = Session.new("pp_diff", str(tmp_path))

        responses = []
        for i in range(8):
            if i % 2 == 0:
                responses.append(_make_tool_response(_tc("read_file", {"path": "/a.txt"})))
            else:
                responses.append(_make_tool_response(_tc("write_file", {"path": "/b.txt"})))
        responses.append(_make_text_response("Done."))

        provider = MockProvider(responses)
        # Each call returns different content (progress being made)
        registry = SequentialToolRegistry(result_sequences={
            "read_file": [ToolResult(content=f"version {i}") for i in range(8)],
            "write_file": [ToolResult(content=f"wrote v{i}") for i in range(8)],
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="work on the file")

        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "ping-pong" in m.get("content", "").lower()
        ]
        assert len(system_msgs) == 0, (
            "Ping-pong should NOT trigger when results differ (progress being made)"
        )

    @pytest.mark.asyncio
    async def test_simple_repetition_still_caught(self, tmp_path):
        """Same tool repeated 5x is still caught — either by single-repetition
        or by ping-pong (since pairs (A,A) also repeat). Either way the loop stops."""
        session = Session.new("pp_single", str(tmp_path))

        responses = [
            _make_tool_response(_tc("read_file", {"path": "/a.txt"}))
            for _ in range(6)
        ]
        responses.append(_make_text_response("Done."))

        provider = MockProvider(responses)
        same_result = ToolResult(content="unchanged content")
        registry = SequentialToolRegistry(result_sequences={
            "read_file": [same_result] * 6,
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="read the file")

        msgs = session.get_messages()
        # Either ping-pong or repetition detection should fire
        detection_msgs = [
            m for m in msgs
            if m["role"] == "system" and (
                "repetitive action" in m.get("content", "").lower()
                or "ping-pong" in m.get("content", "").lower()
            )
        ]
        assert len(detection_msgs) >= 1, (
            "Repeated same-tool calls must be caught by some detector"
        )

    @pytest.mark.asyncio
    async def test_three_tool_cycle_detected(self, tmp_path):
        """A -> B -> C -> A -> B -> C -> A -> B -> C triggers detection."""
        session = Session.new("pp_3tool", str(tmp_path))

        tools = ["read_file", "write_file", "search"]
        responses = []
        for cycle in range(3):
            for tool in tools:
                responses.append(_make_tool_response(_tc(tool, {"q": "test"})))
        responses.append(_make_text_response("Done."))

        provider = MockProvider(responses)
        # All tools return identical results each call
        registry = SequentialToolRegistry(result_sequences={
            "read_file": [ToolResult(content="data")] * 9,
            "write_file": [ToolResult(content="ok")] * 9,
            "search": [ToolResult(content="results")] * 9,
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="cycle through tools")

        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "ping-pong" in m.get("content", "").lower()
        ]
        assert len(system_msgs) >= 1, (
            "Three-tool cycle should trigger ping-pong detection. "
            f"System messages: {[m.get('content', '')[:80] for m in msgs if m['role'] == 'system']}"
        )
