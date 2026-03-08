# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: observation-aware repetition circuit breaker.

Browser tool calls are excluded from hash-based repetition detection entirely.
Repetition for browser is handled by the advisory counter in the browser tool.
For non-browser tools, the hash includes a prefix of the result content.

Threshold: >= 5 identical hashes within a sliding window (non-browser only).
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
        tool_names=["browser"],
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
    """Tool registry that returns a sequence of ToolResult objects per tool name.

    Each call to execute/execute_async pops the next result from the queue
    for that tool. If the queue is exhausted, returns a default result.
    """
    def __init__(self, result_sequences: dict[str, list[ToolResult]]):
        self._sequences = {k: list(v) for k, v in result_sequences.items()}
        self._call_idx: dict[str, int] = {k: 0 for k in result_sequences}
        self.execute_calls: list[tuple[str, dict]] = []

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n}} for n in self._sequences]

    def is_async(self, name: str) -> bool:
        return True  # Use async path to avoid threading issues in tests

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        self.execute_calls.append((name, arguments))
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


def _browser_tc(action: str, **extra) -> list[dict]:
    """Build a single-element tool_calls list for a browser action."""
    args = {"action": action, **extra}
    return [{
        "id": f"call_{id(object())}",
        "type": "function",
        "function": {
            "name": "browser",
            "arguments": json.dumps(args),
        },
    }]


def _generic_tc(name: str, arguments: dict | None = None) -> list[dict]:
    """Build a single-element tool_calls list for a non-browser tool."""
    args = arguments or {}
    return [{
        "id": f"call_{id(object())}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }]


def _browser_result(snippet: str) -> ToolResult:
    """Create a ToolResult with page_signals.visible_text_snippet in meta."""
    return ToolResult(
        content="<screenshot data>",
        meta={"page_signals": {"visible_text_snippet": snippet}},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObservationAwareRepetition:

    @pytest.mark.asyncio
    async def test_screenshots_with_different_content_not_blocked(self, tmp_path):
        """4 screenshots with different visible_text_snippet in meta -> NOT blocked.

        Same action + different observation = progressing, not stuck.
        """
        session = Session.new("diff_obs", str(tmp_path))

        # 4 screenshot calls, each returning different page content
        responses = [
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_text_response("Done taking screenshots."),
        ]
        provider = MockProvider(responses)

        # Each call returns a different snippet
        registry = SequentialToolRegistry(result_sequences={
            "browser": [
                _browser_result("Welcome to Page 1"),
                _browser_result("About Us - Page 2"),
                _browser_result("Contact - Page 3"),
                _browser_result("Products - Page 4"),
            ],
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="take screenshots of each page")

        msgs = session.get_messages()

        # All 4 screenshots should have been executed
        tool_results = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_results) >= 4, (
            f"Expected at least 4 tool results, got {len(tool_results)}"
        )

        # No repetition system message
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "Repetitive action" in m.get("content", "")
        ]
        assert len(system_msgs) == 0, (
            "Different observations should NOT trigger repetition detector"
        )

    @pytest.mark.asyncio
    async def test_screenshots_with_identical_content_not_blocked_by_hash(self, tmp_path):
        """5 screenshots with identical visible_text_snippet -> NOT blocked by hash.

        Browser calls are excluded from hash-based repetition detection.
        The advisory counter in the browser tool handles this instead.
        """
        session = Session.new("same_obs", str(tmp_path))

        # 6 screenshot calls plus fallback text
        responses = [
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_text_response("Stopped."),
        ]
        provider = MockProvider(responses)

        # All calls return the same snippet
        same_result = _browser_result("Stuck on login page")
        registry = SequentialToolRegistry(result_sequences={
            "browser": [same_result] * 6,
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="keep taking screenshots")

        msgs = session.get_messages()

        # All 6 screenshots should have been executed (hash detection skipped for browser)
        tool_results = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_results) >= 6, (
            f"Expected at least 6 tool results (browser skips hash detection), got {len(tool_results)}"
        )

        # No hash-based repetition system message for browser
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "Repetitive action" in m.get("content", "")
        ]
        assert len(system_msgs) == 0, (
            "Browser calls should NOT trigger hash-based repetition detector"
        )

    @pytest.mark.asyncio
    async def test_non_browser_different_results_not_blocked(self, tmp_path):
        """5 identical non-browser tool calls with different results -> NOT blocked.

        Result content differs so the hash differs each time.
        """
        session = Session.new("nb_diff", str(tmp_path))

        responses = [
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_text_response("Done reading."),
        ]
        provider = MockProvider(responses)

        # Each call returns different content
        registry = SequentialToolRegistry(result_sequences={
            "file_read": [
                ToolResult(content=f"content version {i}")
                for i in range(5)
            ],
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="read the file")

        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "Repetitive action" in m.get("content", "")
        ]
        assert len(system_msgs) == 0, (
            "Same tool + different results should NOT trigger repetition"
        )

    @pytest.mark.asyncio
    async def test_non_browser_identical_results_blocked(self, tmp_path):
        """5 identical non-browser tool calls with identical results -> blocked at 5th.
        """
        session = Session.new("nb_same", str(tmp_path))

        responses = [
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_tool_response(_generic_tc("file_read", {"path": "/a.txt"})),
            _make_text_response("Stopped."),
        ]
        provider = MockProvider(responses)

        same_result = ToolResult(content="unchanged content")
        registry = SequentialToolRegistry(result_sequences={
            "file_read": [same_result] * 6,
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="read the file")

        msgs = session.get_messages()
        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and (
                "Repetitive action" in m.get("content", "")
                or "Ping-pong" in m.get("content", "")
            )
        ]
        assert len(system_msgs) >= 1, (
            "5 identical results should trigger repetition or ping-pong detector"
        )

    @pytest.mark.asyncio
    async def test_mixed_actions_different_observations_not_blocked(self, tmp_path):
        """Interleaved browser actions with varying content don't trigger.

        Mix of screenshot, click, scroll — all with different observations.
        """
        session = Session.new("mixed", str(tmp_path))

        responses = [
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("click", ref="e10")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_tool_response(_browser_tc("scroll", direction="down")),
            _make_tool_response(_browser_tc("screenshot")),
            _make_text_response("Done browsing."),
        ]
        provider = MockProvider(responses)

        registry = SequentialToolRegistry(result_sequences={
            "browser": [
                _browser_result("Home page"),
                _browser_result("Clicked link - new page"),
                _browser_result("Products page"),
                _browser_result("Scrolled down - more products"),
                _browser_result("Footer visible"),
            ],
        })
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=20)
        await loop.run(initial_message="browse the site")

        msgs = session.get_messages()

        tool_results = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_results) >= 5, (
            f"Expected at least 5 tool results, got {len(tool_results)}"
        )

        system_msgs = [
            m for m in msgs
            if m["role"] == "system" and "Repetitive action" in m.get("content", "")
        ]
        assert len(system_msgs) == 0, (
            "Mixed actions with different observations should NOT trigger repetition"
        )
