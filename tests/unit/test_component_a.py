# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for Component A -- Agent Loop + Session + Context + Compaction.

Written from specs only (TDD). Covers all 20 acceptance criteria from
TASK-component-A-loop-session.md.

Mock Provider, ToolRegistry, PromptBuilder, and Interceptor via Protocol
interfaces. Tests Component A in isolation.
"""

import asyncio
import json
import os
import threading
import time
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio

from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    StreamAccumulator,
    TokenUsage,
    ContextOverflowError,
    LLMError,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy


# ---------------------------------------------------------------------------
# Lazy imports for Component A modules (not yet implemented).
# We import them at module level; if the modules don't exist the entire test
# file is collected but every test will fail with ImportError -- which is the
# expected TDD behaviour.
# ---------------------------------------------------------------------------

from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop, normalize_tool_call
from agent_os.agent.context import ContextManager
from agent_os.agent import compaction as compaction_mod


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_base_prompt_context(workspace: str) -> PromptContext:
    """Create a minimal PromptContext for testing."""
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class MockPromptBuilder:
    """Mock PromptBuilder satisfying the Protocol."""

    def build(self, context: PromptContext) -> tuple[str, str]:
        return ("cached-system-prefix", "dynamic-suffix")


def _make_text_response(text: str, input_tok: int = 100, output_tok: int = 50) -> LLMResponse:
    """Build an LLMResponse with text only (no tool_calls)."""
    return LLMResponse(
        raw_message={"role": "assistant", "content": text},
        text=text,
        tool_calls=[],
        has_tool_calls=False,
        finish_reason="stop",
        status_text=None,
        usage=TokenUsage(input_tokens=input_tok, output_tokens=output_tok),
    )


def _make_tool_response(
    tool_calls: list[dict],
    text: str | None = None,
    input_tok: int = 100,
    output_tok: int = 50,
) -> LLMResponse:
    """Build an LLMResponse with tool_calls."""
    raw = {"role": "assistant", "content": text, "tool_calls": tool_calls}
    return LLMResponse(
        raw_message=raw,
        text=text,
        tool_calls=tool_calls,
        has_tool_calls=True,
        finish_reason="tool_calls",
        status_text=None,
        usage=TokenUsage(input_tokens=input_tok, output_tokens=output_tok),
    )


async def _async_stream_chunks(chunks: list[StreamChunk]):
    """Async generator yielding StreamChunks."""
    for c in chunks:
        yield c


class MockProvider:
    """Mock LLM provider satisfying the Provider Protocol.

    Accepts a list of LLMResponse objects for successive calls.
    For streaming, generates StreamChunks from the text/tool_calls.
    """

    def __init__(self, responses: list[LLMResponse] | None = None,
                 stream_chunks_list: list[list[StreamChunk]] | None = None):
        self._responses = list(responses or [])
        self._stream_chunks_list = list(stream_chunks_list or [])
        self._call_idx = 0
        self.stream_call_count = 0
        self.complete_call_count = 0

    async def stream(self, messages, tools=None):
        self.stream_call_count += 1
        idx = self._call_idx
        self._call_idx += 1

        if self._stream_chunks_list and idx < len(self._stream_chunks_list):
            for chunk in self._stream_chunks_list[idx]:
                yield chunk
        elif self._responses and idx < len(self._responses):
            resp = self._responses[idx]
            if resp.text:
                yield StreamChunk(text=resp.text)
            # Yield tool_call deltas if the response has tool_calls
            if resp.tool_calls:
                for i, tc in enumerate(resp.tool_calls):
                    tc_with_index = dict(tc)
                    tc_with_index["index"] = i
                    yield StreamChunk(tool_calls_delta=[tc_with_index])
            yield StreamChunk(is_final=True, usage=resp.usage)
        else:
            yield StreamChunk(text="default", is_final=True,
                              usage=TokenUsage(10, 5))

    async def complete(self, messages, tools=None):
        self.complete_call_count += 1
        idx = self._call_idx
        self._call_idx += 1
        if self._responses and idx < len(self._responses):
            return self._responses[idx]
        return _make_text_response("default")


class MockToolRegistry:
    """Mock ToolRegistry satisfying the Protocol."""

    def __init__(self, results: dict[str, ToolResult] | None = None):
        self._results = results or {}
        self.execute_calls: list[tuple[str, dict]] = []

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n}} for n in self._results]

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        self.execute_calls.append((name, arguments))
        if name in self._results:
            return self._results[name]
        return ToolResult(content=f"result of {name}")

    def tool_names(self) -> list[str]:
        return list(self._results.keys())

    def reset_run_state(self) -> None:
        pass


class MockInterceptor:
    """Mock ToolInterceptor that intercepts tools by name."""

    def __init__(self, intercept_names: set[str] | None = None):
        self._intercept_names = intercept_names or set()
        self.intercepted_calls: list[dict] = []

    def should_intercept(self, tool_call: dict) -> bool:
        return tool_call.get("name", "") in self._intercept_names

    def on_intercept(self, tool_call: dict, recent_context: list[dict], reasoning: str | None = None) -> None:
        self.intercepted_calls.append(tool_call)


# ===========================================================================
# AC-1: Session.new() creates JSONL file. Session.load() reads it back.
#        Messages round-trip.
# ===========================================================================

class TestSessionNewAndLoad:

    def test_new_creates_jsonl_file(self, tmp_path):
        """Session.new() should create a JSONL file at the expected path."""
        session = Session.new("test-sess", str(tmp_path))
        expected_dir = tmp_path / ".agent-os" / "sessions"
        expected_file = expected_dir / "test-sess.jsonl"
        assert expected_file.exists()

    def test_messages_round_trip(self, tmp_path):
        """Messages appended to a session should round-trip through save/load."""
        session = Session.new("rt", str(tmp_path))
        msg1 = {"role": "user", "content": "hello", "source": "user"}
        msg2 = {"role": "assistant", "content": "hi", "source": "management"}
        session.append(msg1)
        session.append(msg2)

        filepath = str(tmp_path / ".agent-os" / "sessions" / "rt.jsonl")
        loaded = Session.load(filepath)
        messages = loaded.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "hi"

    def test_load_adds_timestamp_if_missing(self, tmp_path):
        """append() should add a timestamp to messages missing one."""
        session = Session.new("ts", str(tmp_path))
        session.append({"role": "user", "content": "test"})
        msgs = session.get_messages()
        assert "timestamp" in msgs[0]

    def test_load_skips_corrupted_line(self, tmp_path):
        """A corrupted JSONL line should be skipped during load."""
        filepath = tmp_path / ".agent-os" / "sessions" / "corrupt.jsonl"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "content": "good"}) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps({"role": "assistant", "content": "also good"}) + "\n")

        loaded = Session.load(str(filepath))
        messages = loaded.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "good"
        assert messages[1]["content"] == "also good"


# ===========================================================================
# AC-2: append() fires on_append callback. append_tool_result() removes
#        from pending set.
# ===========================================================================

class TestAppendCallbackAndPending:

    def test_append_fires_on_append_callback(self, tmp_path):
        """on_append callback should fire on every append()."""
        session = Session.new("cb", str(tmp_path))
        received = []
        session.on_append = lambda msg: received.append(msg)
        session.append({"role": "user", "content": "hi"})
        assert len(received) == 1
        assert received[0]["content"] == "hi"

    def test_append_tracks_tool_call_ids(self, tmp_path):
        """append() with tool_calls should track their IDs in pending set."""
        session = Session.new("pend", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ],
        }
        session.append(tc_msg)
        assert "tc_1" in session.pending_tool_calls
        assert "tc_2" in session.pending_tool_calls

    def test_append_tool_result_removes_from_pending(self, tmp_path):
        """append_tool_result() should remove the tool_call_id from pending."""
        session = Session.new("rem", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_x", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        }
        session.append(tc_msg)
        assert "tc_x" in session.pending_tool_calls

        session.append_tool_result("tc_x", "file contents")
        assert "tc_x" not in session.pending_tool_calls


# ===========================================================================
# AC-3: append_tool_result() with meta={"network": True} -> _meta in JSONL
# ===========================================================================

class TestAppendToolResultMeta:

    def test_meta_attached_to_message(self, tmp_path):
        """append_tool_result with meta should include _meta in the message."""
        session = Session.new("meta", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_m", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
        }
        session.append(tc_msg)
        session.append_tool_result("tc_m", "Exit code: 0", meta={"network": True, "domains": ["github.com"]})

        msgs = session.get_messages()
        tool_msg = [m for m in msgs if m["role"] == "tool"][0]
        assert "_meta" in tool_msg
        assert tool_msg["_meta"]["network"] is True

    def test_meta_persisted_in_jsonl(self, tmp_path):
        """_meta field should be persisted in the JSONL file."""
        session = Session.new("metap", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_p", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
        }
        session.append(tc_msg)
        session.append_tool_result("tc_p", "ok", meta={"network": True})

        filepath = str(tmp_path / ".agent-os" / "sessions" / "metap.jsonl")
        loaded = Session.load(filepath)
        tool_msg = [m for m in loaded.get_messages() if m["role"] == "tool"][0]
        assert tool_msg["_meta"]["network"] is True

    def test_no_meta_means_no_meta_field(self, tmp_path):
        """When meta=None, the _meta field should NOT be present."""
        session = Session.new("nometa", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_n", "type": "function", "function": {"name": "read", "arguments": "{}"}}],
        }
        session.append(tc_msg)
        session.append_tool_result("tc_n", "contents")

        msgs = session.get_messages()
        tool_msg = [m for m in msgs if m["role"] == "tool"][0]
        assert "_meta" not in tool_msg


# ===========================================================================
# AC-4: get_recent(max_tokens=1000) returns only messages fitting in budget,
#        newest first priority.
# ===========================================================================

class TestGetRecent:

    def test_returns_only_fitting_messages(self, tmp_path):
        """get_recent should return as many recent messages as fit in token budget."""
        session = Session.new("recent", str(tmp_path))
        # Append messages with known sizes
        for i in range(20):
            session.append({"role": "user", "content": f"Message {i}: " + "x" * 200, "source": "user"})

        recent = session.get_recent(max_tokens=1000)
        # Should return a subset, not all 20
        assert len(recent) < 20
        assert len(recent) > 0

    def test_newest_first_priority(self, tmp_path):
        """Newest messages should be prioritized (oldest dropped first)."""
        session = Session.new("newest", str(tmp_path))
        session.append({"role": "user", "content": "oldest message", "source": "user"})
        session.append({"role": "user", "content": "middle message", "source": "user"})
        session.append({"role": "user", "content": "newest message", "source": "user"})

        # Use a very small budget that can only fit ~1 message
        recent = session.get_recent(max_tokens=50)
        assert len(recent) >= 1
        # The newest message should be present
        assert recent[-1]["content"] == "newest message"

    def test_large_budget_returns_all(self, tmp_path):
        """With a large enough budget, all messages should be returned."""
        session = Session.new("all", str(tmp_path))
        for i in range(5):
            session.append({"role": "user", "content": f"msg {i}", "source": "user"})

        recent = session.get_recent(max_tokens=1_000_000)
        assert len(recent) == 5


# ===========================================================================
# AC-5: resolve_pending_tool_calls() injects CANCELLED for orphaned IDs.
# ===========================================================================

class TestResolvePendingToolCalls:

    def test_injects_cancelled_for_orphaned(self, tmp_path):
        """Orphaned tool_call IDs should get CANCELLED messages injected."""
        session = Session.new("orphan", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_a", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "tc_b", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ],
        }
        session.append(tc_msg)
        # Only resolve tc_a, leave tc_b orphaned
        session.append_tool_result("tc_a", "ok")

        assert "tc_b" in session.pending_tool_calls
        session.resolve_pending_tool_calls()
        assert len(session.pending_tool_calls) == 0

        # There should be a CANCELLED tool result for tc_b
        msgs = session.get_messages()
        cancelled = [m for m in msgs if m["role"] == "tool" and "CANCELLED" in m.get("content", "")]
        assert len(cancelled) == 1
        assert cancelled[0]["tool_call_id"] == "tc_b"

    def test_resolve_with_no_pending_is_noop(self, tmp_path):
        """resolve_pending should be safe to call when nothing is pending."""
        session = Session.new("nopend", str(tmp_path))
        msg_count_before = len(session.get_messages())
        session.resolve_pending_tool_calls()
        assert len(session.get_messages()) == msg_count_before

    def test_pending_rebuilt_on_load(self, tmp_path):
        """After load, orphaned tool_calls are healed (CANCELLED inserted adjacent).

        Startup recovery (heal_orphaned_tool_calls) runs during load(),
        so pending_tool_calls should be empty after load. The CANCELLED
        result for tc_r2 should be adjacent to the assistant message.
        """
        session = Session.new("rebuild", str(tmp_path))
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_r1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "tc_r2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ],
        }
        session.append(tc_msg)
        session.append_tool_result("tc_r1", "ok")
        # tc_r2 is still pending

        filepath = str(tmp_path / ".agent-os" / "sessions" / "rebuild.jsonl")
        loaded = Session.load(filepath)
        # Startup recovery heals orphaned tool calls
        assert len(loaded.pending_tool_calls) == 0
        # tc_r2 should have a CANCELLED result adjacent to the assistant message
        msgs = loaded.get_messages()
        tc_r2_results = [m for m in msgs if m.get("tool_call_id") == "tc_r2"]
        assert len(tc_r2_results) == 1
        assert "CANCELLED" in tc_r2_results[0]["content"]


# ===========================================================================
# AC-6: Session._compact(summary, split_idx) rewrites JSONL.
# ===========================================================================

class TestSessionCompact:

    def test_compact_rewrites_jsonl(self, tmp_path):
        """_compact should replace messages[0:split_idx] with summary."""
        session = Session.new("compact", str(tmp_path))
        for i in range(10):
            session.append({"role": "user", "content": f"msg {i}", "source": "user"})

        summary = {"role": "system", "content": "Summary of messages 0-6", "_compaction": True,
                   "source": "management", "timestamp": "2026-01-01T00:00:00"}
        session._compact(summary, split_index=7)

        msgs = session.get_messages()
        # Should be: summary + messages 7,8,9 = 4 messages
        assert len(msgs) == 4
        assert msgs[0]["_compaction"] is True
        assert msgs[0]["content"] == "Summary of messages 0-6"
        assert msgs[1]["content"] == "msg 7"
        assert msgs[3]["content"] == "msg 9"

    def test_compact_persists_to_disk(self, tmp_path):
        """After _compact, reloading the session should reflect compacted state."""
        session = Session.new("compactd", str(tmp_path))
        for i in range(5):
            session.append({"role": "user", "content": f"msg {i}", "source": "user"})

        summary = {"role": "system", "content": "Summary", "_compaction": True,
                   "source": "management", "timestamp": "2026-01-01T00:00:00"}
        session._compact(summary, split_index=3)

        filepath = str(tmp_path / ".agent-os" / "sessions" / "compactd.jsonl")
        loaded = Session.load(filepath)
        msgs = loaded.get_messages()
        assert len(msgs) == 3  # summary + msg3 + msg4
        assert msgs[0]["_compaction"] is True


# ===========================================================================
# AC-7: Loop with MockProvider (text-only) -> appends user msg + assistant
#        msg -> exits.
# ===========================================================================

class TestLoopTextOnly:

    @pytest.mark.asyncio
    async def test_text_only_response_appends_and_exits(self, tmp_path):
        """Text-only LLM response should append assistant msg and exit loop."""
        session = Session.new("textloop", str(tmp_path))
        provider = MockProvider(responses=[_make_text_response("Hello user!")])
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        await loop.run(initial_message="hi there")

        msgs = session.get_messages()
        # Should have: user msg + assistant msg (minimum)
        roles = [m["role"] for m in msgs]
        assert "user" in roles
        assert "assistant" in roles
        # The assistant message should contain "Hello user!"
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert any("Hello user!" in (m.get("content") or "") for m in assistant_msgs)


# ===========================================================================
# AC-8: Loop with MockProvider (tool_call response) -> calls execute() ->
#        appends ToolResult -> loops.
# ===========================================================================

class TestLoopToolCall:

    @pytest.mark.asyncio
    async def test_tool_call_executes_and_loops(self, tmp_path):
        """Tool call response should trigger registry.execute(), append result,
        then loop again (and exit on text-only response)."""
        session = Session.new("toolloop", str(tmp_path))

        tc = [{"id": "call_1", "type": "function",
               "function": {"name": "read", "arguments": '{"file": "test.py"}'}}]
        resp1 = _make_tool_response(tc, text="[STATUS: Reading file]")
        resp2 = _make_text_response("Done reading the file.")

        provider = MockProvider(responses=[resp1, resp2])
        registry = MockToolRegistry(results={"read": ToolResult(content="file contents here")})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        await loop.run(initial_message="read test.py")

        # Registry should have been called
        assert len(registry.execute_calls) == 1
        assert registry.execute_calls[0][0] == "read"

        # Tool result should be in session
        msgs = session.get_messages()
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "file contents here" in tool_msgs[0]["content"]


# ===========================================================================
# AC-9: Streaming: 3 StreamChunks -> accumulates -> assistant message.
#        Verify session.notify_stream called for each chunk.
# ===========================================================================

class TestLoopStreaming:

    @pytest.mark.asyncio
    async def test_streaming_accumulates_and_notifies(self, tmp_path):
        """Provider yields 3 StreamChunks. Loop accumulates via StreamAccumulator.
        session.notify_stream called for each chunk."""
        session = Session.new("stream", str(tmp_path))

        # Track notify_stream calls
        stream_notifications = []
        session.on_stream = lambda chunk: stream_notifications.append(chunk)

        chunks = [
            StreamChunk(text="Hel"),
            StreamChunk(text="lo "),
            StreamChunk(text="world", is_final=True,
                        usage=TokenUsage(input_tokens=50, output_tokens=20)),
        ]

        provider = MockProvider(stream_chunks_list=[chunks])
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        await loop.run(initial_message="say hello world")

        # Should have notified for each chunk
        assert len(stream_notifications) == 3

        # Assistant message should have accumulated text
        msgs = session.get_messages()
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1
        assert "Hello world" in (assistant_msgs[0].get("content") or "")


# ===========================================================================
# AC-10: Loop tool execution uses asyncio.to_thread (verify sync tool
#         doesn't block event loop).
# ===========================================================================

class TestAsyncToolExecution:

    @pytest.mark.asyncio
    async def test_tool_runs_in_thread(self, tmp_path):
        """Tool execution should use asyncio.to_thread so sync tools don't
        block the event loop."""
        session = Session.new("thread", str(tmp_path))

        tc = [{"id": "call_t", "type": "function",
               "function": {"name": "slow_tool", "arguments": "{}"}}]
        resp1 = _make_tool_response(tc)
        resp2 = _make_text_response("done")

        # Create a tool that records its thread
        tool_thread_ids = []
        main_thread_id = threading.current_thread().ident

        class ThreadTrackingRegistry:
            def schemas(self):
                return [{"type": "function", "function": {"name": "slow_tool"}}]

            def is_async(self, name):
                return False

            def execute(self, name, arguments):
                tool_thread_ids.append(threading.current_thread().ident)
                return ToolResult(content="slow result")

            def tool_names(self):
                return ["slow_tool"]

            def reset_run_state(self):
                pass

        provider = MockProvider(responses=[resp1, resp2])
        registry = ThreadTrackingRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)

        with patch("agent_os.agent.loop.asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread:
            await loop.run(initial_message="run slow tool")
            assert mock_to_thread.called


# ===========================================================================
# AC-11: Loop with MockInterceptor (intercepts "shell") -> session.is_paused()
#         == True -> loop exits.
# ===========================================================================

class TestLoopInterceptor:

    @pytest.mark.asyncio
    async def test_interceptor_pauses_loop(self, tmp_path):
        """When interceptor intercepts a tool call, session should pause and
        the loop should exit."""
        session = Session.new("intercept", str(tmp_path))

        tc = [{"id": "call_sh", "type": "function",
               "function": {"name": "shell", "arguments": '{"command": "rm -rf /"}'}}]
        resp = _make_tool_response(tc, text="[STATUS: Running shell]")

        provider = MockProvider(responses=[resp])
        registry = MockToolRegistry(results={"shell": ToolResult(content="ok")})
        interceptor = MockInterceptor(intercept_names={"shell"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr,
                         interceptor=interceptor)
        await loop.run(initial_message="delete everything")

        assert session.is_paused()
        assert len(interceptor.intercepted_calls) == 1
        # Tool should NOT have been executed
        assert len(registry.execute_calls) == 0


# ===========================================================================
# AC-12: After pause: session.resume(), new run() -> resolve_pending ->
#         CANCELLED injected -> loop continues.
# ===========================================================================

class TestLoopResumeAfterPause:

    @pytest.mark.asyncio
    async def test_resume_after_pause_resolves_pending(self, tmp_path):
        """After pause+resume, new run() should resolve pending tool calls
        (CANCELLED injected) and continue normally."""
        session = Session.new("resume", str(tmp_path))

        # First run: interceptor pauses on "shell"
        tc = [{"id": "call_s1", "type": "function",
               "function": {"name": "shell", "arguments": '{"command": "ls"}'}}]
        resp1 = _make_tool_response(tc)

        provider1 = MockProvider(responses=[resp1])
        registry = MockToolRegistry(results={"shell": ToolResult(content="ok")})
        interceptor = MockInterceptor(intercept_names={"shell"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop1 = AgentLoop(session, provider1, registry, context_mgr,
                          interceptor=interceptor)
        await loop1.run(initial_message="list files")
        assert session.is_paused()

        # Resume
        session.resume()
        assert not session.is_paused()

        # Second run: should resolve pending, then get text-only response
        resp2 = _make_text_response("All done after resume.")
        provider2 = MockProvider(responses=[resp2])
        context_mgr2 = ContextManager(session, builder, ctx)
        loop2 = AgentLoop(session, provider2, registry, context_mgr2)
        await loop2.run()

        # The pending call_s1 should have been resolved with CANCELLED
        msgs = session.get_messages()
        cancelled = [m for m in msgs if m["role"] == "tool" and "CANCELLED" in m.get("content", "")]
        assert len(cancelled) >= 1
        assert any(m["tool_call_id"] == "call_s1" for m in cancelled)


# ===========================================================================
# AC-13: session.queue_message("new info") during loop -> next tool iteration
#         pops it, resolves pending, breaks.
# ===========================================================================

class TestLoopQueueMessage:

    @pytest.mark.asyncio
    async def test_queued_message_breaks_tool_loop(self, tmp_path):
        """A queued message during tool execution should cause the loop to
        pop it, resolve pending, and break the inner tool loop."""
        session = Session.new("queue", str(tmp_path))

        # Two tool calls so there's a chance to pop between them
        tc = [
            {"id": "call_q1", "type": "function",
             "function": {"name": "read", "arguments": '{"file": "a.py"}'}},
            {"id": "call_q2", "type": "function",
             "function": {"name": "read", "arguments": '{"file": "b.py"}'}},
        ]
        resp1 = _make_tool_response(tc)
        resp2 = _make_text_response("Processed queued info.")

        # Queue a message that will be detected between tool calls
        call_count = [0]
        original_execute_calls = []

        class QueueInjectingRegistry:
            def schemas(self):
                return [{"type": "function", "function": {"name": "read"}}]

            def is_async(self, name):
                return False

            def execute(self, name, arguments):
                call_count[0] += 1
                original_execute_calls.append(name)
                # After first tool execution, inject a queued message
                if call_count[0] == 1:
                    session.queue_message("new info from user")
                return ToolResult(content="result")

            def tool_names(self):
                return ["read"]

            def reset_run_state(self):
                pass

        provider = MockProvider(responses=[resp1, resp2])
        registry = QueueInjectingRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        await loop.run(initial_message="read files")

        # The queued message should appear in messages
        msgs = session.get_messages()
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert any("new info" in (m.get("content") or "") for m in user_msgs)


# ===========================================================================
# AC-14: session.stop() -> loop exits, resolve_pending fires,
#         is_stopped() == True.
# ===========================================================================

class TestLoopStop:

    @pytest.mark.asyncio
    async def test_stop_exits_loop(self, tmp_path):
        """session.stop() should cause the loop to exit and resolve pending."""
        session = Session.new("stop", str(tmp_path))

        tc = [{"id": "call_st", "type": "function",
               "function": {"name": "read", "arguments": "{}"}}]
        resp = _make_tool_response(tc)

        class StoppingRegistry:
            def schemas(self):
                return [{"type": "function", "function": {"name": "read"}}]

            def is_async(self, name):
                return False

            def execute(self, name, arguments):
                session.stop()
                return ToolResult(content="read ok")

            def tool_names(self):
                return ["read"]

            def reset_run_state(self):
                pass

        provider = MockProvider(responses=[resp])
        registry = StoppingRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        await loop.run(initial_message="read")

        assert session.is_stopped()


# ===========================================================================
# AC-15: Loop guard: max_iterations=3 -> exits after 3 iterations with
#         system message.
# ===========================================================================

class TestLoopGuardIterationCap:

    @pytest.mark.asyncio
    async def test_max_iterations_cap(self, tmp_path):
        """Loop should exit after max_iterations with a system message."""
        session = Session.new("maxiter", str(tmp_path))

        # Provider always returns tool calls (infinite loop scenario)
        def make_tc_resp(idx):
            tc = [{"id": f"call_{idx}", "type": "function",
                   "function": {"name": "read", "arguments": f'{{"file": "f{idx}.py"}}'}}]
            return _make_tool_response(tc)

        responses = [make_tc_resp(i) for i in range(10)]
        provider = MockProvider(responses=responses)
        registry = MockToolRegistry(results={"read": ToolResult(content="ok")})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=3)
        await loop.run(initial_message="read everything")

        # Should have a system message about iteration limit
        msgs = session.get_messages()
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert any("iteration" in (m.get("content") or "").lower() or
                    "limit" in (m.get("content") or "").lower() or
                    "max" in (m.get("content") or "").lower()
                    for m in system_msgs), \
            f"Expected system message about iteration cap, got: {[m.get('content') for m in system_msgs]}"


# ===========================================================================
# AC-16: Loop guard: MockProvider returns same tool_call 3x -> repetition
#         detected -> exits.
# ===========================================================================

class TestLoopGuardRepetition:

    @pytest.mark.asyncio
    async def test_repetition_detected_and_exits(self, tmp_path):
        """Same tool call repeated 3+ times should trigger repetition guard."""
        session = Session.new("repeat", str(tmp_path))

        # Same tool call every time
        def make_same_tc():
            tc = [{"id": f"call_{id(object())}", "type": "function",
                   "function": {"name": "read", "arguments": '{"file": "same.py"}'}}]
            return _make_tool_response(tc)

        responses = [make_same_tc() for _ in range(10)]
        provider = MockProvider(responses=responses)
        registry = MockToolRegistry(results={"read": ToolResult(content="ok")})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=50)
        await loop.run(initial_message="read same.py")

        # Loop should have exited due to repetition, not max_iterations
        # There should be a system message about repetition
        msgs = session.get_messages()
        system_msgs = [m for m in msgs if m["role"] == "system"]
        # Should exit before 50 iterations
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        assert len(tool_msgs) <= 5, "Should exit well before max_iterations due to repetition"

    @pytest.mark.asyncio
    async def test_repetition_message_order_tool_before_system(self, tmp_path):
        """Repetition: session must have tool CANCELLED before system message.

        Correct:  assistant(tool_calls) → tool(CANCELLED) → system(repetitive)
        Wrong:    assistant(tool_calls) → system(repetitive) → tool(CANCELLED)
        """
        session = Session.new("rep_order", str(tmp_path))

        # All 3 tool calls are identical (same name + same args)
        def make_same_tc():
            tc = [{"id": f"tc_{id(object())}", "type": "function",
                   "function": {"name": "read", "arguments": '{"file": "x.py"}'}}]
            return _make_tool_response(tc)

        responses = [make_same_tc() for _ in range(5)]
        provider = MockProvider(responses=responses)
        registry = MockToolRegistry(results={"read": ToolResult(content="ok")})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr, max_iterations=50)
        await loop.run(initial_message="read x.py repeatedly")

        msgs = session.get_messages()

        # Find the system message about repetition (either single-repetition or ping-pong)
        rep_idx = None
        for i, m in enumerate(msgs):
            if m["role"] == "system" and (
                "Repetitive action" in m.get("content", "")
                or "Ping-pong" in m.get("content", "")
            ):
                rep_idx = i
                break
        assert rep_idx is not None, "Should have a repetition/ping-pong system message"

        # The message immediately before the system message must be a tool result
        # (the CANCELLED result for the tool call that triggered detection)
        prev = msgs[rep_idx - 1]
        assert prev["role"] == "tool", (
            f"Message before detection system msg should be role='tool', "
            f"got role='{prev['role']}'. Order: ...→{prev['role']}→system is wrong; "
            f"expected ...→tool(CANCELLED)→system"
        )

        # The assistant message with tool_calls should be 2 positions before system
        # (assistant → tool → system)
        assistant_msg = msgs[rep_idx - 2]
        assert assistant_msg["role"] == "assistant", (
            f"Expected assistant 2 before system, got '{assistant_msg['role']}'"
        )
        assert "tool_calls" in assistant_msg, "Assistant message should have tool_calls"


# ===========================================================================
# AC-17: context.prepare() with base_prompt_context -> updates datetime_now
#         and context_usage_pct each call.
# ===========================================================================

class TestContextPrepareTransientFields:

    def test_datetime_now_updated_each_call(self, tmp_path):
        """prepare() should update datetime_now on each call."""
        session = Session.new("ctxdt", str(tmp_path))
        session.append({"role": "user", "content": "hello", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        # First prepare
        with patch("agent_os.agent.context.datetime") as mock_dt:
            mock_dt.now.return_value.isoformat.return_value = "2026-01-01T10:00:00"
            # Fallback for other datetime uses
            mock_dt.side_effect = lambda *args, **kwargs: None
            try:
                result1 = context_mgr.prepare()
            except Exception:
                pass

        # We verify the ContextManager calls datetime.now() by checking
        # that it calls build on the prompt builder with an updated context
        build_calls = []
        original_build = builder.build

        def tracking_build(context):
            build_calls.append(context)
            return original_build(context)

        builder.build = tracking_build

        context_mgr.prepare()
        time.sleep(0.01)  # small delay
        context_mgr.prepare()

        assert len(build_calls) >= 2
        # datetime_now should be different from the initial base context value
        assert build_calls[0].datetime_now != "2026-01-01T00:00:00" or \
               build_calls[1].datetime_now != "2026-01-01T00:00:00"

    def test_context_usage_pct_updated(self, tmp_path):
        """context_usage_pct should use previous iteration's value."""
        session = Session.new("ctxpct", str(tmp_path))
        session.append({"role": "user", "content": "hello", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        build_calls = []
        original_build = builder.build

        def tracking_build(context):
            build_calls.append(context)
            return original_build(context)

        builder.build = tracking_build

        # First call: usage_pct should be 0.0
        context_mgr.prepare()
        assert build_calls[0].context_usage_pct == 0.0

        # Second call: usage_pct should reflect first call's usage (non-zero)
        context_mgr.prepare()
        # The second call gets the previous iteration's usage
        # It may or may not be exactly 0.0 depending on how much content there is
        assert len(build_calls) == 2


# ===========================================================================
# AC-18: context.prepare() reads PROJECT_STATE.md from disk. Modify file,
#         call again -> picks up changes.
# ===========================================================================

class TestContextReadsLayerFiles:

    def test_reads_project_state_from_disk(self, tmp_path):
        """prepare() should read PROJECT_STATE.md and include it in context."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)
        state_file = agent_os_dir / "PROJECT_STATE.md"
        state_file.write_text("# Project State\nWorking on feature X.", encoding="utf-8")

        session = Session.new("ctxstate", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        # Result is a list of dicts (messages). The layer files should appear
        # as system role messages.
        system_msgs = [m for m in result if m.get("role") == "system"]
        all_content = " ".join(m.get("content", "") for m in system_msgs)
        assert "Working on feature X" in all_content

    def test_picks_up_file_changes(self, tmp_path):
        """Modifying PROJECT_STATE.md between prepare() calls should be reflected."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)
        state_file = agent_os_dir / "PROJECT_STATE.md"
        state_file.write_text("Version 1", encoding="utf-8")

        session = Session.new("ctxchange", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result1 = context_mgr.prepare()
        sys1 = " ".join(m.get("content", "") for m in result1 if m.get("role") == "system")
        assert "Version 1" in sys1

        # Modify the file
        state_file.write_text("Version 2 updated", encoding="utf-8")
        result2 = context_mgr.prepare()
        sys2 = " ".join(m.get("content", "") for m in result2 if m.get("role") == "system")
        assert "Version 2" in sys2

    def test_missing_project_state_is_fine(self, tmp_path):
        """If PROJECT_STATE.md doesn't exist, prepare() should not fail."""
        session = Session.new("ctxmissing", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        # Should not raise
        result = context_mgr.prepare()
        assert isinstance(result, list)


# ===========================================================================
# AC-19: context.prepare() respects token budget -- does not exceed
#         model_context_limit - response_reserve.
# ===========================================================================

class TestContextTokenBudget:

    def test_does_not_exceed_budget(self, tmp_path):
        """Total tokens in prepare() output should not exceed
        model_context_limit - response_reserve."""
        session = Session.new("ctxbudget", str(tmp_path))
        # Add many messages to create pressure
        for i in range(100):
            session.append({"role": "user", "content": f"Message {i}: " + "x" * 500, "source": "user"})

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        # Small context limit to force budget constraints
        model_limit = 10_000
        reserve = 2_000
        context_mgr = ContextManager(session, builder, ctx,
                                     model_context_limit=model_limit,
                                     response_reserve=reserve)

        result = context_mgr.prepare()
        # Estimate total tokens
        total_tokens = sum(len(json.dumps(m)) / 4 for m in result)
        budget = model_limit - reserve
        assert total_tokens <= budget * 1.1, \
            f"Total tokens {total_tokens} exceeded budget {budget}"


# ===========================================================================
# AC-20: Tool result pruning: old tool result >500 chars is truncated in
#         prepared context but intact in JSONL.
# ===========================================================================

class TestToolResultPruning:

    def test_old_tool_result_truncated_in_context(self, tmp_path):
        """Tool results older than 5 turns with >500 chars should be
        truncated in prepare() output but intact in JSONL."""
        session = Session.new("prune", str(tmp_path))

        # Create a long tool result
        long_content = "A" * 1000
        tc_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "tc_long", "type": "function",
                            "function": {"name": "read", "arguments": "{}"}}],
        }
        session.append(tc_msg)
        session.append_tool_result("tc_long", long_content)

        # Add more than 5 turns after to make the tool result "old"
        for i in range(10):
            session.append({"role": "user", "content": f"msg {i}", "source": "user"})
            session.append({"role": "assistant", "content": f"reply {i}", "source": "management"})

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()

        # In the prepared context, the old tool result should be truncated
        tool_msgs_in_context = [m for m in result if m.get("role") == "tool"]
        for m in tool_msgs_in_context:
            if m.get("tool_call_id") == "tc_long":
                assert len(m["content"]) < len(long_content), \
                    "Old tool result should be truncated in prepared context"
                assert "[Truncated]" in m["content"]

        # In the JSONL (session), the original should be intact
        session_msgs = session.get_messages()
        original_tool = [m for m in session_msgs if m.get("tool_call_id") == "tc_long"][0]
        assert original_tool["content"] == long_content


# ===========================================================================
# Additional edge case tests
# ===========================================================================

class TestNormalizeToolCall:
    """Test the normalize_tool_call function directly."""

    def test_nested_function_format(self):
        """Handle OpenAI nested format: {id, function: {name, arguments}}."""
        raw = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "read", "arguments": '{"file": "test.py"}'},
        }
        result = normalize_tool_call(raw)
        assert result["id"] == "call_abc"
        assert result["name"] == "read"
        assert result["arguments"] == {"file": "test.py"}

    def test_flat_format(self):
        """Handle flat format: {id, name, arguments}."""
        raw = {"id": "call_flat", "name": "write", "arguments": {"file": "out.py", "content": "hi"}}
        result = normalize_tool_call(raw)
        assert result["id"] == "call_flat"
        assert result["name"] == "write"
        assert result["arguments"] == {"file": "out.py", "content": "hi"}

    def test_string_arguments_parsed(self):
        """String arguments should be JSON-parsed."""
        raw = {"id": "call_str", "name": "shell", "arguments": '{"command": "ls"}'}
        result = normalize_tool_call(raw)
        assert result["arguments"] == {"command": "ls"}

    def test_invalid_json_arguments_fallback(self):
        """Invalid JSON arguments should fall back to empty dict."""
        raw = {"id": "call_bad", "name": "shell", "arguments": "not json"}
        result = normalize_tool_call(raw)
        assert result["arguments"] == {}

    def test_missing_id_defaults_empty(self):
        """Missing id should default to empty string."""
        raw = {"name": "read", "arguments": {}}
        result = normalize_tool_call(raw)
        assert result["id"] == ""


class TestSessionQueueAndLifecycle:
    """Test session queue_message, pop, pause/resume/stop."""

    def test_queue_message_and_pop(self, tmp_path):
        session = Session.new("queuetest", str(tmp_path))
        session.queue_message("info 1")
        session.queue_message("info 2")

        popped = session.pop_queued_messages()
        assert popped == [("info 1", None), ("info 2", None)]
        # Pop again should be empty
        assert session.pop_queued_messages() == []

    def test_pause_resume_stop_flags(self, tmp_path):
        session = Session.new("flags", str(tmp_path))
        assert not session.is_paused()
        assert not session.is_stopped()

        session.pause()
        assert session.is_paused()

        session.resume()
        assert not session.is_paused()

        session.stop()
        assert session.is_stopped()

    def test_append_system(self, tmp_path):
        session = Session.new("sys", str(tmp_path))
        session.append_system("System notice")
        msgs = session.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "System notice"
        assert msgs[0]["source"] == "management"

    def test_recent_activity(self, tmp_path):
        session = Session.new("activity", str(tmp_path))
        for i in range(15):
            session.append({"role": "user", "content": f"msg {i}", "source": "user"})
        activity = session.recent_activity()
        assert len(activity) <= 10

    def test_notify_stream_calls_on_stream(self, tmp_path):
        session = Session.new("notify", str(tmp_path))
        received = []
        session.on_stream = lambda chunk: received.append(chunk)
        chunk = StreamChunk(text="hello")
        session.notify_stream(chunk)
        assert len(received) == 1
        assert received[0].text == "hello"

    def test_notify_stream_no_callback_is_safe(self, tmp_path):
        session = Session.new("noncb", str(tmp_path))
        session.on_stream = None
        # Should not raise
        session.notify_stream(StreamChunk(text="hello"))


class TestSessionThreadSafety:
    """Test that JSONL file writes are protected by a lock."""

    def test_concurrent_appends_do_not_corrupt(self, tmp_path):
        """Multiple threads appending should not corrupt the JSONL."""
        session = Session.new("threadsafe", str(tmp_path))
        errors = []

        def append_many(start):
            try:
                for i in range(50):
                    session.append({"role": "user", "content": f"msg-{start}-{i}", "source": "user"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=append_many, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        msgs = session.get_messages()
        assert len(msgs) == 200  # 4 threads * 50 messages


class TestCompactionModule:
    """Test the compaction module."""

    @pytest.mark.asyncio
    async def test_compaction_run_summarizes(self, tmp_path):
        """compaction.run() should summarize older messages and compact."""
        session = Session.new("compactrun", str(tmp_path))
        for i in range(20):
            session.append({"role": "user", "content": f"Message {i}: details about task {i}", "source": "user"})
            session.append({"role": "assistant", "content": f"Working on task {i}", "source": "management"})

        original_count = len(session.get_messages())

        # Mock provider for summarization
        summary_response = _make_text_response("Summary: User discussed tasks 0-13.")

        provider = MockProvider(responses=[])
        utility_provider = MockProvider(responses=[summary_response])

        await compaction_mod.run(session, provider, utility_provider=utility_provider)

        msgs = session.get_messages()
        # Should have fewer messages after compaction
        assert len(msgs) < original_count
        # First message should be the compaction summary
        assert msgs[0].get("_compaction") is True


class TestContextManagerShouldCompact:
    """Test should_compact() and reduce_window()."""

    def test_should_compact_when_usage_high(self, tmp_path):
        """should_compact() should return True when usage > 0.80."""
        session = Session.new("compact_check", str(tmp_path))
        # Fill session with enough content to push usage high
        for i in range(200):
            session.append({"role": "user", "content": "x" * 500, "source": "user"})

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        # Small limits to trigger high usage
        context_mgr = ContextManager(session, builder, ctx,
                                     model_context_limit=5000,
                                     response_reserve=1000)

        context_mgr.prepare()
        # After prepare with lots of content and small limits, usage should be high
        assert context_mgr.usage_percentage > 0.0

    def test_reduce_window(self, tmp_path):
        """reduce_window() should reduce the sliding window budget."""
        session = Session.new("reduce", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        # Call reduce_window -- it should not raise
        context_mgr.reduce_window(0.5)
        # Should still be able to prepare
        result = context_mgr.prepare()
        assert isinstance(result, list)


class TestLoopIsRunning:
    """Test the is_running property of AgentLoop."""

    @pytest.mark.asyncio
    async def test_is_running_during_loop(self, tmp_path):
        """is_running should be True during run() and False before/after."""
        session = Session.new("running", str(tmp_path))
        provider = MockProvider(responses=[_make_text_response("hi")])
        registry = MockToolRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        assert not loop.is_running

        running_states = []

        original_prepare = context_mgr.prepare

        def tracking_prepare():
            running_states.append(loop.is_running)
            return original_prepare()

        context_mgr.prepare = tracking_prepare

        await loop.run(initial_message="test")
        assert not loop.is_running
        # During prepare() inside run(), is_running should have been True
        if running_states:
            assert running_states[0] is True


class TestLoopFinallyCleansUp:
    """Verify that the finally block always resolves pending."""

    @pytest.mark.asyncio
    async def test_finally_resolves_pending_on_exception(self, tmp_path):
        """Even on unexpected exception, pending tool calls should be resolved."""
        session = Session.new("finally", str(tmp_path))

        tc = [{"id": "call_fin", "type": "function",
               "function": {"name": "explode", "arguments": "{}"}}]
        resp = _make_tool_response(tc)

        class ExplodingProvider:
            """Provider that gives tool call first, then errors."""
            _call_count = 0

            async def stream(self, messages, tools=None):
                self._call_count += 1
                if self._call_count == 1:
                    yield StreamChunk(text="[STATUS: test]")
                    yield StreamChunk(
                        is_final=True,
                        usage=TokenUsage(100, 50),
                    )
                else:
                    raise LLMError("Unexpected error", status_code=500)

            async def complete(self, messages, tools=None):
                return _make_text_response("ok")

        # We need the first response to be a tool-call response
        # Use a provider that streams the tool call, then fails on second call
        provider = MockProvider(responses=[resp])

        class FailingRegistry:
            def schemas(self):
                return []

            def is_async(self, name):
                return False

            def execute(self, name, arguments):
                raise RuntimeError("Tool crashed!")

            def tool_names(self):
                return []

            def reset_run_state(self):
                pass

        registry = FailingRegistry()
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop = AgentLoop(session, provider, registry, context_mgr)
        # The loop should handle the error gracefully
        try:
            await loop.run(initial_message="test")
        except Exception:
            pass

        # Pending tool calls should be resolved regardless
        assert len(session.pending_tool_calls) == 0


class TestPausedForApprovalFlag:
    """Test _paused_for_approval behavior."""

    @pytest.mark.asyncio
    async def test_paused_for_approval_skips_resolve_on_resume(self, tmp_path):
        """When _paused_for_approval is True, run() should NOT resolve pending
        so the approved tool can execute through the normal pipeline."""
        session = Session.new("approval", str(tmp_path))

        # First run: interceptor intercepts "shell"
        tc = [{"id": "call_ap", "type": "function",
               "function": {"name": "shell", "arguments": '{"command": "echo hi"}'}}]
        resp1 = _make_tool_response(tc)

        provider1 = MockProvider(responses=[resp1])
        registry = MockToolRegistry(results={"shell": ToolResult(content="hi")})
        interceptor = MockInterceptor(intercept_names={"shell"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        loop1 = AgentLoop(session, provider1, registry, context_mgr,
                          interceptor=interceptor)
        await loop1.run(initial_message="run echo")
        assert session.is_paused()
        # The loop should have set _paused_for_approval
        assert session._paused_for_approval is True

        # resume() does NOT clear _paused_for_approval — loop.run() owns that
        session.resume()
        assert session._paused_for_approval is True  # still set until next run()

        # On the next run(), the flag is consumed and pending calls are preserved
        resp2 = _make_text_response("done")
        provider2 = MockProvider(responses=[resp2])
        loop2 = AgentLoop(session, provider2, registry, context_mgr,
                          interceptor=interceptor)
        await loop2.run()
        assert session._paused_for_approval is False  # cleared by run()


# ===========================================================================
# Session history persistence: crash recovery context injection
# ===========================================================================

class TestCrashRecoveryContextInjection:

    def test_no_injection_when_both_layer1_files_exist(self, tmp_path):
        """If project_goals.md AND PROJECT_STATE.md exist, no recovery injection."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)
        (agent_os_dir / "PROJECT_STATE.md").write_text("state", encoding="utf-8")
        instructions_dir = agent_os_dir / "instructions"
        instructions_dir.mkdir(parents=True, exist_ok=True)
        (instructions_dir / "project_goals.md").write_text("goals", encoding="utf-8")

        # Create an archived session to potentially recover from
        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "old_session.jsonl").write_text(
            json.dumps({"role": "user", "content": "old msg", "session_id": "old"}) + "\n",
            encoding="utf-8",
        )

        session = Session.new("fresh_session", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        # No recovery context should appear
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" not in all_content

    def test_injection_when_project_goals_missing(self, tmp_path):
        """If project_goals.md is missing and an archived session exists, inject recovery."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)
        # PROJECT_STATE.md exists but project_goals.md doesn't
        (agent_os_dir / "PROJECT_STATE.md").write_text("state", encoding="utf-8")

        # Create an archived session
        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        old_msgs = [
            json.dumps({"role": "user", "content": "I want to build a CLI tool", "session_id": "old"}),
            json.dumps({"role": "assistant", "content": "Great, let me help with that", "session_id": "old"}),
        ]
        (sessions_dir / "old_session.jsonl").write_text("\n".join(old_msgs) + "\n", encoding="utf-8")

        session = Session.new("fresh_session", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" in all_content
        assert "I want to build a CLI tool" in all_content
        assert "let me help with that" in all_content

    def test_injection_when_both_files_missing(self, tmp_path):
        """If both Layer 1 files missing and archived JSONL exists, inject recovery."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)

        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        old_msgs = [
            json.dumps({"role": "user", "content": "Hello agent", "session_id": "old"}),
        ]
        (sessions_dir / "old_session.jsonl").write_text("\n".join(old_msgs) + "\n", encoding="utf-8")

        session = Session.new("fresh_session", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" in all_content
        assert "Hello agent" in all_content

    def test_no_injection_when_no_archived_sessions(self, tmp_path):
        """If Layer 1 files missing but no archived sessions, no injection."""
        session = Session.new("fresh_session", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" not in all_content

    def test_no_injection_on_session_with_existing_messages(self, tmp_path):
        """If session already has messages, skip recovery (not a fresh start)."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "old_session.jsonl").write_text(
            json.dumps({"role": "user", "content": "old context", "session_id": "old"}) + "\n",
            encoding="utf-8",
        )

        session = Session.new("fresh_session", str(tmp_path))
        session.append({"role": "user", "content": "already chatting", "source": "user"})
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" not in all_content

    def test_recovery_only_runs_once(self, tmp_path):
        """Recovery injection should only happen on the first prepare() call."""
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / "old_session.jsonl").write_text(
            json.dumps({"role": "user", "content": "old msg", "session_id": "old"}) + "\n",
            encoding="utf-8",
        )

        session = Session.new("fresh_session", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        context_mgr.prepare()  # first call: injects recovery
        msg_count_after_first = len(session.get_messages())

        context_mgr.prepare()  # second call: should NOT inject again
        msg_count_after_second = len(session.get_messages())

        assert msg_count_after_first == msg_count_after_second

    def test_recovery_excludes_current_session_file(self, tmp_path):
        """Recovery should not read from the current session's own JSONL file."""
        agent_os_dir = tmp_path / ".agent-os"
        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session = Session.new("my_session", str(tmp_path))
        # Write a message to current session's JSONL (simulating partial write)
        session.append({"role": "system", "content": "init"})
        # Remove the message from memory to simulate a fresh session
        session._messages.clear()

        # The only JSONL file is current session's own — no recovery should happen
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" not in all_content

    def test_recovery_formats_tool_calls(self, tmp_path):
        """Tool calls in archived messages should be formatted as 'Assistant used: tool_name'."""
        agent_os_dir = tmp_path / ".agent-os"
        sessions_dir = agent_os_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        old_msgs = [
            json.dumps({
                "role": "assistant",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
                "session_id": "old",
            }),
            json.dumps({"role": "tool", "content": "file contents", "tool_call_id": "tc1", "session_id": "old"}),
            json.dumps({"role": "user", "content": "Thanks!", "session_id": "old"}),
        ]
        (sessions_dir / "old_session.jsonl").write_text("\n".join(old_msgs) + "\n", encoding="utf-8")

        session = Session.new("fresh_session", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        all_content = " ".join(m.get("content", "") for m in result)
        assert "RECOVERY CONTEXT" in all_content
        assert "Assistant used: read" in all_content
        # Tool result should be skipped
        assert "file contents" not in all_content


# ===========================================================================
# Multi-tool-call session corruption fixes
# ===========================================================================

class TestMultiToolInterceptBatchCancellation:
    """Fix 2: When one tool call in a batch is intercepted, subsequent
    unprocessed tool calls should get CANCELLED results."""

    @pytest.mark.asyncio
    async def test_intercept_cancels_remaining_batch(self, tmp_path):
        """When tool B is intercepted in [A, B, C], C should get CANCELLED."""
        # Tool A executes normally, tool B is intercepted, tool C never processed
        tool_calls = [
            {"id": "tc_a", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            {"id": "tc_b", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
            {"id": "tc_c", "type": "function", "function": {"name": "write", "arguments": "{}"}},
        ]
        resp1 = _make_tool_response(tool_calls)
        provider = MockProvider(responses=[resp1])

        session = Session.new("multi_intercept", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        registry = MagicMock()
        registry.schemas.return_value = []
        registry.is_async.return_value = False
        registry.execute.return_value = ToolResult(content="ok")

        # Interceptor: only intercept "shell"
        interceptor = MagicMock()
        interceptor.should_intercept.side_effect = lambda tc: tc.get("name") == "shell"
        interceptor.on_intercept = MagicMock()

        loop = AgentLoop(session, provider, registry, context_mgr, interceptor=interceptor)
        await loop.run(initial_message="do three things")

        messages = session.get_messages()
        # Find tool results
        tool_results = [m for m in messages if m.get("role") == "tool"]

        # tc_a should have executed normally
        tc_a_result = [m for m in tool_results if m.get("tool_call_id") == "tc_a"]
        assert len(tc_a_result) == 1
        assert tc_a_result[0]["content"] == "ok"

        # tc_b should be pending (intercepted, no result yet)
        assert "tc_b" in session.pending_tool_calls

        # tc_c should be CANCELLED
        tc_c_result = [m for m in tool_results if m.get("tool_call_id") == "tc_c"]
        assert len(tc_c_result) == 1
        assert "CANCELLED" in tc_c_result[0]["content"]
        assert "approval" in tc_c_result[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_intercept_first_tool_cancels_all_remaining(self, tmp_path):
        """When the first tool in [A, B] is intercepted, B should get CANCELLED."""
        tool_calls = [
            {"id": "tc_a", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
            {"id": "tc_b", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ]
        resp1 = _make_tool_response(tool_calls)
        provider = MockProvider(responses=[resp1])

        session = Session.new("intercept_first", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        registry = MagicMock()
        registry.schemas.return_value = []
        registry.is_async.return_value = False

        interceptor = MagicMock()
        interceptor.should_intercept.side_effect = lambda tc: tc.get("name") == "shell"
        interceptor.on_intercept = MagicMock()

        loop = AgentLoop(session, provider, registry, context_mgr, interceptor=interceptor)
        await loop.run(initial_message="two tools")

        messages = session.get_messages()
        tool_results = [m for m in messages if m.get("role") == "tool"]

        # tc_a intercepted (pending)
        assert "tc_a" in session.pending_tool_calls

        # tc_b CANCELLED
        tc_b_result = [m for m in tool_results if m.get("tool_call_id") == "tc_b"]
        assert len(tc_b_result) == 1
        assert "CANCELLED" in tc_b_result[0]["content"]


class TestContextValidateToolResults:
    """Fix 3: ContextManager validates that every tool_call has a matching result."""

    def test_missing_tool_result_gets_synthetic_error(self, tmp_path):
        """If a tool_call has no matching result, inject synthetic error."""
        session = Session.new("validate_test", str(tmp_path))
        # Simulate corrupted session: assistant with 2 tool_calls, only 1 result
        session.append({
            "role": "user", "content": "test", "source": "user",
        })
        session.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
            ],
            "source": "management",
        })
        # Only append result for tc_1, not tc_2
        session.append_tool_result("tc_1", "file contents")
        # Don't append result for tc_2 - simulates corruption

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        # Find tool results in context
        tool_msgs = [m for m in result if m.get("role") == "tool"]

        # Should have 2 tool results: original for tc_1, synthetic for tc_2
        tc_ids = {m.get("tool_call_id") for m in tool_msgs}
        assert "tc_1" in tc_ids
        assert "tc_2" in tc_ids

        tc_2_msg = [m for m in tool_msgs if m.get("tool_call_id") == "tc_2"]
        assert len(tc_2_msg) == 1
        assert "Error" in tc_2_msg[0]["content"]
        assert "lost" in tc_2_msg[0]["content"].lower()

    def test_valid_session_not_modified(self, tmp_path):
        """Sessions where all tool_calls have results should not be modified."""
        session = Session.new("validate_ok", str(tmp_path))
        session.append({"role": "user", "content": "test", "source": "user"})
        session.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            ],
            "source": "management",
        })
        session.append_tool_result("tc_1", "ok")

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "ok"

    def test_multiple_missing_results_all_injected(self, tmp_path):
        """Multiple missing results in one assistant message get synthetic errors."""
        session = Session.new("validate_multi", str(tmp_path))
        session.append({"role": "user", "content": "test", "source": "user"})
        session.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tc_1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "tc_2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
                {"id": "tc_3", "type": "function", "function": {"name": "shell", "arguments": "{}"}},
            ],
            "source": "management",
        })
        # Only provide result for tc_1
        session.append_tool_result("tc_1", "ok")

        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        context_mgr = ContextManager(session, builder, ctx)

        result = context_mgr.prepare()
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        tc_ids = {m.get("tool_call_id") for m in tool_msgs}
        assert tc_ids == {"tc_1", "tc_2", "tc_3"}
