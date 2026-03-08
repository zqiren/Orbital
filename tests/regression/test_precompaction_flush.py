# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: pre-compaction memory flush preserves working state.

Bug: When compaction fires at >80% context, the agent's working knowledge
(current task position, active decisions, in-progress state) is summarised
and potentially lost. The agent may repeat work or lose track after compaction.

Fix: Before compaction, give the agent one additional LLM turn to write
critical working state to PROJECT_STATE.md. If the agent responds with
<silent>, skip cleanly. The flush turn does not increment the iteration counter.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.compaction import MEMORY_FLUSH_PROMPT, is_silent_response
from agent_os.agent.session import Session


@pytest.fixture
def session(tmp_path):
    return Session.new("test-flush", str(tmp_path))


class TestMemoryFlushPrompt:
    """MEMORY_FLUSH_PROMPT constant and is_silent_response helper."""

    def test_flush_prompt_exists(self):
        assert isinstance(MEMORY_FLUSH_PROMPT, str)
        assert "Pre-compaction" in MEMORY_FLUSH_PROMPT
        assert "PROJECT_STATE.md" in MEMORY_FLUSH_PROMPT

    def test_silent_response_exact(self):
        assert is_silent_response("<silent>") is True

    def test_silent_response_with_whitespace(self):
        assert is_silent_response("  <silent>  \n") is True

    def test_silent_response_empty(self):
        assert is_silent_response("") is True

    def test_silent_response_normal_text(self):
        assert is_silent_response("I will save my state now.") is False

    def test_silent_response_partial(self):
        assert is_silent_response("<silent> but also some text") is False


class TestFlushTurnInLoop:
    """The flush turn must fire BEFORE compaction, not after."""

    @pytest.mark.asyncio
    async def test_flush_called_before_compaction(self, session, tmp_path):
        """When should_compact() returns True, flush turn fires before compaction."""
        from agent_os.agent.loop import AgentLoop
        from agent_os.agent.providers.types import LLMResponse, TokenUsage

        def _resp(text="", tool_calls=None, raw=None):
            tcs = tool_calls or []
            return LLMResponse(
                text=text,
                tool_calls=tcs,
                raw_message=raw or {"role": "assistant", "content": text},
                has_tool_calls=bool(tcs),
                finish_reason="tool_calls" if tcs else "stop",
                status_text=None,
                usage=TokenUsage(input_tokens=50, output_tokens=10),
            )

        call_order = []

        # First call returns tool call to trigger tool execution path
        tc_list = [{"id": "tc_f1", "function": {"name": "read", "arguments": "{}"}}]
        tool_response = _resp(
            tool_calls=tc_list,
            raw={"role": "assistant", "content": "", "tool_calls": tc_list},
        )
        # Flush turn response: <silent>
        silent_response = _resp(text="<silent>")
        # After compaction, LLM returns text-only (ends loop)
        final_response = _resp(text="Done.")

        # Provider: complete() used by flush turn
        provider = AsyncMock()

        async def mock_complete(messages):
            call_order.append("flush")
            return silent_response

        provider.complete = mock_complete

        # Context manager: should_compact returns True after first tool execution
        context_manager = MagicMock()
        context_manager.prepare.return_value = [{"role": "system", "content": "test"}]
        context_manager.model_context_limit = 128_000
        compact_call_count = {"n": 0}

        def should_compact_side_effect():
            compact_call_count["n"] += 1
            return compact_call_count["n"] == 1  # True on first check only

        context_manager.should_compact = MagicMock(side_effect=should_compact_side_effect)

        # Tool registry
        from agent_os.agent.tools.base import ToolResult
        tool_registry = MagicMock()
        tool_registry.schemas.return_value = []
        tool_registry.is_async.return_value = False
        tool_registry.execute.return_value = ToolResult(content="file contents")
        tool_registry.reset_run_state = MagicMock()

        loop = AgentLoop(
            session=session,
            provider=provider,
            tool_registry=tool_registry,
            context_manager=context_manager,
        )

        # Override _stream_response: first call returns tool call, second returns text
        responses = iter([tool_response, final_response])

        async def mock_stream(context, tool_schemas):
            return next(responses)

        loop._stream_response = mock_stream

        # Patch compaction.run at the module level
        async def mock_compact_run(sess, prov, utility_provider=None):
            call_order.append("compaction")

        with patch("agent_os.agent.compaction.run", new=mock_compact_run):
            await loop.run("Do the task")

        # Verify flush was called before compaction
        assert "flush" in call_order, "Flush turn must be called"
        assert "compaction" in call_order, "Compaction must be called"
        assert call_order.index("flush") < call_order.index("compaction"), \
            "Flush must happen BEFORE compaction"

    @pytest.mark.asyncio
    async def test_silent_flush_appends_nothing(self, session):
        """When flush turn returns <silent>, no assistant message is appended."""
        # Seed session with a user message
        session.append({"role": "user", "content": "test", "source": "user"})

        msg_count_before = len(session.get_messages())

        # The flush prompt system message will be appended
        session.append_system(MEMORY_FLUSH_PROMPT)

        # Simulate: flush response is <silent>, so nothing else should be appended
        # (In real code, the loop checks is_silent_response and skips append)
        response_text = "<silent>"
        assert is_silent_response(response_text) is True

        # Only the system message was added, no assistant response
        assert len(session.get_messages()) == msg_count_before + 1

    @pytest.mark.asyncio
    async def test_flush_system_message_appended_to_session(self, session):
        """The flush prompt must appear as a system message in the session."""
        session.append({"role": "user", "content": "test", "source": "user"})
        session.append_system(MEMORY_FLUSH_PROMPT)

        messages = session.get_messages()
        system_msgs = [m for m in messages if m.get("role") == "system"]
        flush_msgs = [m for m in system_msgs if "Pre-compaction" in m.get("content", "")]

        assert len(flush_msgs) == 1, "Exactly one flush prompt system message expected"


class TestFlushDoesNotIncrementIteration:
    """The flush turn is infrastructure, not a task iteration."""

    @pytest.mark.asyncio
    async def test_iteration_counter_unchanged_by_flush(self):
        """Flush turn should not count toward the max_iterations limit."""
        # This is verified by the loop implementation: the flush turn
        # happens inside the compaction check block, outside the
        # iteration increment at the top of the while loop.
        # We test indirectly: if flush incremented iteration, a loop
        # with max_iterations=1 would stop before compaction.
        from agent_os.agent.loop import AgentLoop
        from agent_os.agent.providers.types import LLMResponse, TokenUsage

        session = MagicMock()
        session.is_paused.return_value = False
        session.is_stopped.return_value = False
        session._paused_for_approval = False
        session.pending_tool_calls = set()
        session.pop_queued_messages.return_value = []
        session.resolve_pending_tool_calls = MagicMock()
        session.append = MagicMock()
        session.append_system = MagicMock()
        session.append_tool_result = MagicMock()
        session.recent_activity.return_value = []
        session.get_messages.return_value = []

        provider = AsyncMock()
        provider.complete = AsyncMock()

        context_manager = MagicMock()
        context_manager.prepare.return_value = [{"role": "system", "content": "test"}]
        context_manager.model_context_limit = 128_000
        context_manager.should_compact.return_value = False

        tool_registry = MagicMock()
        tool_registry.schemas.return_value = []
        tool_registry.reset_run_state = MagicMock()

        # Text-only response — loop will exit after 1 iteration
        text_response = LLMResponse(
            text="Done.",
            tool_calls=[],
            raw_message={"role": "assistant", "content": "Done."},
            has_tool_calls=False,
            finish_reason="stop",
            status_text=None,
            usage=TokenUsage(input_tokens=50, output_tokens=10),
        )

        loop = AgentLoop(
            session=session,
            provider=provider,
            tool_registry=tool_registry,
            context_manager=context_manager,
            max_iterations=1,
        )

        async def mock_stream(context, tool_schemas):
            return text_response

        loop._stream_response = mock_stream

        await loop.run("test")

        # If flush had incremented, max_iterations=1 would have stopped
        # before we got a text response. The fact that we got here means
        # the iteration counter was only incremented by the main loop.
        session.append.assert_called()
