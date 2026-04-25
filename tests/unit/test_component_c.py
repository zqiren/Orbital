# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for Component C — Provider (LLMProvider, types, StreamAccumulator).

Written from specs only (TDD). Covers all 12 acceptance criteria from
TASK-component-C-provider.md.
"""

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

from agent_os.agent.providers.types import (
    TokenUsage,
    StreamChunk,
    LLMResponse,
    StreamAccumulator,
    ContextOverflowError,
    LLMError,
)
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.providers.anthropic_adapter import (
    translate_messages_to_anthropic,
    translate_response_to_openai,
    translate_stream_event,
    StreamState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_chunk(*, content=None, tool_calls=None, finish_reason=None, usage=None):
    """Build a mock object that mimics an OpenAI streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


def _make_openai_response(*, content="hello", tool_calls=None, finish_reason="stop",
                          input_tokens=10, output_tokens=5, cache_read_tokens=0):
    """Build a mock object that mimics a non-streaming OpenAI completion response."""
    message = MagicMock()
    message.content = content
    # tool_calls on the message object
    message.tool_calls = tool_calls

    # Make message dict-convertible for raw_message capture
    message_dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message_dict["tool_calls"] = tool_calls
    message.model_dump = MagicMock(return_value=message_dict)

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    # Anthropic-style cache token attribute
    usage.cache_read_input_tokens = cache_read_tokens
    # Some providers may not have this attribute; we test both paths separately

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


async def _async_iter(items):
    """Turn a list into an async iterator (simulates streaming chunks)."""
    for item in items:
        yield item


# ===========================================================================
# AC-1: LLMProvider routing — sdk="openai" → OpenAI SDK, sdk="anthropic" → Anthropic SDK
# ===========================================================================

class TestProviderRouting:
    """AC-1: LLMProvider with sdk='openai' creates OpenAI SDK client.
    sdk='anthropic' creates Anthropic SDK client."""

    @patch("agent_os.agent.providers.openai_compat.openai")
    def test_base_url_set_creates_openai_client(self, mock_openai_mod):
        """When sdk='openai' (default), the provider should create an OpenAI
        AsyncOpenAI client with that base_url."""
        provider = LLMProvider(
            model="gpt-4", api_key="sk-test", base_url="https://my-proxy.example.com"
        )
        mock_openai_mod.AsyncOpenAI.assert_called_once_with(
            base_url="https://my-proxy.example.com", api_key="sk-test", timeout=None
        )
        assert provider.base_url == "https://my-proxy.example.com"

    @patch("agent_os.agent.providers.openai_compat.openai")
    def test_anthropic_sdk_creates_anthropic_client(self, mock_openai_mod):
        """When sdk='anthropic', the provider should create an Anthropic
        AsyncAnthropic client and NOT an OpenAI client."""
        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            provider = LLMProvider(model="claude-opus-4-6", api_key="sk-test", sdk="anthropic")
            mock_openai_mod.AsyncOpenAI.assert_not_called()
            mock_anthropic.assert_called_once_with(api_key="sk-test")
            assert provider.sdk == "anthropic"
            assert provider._openai_client is None

    @patch("agent_os.agent.providers.openai_compat.openai")
    def test_sdk_defaults_to_openai(self, mock_openai_mod):
        """LLMProvider without sdk param defaults to sdk='openai'."""
        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        assert provider.sdk == "openai"
        mock_openai_mod.AsyncOpenAI.assert_called_once()
        assert provider._anthropic_client is None


# ===========================================================================
# AC-2: stream() yields StreamChunk; final chunk has is_final + usage
# ===========================================================================

class TestStreamYieldsChunks:
    """AC-2: stream() yields StreamChunk objects. Final chunk has is_final=True
    and usage populated."""

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_stream_yields_stream_chunks_openai_route(self, mock_openai_mod):
        """OpenAI SDK route: stream yields StreamChunk objects and the last one
        has is_final=True with usage."""
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        # Build fake streaming chunks
        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 100
        usage_obj.completion_tokens = 20
        usage_obj.cache_read_input_tokens = 0

        chunks_data = [
            _make_openai_chunk(content="Hello"),
            _make_openai_chunk(content=" world"),
            _make_openai_chunk(content=None, finish_reason="stop", usage=usage_obj),
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks_data))

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        collected = []
        async for chunk in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            assert isinstance(chunk, StreamChunk)
            collected.append(chunk)

        assert len(collected) >= 2  # at least text chunks + final
        final = collected[-1]
        assert final.is_final is True
        assert final.usage is not None
        assert isinstance(final.usage, TokenUsage)

    @pytest.mark.asyncio
    async def test_stream_yields_stream_chunks_anthropic_route(self):
        """Anthropic SDK route: stream yields StreamChunk objects and the last one
        has is_final=True with usage."""
        # Build fake Anthropic streaming events
        events = []
        # message_start with usage
        msg_start = MagicMock()
        msg_start.type = "message_start"
        msg_usage = MagicMock()
        msg_usage.input_tokens = 50
        msg_usage.output_tokens = 0
        msg_usage.cache_read_input_tokens = 0
        msg_start.message = MagicMock()
        msg_start.message.usage = msg_usage
        events.append(msg_start)

        # content_block_start (text)
        cbs = MagicMock()
        cbs.type = "content_block_start"
        cbs.index = 0
        cbs.content_block = MagicMock()
        cbs.content_block.type = "text"
        events.append(cbs)

        # content_block_delta (text)
        cbd = MagicMock()
        cbd.type = "content_block_delta"
        cbd.index = 0
        cbd.delta = MagicMock()
        cbd.delta.type = "text_delta"
        cbd.delta.text = "Hi"
        events.append(cbd)

        # content_block_stop
        cbstop = MagicMock()
        cbstop.type = "content_block_stop"
        cbstop.index = 0
        events.append(cbstop)

        # message_delta
        md = MagicMock()
        md.type = "message_delta"
        md.delta = MagicMock()
        md.delta.stop_reason = "end_turn"
        md.usage = MagicMock()
        md.usage.output_tokens = 10
        events.append(md)

        # message_stop
        ms = MagicMock()
        ms.type = "message_stop"
        events.append(ms)

        with patch("anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_client = AsyncMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=_async_iter(events))

            provider = LLMProvider(model="claude-opus-4-6", api_key="sk-test", sdk="anthropic")
            collected = []
            async for chunk in provider.stream(messages=[{"role": "user", "content": "hi"}]):
                assert isinstance(chunk, StreamChunk)
                collected.append(chunk)

            assert len(collected) >= 2  # at least text chunk + final
            final = collected[-1]
            assert final.is_final is True
            assert final.usage is not None
            assert isinstance(final.usage, TokenUsage)


# ===========================================================================
# AC-3: StreamAccumulator — 5 text chunks → finalize() produces joined text
# ===========================================================================

class TestStreamAccumulatorText:
    """AC-3: StreamAccumulator: feed 5 text chunks → finalize() produces
    LLMResponse with joined text."""

    def test_five_text_chunks_joined(self):
        acc = StreamAccumulator()
        parts = ["The ", "quick ", "brown ", "fox ", "jumps"]
        for p in parts:
            acc.add(StreamChunk(text=p))
        # Feed final chunk with usage
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(input_tokens=10, output_tokens=5)))

        result = acc.finalize()
        assert isinstance(result, LLMResponse)
        assert result.text == "The quick brown fox jumps"

    def test_finalize_empty_text_is_none(self):
        """When no text chunks were fed, text should be None."""
        acc = StreamAccumulator()
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(0, 0)))
        result = acc.finalize()
        assert result.text is None

    def test_finalize_returns_usage(self):
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="x"))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(100, 50, cache_read_tokens=25)))
        result = acc.finalize()
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_read_tokens == 25


# ===========================================================================
# AC-4: StreamAccumulator — tool_call deltas across chunks → assembled
# ===========================================================================

class TestStreamAccumulatorToolCalls:
    """AC-4: StreamAccumulator: feed tool_call deltas across chunks →
    finalize() assembles complete tool_calls."""

    def test_tool_call_deltas_assembled(self):
        """Tool call fragments spread across multiple chunks should be
        assembled into a complete tool_call entry."""
        acc = StreamAccumulator()

        # Chunk 1: start of tool call
        delta1 = MagicMock()
        delta1.index = 0
        delta1.id = "call_abc"
        delta1.type = "function"
        delta1.function = MagicMock()
        delta1.function.name = "read"
        delta1.function.arguments = '{"file'

        acc.add(StreamChunk(tool_calls_delta=[delta1]))

        # Chunk 2: continuation of arguments
        delta2 = MagicMock()
        delta2.index = 0
        delta2.id = None
        delta2.type = None
        delta2.function = MagicMock()
        delta2.function.name = None
        delta2.function.arguments = '": "test.py"}'

        acc.add(StreamChunk(tool_calls_delta=[delta2]))

        # Final chunk with usage
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 20)))

        result = acc.finalize()
        assert result.has_tool_calls is True
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        # The assembled tool call should have the complete function arguments
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "read"
        assert '{"file": "test.py"}' in tc["function"]["arguments"]

    def test_multiple_tool_calls_assembled(self):
        """Multiple tool calls (different indices) should each be assembled."""
        acc = StreamAccumulator()

        # Tool call 0
        d0 = MagicMock()
        d0.index = 0
        d0.id = "call_1"
        d0.type = "function"
        d0.function = MagicMock()
        d0.function.name = "read"
        d0.function.arguments = '{"file": "a.py"}'

        # Tool call 1
        d1 = MagicMock()
        d1.index = 1
        d1.id = "call_2"
        d1.type = "function"
        d1.function = MagicMock()
        d1.function.name = "write"
        d1.function.arguments = '{"file": "b.py", "content": "hi"}'

        acc.add(StreamChunk(tool_calls_delta=[d0, d1]))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 20)))

        result = acc.finalize()
        assert result.has_tool_calls is True
        assert len(result.tool_calls) == 2


# ===========================================================================
# AC-5: complete() returns LLMResponse with raw_message preserved
# ===========================================================================

class TestCompleteRawMessage:
    """AC-5: complete() returns LLMResponse with raw_message preserved
    as-is from API."""

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_complete_preserves_raw_message(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        raw_msg_dict = {"role": "assistant", "content": "Summary of conversation."}
        resp = _make_openai_response(content="Summary of conversation.")
        resp.choices[0].message.model_dump.return_value = raw_msg_dict

        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        result = await provider.complete(messages=[{"role": "user", "content": "summarize"}])

        assert isinstance(result, LLMResponse)
        assert result.raw_message == raw_msg_dict

    @pytest.mark.asyncio
    async def test_complete_anthropic_returns_openai_format_raw_message(self):
        """Anthropic SDK route: complete() returns LLMResponse with raw_message
        in OpenAI format (translated from Anthropic response)."""
        # Build fake Anthropic response
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Done."

        mock_response = MagicMock()
        mock_response.content = [text_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.usage.cache_read_input_tokens = 0

        with patch("anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_client = AsyncMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=mock_response)

            provider = LLMProvider(model="claude-opus-4-6", api_key="sk-test", sdk="anthropic")
            result = await provider.complete(messages=[{"role": "user", "content": "hi"}])

            assert isinstance(result, LLMResponse)
            assert result.raw_message == {"role": "assistant", "content": "Done."}
            assert result.text == "Done."
            assert result.finish_reason == "stop"


# ===========================================================================
# AC-6: Context overflow → ContextOverflowError, not LLMError
# ===========================================================================

class TestContextOverflow:
    """AC-6: Context overflow (mock 400 response) → raises
    ContextOverflowError, not LLMError."""

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_400_context_length_raises_context_overflow_openai(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        # Simulate OpenAI 400 error with context_length message
        error = Exception("context_length_exceeded")
        error.status_code = 400
        error.message = "This model's maximum context length is 8192 tokens"
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")

        with pytest.raises(ContextOverflowError):
            async for _ in provider.stream(messages=[{"role": "user", "content": "x" * 100000}]):
                pass

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_400_context_overflow_not_llm_error(self, mock_openai_mod):
        """ContextOverflowError must NOT be caught as a generic LLMError."""
        assert not issubclass(ContextOverflowError, LLMError)

    @pytest.mark.asyncio
    async def test_400_context_length_raises_context_overflow_anthropic(self):
        """Anthropic SDK route: 400 context length error raises ContextOverflowError."""
        import anthropic as anthropic_mod
        # Create a realistic APIStatusError
        error = anthropic_mod.APIStatusError.__new__(anthropic_mod.APIStatusError)
        error.status_code = 400
        error.message = "maximum context length exceeded"

        with patch("anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_client = AsyncMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(side_effect=error)

            provider = LLMProvider(model="claude-opus-4-6", api_key="sk-test", sdk="anthropic")

            with pytest.raises(ContextOverflowError):
                async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
                    pass

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_complete_context_overflow(self, mock_openai_mod):
        """complete() also raises ContextOverflowError on 400 context length."""
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        error = Exception("context_length_exceeded")
        error.status_code = 400
        error.message = "context length exceeded"
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        with pytest.raises(ContextOverflowError):
            await provider.complete(messages=[{"role": "user", "content": "hi"}])


# ===========================================================================
# AC-7: Rate limit (429) → LLMError with status_code=429
# ===========================================================================

class TestRateLimitError:
    """AC-7: Rate limit (mock 429 response) → raises LLMError with
    status_code=429."""

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_429_raises_llm_error_openai(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        error = Exception("Rate limit exceeded")
        error.status_code = 429
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        with pytest.raises(LLMError) as exc_info:
            async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
                pass
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_429_raises_llm_error_anthropic(self):
        """Anthropic SDK route: 429 rate limit raises LLMError with status_code=429."""
        import anthropic as anthropic_mod
        error = anthropic_mod.APIStatusError.__new__(anthropic_mod.APIStatusError)
        error.status_code = 429
        error.message = "Rate limit exceeded"

        with patch("anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_client = AsyncMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(side_effect=error)

            provider = LLMProvider(model="claude-opus-4-6", api_key="sk-test", sdk="anthropic")
            with pytest.raises(LLMError) as exc_info:
                async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
                    pass
            assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_429_complete_raises_llm_error(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        error = Exception("Rate limit exceeded")
        error.status_code = 429
        mock_client.chat.completions.create = AsyncMock(side_effect=error)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        with pytest.raises(LLMError) as exc_info:
            await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429


# ===========================================================================
# AC-8: Timeout → LLMError with descriptive message
# ===========================================================================

class TestTimeoutError:
    """AC-8: Timeout → raises LLMError with descriptive message."""

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_timeout_raises_llm_error_openai(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        mock_client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        with pytest.raises(LLMError) as exc_info:
            async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
                pass
        assert "timed out" in exc_info.value.message.lower() or "timeout" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_error_anthropic(self):
        """Anthropic SDK route: timeout raises LLMError with descriptive message."""
        with patch("anthropic.AsyncAnthropic") as mock_anthropic_cls:
            mock_client = AsyncMock()
            mock_anthropic_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(
                side_effect=asyncio.TimeoutError("Request timed out")
            )

            provider = LLMProvider(model="claude-opus-4-6", api_key="sk-test", sdk="anthropic")
            with pytest.raises(LLMError) as exc_info:
                async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
                    pass
            assert "timed out" in exc_info.value.message.lower() or "timeout" in exc_info.value.message.lower()

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_timeout_complete_raises_llm_error(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        mock_client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        with pytest.raises(LLMError) as exc_info:
            await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert "timed out" in exc_info.value.message.lower() or "timeout" in exc_info.value.message.lower()


# ===========================================================================
# AC-9: Text with [STATUS: Building project] → status_text = "Building project"
# ===========================================================================

class TestStatusExtraction:
    """AC-9: Text containing "[STATUS: Building project]" → status_text =
    "Building project"."""

    def test_status_extracted_from_accumulator(self):
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="Working on it. [STATUS: Building project] Done."))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5)))
        result = acc.finalize()
        assert result.status_text == "Building project"

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_status_extracted_from_complete(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        content = "Processing your request. [STATUS: Building project] All done."
        resp = _make_openai_response(content=content)
        resp.choices[0].message.model_dump.return_value = {
            "role": "assistant", "content": content
        }
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        result = await provider.complete(messages=[{"role": "user", "content": "go"}])
        assert result.status_text == "Building project"

    def test_status_with_extra_whitespace(self):
        """[STATUS:  Running tests  ] → 'Running tests' (trimmed)."""
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="[STATUS:  Running tests  ]"))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(0, 0)))
        result = acc.finalize()
        assert result.status_text == "Running tests"


# ===========================================================================
# AC-10: Text with no [STATUS: ...] → status_text = None
# ===========================================================================

class TestNoStatusExtraction:
    """AC-10: Text with no [STATUS: ...] → status_text = None."""

    def test_no_status_marker_returns_none(self):
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="Just a normal response without any status."))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5)))
        result = acc.finalize()
        assert result.status_text is None

    def test_none_text_returns_none(self):
        """When there is no text at all, status_text is None."""
        acc = StreamAccumulator()
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(0, 0)))
        result = acc.finalize()
        assert result.status_text is None

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_complete_no_status_returns_none(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        content = "Here is your answer."
        resp = _make_openai_response(content=content)
        resp.choices[0].message.model_dump.return_value = {
            "role": "assistant", "content": content
        }
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.status_text is None


# ===========================================================================
# AC-11: raw_message contains tool_calls in original API format (not normalized)
# ===========================================================================

class TestRawMessageToolCalls:
    """AC-11: raw_message in LLMResponse contains tool_calls in original API
    format (not normalized)."""

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_raw_message_has_original_tool_calls(self, mock_openai_mod):
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        raw_tool_calls = [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "read", "arguments": '{"file": "test.py"}'},
            }
        ]
        raw_msg_dict = {
            "role": "assistant",
            "content": None,
            "tool_calls": raw_tool_calls,
        }

        resp = _make_openai_response(content=None, tool_calls=raw_tool_calls,
                                     finish_reason="tool_calls")
        resp.choices[0].message.model_dump.return_value = raw_msg_dict

        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        result = await provider.complete(messages=[{"role": "user", "content": "read file"}])

        # raw_message should contain the original tool_calls, not a normalized version
        assert "tool_calls" in result.raw_message
        assert result.raw_message["tool_calls"] == raw_tool_calls
        # tool_calls on LLMResponse are also raw (not normalized)
        assert result.tool_calls == raw_tool_calls

    def test_accumulator_finalize_raw_message_has_tool_calls(self):
        """StreamAccumulator.finalize() should include tool_calls in raw_message."""
        acc = StreamAccumulator()

        delta = MagicMock()
        delta.index = 0
        delta.id = "call_abc"
        delta.type = "function"
        delta.function = MagicMock()
        delta.function.name = "shell"
        delta.function.arguments = '{"command": "ls"}'

        acc.add(StreamChunk(tool_calls_delta=[delta]))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 20)))

        result = acc.finalize()
        assert "tool_calls" in result.raw_message
        assert len(result.raw_message["tool_calls"]) == 1
        assert result.raw_message["tool_calls"][0]["id"] == "call_abc"


# ===========================================================================
# AC-12: TokenUsage includes cache_read_tokens (Anthropic), 0 otherwise
# ===========================================================================

class TestTokenUsageCacheReadTokens:
    """AC-12: TokenUsage includes cache_read_tokens when Anthropic returns
    them (0 otherwise)."""

    def test_token_usage_default_cache_read_is_zero(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.cache_read_tokens == 0

    def test_token_usage_with_cache_read_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=30)
        assert usage.cache_read_tokens == 30

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_complete_extracts_cache_read_tokens(self, mock_openai_mod):
        """When the API response includes cache_read_input_tokens (Anthropic),
        it should be captured in TokenUsage.cache_read_tokens."""
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        resp = _make_openai_response(
            content="cached response",
            input_tokens=200, output_tokens=50, cache_read_tokens=150
        )
        resp.choices[0].message.model_dump.return_value = {
            "role": "assistant", "content": "cached response"
        }
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.usage.cache_read_tokens == 150

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_complete_no_cache_read_tokens_defaults_zero(self, mock_openai_mod):
        """When cache_read_input_tokens is not present, cache_read_tokens
        should default to 0."""
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        resp = _make_openai_response(content="no cache", cache_read_tokens=0)
        # Remove the cache attribute to simulate a provider that doesn't supply it
        del resp.usage.cache_read_input_tokens
        resp.choices[0].message.model_dump.return_value = {
            "role": "assistant", "content": "no cache"
        }
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        result = await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert result.usage.cache_read_tokens == 0

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_stream_extracts_cache_read_tokens(self, mock_openai_mod):
        """Streaming final chunk should also capture cache_read_tokens."""
        mock_client = AsyncMock()
        mock_openai_mod.AsyncOpenAI.return_value = mock_client

        usage_obj = MagicMock()
        usage_obj.prompt_tokens = 200
        usage_obj.completion_tokens = 30
        usage_obj.cache_read_input_tokens = 100

        chunks = [
            _make_openai_chunk(content="hello"),
            _make_openai_chunk(content=None, finish_reason="stop", usage=usage_obj),
        ]

        mock_client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))

        provider = LLMProvider(model="gpt-4", api_key="sk-test", base_url="https://proxy")
        collected = []
        async for chunk in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            collected.append(chunk)

        final = collected[-1]
        assert final.usage.cache_read_tokens == 100


# ===========================================================================
# Anthropic Adapter — Outbound (translate_messages_to_anthropic)
# ===========================================================================

class TestAnthropicAdapterOutbound:
    """Test translate_messages_to_anthropic."""

    def test_system_message_extraction(self):
        """System messages extracted as separate parameter."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = translate_messages_to_anthropic(messages)
        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self):
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "Hi"},
        ]
        result = translate_messages_to_anthropic(messages)
        assert result["system"] == "System 1\nSystem 2"

    def test_tool_schema_translation(self):
        tools = [{"type": "function", "function": {"name": "read", "description": "Read file", "parameters": {"type": "object"}}}]
        result = translate_messages_to_anthropic([], tools)
        assert result["tools"][0]["name"] == "read"
        assert result["tools"][0]["input_schema"] == {"type": "object"}

    def test_tool_result_messages_merged(self):
        """Consecutive tool results merged into single user message."""
        messages = [
            {"role": "user", "content": "do both"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "write", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result1", "tool_call_id": "c1"},
            {"role": "tool", "content": "result2", "tool_call_id": "c2"},
        ]
        result = translate_messages_to_anthropic(messages)
        # The two tool results should be merged into ONE user message
        user_msgs = [m for m in result["messages"] if m["role"] == "user"]
        tool_result_msg = [m for m in user_msgs if isinstance(m.get("content"), list)]
        assert len(tool_result_msg) == 1
        assert len(tool_result_msg[0]["content"]) == 2

    def test_assistant_tool_calls_to_content_blocks(self):
        msg = {
            "role": "assistant",
            "content": "Let me read that.",
            "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "read", "arguments": '{"path": "test.py"}'}}],
        }
        result = translate_messages_to_anthropic([msg])
        assistant_msg = result["messages"][0]
        assert assistant_msg["role"] == "assistant"
        blocks = assistant_msg["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["input"] == {"path": "test.py"}


# ===========================================================================
# Anthropic Adapter — Inbound (translate_response_to_openai)
# ===========================================================================

class TestAnthropicAdapterInbound:
    """Test translate_response_to_openai."""

    def test_text_response(self):
        """Simple text response translated to OpenAI format."""
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello"
        response.content = [text_block]
        response.stop_reason = "end_turn"
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_read_input_tokens = 0
        response.usage = usage

        result = translate_response_to_openai(response)
        assert result["text"] == "Hello"
        assert result["finish_reason"] == "stop"
        assert result["has_tool_calls"] is False

    def test_tool_use_response(self):
        response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Reading..."
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_abc"
        tool_block.name = "read"
        tool_block.input = {"path": "test.py"}
        response.content = [text_block, tool_block]
        response.stop_reason = "tool_use"
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 50
        usage.cache_read_input_tokens = 30
        response.usage = usage

        result = translate_response_to_openai(response)
        assert result["has_tool_calls"] is True
        assert result["finish_reason"] == "tool_calls"
        tc = result["tool_calls"][0]
        assert tc["id"] == "toolu_abc"
        assert tc["function"]["name"] == "read"
        assert json.loads(tc["function"]["arguments"]) == {"path": "test.py"}
        assert result["usage"].cache_read_tokens == 30

    def test_finish_reason_mapping(self):
        for anth_reason, oai_reason in [("end_turn", "stop"), ("tool_use", "tool_calls"), ("max_tokens", "length")]:
            response = MagicMock()
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "x"
            response.content = [text_block]
            response.stop_reason = anth_reason
            response.usage = MagicMock(input_tokens=0, output_tokens=0, cache_read_input_tokens=0)
            result = translate_response_to_openai(response)
            assert result["finish_reason"] == oai_reason


# ===========================================================================
# Anthropic Adapter — Streaming (translate_stream_event + StreamState)
# ===========================================================================

class TestAnthropicAdapterStreaming:
    """Test translate_stream_event + StreamState."""

    def test_text_delta_produces_chunk(self):
        state = StreamState()

        # message_start
        event = MagicMock(type="message_start")
        event.message = MagicMock()
        event.message.usage = MagicMock(input_tokens=50, output_tokens=0, cache_read_input_tokens=10)
        assert translate_stream_event(event, state) is None

        # content_block_start
        event = MagicMock(type="content_block_start", index=0)
        event.content_block = MagicMock(type="text")
        assert translate_stream_event(event, state) is None

        # content_block_delta with text
        event = MagicMock(type="content_block_delta", index=0)
        event.delta = MagicMock(type="text_delta", text="Hello")
        chunk = translate_stream_event(event, state)
        assert chunk is not None
        assert chunk.text == "Hello"

    def test_tool_use_delta_produces_tool_calls_delta(self):
        state = StreamState()

        # content_block_start for tool_use
        content_block = MagicMock(type="tool_use", id="toolu_abc")
        content_block.name = "read"  # name is special on MagicMock, set explicitly
        event = MagicMock(type="content_block_start", index=0)
        event.content_block = content_block
        translate_stream_event(event, state)

        # First delta — should include id and name
        event = MagicMock(type="content_block_delta", index=0)
        event.delta = MagicMock(type="input_json_delta", partial_json='{"path":')
        chunk = translate_stream_event(event, state)
        assert chunk is not None
        assert len(chunk.tool_calls_delta) == 1
        tc = chunk.tool_calls_delta[0]
        assert tc["id"] == "toolu_abc"
        assert tc["function"]["name"] == "read"

        # Second delta — should NOT include id and name
        event = MagicMock(type="content_block_delta", index=0)
        event.delta = MagicMock(type="input_json_delta", partial_json='"test.py"}')
        chunk = translate_stream_event(event, state)
        tc2 = chunk.tool_calls_delta[0]
        assert "id" not in tc2
        assert "name" not in tc2.get("function", {})

    def test_message_stop_produces_final_chunk(self):
        state = StreamState()
        state.input_tokens = 100
        state.output_tokens = 50
        state.cache_read_tokens = 25

        event = MagicMock(type="message_stop")
        chunk = translate_stream_event(event, state)
        assert chunk is not None
        assert chunk.is_final is True
        assert chunk.usage.input_tokens == 100
        assert chunk.usage.output_tokens == 50
        assert chunk.usage.cache_read_tokens == 25


# ===========================================================================
# Anthropic Errors — via LLMProvider error classification
# ===========================================================================

class TestAnthropicErrors:

    @pytest.mark.asyncio
    @patch("agent_os.agent.providers.openai_compat.openai")
    async def test_anthropic_rate_limit(self, mock_openai_mod):
        """Anthropic 429 -> LLMError(429)"""
        import anthropic
        provider = LLMProvider(model="claude-sonnet-4-5-20250929", api_key="sk-test", sdk="anthropic")
        # Mock the anthropic client to raise rate limit
        error = anthropic.RateLimitError(message="rate limited", response=MagicMock(status_code=429), body=None)
        provider._anthropic_client = AsyncMock()
        provider._anthropic_client.messages.create = AsyncMock(side_effect=error)
        with pytest.raises(LLMError) as exc_info:
            await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_anthropic_auth_error(self):
        import anthropic
        provider = LLMProvider(model="claude-sonnet-4-5-20250929", api_key="bad-key", sdk="anthropic")
        error = anthropic.AuthenticationError(message="invalid api key", response=MagicMock(status_code=401), body=None)
        provider._anthropic_client = AsyncMock()
        provider._anthropic_client.messages.create = AsyncMock(side_effect=error)
        with pytest.raises(LLMError) as exc_info:
            await provider.complete(messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 401


# ===========================================================================
# Edge cases and LLMError structure
# ===========================================================================

class TestLLMErrorStructure:
    """Verify LLMError has the expected attributes."""

    def test_llm_error_has_status_code(self):
        err = LLMError("Something went wrong", status_code=500)
        assert err.status_code == 500
        assert err.message == "Something went wrong"
        assert str(err) == "Something went wrong"

    def test_llm_error_status_code_none(self):
        err = LLMError("Connection failed")
        assert err.status_code is None
        assert err.message == "Connection failed"

    def test_context_overflow_is_not_llm_error(self):
        """ContextOverflowError should be a separate exception, not a subclass
        of LLMError, so callers can distinguish them."""
        assert not issubclass(ContextOverflowError, LLMError)


class TestStreamChunkDataclass:
    """Verify StreamChunk defaults and structure."""

    def test_defaults(self):
        chunk = StreamChunk()
        assert chunk.text == ""
        assert chunk.tool_calls_delta == []
        assert chunk.is_final is False
        assert chunk.usage is None

    def test_final_chunk_with_usage(self):
        usage = TokenUsage(input_tokens=10, output_tokens=5, cache_read_tokens=2)
        chunk = StreamChunk(is_final=True, usage=usage)
        assert chunk.is_final is True
        assert chunk.usage.input_tokens == 10


class TestLLMResponseDataclass:
    """Verify LLMResponse structure."""

    def test_has_tool_calls_flag(self):
        resp = LLMResponse(
            raw_message={"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            text=None,
            tool_calls=[{"id": "1"}],
            has_tool_calls=True,
            finish_reason="tool_calls",
            status_text=None,
            usage=TokenUsage(10, 5),
        )
        assert resp.has_tool_calls is True
        assert resp.finish_reason == "tool_calls"

    def test_no_tool_calls(self):
        resp = LLMResponse(
            raw_message={"role": "assistant", "content": "hello"},
            text="hello",
            tool_calls=[],
            has_tool_calls=False,
            finish_reason="stop",
            status_text=None,
            usage=TokenUsage(10, 5),
        )
        assert resp.has_tool_calls is False
        assert resp.finish_reason == "stop"
