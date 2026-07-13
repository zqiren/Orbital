# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: real finish_reason passthrough in the streaming layer.

The streaming accumulator used to SYNTHESIZE finish_reason
("tool_calls" if tool_calls else "stop"), discarding the provider's own
signal. That destroyed the one in-band bit that distinguishes a completed
response from one the server truncated mid-generation (the MiniMax-M3
reasoning-only silent-stall, BACKLOG-m3-reasoning-only-silent-stall.md).

Contract pinned here:
- StreamChunk carries an optional finish_reason.
- StreamAccumulator.finalize() prefers the real value when any chunk
  delivered one, and falls back to the synthesized value otherwise
  (providers that never report one keep today's behavior).
- The Anthropic streaming translator maps stop_reason onto the final
  chunk using the existing _STOP_REASON_MAP vocabulary.
"""

from types import SimpleNamespace

from agent_os.agent.providers.types import (
    StreamAccumulator,
    StreamChunk,
    TokenUsage,
)
from agent_os.agent.providers.anthropic_adapter import (
    StreamState,
    translate_stream_event,
)


class TestAccumulatorFinishReason:
    def test_passes_through_real_finish_reason_on_truncation(self):
        """A stream whose final chunk reports length must finalize with
        finish_reason='length' — even though there are no tool calls (the
        old synthesis would have reported 'stop')."""
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="\n\n", reasoning_content="cut mid-sent"))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5),
                            finish_reason="length"))
        response = acc.finalize()
        assert response.finish_reason == "length"

    def test_passes_through_real_stop(self):
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="hello"))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5),
                            finish_reason="stop"))
        assert acc.finalize().finish_reason == "stop"

    def test_finish_reason_on_non_final_chunk_is_kept(self):
        """Standard OpenAI streams deliver finish_reason on a choices chunk
        and usage on a separate final chunk — the value must survive."""
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="hi", finish_reason="length"))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5)))
        assert acc.finalize().finish_reason == "length"

    def test_synthesizes_stop_when_provider_reports_nothing(self):
        """Fallback contract: providers that never report finish_reason keep
        the historical synthesized value."""
        acc = StreamAccumulator()
        acc.add(StreamChunk(text="hello"))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5)))
        assert acc.finalize().finish_reason == "stop"

    def test_synthesizes_tool_calls_when_provider_reports_nothing(self):
        acc = StreamAccumulator()
        acc.add(StreamChunk(tool_calls_delta=[{
            "index": 0, "id": "c1", "type": "function",
            "function": {"name": "shell", "arguments": "{}"},
        }]))
        acc.add(StreamChunk(is_final=True, usage=TokenUsage(10, 5)))
        assert acc.finalize().finish_reason == "tool_calls"


class TestAnthropicStreamStopReason:
    def _drive(self, stop_reason: str) -> StreamChunk:
        state = StreamState()
        translate_stream_event(
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason=stop_reason),
                usage=SimpleNamespace(output_tokens=5),
            ),
            state,
        )
        final = translate_stream_event(SimpleNamespace(type="message_stop"), state)
        assert final is not None and final.is_final
        return final

    def test_max_tokens_maps_to_length(self):
        assert self._drive("max_tokens").finish_reason == "length"

    def test_end_turn_maps_to_stop(self):
        assert self._drive("end_turn").finish_reason == "stop"
