# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration of InlineThinkSplitter into OpenAICompatProvider.

For a model whose ReasoningInfo declares inline-think (field is None,
supported=True), the streaming AND non-streaming paths must move
``<think>…</think>`` out of ``content`` into ``reasoning_content``. For a
model with a named reasoning field (e.g. reasoning_content), content is left
untouched (no accidental stripping).
"""

from types import SimpleNamespace

import pytest

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.providers.types import StreamAccumulator
from agent_os.config.provider_registry import ReasoningInfo


def _chunk(content=None, finish_reason=None, usage=None, reasoning_content=None):
    delta = SimpleNamespace(
        content=content, reasoning_content=reasoning_content, tool_calls=None,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def __init__(self, chunks):
        self._chunks = chunks

    async def create(self, **kwargs):
        chunks = self._chunks

        async def _gen():
            for c in chunks:
                yield c

        return _gen()


class _FakeClient:
    def __init__(self, chunks):
        self.chat = SimpleNamespace(completions=_FakeCompletions(chunks))


def _usage():
    return SimpleNamespace(prompt_tokens=10, completion_tokens=5)


def _provider(reasoning):
    p = LLMProvider("test-model", "key", sdk="openai", reasoning=reasoning)
    return p


async def _drain(provider, chunks):
    provider._openai_client = _FakeClient(chunks)
    acc = StreamAccumulator()
    async for ch in provider.stream([{"role": "user", "content": "hi"}]):
        acc.add(ch)
    return acc.finalize()


INLINE = ReasoningInfo(supported=True, field=None, enable="model_only")
NAMED = ReasoningInfo(supported=True, field="reasoning_content", enable="model_only")


@pytest.mark.asyncio
async def test_inline_think_split_in_stream():
    chunks = [
        _chunk(content="<think>reason"),
        _chunk(content="ing</think>ans"),
        _chunk(content="wer", finish_reason="stop", usage=_usage()),
    ]
    resp = await _drain(_provider(INLINE), chunks)
    assert resp.text == "answer"
    assert "<think>" not in (resp.text or "")
    assert resp.raw_message.get("reasoning_content") == "reasoning"


@pytest.mark.asyncio
async def test_stream_tolerates_none_token_usage():
    # MiniMax's China endpoint (api.minimaxi.com) sometimes returns a usage
    # object with prompt_tokens / completion_tokens = None. The stream must not
    # crash (regression: TypeError "'>' not supported between instances of
    # 'NoneType' and 'int'" in _log_cache_audit killed the whole turn before any
    # row was appended). Tokens coerce to 0; the answer still streams through.
    usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None)
    chunks = [_chunk(content="<think>r</think>ans", finish_reason="stop", usage=usage)]
    resp = await _drain(_provider(INLINE), chunks)
    assert resp.text == "ans"
    assert resp.raw_message.get("reasoning_content") == "r"
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


@pytest.mark.asyncio
async def test_named_field_model_not_stripped():
    # A non-inline model that happens to emit literal <think> text must NOT be
    # stripped — the splitter is gated to inline-think models only.
    chunks = [
        _chunk(content="<think>kept</think>body", finish_reason="stop", usage=_usage()),
    ]
    resp = await _drain(_provider(NAMED), chunks)
    assert resp.text == "<think>kept</think>body"
    assert not resp.raw_message.get("reasoning_content")


@pytest.mark.asyncio
async def test_inline_think_non_streaming_complete():
    # Non-streaming complete() also separates inline think.
    msg = SimpleNamespace(
        content="<think>my reasoning</think>final answer",
        tool_calls=None,
        reasoning_content=None,
    )
    msg.model_dump = lambda: {
        "role": "assistant",
        "content": "<think>my reasoning</think>final answer",
    }
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    response = SimpleNamespace(choices=[choice], usage=_usage())

    class _C:
        async def create(self, **kwargs):
            return response

    prov = _provider(INLINE)
    prov._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_C()))
    resp = await prov.complete([{"role": "user", "content": "hi"}])
    assert resp.text == "final answer"
    assert resp.raw_message.get("reasoning_content") == "my reasoning"
    assert "<think>" not in resp.raw_message.get("content", "")
