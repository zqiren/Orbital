# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The OpenAI-compatible request must OMIT `tools`, not send it as null.

`tools=tools or None` serializes to `"tools": null`, which lenient gateways
ignore and strict ones reject outright:

    400 invalid_request_error — "Input should be a valid list,
                                 field: 'tools', value: None"

Observed from GLM-5.2 through OpenCode Go on the Test Connection path, which
sends no tools. The Anthropic path already gets this right (it only sets the
key when translated["tools"] is truthy); the OpenAI path did not.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.providers.openai_compat import LLMProvider


def _provider_with_captured_client():
    provider = LLMProvider("m-1", "sk-x", "https://gw.example/v1", sdk="openai")
    captured: dict = {}

    async def create(**kwargs):
        captured.update(kwargs)
        msg = MagicMock()
        msg.content = "ok"
        msg.tool_calls = None
        msg.reasoning_content = None
        choice = MagicMock()
        choice.message = msg
        choice.finish_reason = "stop"
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    provider._openai_client = MagicMock()
    provider._openai_client.chat.completions.create = AsyncMock(side_effect=create)
    return provider, captured


@pytest.mark.asyncio
async def test_complete_omits_tools_when_there_are_none():
    provider, captured = _provider_with_captured_client()
    await provider.complete(messages=[{"role": "user", "content": "hi"}])
    assert "tools" not in captured, f"sent tools={captured.get('tools')!r}"


@pytest.mark.asyncio
async def test_complete_still_sends_tools_when_present():
    provider, captured = _provider_with_captured_client()
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    await provider.complete(messages=[{"role": "user", "content": "hi"}], tools=tools)
    assert captured["tools"] == tools


@pytest.mark.asyncio
async def test_stream_omits_tools_when_there_are_none():
    provider, captured = _provider_with_captured_client()

    async def empty_stream(**kwargs):
        captured.update(kwargs)

        class _It:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        return _It()

    provider._openai_client.chat.completions.create = AsyncMock(side_effect=empty_stream)
    async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
        pass
    assert "tools" not in captured, f"sent tools={captured.get('tools')!r}"
