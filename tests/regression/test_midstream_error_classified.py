# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""A stream that dies mid-flight must surface as LLMError, not a raw SDK error.

`_stream_openai` wrapped only the `chat.completions.create()` call in the
error classifier; the `async for chunk in response_iter` that follows it was
outside the try. A connection dropped *mid-stream* therefore propagated as a
raw `httpx.RemoteProtocolError`, which is not an `LLMError` — so it bypassed
the agent loop's retry/rotate logic AND its error-row writer. The session was
left holding the user's message with nothing after it: on a scheduled trigger
run that reads as "the trigger fired and the agent never answered", with no
error anywhere in the UI.

Observed on the live daemon twice (2026-07-29, 2026-08-09); the July run hung
~30 minutes before the drop. `_stream_anthropic` already guards its iteration
this way (openai_compat.py) — the OpenAI path just never did.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.agent.providers.types import ErrorCategory, LLMError


def _chunk(text: str):
    delta = MagicMock()
    delta.content = text
    delta.tool_calls = None
    delta.reasoning_content = None
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


def _provider_whose_stream_dies(exc: Exception, *, after: int = 1):
    """A provider whose stream yields `after` chunks and then raises."""
    provider = LLMProvider("m-1", "sk-x", "https://gw.example/v1", sdk="openai")

    class _DyingStream:
        def __aiter__(self):
            return self

        def __init__(self):
            self._sent = 0

        async def __anext__(self):
            if self._sent < after:
                self._sent += 1
                return _chunk("partial ")
            raise exc

    provider._openai_client = MagicMock()
    provider._openai_client.chat.completions.create = AsyncMock(
        return_value=_DyingStream()
    )
    return provider


@pytest.mark.asyncio
async def test_midstream_protocol_error_becomes_llm_error():
    dropped = httpx.RemoteProtocolError(
        "peer closed connection without sending complete message body "
        "(incomplete chunked read)"
    )
    provider = _provider_whose_stream_dies(dropped)

    with pytest.raises(LLMError) as excinfo:
        async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            pass

    # RETRY, not ABORT: a dropped connection is worth retrying, and that is
    # what routes it into the loop's retry/rotate path instead of killing
    # the run outright.
    assert excinfo.value.category == ErrorCategory.RETRY
    assert excinfo.value.__cause__ is dropped


@pytest.mark.asyncio
async def test_midstream_connection_reset_becomes_llm_error():
    provider = _provider_whose_stream_dies(ConnectionResetError("reset by peer"))

    with pytest.raises(LLMError):
        async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            pass


@pytest.mark.asyncio
async def test_midstream_llm_error_passes_through_unwrapped():
    """An LLMError raised inside the stream keeps its own classification —
    it must not be re-wrapped into a generic RETRY."""
    original = LLMError("bad key", status_code=401)
    provider = _provider_whose_stream_dies(original)

    with pytest.raises(LLMError) as excinfo:
        async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            pass

    assert excinfo.value is original
    assert excinfo.value.category == ErrorCategory.ABORT


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed_as_an_llm_error():
    """cancel_turn() cancels the in-flight stream task. That must stay a
    CancelledError so the loop's cancel branch runs — turning it into an
    LLMError would make a user-cancelled turn look like a provider failure."""
    import asyncio

    provider = _provider_whose_stream_dies(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            pass
