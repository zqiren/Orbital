# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared helpers for the test suite."""

from __future__ import annotations


def streamable(provider):
    """Give a ``complete()``-style mock provider a ``stream()``.

    The session-end merge streams rather than issuing one non-streaming
    request — a 15-minute idle connection does not reliably survive, see
    ``workspace_files._stream_merge_text``. Tests whose subject is the
    routine's *behaviour* (which files it writes, OCC aborts, archiving)
    should not have to restate that transport, so this replays whatever the
    existing ``complete`` mock is configured to return, as chunks. Exceptions
    configured via ``complete.side_effect`` propagate unchanged.

    Tests whose subject IS the transport build their streams directly — see
    ``test_merge_compression_and_streaming.py``.
    """
    from agent_os.agent.providers.types import StreamChunk, TokenUsage

    async def _stream(messages, tools=None, **kwargs):
        resp = await provider.complete(messages)
        text = getattr(resp, "text", "") or ""
        for i in range(0, len(text), 256):
            yield StreamChunk(text=text[i:i + 256])
        yield StreamChunk(
            is_final=True, usage=TokenUsage(input_tokens=1, output_tokens=1)
        )

    provider.stream = _stream
    return provider
