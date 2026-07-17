# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the oversized-line-tolerant JSONL stream reader.

Regression scope: asyncio's StreamReader.readline() raises
``ValueError('Separator is not found, and chunk exceed the limit')`` on any
line longer than the reader's limit (default 64 KiB) AND destroys the
buffered bytes. The codex app-server emits single-line JSON-RPC events
carrying full command output — a broad ``rg`` over a repo routinely crosses
64 KiB, which killed CodexTransport._read_loop mid-turn (incidents
2026-07-02, 2026-07-17 13:21, 2026-07-17 15:40). ``read_jsonl_line`` must
survive lines of any size up to an explicit ceiling.
"""

import asyncio

import pytest

from agent_os.agent.transports.jsonl_stream import (
    LineTooLongError,
    read_jsonl_line,
)


def _reader(limit: int = 64) -> asyncio.StreamReader:
    # Tiny limit so the multi-chunk path is exercised without megabyte
    # payloads. Production readers use the asyncio default (64 KiB) or the
    # transport's own limit — the helper must not care.
    return asyncio.StreamReader(limit=limit)


class TestReadJsonlLine:
    @pytest.mark.asyncio
    async def test_returns_small_lines_unchanged(self):
        r = _reader()
        r.feed_data(b'{"a":1}\n{"b":2}\n')
        r.feed_eof()
        assert await read_jsonl_line(r) == b'{"a":1}\n'
        assert await read_jsonl_line(r) == b'{"b":2}\n'

    @pytest.mark.asyncio
    async def test_survives_line_longer_than_reader_limit(self):
        # THE regression: readline() would raise ValueError here.
        r = _reader(limit=64)
        big = b'{"output":"' + b"x" * 4096 + b'"}\n'
        r.feed_data(big + b'{"next":true}\n')
        r.feed_eof()
        assert await read_jsonl_line(r) == big
        # Stream stays aligned: the following line is intact.
        assert await read_jsonl_line(r) == b'{"next":true}\n'

    @pytest.mark.asyncio
    async def test_clean_eof_returns_empty(self):
        r = _reader()
        r.feed_eof()
        assert await read_jsonl_line(r) == b""

    @pytest.mark.asyncio
    async def test_eof_mid_line_returns_partial(self):
        r = _reader(limit=64)
        r.feed_data(b'{"truncated":"' + b"y" * 300)
        r.feed_eof()
        line = await read_jsonl_line(r)
        assert line.startswith(b'{"truncated":"')
        assert len(line) == len(b'{"truncated":"') + 300

    @pytest.mark.asyncio
    async def test_ceiling_raises_and_realigns_stream(self):
        # A line beyond max_line_bytes is discarded (not buffered into RAM),
        # the error is surfaced, and the NEXT line is still readable.
        r = _reader(limit=64)
        r.feed_data(b"z" * 1000 + b"\n" + b'{"after":1}\n')
        r.feed_eof()
        with pytest.raises(LineTooLongError):
            await read_jsonl_line(r, max_line_bytes=256)
        assert await read_jsonl_line(r, max_line_bytes=256) == b'{"after":1}\n'

    @pytest.mark.asyncio
    async def test_ceiling_exceeded_at_eof_raises(self):
        r = _reader(limit=64)
        r.feed_data(b"z" * 1000)  # no newline, ever
        r.feed_eof()
        with pytest.raises(LineTooLongError):
            await read_jsonl_line(r, max_line_bytes=256)
