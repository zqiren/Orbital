# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Oversized-line-tolerant reading for newline-delimited JSON streams.

``asyncio.StreamReader.readline()`` raises ``ValueError`` on any line longer
than the reader's limit (64 KiB unless the subprocess was created with an
explicit ``limit=``) — and destroys the buffered bytes as it goes, so the
error is not recoverable at the call site. CLI agent protocols (codex
app-server, ACP) put full command output on single JSON-RPC lines, which
crosses 64 KiB routinely. ``read_jsonl_line`` accumulates across
``LimitOverrunError`` so a line of any size up to ``max_line_bytes`` comes
back intact; beyond the ceiling the line is discarded without buffering it
into RAM and ``LineTooLongError`` is raised with the stream left aligned on
the next line. The same chunk-accumulation pattern is what the ``acp``
package's ``Connection._read_line`` ships for the cursor transport.
"""

from __future__ import annotations

import asyncio

# Generous ceiling: a real event line (full rg output over a large repo) is
# hundreds of KiB; anything past this is a protocol violation, not data.
DEFAULT_MAX_LINE_BYTES = 64 * 1024 * 1024


class LineTooLongError(Exception):
    """A single JSONL line exceeded ``max_line_bytes``.

    The oversized line has been consumed (through its newline when one
    arrived, or to EOF) — the caller may keep reading the stream.
    """

    def __init__(self, size: int, cap: int):
        super().__init__(f"JSONL line exceeded {cap} bytes (got >= {size})")
        self.size = size
        self.cap = cap


async def read_jsonl_line(reader: asyncio.StreamReader, *,
                          max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
                          ) -> bytes:
    """Read one newline-terminated line regardless of the reader's limit.

    Returns ``b""`` at clean EOF and the unterminated tail at EOF-mid-line,
    mirroring ``readline()``'s contract for both.
    """
    chunks: list[bytes] = []
    size = 0
    discarding = False
    while True:
        try:
            chunk = await reader.readuntil(b"\n")
        except asyncio.LimitOverrunError as e:
            # More than `limit` bytes buffered around the separator search.
            # Drain exactly what the reader has measured and keep going —
            # for the separator-found-past-limit case the separator itself
            # stays in the buffer, so the next readuntil() terminates.
            chunk = await reader.readexactly(e.consumed)
            size += len(chunk)
            if not discarding:
                if size > max_line_bytes:
                    discarding = True
                    chunks = []
                else:
                    chunks.append(chunk)
            continue
        except asyncio.IncompleteReadError as e:
            size += len(e.partial)
            if discarding or size > max_line_bytes:
                raise LineTooLongError(size, max_line_bytes) from None
            chunks.append(e.partial)
            return b"".join(chunks)
        size += len(chunk)
        if discarding or size > max_line_bytes:
            raise LineTooLongError(size, max_line_bytes)
        chunks.append(chunk)
        return b"".join(chunks)
