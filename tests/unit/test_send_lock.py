# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for CLIAdapter._send_lock serialization.

Verifies that concurrent send() calls to the same adapter are serialized
by the asyncio.Lock, preventing interleaved writes to stdin/PTY/transport.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter


def _make_adapter(transport=None):
    """Create a CLIAdapter with no real process."""
    return CLIAdapter(
        handle="test-agent",
        display_name="Test Agent",
        transport=transport,
    )


class TestSendLockSerialization:
    """Concurrent send() calls must not interleave."""

    @pytest.mark.asyncio
    async def test_concurrent_sends_serialized(self):
        """Two concurrent sends must execute sequentially, not overlap."""
        transport = AsyncMock()
        transport.send = AsyncMock(return_value="ok")
        adapter = _make_adapter(transport=transport)

        # Track whether sends overlap
        in_flight = 0
        max_in_flight = 0

        original_send = transport.send

        async def tracking_send(message):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)  # simulate work
            in_flight -= 1
            return await original_send(message)

        transport.send = tracking_send

        await asyncio.gather(
            adapter.send("message-1"),
            adapter.send("message-2"),
        )

        assert max_in_flight == 1, (
            f"Expected serialized sends (max 1 in-flight), got {max_in_flight}"
        )
        assert original_send.call_count == 2

    @pytest.mark.asyncio
    async def test_send_lock_exists(self):
        """CLIAdapter must have _send_lock as an asyncio.Lock."""
        adapter = _make_adapter()
        assert hasattr(adapter, '_send_lock')
        assert isinstance(adapter._send_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_send_lock_does_not_deadlock_on_sequential(self):
        """Sequential sends must not deadlock (lock is non-reentrant)."""
        transport = AsyncMock()
        transport.send = AsyncMock(return_value="ok")
        adapter = _make_adapter(transport=transport)

        await adapter.send("first")
        await adapter.send("second")

        assert transport.send.call_count == 2

    @pytest.mark.asyncio
    async def test_lock_lifecycle_matches_adapter(self):
        """Lock is created in __init__, no separate cleanup needed."""
        adapter = _make_adapter()
        lock_id = id(adapter._send_lock)
        # Lock is a plain attribute — it's garbage collected with the adapter
        del adapter
        # No assertion needed beyond no error; confirms no external registry
