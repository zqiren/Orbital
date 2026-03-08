# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: _heartbeat_loop sends first heartbeat immediately, not after 30s.

Bug: _heartbeat_loop() slept before sending the first heartbeat, creating a
30-second window where phones had no heartbeat confirmation of daemon status.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from agent_os.relay.client import RelayClient


class TestImmediateFirstHeartbeat:
    """_heartbeat_loop sends a heartbeat before the first sleep."""

    @pytest.mark.asyncio
    async def test_first_heartbeat_sent_within_one_second(self):
        """send() is called within 1s, proving sleep is NOT first."""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        client = RelayClient.__new__(RelayClient)

        # Start heartbeat loop, cancel after 1s
        task = asyncio.create_task(client._heartbeat_loop(mock_ws))
        await asyncio.sleep(0.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # First heartbeat should have been sent within 0.5s
        mock_ws.send.assert_called()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent == {"type": "device.status", "status": "online"}
