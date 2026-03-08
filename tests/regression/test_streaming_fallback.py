# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: Mobile streaming REST fallback path.

Fix 1 — When is_final stream delta arrives with null prev stream state
         (intermediate deltas missed), REST fallback must still be triggered.
Fix 2 — When agent status transitions to idle after running, a catch-up
         REST fetch must fire to recover any missed messages.
Fix 3 — RelayClient logs warnings when events are silently dropped due
         to a missing WebSocket connection.
"""

import asyncio
import logging

import pytest

from agent_os.relay.client import RelayClient


def _make_client():
    return RelayClient(
        relay_url="https://relay.example.com",
        device_id="dev_test",
        device_secret="secret",
    )


class TestRelayForwardDropLogging:
    """Fix 3: forward_event and _send log warnings when self._ws is None."""

    @pytest.mark.asyncio
    async def test_forward_event_logs_warning_when_no_ws(self, caplog):
        """forward_event should log relay_forward_drop_no_ws when WS is None."""
        client = _make_client()
        client._ws = None

        with caplog.at_level(logging.WARNING, logger="agent_os.relay.client"):
            await client.forward_event("project_123", {"type": "chat.stream_delta"})

        assert any("relay_forward_drop_no_ws" in r.message for r in caplog.records), (
            "Expected relay_forward_drop_no_ws warning in logs"
        )
        assert any("project_123" in r.message for r in caplog.records)
        assert any("chat.stream_delta" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_send_logs_warning_when_no_ws(self, caplog):
        """_send should log relay_send_drop_no_ws when WS is None."""
        client = _make_client()
        client._ws = None

        with caplog.at_level(logging.WARNING, logger="agent_os.relay.client"):
            await client._send({"type": "event.forward", "event": {"type": "test"}})

        assert any("relay_send_drop_no_ws" in r.message for r in caplog.records), (
            "Expected relay_send_drop_no_ws warning in logs"
        )

    @pytest.mark.asyncio
    async def test_forward_event_still_returns_silently(self):
        """forward_event should not raise when WS is None — just log and return."""
        client = _make_client()
        client._ws = None

        # Should not raise
        await client.forward_event("proj", {"type": "agent.status", "status": "idle"})

    @pytest.mark.asyncio
    async def test_forward_event_no_warning_when_ws_connected(self, caplog):
        """forward_event should NOT log drop warnings when WS is connected."""

        class FakeWS:
            def __init__(self):
                self.sent = []

            async def send(self, data):
                self.sent.append(data)

            async def close(self):
                pass

        client = _make_client()
        client._ws = FakeWS()

        with caplog.at_level(logging.WARNING, logger="agent_os.relay.client"):
            await client.forward_event("proj", {"type": "chat.stream_delta"})

        drop_warnings = [r for r in caplog.records if "drop_no_ws" in r.message]
        assert len(drop_warnings) == 0, "Should not log drop warnings when WS is connected"
        assert len(client._ws.sent) == 1
