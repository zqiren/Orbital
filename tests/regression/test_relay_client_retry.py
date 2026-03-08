# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: RelayClient._send() retries event.forward messages.

Fix 1A — the relay tunnel can have transient failures. event.forward
messages should be retried up to 3 times with exponential backoff.
Non-retryable messages (rest.response) should fail immediately.
"""

import asyncio
import json

import pytest

from agent_os.relay.client import RelayClient


class FailNTimesThenSucceed:
    """Fake WebSocket that fails N sends, then succeeds."""

    def __init__(self, fail_count: int):
        self.fail_count = fail_count
        self.attempt = 0
        self.sent: list[dict] = []

    async def send(self, data: str):
        self.attempt += 1
        if self.attempt <= self.fail_count:
            raise ConnectionError(f"Transient failure #{self.attempt}")
        self.sent.append(json.loads(data))

    async def close(self):
        pass


class AlwaysFailWebSocket:
    """Fake WebSocket that always fails."""

    def __init__(self):
        self.attempt = 0

    async def send(self, data: str):
        self.attempt += 1
        raise ConnectionError(f"Permanent failure #{self.attempt}")

    async def close(self):
        pass


def _make_client():
    return RelayClient(
        relay_url="https://relay.example.com",
        device_id="dev_test",
        device_secret="secret",
    )


class TestRelaySendRetry:
    @pytest.mark.asyncio
    async def test_event_forward_retries_on_transient_failure(self):
        """event.forward should succeed after 2 transient failures."""
        client = _make_client()
        ws = FailNTimesThenSucceed(fail_count=2)
        client._ws = ws

        await client._send({
            "type": "event.forward",
            "event": {"type": "agent.status", "project_id": "p1", "status": "idle", "seq": ""},
        })

        assert len(ws.sent) == 1
        assert ws.sent[0]["type"] == "event.forward"
        assert ws.attempt == 3  # 2 failures + 1 success

    @pytest.mark.asyncio
    async def test_event_forward_drops_after_max_retries(self):
        """event.forward should be dropped after 3 consecutive failures."""
        client = _make_client()
        ws = AlwaysFailWebSocket()
        client._ws = ws

        # Should not raise — drops silently with error log
        await client._send({
            "type": "event.forward",
            "event": {"type": "chat.stream_delta", "project_id": "p1", "seq": 5},
        })

        assert ws.attempt == 3  # 3 attempts total

    @pytest.mark.asyncio
    async def test_rest_response_not_retried(self):
        """rest.response messages should NOT be retried — they are request-scoped."""
        client = _make_client()
        ws = AlwaysFailWebSocket()
        client._ws = ws

        await client._send({
            "type": "rest.response",
            "request_id": "req_001",
            "status": 200,
            "body": {},
        })

        assert ws.attempt == 1  # only 1 attempt, no retries

    @pytest.mark.asyncio
    async def test_send_noop_when_disconnected(self):
        """_send should do nothing when WebSocket is None."""
        client = _make_client()
        client._ws = None

        # Should not raise
        await client._send({"type": "event.forward", "event": {"type": "test"}})
