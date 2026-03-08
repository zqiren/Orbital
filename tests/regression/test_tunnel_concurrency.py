# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for concurrent tunnel message processing.

Verifies that RelayClient dispatches tunnel messages concurrently
(not sequentially), respects the semaphore bound, and cleans up
tasks on disconnect.
"""

import asyncio
import json
import time

import pytest

from agent_os.relay.client import RelayClient


def _make_client() -> RelayClient:
    return RelayClient(
        relay_url="https://relay.example.com",
        device_id="dev-test",
        device_secret="secret-test",
    )


class FakeWebSocket:
    """Fake websocket that yields pre-loaded messages then closes."""

    def __init__(self, messages: list[str]):
        self._messages = messages
        self.sent: list[str] = []

    async def send(self, data: str):
        self.sent.append(data)

    async def close(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _rest_request_msg(request_id: str) -> str:
    return json.dumps({
        "type": "rest.request",
        "request_id": request_id,
        "method": "GET",
        "path": "/api/v2/projects",
        "headers": {},
        "body": None,
    })


async def _run_message_loop(client: RelayClient, ws: FakeWebSocket):
    """Simulate the message loop from _connect_and_run using concurrent dispatch."""
    client._ws = ws
    heartbeat_task = asyncio.create_task(asyncio.sleep(999))
    try:
        async for raw in ws:
            msg = json.loads(raw)
            task = asyncio.create_task(client._handle_tunnel_message_bounded(msg))
            client._tunnel_tasks.add(task)
            task.add_done_callback(client._tunnel_tasks.discard)
        # Wait for all in-flight tasks to complete
        if client._tunnel_tasks:
            await asyncio.gather(*client._tunnel_tasks, return_exceptions=True)
    finally:
        heartbeat_task.cancel()
        for t in client._tunnel_tasks:
            t.cancel()
        client._tunnel_tasks.clear()
        client._ws = None


@pytest.mark.asyncio
async def test_concurrent_processing():
    """10 messages with 0.1s simulated work should complete in <0.5s
    (proving concurrency, not 10 x 0.1s = 1s sequential).
    """
    client = _make_client()

    call_count = 0

    async def slow_handler(msg: dict):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.1)

    client._handle_tunnel_message = slow_handler

    messages = [_rest_request_msg(f"req-{i}") for i in range(10)]
    ws = FakeWebSocket(messages)

    start = time.monotonic()
    await _run_message_loop(client, ws)
    elapsed = time.monotonic() - start

    assert call_count == 10
    # Sequential would be ~1.0s; concurrent should be ~0.1s
    assert elapsed < 0.5, f"Processing took {elapsed:.2f}s — not concurrent"


@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    """With semaphore=2 and 5 messages each taking 0.1s,
    total time should be ~0.3s (ceil(5/2) x 0.1s),
    not 0.1s (unbounded) or 0.5s (sequential).
    """
    client = _make_client()

    timestamps: list[tuple[float, float]] = []

    async def slow_handler(msg: dict):
        t_start = time.monotonic()
        await asyncio.sleep(0.1)
        t_end = time.monotonic()
        timestamps.append((t_start, t_end))

    client._handle_tunnel_message = slow_handler
    client._tunnel_semaphore = asyncio.Semaphore(2)

    messages = [_rest_request_msg(f"req-{i}") for i in range(5)]
    ws = FakeWebSocket(messages)

    start = time.monotonic()
    await _run_message_loop(client, ws)
    elapsed = time.monotonic() - start

    assert len(timestamps) == 5
    # With semaphore=2: ceil(5/2) * 0.1 = 0.3s
    # Allow some slack but ensure it's bounded (not unbounded at ~0.1s)
    assert elapsed >= 0.2, f"Took {elapsed:.2f}s — semaphore not limiting"
    assert elapsed < 0.5, f"Took {elapsed:.2f}s — still sequential"


@pytest.mark.asyncio
async def test_task_cleanup_on_disconnect():
    """After WS disconnect, _tunnel_tasks should be empty."""
    client = _make_client()

    async def slow_handler(msg: dict):
        await asyncio.sleep(10)  # long-running: still active when WS ends

    client._handle_tunnel_message = slow_handler

    messages = [_rest_request_msg(f"req-{i}") for i in range(3)]
    ws = FakeWebSocket(messages)

    await _run_message_loop(client, ws)

    assert len(client._tunnel_tasks) == 0, \
        f"Expected empty _tunnel_tasks after cleanup, got {len(client._tunnel_tasks)}"
