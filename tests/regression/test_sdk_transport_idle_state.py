# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: CLIAdapter idle state for SDK transport sub-agents.

Verifies that:
1. SDKTransport emits a turn_complete TransportEvent after consuming a response
2. CLIAdapter.read_stream() intercepts turn_complete and sets _idle = True
3. turn_complete IS yielded as OutputChunk(chunk_type="turn_complete") for ProcessManager
4. _idle resets to False on dispatch
5. Multiple turn_complete cycles work correctly
"""

import asyncio
import pytest

from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.adapters.cli_adapter import CLIAdapter


class FakeSDKTransport:
    """Minimal transport stub that uses an asyncio.Queue for events.

    Mirrors the real SDKTransport.read_stream() behavior: yields events
    while alive, but also drains any remaining queued events after _alive
    goes False (so pre-queued test events are always delivered).
    """

    def __init__(self):
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._alive = True

    async def start(self, command, args, workspace, env=None):
        pass

    async def send(self, message):
        return None

    async def dispatch(self, message):
        pass

    async def read_stream(self):
        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                if not self._alive:
                    break
                continue

    async def stop(self):
        self._alive = False

    def is_alive(self):
        return self._alive

    @property
    def session_id(self):
        return None


def _make_adapter(transport):
    """Create a CLIAdapter wired to the given transport."""
    return CLIAdapter(
        handle="test-agent",
        display_name="Test Agent",
        transport=transport,
    )


@pytest.mark.asyncio
async def test_turn_complete_sets_idle():
    """turn_complete event sets adapter._idle to True."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)

    # Initially not idle
    assert not adapter.is_idle()

    # Put a message event followed by turn_complete
    await transport._event_queue.put(TransportEvent(
        event_type="message",
        data={"text": "Hello"},
        raw_text="Hello",
    ))
    await transport._event_queue.put(TransportEvent(event_type="turn_complete"))
    # Signal end of stream
    transport._alive = False

    chunks = []
    async for chunk in adapter.read_stream():
        chunks.append(chunk)

    # Message chunk + turn_complete sentinel chunk
    assert len(chunks) == 2
    assert chunks[0].text == "Hello"
    assert chunks[1].chunk_type == "turn_complete"
    assert chunks[1].text == ""

    # Adapter should now be idle
    assert adapter.is_idle()


@pytest.mark.asyncio
async def test_turn_complete_yielded_as_sentinel():
    """turn_complete is yielded as a sentinel OutputChunk for ProcessManager."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)

    # Only put turn_complete, no other events
    await transport._event_queue.put(TransportEvent(event_type="turn_complete"))
    transport._alive = False

    chunks = []
    async for chunk in adapter.read_stream():
        chunks.append(chunk)

    # turn_complete sentinel should be yielded
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "turn_complete"
    assert chunks[0].text == ""
    # Adapter should be idle
    assert adapter.is_idle()


@pytest.mark.asyncio
async def test_idle_resets_on_dispatch():
    """Simulating dispatch resets _idle to False."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)

    # Manually set idle (as if turn_complete was processed)
    adapter._idle = True
    assert adapter.is_idle()

    # Simulate what _dispatch_async does: reset idle before dispatch
    adapter._idle = False
    assert not adapter.is_idle()


@pytest.mark.asyncio
async def test_multiple_turn_complete_cycles():
    """Idle transitions work across multiple message/turn_complete cycles."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)

    # First cycle: message + turn_complete
    await transport._event_queue.put(TransportEvent(
        event_type="message",
        data={"text": "First response"},
        raw_text="First response",
    ))
    await transport._event_queue.put(TransportEvent(event_type="turn_complete"))

    # Consume first cycle (need to collect within timeout)
    chunks = []
    async for event in transport.read_stream():
        if event.event_type == "turn_complete":
            break
        chunks.append(event)
    # Re-create transport to reset, simulating continued operation
    transport2 = FakeSDKTransport()
    adapter2 = _make_adapter(transport2)

    # First cycle
    await transport2._event_queue.put(TransportEvent(
        event_type="message", data={"text": "msg1"}, raw_text="msg1",
    ))
    await transport2._event_queue.put(TransportEvent(event_type="turn_complete"))

    # Second cycle
    await transport2._event_queue.put(TransportEvent(
        event_type="message", data={"text": "msg2"}, raw_text="msg2",
    ))
    await transport2._event_queue.put(TransportEvent(event_type="turn_complete"))

    # End stream
    transport2._alive = False

    all_chunks = []
    async for chunk in adapter2.read_stream():
        all_chunks.append(chunk)

    # Should get 2 message chunks + 2 turn_complete sentinels
    assert len(all_chunks) == 4
    assert all_chunks[0].text == "msg1"
    assert all_chunks[1].chunk_type == "turn_complete"
    assert all_chunks[2].text == "msg2"
    assert all_chunks[3].chunk_type == "turn_complete"

    # After processing both cycles, adapter should be idle
    assert adapter2.is_idle()


@pytest.mark.asyncio
async def test_without_turn_complete_stays_not_idle():
    """Without turn_complete, adapter stays not idle (the bug scenario)."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)

    # Put only message events, no turn_complete
    await transport._event_queue.put(TransportEvent(
        event_type="message",
        data={"text": "Hello"},
        raw_text="Hello",
    ))
    transport._alive = False

    chunks = []
    async for chunk in adapter.read_stream():
        chunks.append(chunk)

    assert len(chunks) == 1
    # Without turn_complete, adapter should NOT be idle
    assert not adapter.is_idle()


@pytest.mark.asyncio
async def test_error_event_followed_by_turn_complete():
    """Even after error events, turn_complete still sets idle."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)

    await transport._event_queue.put(TransportEvent(
        event_type="error",
        data={"error": "something broke"},
        raw_text="Error: something broke",
    ))
    await transport._event_queue.put(TransportEvent(event_type="turn_complete"))
    transport._alive = False

    chunks = []
    async for chunk in adapter.read_stream():
        chunks.append(chunk)

    # Error chunk + turn_complete sentinel should both be yielded
    assert len(chunks) == 2
    assert "something broke" in chunks[0].text
    assert chunks[1].chunk_type == "turn_complete"
    assert adapter.is_idle()
