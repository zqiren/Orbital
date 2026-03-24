# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: SDK sub-agent completion notification.

Verifies that when a sub-agent completes a turn via SDK transport,
ProcessManager fires on_completed() on the lifecycle observer with the
response text as summary.  Covers single-turn, multi-turn, no-output,
and no-observer scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.daemon_v2.process_manager import ProcessManager


# ---------------------------------------------------------------------------
# Helpers (same FakeSDKTransport / _make_adapter pattern as
# test_sdk_transport_idle_state.py)
# ---------------------------------------------------------------------------

class FakeSDKTransport:
    """Minimal transport stub backed by an asyncio.Queue.

    Drains remaining queued events after ``_alive`` goes False so that
    pre-queued test events are always delivered.
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


def _make_mocks():
    """Return (ws_manager, activity_translator, lifecycle_observer, transcript) mocks."""
    ws = MagicMock()
    ws.broadcast = MagicMock()

    activity = MagicMock()
    activity.on_message = MagicMock()

    observer = MagicMock()
    observer.on_completed = AsyncMock()
    observer.on_error = AsyncMock()

    transcript = MagicMock()
    transcript.append = MagicMock()
    transcript.filepath = "/tmp/test-transcript.jsonl"

    return ws, activity, observer, transcript


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_complete_fires_on_completed():
    """A single turn_complete event triggers on_completed with the response summary."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)
    ws, activity, observer, transcript = _make_mocks()
    pm = ProcessManager(ws, activity, lifecycle_observer=observer)

    # Queue a message followed by turn_complete
    await transport._event_queue.put(
        TransportEvent("message", data={"text": "hello"}, raw_text="hello")
    )
    await transport._event_queue.put(
        TransportEvent("turn_complete", data={}, raw_text="")
    )

    # Start consumer
    await pm.start("proj1", "test-agent", adapter, transcript=transcript)

    # Let the consume loop process the queued events
    await asyncio.sleep(0.3)

    # Stop the transport so the consume loop exits
    transport._alive = False
    await asyncio.sleep(0.6)

    # on_completed should have been called: once for turn_complete, once for stream end
    # The turn_complete call is the one we care about — it must contain "hello"
    calls = observer.on_completed.call_args_list
    assert len(calls) >= 1, f"Expected on_completed to be called, got {len(calls)} calls"

    def _get_summary(c):
        if "summary" in c.kwargs:
            return c.kwargs["summary"]
        return c[0][2]  # positional arg index 2

    # First call (from turn_complete) should have summary containing "hello"
    first_call = calls[0]
    assert first_call[0][0] == "proj1"
    assert first_call[0][1] == "test-agent"
    assert "hello" in _get_summary(first_call)

    # Adapter should be idle after turn_complete
    assert adapter.is_idle()


@pytest.mark.asyncio
async def test_multi_turn_completion_resets_response():
    """Multiple turn_complete cycles each report their own response, not accumulated text."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)
    ws, activity, observer, transcript = _make_mocks()
    pm = ProcessManager(ws, activity, lifecycle_observer=observer)

    # First turn
    await transport._event_queue.put(
        TransportEvent("message", data={"text": "first response"}, raw_text="first response")
    )
    await transport._event_queue.put(
        TransportEvent("turn_complete", data={}, raw_text="")
    )
    # Second turn
    await transport._event_queue.put(
        TransportEvent("message", data={"text": "second response"}, raw_text="second response")
    )
    await transport._event_queue.put(
        TransportEvent("turn_complete", data={}, raw_text="")
    )

    await pm.start("proj1", "test-agent", adapter, transcript=transcript)
    await asyncio.sleep(0.3)

    # Stop transport
    transport._alive = False
    await asyncio.sleep(0.6)

    calls = observer.on_completed.call_args_list

    # We expect at least 2 calls from the two turn_complete events
    # (there may be a 3rd from stream-end with "(no output)" since response was reset)
    assert len(calls) >= 2, f"Expected at least 2 on_completed calls, got {len(calls)}"

    # Extract summaries from the first two calls
    def _get_summary(c):
        if "summary" in c.kwargs:
            return c.kwargs["summary"]
        return c[0][2]  # positional arg index 2

    first_summary = _get_summary(calls[0])
    second_summary = _get_summary(calls[1])

    assert "first response" in first_summary
    assert "second response" in second_summary
    # Second summary must NOT contain first response (reset between turns)
    assert "first response" not in second_summary


@pytest.mark.asyncio
async def test_turn_complete_without_response():
    """turn_complete with no preceding message fires on_completed with '(no output)'."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)
    ws, activity, observer, transcript = _make_mocks()
    pm = ProcessManager(ws, activity, lifecycle_observer=observer)

    # Only queue turn_complete, no message before it
    await transport._event_queue.put(
        TransportEvent("turn_complete", data={}, raw_text="")
    )

    await pm.start("proj1", "test-agent", adapter, transcript=transcript)
    await asyncio.sleep(0.3)

    transport._alive = False
    await asyncio.sleep(0.6)

    calls = observer.on_completed.call_args_list
    assert len(calls) >= 1, f"Expected on_completed to be called, got {len(calls)} calls"

    def _get_summary(c):
        if "summary" in c.kwargs:
            return c.kwargs["summary"]
        return c[0][2]

    first_summary = _get_summary(calls[0])
    assert first_summary == "(no output)"


@pytest.mark.asyncio
async def test_on_completed_not_called_without_lifecycle():
    """ProcessManager with lifecycle_observer=None does not crash on turn_complete."""
    transport = FakeSDKTransport()
    adapter = _make_adapter(transport)
    ws, activity, _observer, transcript = _make_mocks()

    # Create ProcessManager WITHOUT lifecycle observer
    pm = ProcessManager(ws, activity, lifecycle_observer=None)

    await transport._event_queue.put(
        TransportEvent("message", data={"text": "hello"}, raw_text="hello")
    )
    await transport._event_queue.put(
        TransportEvent("turn_complete", data={}, raw_text="")
    )

    await pm.start("proj1", "test-agent", adapter, transcript=transcript)
    await asyncio.sleep(0.3)

    transport._alive = False
    await asyncio.sleep(0.6)

    # No crash is the success criterion. The unused _observer should never be called.
    _observer.on_completed.assert_not_called()
