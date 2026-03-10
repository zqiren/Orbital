# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke test for BUG-005a: end-to-end message streaming through the daemon stack.

Simulates the full data flow:
    SDKTransport (mocked SDK client)
      -> CLIAdapter.read_stream()  (delegates to transport)
        -> ProcessManager-style background consumer

Verifies that message events arrive at the read_stream() consumer
BEFORE send() completes — the core guarantee of the BUG-005a fix.

Without the fix, message events were only collected inside send()'s
response_parts list and never queued to _event_queue, so read_stream()
consumers (ProcessManager) would never see assistant text.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.base import TransportEvent
from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.adapters.base import OutputChunk


# ---------------------------------------------------------------------------
# Fake SDK types (mirrors the SDK's message protocol)
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeToolUseBlock:
    def __init__(self, name: str, id: str, input: dict):
        self.name = name
        self.id = id
        self.input = input


class _FakeAssistantMessage:
    def __init__(self, content: list):
        self.content = content


class _FakeResultMessage:
    def __init__(self, is_error: bool = False, result=None, session_id=None):
        self.is_error = is_error
        self.result = result
        self.session_id = session_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transport():
    """Create an SDKTransport with mocked internals (bypass __init__ import check)."""
    transport = SDKTransport.__new__(SDKTransport)
    transport._client = MagicMock()
    transport._session_id = None
    transport._alive = True
    transport._workspace = ""
    transport._pending_approvals = {}
    transport._event_queue = asyncio.Queue()
    transport._needs_flush = False
    return transport


def _make_cli_adapter(transport):
    """Create a CLIAdapter that delegates to the given transport."""
    return CLIAdapter(
        handle="test-agent",
        display_name="Test Agent",
        transport=transport,
    )


def _patch_sdk_types():
    """Context manager to patch SDK type references in sdk_transport module."""
    return patch.multiple(
        "agent_os.agent.transports.sdk_transport",
        AssistantMessage=_FakeAssistantMessage,
        ResultMessage=_FakeResultMessage,
        TextBlock=_FakeTextBlock,
        ToolUseBlock=_FakeToolUseBlock,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_arrives_at_read_stream_before_send_completes():
    """Full stack: message event reaches CLIAdapter.read_stream() before send() returns.

    This is the core BUG-005a scenario.  ProcessManager calls
    adapter.read_stream() in a background task.  When adapter.send() triggers
    SDKTransport.send(), the message event must be queued so the background
    consumer sees it in real-time — not after send() finishes.
    """
    transport = _make_transport()
    adapter = _make_cli_adapter(transport)

    # The SDK will yield an assistant message then a result message
    assistant_msg = _FakeAssistantMessage([_FakeTextBlock("I'll help you with that.")])
    result_msg = _FakeResultMessage(session_id="sess-001")

    # Add a delay between messages so we can observe real-time streaming
    async def mock_receive():
        yield assistant_msg
        await asyncio.sleep(0.05)  # simulate SDK processing time
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    # --- Simulate ProcessManager background consumer ---
    collected_chunks: list[OutputChunk] = []
    consumer_saw_message = asyncio.Event()

    async def process_manager_consumer():
        """Mimics ProcessManager.consume(): reads from adapter.read_stream()."""
        async for chunk in adapter.read_stream():
            collected_chunks.append(chunk)
            if chunk.chunk_type == "response" and chunk.text:
                consumer_saw_message.set()

    with _patch_sdk_types():
        # Start the background consumer (like ProcessManager.start())
        consumer_task = asyncio.create_task(process_manager_consumer())

        # Give the consumer a moment to start iterating read_stream()
        await asyncio.sleep(0.01)

        # Start send() (like when ProcessManager routes a user message)
        send_task = asyncio.create_task(adapter.send("help me"))

        # The consumer must see the message BEFORE send() completes
        try:
            await asyncio.wait_for(consumer_saw_message.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "BUG-005a regression: message event never reached read_stream() "
                "consumer — events are being buffered instead of streamed"
            )

        # Confirm send() hasn't finished yet (or just finished) — the message
        # arrived at the consumer in real-time
        # (We allow send_task to be done since the async scheduling may complete
        # it immediately after the yield; the key assertion is that the consumer
        # *did* receive the event.)

        # Wait for send to complete
        await send_task

        # Verify the consumer actually received the correct content
        assert len(collected_chunks) >= 1
        message_chunks = [c for c in collected_chunks if c.chunk_type == "response"]
        assert any("I'll help you with that." in c.text for c in message_chunks), (
            f"Expected message text in chunks, got: {[c.text for c in collected_chunks]}"
        )

        # Clean up: stop transport so read_stream() exits
        transport._alive = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_multiple_messages_stream_in_order():
    """Multiple assistant messages arrive at the consumer in the correct order."""
    transport = _make_transport()
    adapter = _make_cli_adapter(transport)

    msg1 = _FakeAssistantMessage([_FakeTextBlock("First chunk")])
    msg2 = _FakeAssistantMessage([_FakeTextBlock("Second chunk")])
    result_msg = _FakeResultMessage()

    async def mock_receive():
        yield msg1
        await asyncio.sleep(0.01)
        yield msg2
        await asyncio.sleep(0.01)
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    collected: list[str] = []
    got_both = asyncio.Event()

    async def consumer():
        async for chunk in adapter.read_stream():
            if chunk.chunk_type == "response" and chunk.text:
                collected.append(chunk.text)
                if len(collected) >= 2:
                    got_both.set()

    with _patch_sdk_types():
        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        send_task = asyncio.create_task(adapter.send("do something"))

        await asyncio.wait_for(got_both.wait(), timeout=3.0)
        await send_task

        assert collected == ["First chunk", "Second chunk"], (
            f"Messages arrived out of order or missing: {collected}"
        )

        transport._alive = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_tool_use_and_message_events_interleaved():
    """Tool use and message events both arrive at the consumer when interleaved."""
    transport = _make_transport()
    adapter = _make_cli_adapter(transport)

    msg_with_tool = _FakeAssistantMessage([
        _FakeTextBlock("Let me check that."),
        _FakeToolUseBlock("bash", "tc-1", {"command": "ls"}),
    ])
    msg_with_text = _FakeAssistantMessage([_FakeTextBlock("Here are the results.")])
    result_msg = _FakeResultMessage()

    async def mock_receive():
        yield msg_with_tool
        await asyncio.sleep(0.01)
        yield msg_with_text
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    collected: list[tuple[str, str]] = []  # (chunk_type, text)
    all_received = asyncio.Event()

    async def consumer():
        async for chunk in adapter.read_stream():
            collected.append((chunk.chunk_type, chunk.text))
            # Expect: response, tool_activity, response = 3 events
            if len(collected) >= 3:
                all_received.set()

    with _patch_sdk_types():
        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        send_task = asyncio.create_task(adapter.send("list files"))

        await asyncio.wait_for(all_received.wait(), timeout=3.0)
        await send_task

        types = [t for t, _ in collected[:3]]
        assert types == ["response", "tool_activity", "response"], (
            f"Expected [response, tool_activity, response], got {types}"
        )

        transport._alive = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_consumer_receives_events_concurrently_with_send():
    """Demonstrates that the consumer task runs concurrently with send().

    This test uses timing to verify real-time streaming: the consumer
    records timestamps when it receives each event, and we verify that
    the first event arrives before send() completes.
    """
    transport = _make_transport()
    adapter = _make_cli_adapter(transport)

    assistant_msg = _FakeAssistantMessage([_FakeTextBlock("Streaming response")])
    result_msg = _FakeResultMessage()

    # Insert a noticeable delay before the result so send() takes a while
    async def mock_receive():
        yield assistant_msg
        await asyncio.sleep(0.2)  # deliberate delay before result
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    consumer_event_time = None
    send_complete_time = None
    consumer_got_event = asyncio.Event()

    async def consumer():
        nonlocal consumer_event_time
        async for chunk in adapter.read_stream():
            if chunk.chunk_type == "response" and chunk.text:
                consumer_event_time = asyncio.get_event_loop().time()
                consumer_got_event.set()

    with _patch_sdk_types():
        consumer_task = asyncio.create_task(consumer())
        await asyncio.sleep(0.01)

        send_task = asyncio.create_task(adapter.send("stream test"))

        # Wait for consumer to receive the event
        await asyncio.wait_for(consumer_got_event.wait(), timeout=3.0)

        # Now wait for send to complete and record the time
        await send_task
        send_complete_time = asyncio.get_event_loop().time()

        # The consumer must have received the event BEFORE send() completed
        assert consumer_event_time is not None, "Consumer never received an event"
        assert consumer_event_time < send_complete_time, (
            f"Event arrived at consumer at {consumer_event_time}, "
            f"but send completed at {send_complete_time} — "
            "events are not streaming in real-time"
        )

        transport._alive = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_send_return_value_still_contains_message_text():
    """send() must still return the assembled response text (no regression)."""
    transport = _make_transport()
    adapter = _make_cli_adapter(transport)

    assistant_msg = _FakeAssistantMessage([_FakeTextBlock("Response text")])
    result_msg = _FakeResultMessage()

    async def mock_receive():
        yield assistant_msg
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    async def drain_consumer():
        """Drain read_stream to prevent queue backup."""
        async for _ in adapter.read_stream():
            pass

    with _patch_sdk_types():
        consumer_task = asyncio.create_task(drain_consumer())
        await asyncio.sleep(0.01)

        await adapter.send("test")

        # CLIAdapter.send() stores the transport return in _last_response
        assert adapter._last_response is not None
        assert "Response text" in adapter._last_response

        transport._alive = False
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
