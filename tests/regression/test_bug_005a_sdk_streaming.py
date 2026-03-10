# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for BUG-005a: SDK transport message streaming.

Verifies that message events are queued to read_stream() consumers
in real-time (not buffered until send() returns).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.base import TransportEvent


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


@pytest.mark.asyncio
async def test_message_events_queued_for_read_stream():
    """Message events should appear in read_stream() BEFORE send() returns."""
    transport = SDKTransport.__new__(SDKTransport)
    transport._client = MagicMock()
    transport._session_id = None
    transport._alive = True
    transport._workspace = ""
    transport._pending_approvals = {}
    transport._event_queue = asyncio.Queue()
    transport._needs_flush = False

    assistant_msg = _FakeAssistantMessage([_FakeTextBlock("Hello from Claude Code")])
    result_msg = _FakeResultMessage()

    async def mock_receive():
        yield assistant_msg
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    with patch("agent_os.agent.transports.sdk_transport.AssistantMessage", _FakeAssistantMessage), \
         patch("agent_os.agent.transports.sdk_transport.ResultMessage", _FakeResultMessage), \
         patch("agent_os.agent.transports.sdk_transport.TextBlock", _FakeTextBlock), \
         patch("agent_os.agent.transports.sdk_transport.ToolUseBlock", _FakeToolUseBlock):

        send_task = asyncio.create_task(transport.send("hello"))

        # Message event should reach queue BEFORE send() returns
        event = await asyncio.wait_for(transport._event_queue.get(), timeout=2.0)
        assert event.event_type == "message"
        assert event.raw_text == "Hello from Claude Code"

        result = await send_task
        assert "Hello from Claude Code" in result


@pytest.mark.asyncio
async def test_tool_use_events_still_queued():
    """Tool use events should still be queued (existing behavior preserved)."""
    transport = SDKTransport.__new__(SDKTransport)
    transport._client = MagicMock()
    transport._session_id = None
    transport._alive = True
    transport._workspace = ""
    transport._pending_approvals = {}
    transport._event_queue = asyncio.Queue()
    transport._needs_flush = False

    assistant_msg = _FakeAssistantMessage([_FakeToolUseBlock("bash", "tc-1", {"command": "ls"})])
    result_msg = _FakeResultMessage()

    async def mock_receive():
        yield assistant_msg
        yield result_msg

    transport._client.query = AsyncMock()
    transport._client.receive_response = mock_receive

    with patch("agent_os.agent.transports.sdk_transport.AssistantMessage", _FakeAssistantMessage), \
         patch("agent_os.agent.transports.sdk_transport.ResultMessage", _FakeResultMessage), \
         patch("agent_os.agent.transports.sdk_transport.TextBlock", _FakeTextBlock), \
         patch("agent_os.agent.transports.sdk_transport.ToolUseBlock", _FakeToolUseBlock):

        send_task = asyncio.create_task(transport.send("run ls"))

        event = await asyncio.wait_for(transport._event_queue.get(), timeout=2.0)
        assert event.event_type == "tool_use"
        assert event.data["tool_name"] == "bash"

        await send_task
