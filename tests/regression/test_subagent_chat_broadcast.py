# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: ProcessManager must broadcast chat.sub_agent_message for sub-agent text."""

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.process_manager import ProcessManager


class _FakeChunk:
    def __init__(self, text, chunk_type="response", metadata=None):
        self.text = text
        self.chunk_type = chunk_type
        self.metadata = metadata or {}


class _FakeAdapter:
    """Adapter that yields predetermined chunks from read_stream()."""
    def __init__(self, chunks):
        self._chunks = chunks

    async def read_stream(self):
        for chunk in self._chunks:
            yield chunk


class TestSubAgentChatBroadcast:

    @pytest.mark.asyncio
    async def test_process_manager_broadcasts_sub_agent_message(self):
        """Response chunks must produce a chat.sub_agent_message broadcast."""
        ws = MagicMock()
        broadcasts = []
        ws.broadcast = lambda pid, payload: broadcasts.append((pid, payload))
        activity = MagicMock()

        pm = ProcessManager(ws, activity)

        adapter = _FakeAdapter([_FakeChunk("Hello from Claude Code", "response")])
        await pm.start("proj-1", "claude-code", adapter)
        await asyncio.sleep(0.5)

        chat_msgs = [b for _, b in broadcasts if b.get("type") == "chat.sub_agent_message"]
        assert len(chat_msgs) == 1
        assert chat_msgs[0]["content"] == "Hello from Claude Code"
        assert chat_msgs[0]["source"] == "claude-code"
        assert chat_msgs[0]["project_id"] == "proj-1"
        assert "timestamp" in chat_msgs[0]

    @pytest.mark.asyncio
    async def test_tool_activity_not_broadcast_as_chat_message(self):
        """tool_activity chunks must NOT produce chat.sub_agent_message."""
        ws = MagicMock()
        broadcasts = []
        ws.broadcast = lambda pid, payload: broadcasts.append((pid, payload))
        activity = MagicMock()

        pm = ProcessManager(ws, activity)

        adapter = _FakeAdapter([_FakeChunk("Running shell: ls", "tool_activity")])
        await pm.start("proj-1", "claude-code", adapter)
        await asyncio.sleep(0.5)

        chat_msgs = [b for _, b in broadcasts if b.get("type") == "chat.sub_agent_message"]
        assert len(chat_msgs) == 0, "tool_activity should not be broadcast as chat message"

        # Activity translator should still be called
        activity.on_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_approval_request_not_duplicated(self):
        """approval_request chunks must NOT also produce chat.sub_agent_message."""
        ws = MagicMock()
        broadcasts = []
        ws.broadcast = lambda pid, payload: broadcasts.append((pid, payload))
        activity = MagicMock()

        pm = ProcessManager(ws, activity)

        adapter = _FakeAdapter([_FakeChunk(
            "Approve tool?", "approval_request",
            metadata={"tool_name": "shell", "request_id": "req-1", "tool_input": {"command": "ls"}}
        )])
        await pm.start("proj-1", "claude-code", adapter)
        await asyncio.sleep(0.5)

        approval_msgs = [b for _, b in broadcasts if b.get("type") == "approval.request"]
        chat_msgs = [b for _, b in broadcasts if b.get("type") == "chat.sub_agent_message"]
        assert len(approval_msgs) == 1, "approval.request should be broadcast"
        assert len(chat_msgs) == 0, "approval_request should NOT also be a chat message"
