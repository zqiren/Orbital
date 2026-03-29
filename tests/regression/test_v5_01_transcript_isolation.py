# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-V5-01: sub-agent transcript isolation."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript


class TestSubAgentTranscript:
    def test_append_and_read(self, tmp_path):
        t = SubAgentTranscript(str(tmp_path), "claude-code", "t001")
        t.append({"source": "claude-code", "content": "Hello", "timestamp": "t1", "chunk_type": "response"})
        t.append({"source": "claude-code", "content": "World", "timestamp": "t2", "chunk_type": "response"})
        t.append({"source": "claude-code", "content": "Done", "timestamp": "t3", "chunk_type": "response"})
        entries = SubAgentTranscript.read(t.filepath)
        assert len(entries) == 3
        assert entries[0]["content"] == "Hello"
        assert entries[2]["content"] == "Done"
        # Verify file on disk
        with open(t.filepath) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 3

    def test_directory_creation(self, tmp_path):
        t = SubAgentTranscript(str(tmp_path), "deep-agent", "t002")
        assert os.path.isdir(os.path.join(str(tmp_path), "orbital", "sub_agents", "deep-agent"))
        assert os.path.isfile(t.filepath)

    def test_latest_pointer_updated(self, tmp_path):
        t1 = SubAgentTranscript(str(tmp_path), "claude-code", "first")
        latest_path = os.path.join(str(tmp_path), "orbital", "sub_agents", "claude-code", ".latest")
        assert os.path.isfile(latest_path)
        with open(latest_path) as f:
            assert f.read().strip() == "first.jsonl"
        t2 = SubAgentTranscript(str(tmp_path), "claude-code", "second")
        with open(latest_path) as f:
            assert f.read().strip() == "second.jsonl"

    def test_filepath_property(self, tmp_path):
        t = SubAgentTranscript(str(tmp_path), "test-agent", "abc")
        assert t.filepath.endswith("abc.jsonl")
        assert "sub_agents" in t.filepath
        assert "test-agent" in t.filepath


class TestProcessManagerIsolation:
    @pytest.mark.asyncio
    async def test_process_manager_does_not_write_to_session(self, tmp_path):
        """ProcessManager must write to transcript, not session."""
        from agent_os.agent.adapters.base import OutputChunk
        from agent_os.daemon_v2.process_manager import ProcessManager

        ws = MagicMock()
        ws.broadcast = MagicMock()
        activity = MagicMock()
        activity.on_message = MagicMock()

        pm = ProcessManager(ws, activity)

        # Create a mock session and register it (should NOT be called)
        mock_session = MagicMock()
        mock_session.append = MagicMock()

        # Create real transcript
        transcript = SubAgentTranscript(str(tmp_path), "test-agent", "t001")

        # Create mock adapter that yields chunks
        chunks = [
            OutputChunk(text="Hello world", chunk_type="response", timestamp="t1"),
            OutputChunk(text="Reading file", chunk_type="tool_activity", timestamp="t2"),
        ]
        adapter = MagicMock()
        adapter.read_stream = MagicMock(return_value=_async_iter(chunks))

        await pm.start("proj1", "test-agent", adapter, transcript=transcript)
        await asyncio.sleep(0.1)  # Let consume task run

        # Session must NOT have been called
        mock_session.append.assert_not_called()

        # Transcript should have entries
        entries = SubAgentTranscript.read(transcript.filepath)
        assert len(entries) == 2
        assert entries[0]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_process_manager_broadcasts_chat_messages(self, tmp_path):
        """Response chunks should broadcast as chat.sub_agent_message."""
        from agent_os.agent.adapters.base import OutputChunk
        from agent_os.daemon_v2.process_manager import ProcessManager

        ws = MagicMock()
        ws.broadcast = MagicMock()
        activity = MagicMock()
        activity.on_message = MagicMock()

        pm = ProcessManager(ws, activity)
        transcript = SubAgentTranscript(str(tmp_path), "test-agent", "t001")

        chunks = [OutputChunk(text="Hello", chunk_type="response", timestamp="t1")]
        adapter = MagicMock()
        adapter.read_stream = MagicMock(return_value=_async_iter(chunks))

        await pm.start("proj1", "test-agent", adapter, transcript=transcript)
        await asyncio.sleep(0.1)

        # Check broadcast was called with chat.sub_agent_message
        calls = [c for c in ws.broadcast.call_args_list
                 if c[0][1].get("type") == "chat.sub_agent_message"]
        assert len(calls) == 1
        assert calls[0][0][1]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_process_manager_does_not_broadcast_tool_activity_as_chat(self, tmp_path):
        """Tool activity chunks should not broadcast as chat messages."""
        from agent_os.agent.adapters.base import OutputChunk
        from agent_os.daemon_v2.process_manager import ProcessManager

        ws = MagicMock()
        ws.broadcast = MagicMock()
        activity = MagicMock()
        activity.on_message = MagicMock()

        pm = ProcessManager(ws, activity)
        transcript = SubAgentTranscript(str(tmp_path), "test-agent", "t001")

        chunks = [OutputChunk(text="Reading file.py", chunk_type="tool_activity", timestamp="t1")]
        adapter = MagicMock()
        adapter.read_stream = MagicMock(return_value=_async_iter(chunks))

        await pm.start("proj1", "test-agent", adapter, transcript=transcript)
        await asyncio.sleep(0.1)

        # No chat.sub_agent_message broadcast
        chat_calls = [c for c in ws.broadcast.call_args_list
                      if c[0][1].get("type") == "chat.sub_agent_message"]
        assert len(chat_calls) == 0

        # But activity_translator was called
        activity.on_message.assert_called()


async def _async_iter(items):
    """Helper: create an async iterator from a list."""
    for item in items:
        yield item
