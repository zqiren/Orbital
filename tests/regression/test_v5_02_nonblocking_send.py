# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-V5-02: non-blocking agent_message(send)."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNonBlockingSend:
    @pytest.mark.asyncio
    async def test_send_returns_immediately(self, tmp_path):
        """send() must return in under 1 second even if adapter.send() is slow."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        pm = MagicMock()
        pm.start = AsyncMock()
        sam = SubAgentManager(pm)

        # Create a slow adapter (5 second sleep in send)
        slow_adapter = MagicMock()

        async def slow_send(msg):
            await asyncio.sleep(5)

        slow_adapter.send = slow_send
        slow_adapter._transport = None  # Legacy path: uses background task
        slow_adapter.is_alive = MagicMock(return_value=True)
        slow_adapter.is_idle = MagicMock(return_value=False)

        sam._adapters["proj1"] = {"slow-agent": slow_adapter}

        # Create transcript
        transcript = SubAgentTranscript(str(tmp_path), "slow-agent", "t001")
        sam._transcripts[("proj1", "slow-agent")] = transcript

        start = time.monotonic()
        result = await sam.send("proj1", "slow-agent", "do something slow")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"send() took {elapsed:.1f}s — should be under 1s"
        assert "Message sent to slow-agent" in result
        assert "Transcript:" in result

    @pytest.mark.asyncio
    async def test_send_result_includes_transcript_path(self, tmp_path):
        """Return value must include transcript file path."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        pm = MagicMock()
        sam = SubAgentManager(pm)

        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter._transport = None
        adapter.is_alive = MagicMock(return_value=True)

        sam._adapters["proj1"] = {"claude-code": adapter}
        transcript = SubAgentTranscript(str(tmp_path), "claude-code", "t001")
        sam._transcripts[("proj1", "claude-code")] = transcript

        result = await sam.send("proj1", "claude-code", "hello")

        assert ".agent-os/sub_agents/" in result.replace("\\", "/")
        assert "claude-code" in result

    @pytest.mark.asyncio
    async def test_send_error_for_unknown_agent(self):
        """send() returns error for non-existent agent."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        sam = SubAgentManager(pm)

        result = await sam.send("proj1", "nonexistent", "hello")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_send_transcript_unknown_when_no_transcript(self):
        """send() returns 'unknown' transcript path if no transcript registered."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        sam = SubAgentManager(pm)

        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter._transport = None

        sam._adapters["proj1"] = {"agent-x": adapter}
        # No transcript registered

        result = await sam.send("proj1", "agent-x", "hello")
        assert "Transcript: unknown" in result

    @pytest.mark.asyncio
    async def test_dispatch_uses_transport_dispatch_when_available(self, tmp_path):
        """_dispatch_async prefers transport.dispatch() over background task."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        pm = MagicMock()
        sam = SubAgentManager(pm)

        transport = MagicMock()
        transport.dispatch = AsyncMock()

        adapter = MagicMock()
        adapter._transport = transport

        sam._adapters["proj1"] = {"sdk-agent": adapter}
        transcript = SubAgentTranscript(str(tmp_path), "sdk-agent", "t001")
        sam._transcripts[("proj1", "sdk-agent")] = transcript

        await sam.send("proj1", "sdk-agent", "test message")

        transport.dispatch.assert_awaited_once_with("test message")
        # adapter.send should NOT have been called
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_falls_back_to_background_send(self, tmp_path):
        """_dispatch_async uses background task when transport has no dispatch()."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        pm = MagicMock()
        sam = SubAgentManager(pm)

        # Transport without dispatch method
        transport = MagicMock(spec=["send", "read_stream", "start", "stop", "is_alive"])
        adapter = MagicMock()
        adapter._transport = transport
        adapter.send = AsyncMock()

        sam._adapters["proj1"] = {"pipe-agent": adapter}
        transcript = SubAgentTranscript(str(tmp_path), "pipe-agent", "t001")
        sam._transcripts[("proj1", "pipe-agent")] = transcript

        result = await sam.send("proj1", "pipe-agent", "test")
        assert "Message sent to pipe-agent" in result

        # Give the background task a chance to run
        await asyncio.sleep(0.05)
        adapter.send.assert_awaited_once_with("test")

    @pytest.mark.asyncio
    async def test_sdk_dispatch_non_blocking(self):
        """SDKTransport.dispatch() should return before response is consumed."""
        try:
            from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
        except ImportError:
            pytest.skip("SDK transport not available")

        if not HAS_SDK:
            pytest.skip("SDK not available")

        transport = SDKTransport()
        transport._client = MagicMock()
        transport._client.query = AsyncMock()
        transport._needs_flush = False

        # Mock receive_response to be slow (async generator)
        async def slow_receive():
            await asyncio.sleep(5)
            return
            yield  # Make it an async generator

        transport._client.receive_response = slow_receive

        start = time.monotonic()
        await transport.dispatch("hello")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"dispatch() took {elapsed:.1f}s — should be non-blocking"
        transport._client.query.assert_awaited_once_with("hello")

    @pytest.mark.asyncio
    async def test_sdk_dispatch_raises_when_client_not_initialized(self):
        """SDKTransport.dispatch() raises RuntimeError if client is None."""
        try:
            from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
        except ImportError:
            pytest.skip("SDK transport not available")

        if not HAS_SDK:
            pytest.skip("SDK not available")

        transport = SDKTransport()
        transport._client = None

        with pytest.raises(RuntimeError, match="not initialized"):
            await transport.dispatch("hello")

    @pytest.mark.asyncio
    async def test_sdk_dispatch_flushes_stale_messages(self):
        """SDKTransport.dispatch() flushes stale messages before querying."""
        try:
            from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
        except ImportError:
            pytest.skip("SDK transport not available")

        if not HAS_SDK:
            pytest.skip("SDK not available")

        transport = SDKTransport()
        transport._client = MagicMock()
        transport._client.query = AsyncMock()
        transport._needs_flush = True

        flush_called = False
        original_flush = transport._flush_stale_messages

        async def mock_flush():
            nonlocal flush_called
            flush_called = True
            transport._needs_flush = False

        transport._flush_stale_messages = mock_flush

        # Mock receive_response as async generator
        async def empty_receive():
            return
            yield

        transport._client.receive_response = empty_receive

        await transport.dispatch("hello")
        assert flush_called, "_flush_stale_messages should have been called"

    @pytest.mark.asyncio
    async def test_background_send_error_is_logged(self, tmp_path):
        """Errors in background send should be logged, not crash."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        pm = MagicMock()
        sam = SubAgentManager(pm)

        adapter = MagicMock()

        async def failing_send(msg):
            raise RuntimeError("connection lost")

        adapter.send = failing_send
        adapter._transport = None

        sam._adapters["proj1"] = {"fail-agent": adapter}
        transcript = SubAgentTranscript(str(tmp_path), "fail-agent", "t001")
        sam._transcripts[("proj1", "fail-agent")] = transcript

        # Should not raise
        result = await sam.send("proj1", "fail-agent", "hello")
        assert "Message sent to fail-agent" in result

        # Let background task run and fail gracefully
        await asyncio.sleep(0.05)
