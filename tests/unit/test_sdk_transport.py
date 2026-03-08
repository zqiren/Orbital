# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for SDKTransport.

Tests cover: init, start, send, read_stream, stop, permission handling,
message-to-event conversion, and approval routing.
All SDK classes are mocked — no API key needed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Mock SDK types before importing SDKTransport
@pytest.fixture(autouse=True)
def mock_sdk(monkeypatch):
    """Ensure SDK imports succeed even if not installed."""
    pass  # SDK is installed in this env, so no patching needed


from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
from agent_os.agent.transports.base import TransportEvent


class TestSDKTransportInit:
    def test_has_sdk_flag(self):
        assert HAS_SDK is True

    def test_init(self):
        transport = SDKTransport()
        assert transport._client is None
        assert transport._session_id is None
        assert transport._alive is False
        assert transport._pending_approvals == {}


class TestSDKTransportStart:
    @pytest.mark.asyncio
    async def test_start_creates_client(self):
        transport = SDKTransport()

        with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as MockClient, \
             patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as MockOptions:
            MockClient.return_value.connect = AsyncMock()
            await transport.start("claude", ["--verbose"], "/tmp/workspace")

        assert transport._client is not None
        assert transport._alive is True
        assert transport._workspace == "/tmp/workspace"

    @pytest.mark.asyncio
    async def test_start_passes_options(self):
        transport = SDKTransport()

        with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as MockClient, \
             patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as MockOptions:
            MockClient.return_value.connect = AsyncMock()
            await transport.start("claude", [], "/workspace")

            MockOptions.assert_called_once()
            opts_kwargs = MockOptions.call_args[1]
            assert opts_kwargs["cwd"] == "/workspace"
            assert opts_kwargs["permission_mode"] == "default"
            assert opts_kwargs["can_use_tool"] is not None
            assert opts_kwargs["cli_path"] == "claude"


class TestSDKTransportSend:
    @pytest.mark.asyncio
    async def test_send_returns_text(self):
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

        transport = SDKTransport()
        mock_client = AsyncMock()

        real_text = TextBlock(text="Hello from SDK!")
        real_assistant = AssistantMessage(
            content=[real_text], model="claude-3",
            parent_tool_use_id=None, error=None,
        )
        real_result = ResultMessage(
            subtype="success", duration_ms=100, duration_api_ms=80,
            is_error=False, num_turns=1, session_id="sdk-session-1",
            total_cost_usd=0.01, usage={}, result=None, structured_output=None,
        )

        async def real_receive():
            yield real_assistant
            yield real_result

        mock_client.receive_response = lambda: real_receive()
        transport._client = mock_client
        transport._alive = True

        response = await transport.send("hello")

        mock_client.query.assert_called_once_with("hello")
        assert response == "Hello from SDK!"
        assert transport.session_id == "sdk-session-1"

    @pytest.mark.asyncio
    async def test_send_no_client_returns_error(self):
        transport = SDKTransport()
        transport._client = None
        response = await transport.send("hello")
        assert "Error" in response

    @pytest.mark.asyncio
    async def test_send_query_exception(self):
        transport = SDKTransport()
        mock_client = AsyncMock()
        mock_client.query.side_effect = Exception("connection failed")
        transport._client = mock_client
        transport._alive = True

        response = await transport.send("hello")
        assert "Error" in response
        assert "connection failed" in response

    @pytest.mark.asyncio
    async def test_send_no_response(self):
        transport = SDKTransport()
        mock_client = AsyncMock()

        async def empty_receive():
            return
            yield  # make it an async generator

        mock_client.receive_response = lambda: empty_receive()
        transport._client = mock_client
        transport._alive = True

        response = await transport.send("hello")
        assert response == "(no response)"


class TestSDKTransportLifecycle:
    def test_is_alive_false_initially(self):
        transport = SDKTransport()
        assert transport.is_alive() is False

    @pytest.mark.asyncio
    async def test_stop_sets_alive_false(self):
        transport = SDKTransport()
        transport._alive = True
        mock_client = AsyncMock()
        transport._client = mock_client

        await transport.stop()

        assert transport.is_alive() is False
        assert transport._client is None
        assert transport.session_id is None
        mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_denies_pending_approvals(self):
        transport = SDKTransport()
        transport._alive = True
        transport._client = AsyncMock()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        transport._pending_approvals["req-1"] = future

        await transport.stop()

        assert future.done()
        assert future.result() is False
        assert len(transport._pending_approvals) == 0

    def test_session_id_default_none(self):
        transport = SDKTransport()
        assert transport.session_id is None


class TestSDKTransportApproval:
    @pytest.mark.asyncio
    async def test_respond_to_permission_approve(self):
        transport = SDKTransport()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        transport._pending_approvals["req-1"] = future

        await transport.respond_to_permission("req-1", True)

        assert future.done()
        assert future.result() is True
        assert "req-1" not in transport._pending_approvals

    @pytest.mark.asyncio
    async def test_respond_to_permission_deny(self):
        transport = SDKTransport()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        transport._pending_approvals["req-1"] = future

        await transport.respond_to_permission("req-1", False)

        assert future.done()
        assert future.result() is False

    @pytest.mark.asyncio
    async def test_respond_to_permission_unknown_id(self):
        transport = SDKTransport()
        # Should not raise
        await transport.respond_to_permission("unknown-id", True)

    @pytest.mark.asyncio
    async def test_handle_permission_emits_event_and_waits(self):
        transport = SDKTransport()
        transport._alive = True

        async def approve_after_delay():
            await asyncio.sleep(0.05)
            # Find the pending approval and resolve it
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(True)
                    break

        from claude_agent_sdk import PermissionResultAllow

        task = asyncio.create_task(approve_after_delay())
        result = await transport._handle_permission("Bash", {"command": "ls"}, None)
        await task

        assert isinstance(result, PermissionResultAllow)
        # Check that event was queued
        assert not transport._event_queue.empty()
        event = transport._event_queue.get_nowait()
        assert event.event_type == "permission_request"
        assert event.data["tool_name"] == "Bash"

    @pytest.mark.asyncio
    async def test_handle_permission_deny(self):
        transport = SDKTransport()
        transport._alive = True

        async def deny_after_delay():
            await asyncio.sleep(0.05)
            for req_id, future in transport._pending_approvals.items():
                if not future.done():
                    future.set_result(False)
                    break

        from claude_agent_sdk import PermissionResultDeny

        task = asyncio.create_task(deny_after_delay())
        result = await transport._handle_permission("Bash", {"command": "rm -rf /"}, None)
        await task

        assert isinstance(result, PermissionResultDeny)
        assert result.message == "Denied by user"


class TestMessageToEvents:
    def test_assistant_text_block(self):
        from claude_agent_sdk.types import AssistantMessage, TextBlock
        transport = SDKTransport()

        msg = AssistantMessage(
            content=[TextBlock(text="Hello!")],
            model="claude-3", parent_tool_use_id=None, error=None,
        )
        events = transport._message_to_events(msg)

        assert len(events) == 1
        assert events[0].event_type == "message"
        assert events[0].raw_text == "Hello!"

    def test_assistant_tool_use_block(self):
        from claude_agent_sdk.types import AssistantMessage, ToolUseBlock
        transport = SDKTransport()

        msg = AssistantMessage(
            content=[ToolUseBlock(id="tu-1", name="Read", input={"path": "/foo"})],
            model="claude-3", parent_tool_use_id=None, error=None,
        )
        events = transport._message_to_events(msg)

        assert len(events) == 1
        assert events[0].event_type == "tool_use"
        assert events[0].data["tool_name"] == "Read"
        assert events[0].data["tool_id"] == "tu-1"

    def test_result_message_captures_session_id(self):
        from claude_agent_sdk.types import ResultMessage
        transport = SDKTransport()

        msg = ResultMessage(
            subtype="success", duration_ms=100, duration_api_ms=80,
            is_error=False, num_turns=1, session_id="sess-abc",
            total_cost_usd=0.01, usage={}, result=None, structured_output=None,
        )
        events = transport._message_to_events(msg)

        assert len(events) == 0  # success result produces no events
        assert transport._session_id == "sess-abc"

    def test_result_message_error(self):
        from claude_agent_sdk.types import ResultMessage
        transport = SDKTransport()

        msg = ResultMessage(
            subtype="error", duration_ms=100, duration_api_ms=80,
            is_error=True, num_turns=1, session_id="sess-abc",
            total_cost_usd=None, usage=None, result="API key invalid",
            structured_output=None,
        )
        events = transport._message_to_events(msg)

        assert len(events) == 1
        assert events[0].event_type == "error"
        assert "API key invalid" in events[0].raw_text

    def test_mixed_content(self):
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolUseBlock
        transport = SDKTransport()

        msg = AssistantMessage(
            content=[
                TextBlock(text="Let me check that."),
                ToolUseBlock(id="tu-1", name="Read", input={"path": "/foo"}),
            ],
            model="claude-3", parent_tool_use_id=None, error=None,
        )
        events = transport._message_to_events(msg)

        assert len(events) == 2
        assert events[0].event_type == "message"
        assert events[1].event_type == "tool_use"


class TestSubAgentManagerApprovalRouting:
    """Test that SubAgentManager routes approvals to SDK transport."""

    @pytest.mark.asyncio
    async def test_resolve_sub_agent_approval_found(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        # Create mock adapter with transport that has pending approval
        mock_transport = AsyncMock()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        mock_transport._pending_approvals = {"req-123": future}

        mock_adapter = MagicMock()
        mock_adapter._transport = mock_transport
        mgr._adapters["proj_1"] = {"test-agent": mock_adapter}

        result = await mgr.resolve_sub_agent_approval("proj_1", "req-123", approved=True)

        assert result is True
        mock_transport.respond_to_permission.assert_called_once_with("req-123", True)

    @pytest.mark.asyncio
    async def test_resolve_sub_agent_approval_not_found(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        result = await mgr.resolve_sub_agent_approval("proj_1", "req-123", approved=True)

        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_sub_agent_approval_wrong_request_id(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        mock_transport = AsyncMock()
        mock_transport._pending_approvals = {"other-id": asyncio.get_event_loop().create_future()}

        mock_adapter = MagicMock()
        mock_adapter._transport = mock_transport
        mgr._adapters["proj_1"] = {"test-agent": mock_adapter}

        result = await mgr.resolve_sub_agent_approval("proj_1", "req-123", approved=True)

        assert result is False


class TestFlushStaleMessages:
    """Test the defensive flush mechanism for unknown message types."""

    @pytest.mark.asyncio
    async def test_needs_flush_initially_false(self):
        transport = SDKTransport()
        assert transport._needs_flush is False

    @pytest.mark.asyncio
    async def test_send_sets_needs_flush_when_no_result_message(self):
        """When receive_response() crashes before ResultMessage, _needs_flush is set."""
        from claude_agent_sdk.types import AssistantMessage, TextBlock

        transport = SDKTransport()
        mock_client = AsyncMock()

        real_text = TextBlock(text="partial answer")
        real_assistant = AssistantMessage(
            content=[real_text], model="claude-3",
            parent_tool_use_id=None, error=None,
        )

        call_count = 0

        async def crash_receive():
            nonlocal call_count
            call_count += 1
            yield real_assistant
            raise Exception("Unknown message type: rate_limit_event")

        mock_client.receive_response = crash_receive
        transport._client = mock_client
        transport._alive = True

        response = await transport.send("hello")

        assert response == "partial answer"
        assert transport._needs_flush is True

    @pytest.mark.asyncio
    async def test_send_clears_needs_flush_on_normal_response(self):
        """When receive_response() completes with ResultMessage, _needs_flush stays False."""
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

        transport = SDKTransport()
        mock_client = AsyncMock()

        async def normal_receive():
            yield AssistantMessage(
                content=[TextBlock(text="answer")], model="claude-3",
                parent_tool_use_id=None, error=None,
            )
            yield ResultMessage(
                subtype="success", duration_ms=100, duration_api_ms=80,
                is_error=False, num_turns=1, session_id="s1",
                total_cost_usd=0.01, usage={}, result=None, structured_output=None,
            )

        mock_client.receive_response = lambda: normal_receive()
        transport._client = mock_client
        transport._alive = True

        response = await transport.send("hello")

        assert response == "answer"
        assert transport._needs_flush is False

    @pytest.mark.asyncio
    async def test_flush_drains_stale_result_message(self):
        """_flush_stale_messages() consumes a leftover ResultMessage."""
        from claude_agent_sdk.types import ResultMessage

        transport = SDKTransport()
        mock_client = AsyncMock()
        transport._needs_flush = True

        stale_result = ResultMessage(
            subtype="success", duration_ms=50, duration_api_ms=40,
            is_error=False, num_turns=1, session_id="stale-sess",
            total_cost_usd=0.01, usage={}, result=None, structured_output=None,
        )

        async def flush_receive():
            yield stale_result

        mock_client.receive_response = lambda: flush_receive()
        transport._client = mock_client

        await transport._flush_stale_messages()

        assert transport._needs_flush is False
        assert transport._session_id == "stale-sess"

    @pytest.mark.asyncio
    async def test_flush_called_before_query_when_needed(self):
        """When _needs_flush is True, send() flushes before issuing query()."""
        from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

        transport = SDKTransport()
        mock_client = AsyncMock()
        transport._needs_flush = True
        transport._client = mock_client
        transport._alive = True

        call_sequence = []

        stale_result = ResultMessage(
            subtype="success", duration_ms=50, duration_api_ms=40,
            is_error=False, num_turns=1, session_id="stale-sess",
            total_cost_usd=0.01, usage={}, result=None, structured_output=None,
        )
        fresh_assistant = AssistantMessage(
            content=[TextBlock(text="correct answer")], model="claude-3",
            parent_tool_use_id=None, error=None,
        )
        fresh_result = ResultMessage(
            subtype="success", duration_ms=100, duration_api_ms=80,
            is_error=False, num_turns=1, session_id="fresh-sess",
            total_cost_usd=0.01, usage={}, result=None, structured_output=None,
        )

        receive_call = 0

        async def multi_receive():
            nonlocal receive_call
            receive_call += 1
            if receive_call == 1:
                # Flush call: yield stale result
                call_sequence.append("flush")
                yield stale_result
            else:
                # Real response
                call_sequence.append("response")
                yield fresh_assistant
                yield fresh_result

        mock_client.receive_response = multi_receive

        response = await transport.send("question 2")

        assert response == "correct answer"
        assert transport._needs_flush is False
        assert transport._session_id == "fresh-sess"
        assert call_sequence == ["flush", "response"]

    @pytest.mark.asyncio
    async def test_flush_handles_error_gracefully(self):
        """_flush_stale_messages() handles errors without crashing."""
        transport = SDKTransport()
        mock_client = AsyncMock()
        transport._needs_flush = True

        async def error_receive():
            raise Exception("stream broken")
            yield  # make it async gen

        mock_client.receive_response = error_receive
        transport._client = mock_client

        # Should not raise
        await transport._flush_stale_messages()

        assert transport._needs_flush is False


class TestReadStream:
    @pytest.mark.asyncio
    async def test_read_stream_yields_queued_events(self):
        transport = SDKTransport()
        transport._alive = True

        event = TransportEvent(event_type="tool_use", data={"tool_name": "Bash"}, raw_text="[Using tool: Bash]")
        await transport._event_queue.put(event)

        # Set alive to False after event so read_stream terminates
        async def stop_later():
            await asyncio.sleep(0.1)
            transport._alive = False

        task = asyncio.create_task(stop_later())

        events = []
        async for e in transport.read_stream():
            events.append(e)
            transport._alive = False  # stop after first event

        await task

        assert len(events) == 1
        assert events[0].event_type == "tool_use"
