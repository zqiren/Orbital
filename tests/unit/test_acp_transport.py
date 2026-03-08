# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for ACP transport (JSON-RPC over stdio)."""
import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch
from agent_os.agent.transports.acp_transport import ACPTransport


class MockACPProcess:
    """Simulates an ACP agent subprocess for testing."""
    def __init__(self, responses=None):
        self._responses = responses or []
        self._response_idx = 0
        self._stdin_lines = []
        self.returncode = None

        # Create mock stdin/stdout
        self.stdin = MagicMock()
        self.stdin.write = self._capture_stdin
        self.stdin.flush = MagicMock()

        self.stdout = MagicMock()
        self.stdout.readline = self._readline
        self.stderr = MagicMock()

    def _capture_stdin(self, data):
        self._stdin_lines.append(data)

    def _readline(self):
        if self._response_idx < len(self._responses):
            resp = self._responses[self._response_idx]
            self._response_idx += 1
            if isinstance(resp, dict):
                return (json.dumps(resp) + "\n").encode()
            return resp
        return b""

    def poll(self):
        return self.returncode

    def terminate(self): self.returncode = -1
    def kill(self): self.returncode = -9
    def wait(self, timeout=None): pass


class TestACPTransportWindowsCompat:
    @pytest.mark.asyncio
    async def test_start_uses_shell_for_cmd_files(self):
        """On Windows, .cmd files need shell=True."""
        import sys
        if sys.platform != "win32":
            pytest.skip("Windows-only test")
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        mock_proc = MockACPProcess(responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc) as mock_popen:
            await t.start("C:/path/to/claude.cmd", ["acp"], ".")
        call_args = mock_popen.call_args
        assert call_args[1].get("shell") is True

    @pytest.mark.asyncio
    async def test_start_no_shell_for_non_cmd(self):
        """Non-.cmd files should not use shell=True."""
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        mock_proc = MockACPProcess(responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc) as mock_popen:
            await t.start("claude", ["acp"], ".")
        call_args = mock_popen.call_args
        assert call_args[1].get("shell") is False


class TestACPTransportInitialize:
    @pytest.mark.asyncio
    async def test_start_sends_initialize_and_session_new(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {
                "protocolVersion": 1,
                "agentInfo": {"name": "test", "version": "0.1"},
                "agentCapabilities": {},
            }},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        mock_proc = MockACPProcess(responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc):
            await t.start("agent", [], ".")
        assert t.session_id == "sess-001"
        # Verify initialize and session/new were sent
        assert len(mock_proc._stdin_lines) == 2
        init_msg = json.loads(mock_proc._stdin_lines[0])
        assert init_msg["method"] == "initialize"
        new_msg = json.loads(mock_proc._stdin_lines[1])
        assert new_msg["method"] == "session/new"

    @pytest.mark.asyncio
    async def test_start_emits_session_created_event(self):
        responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-002"}},
        ]
        mock_proc = MockACPProcess(responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc):
            await t.start("agent", [], ".")
        # Check event queue has session_created
        event = t._event_queue.get_nowait()
        assert event.event_type == "session_created"
        assert event.data["session_id"] == "sess-002"


class TestACPTransportSend:
    @pytest.mark.asyncio
    async def test_send_sends_session_prompt(self):
        init_responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        # After init, send will read: text update + prompt response
        prompt_responses = [
            {"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-001",
                "update": {"kind": "text", "text": "Echo: hello"},
            }},
            {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "endTurn"}},
        ]
        mock_proc = MockACPProcess(init_responses + prompt_responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc):
            await t.start("agent", [], ".")
        result = await t.send("hello")
        assert "Echo: hello" in result


class TestACPTransportPermissions:
    @pytest.mark.asyncio
    async def test_respond_to_permission_sends_json_rpc(self):
        init_responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        mock_proc = MockACPProcess(init_responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc):
            await t.start("agent", [], ".")
        await t.respond_to_permission("perm_001", True)
        last_msg = json.loads(mock_proc._stdin_lines[-1])
        assert last_msg["method"] == "session/permissionResponse"
        assert last_msg["params"]["permissionId"] == "perm_001"
        assert last_msg["params"]["granted"] is True


class TestACPTransportLifecycle:
    @pytest.mark.asyncio
    async def test_stop_sends_shutdown(self):
        init_responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        shutdown_response = [
            {"jsonrpc": "2.0", "id": 3, "result": {}},
        ]
        mock_proc = MockACPProcess(init_responses + shutdown_response)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc):
            await t.start("agent", [], ".")
        await t.stop()
        # Verify shutdown was sent
        shutdown_msg = json.loads(mock_proc._stdin_lines[-1])
        assert shutdown_msg["method"] == "shutdown"
        assert not t.is_alive()

    def test_is_alive_false_without_start(self):
        t = ACPTransport()
        assert t.is_alive() is False

    @pytest.mark.asyncio
    async def test_parse_session_update_text(self):
        t = ACPTransport()
        event = t._parse_session_update({
            "sessionId": "s1",
            "update": {"kind": "text", "text": "hello world"},
        })
        assert event.event_type == "message"
        assert event.raw_text == "hello world"

    @pytest.mark.asyncio
    async def test_parse_session_update_permission_request(self):
        t = ACPTransport()
        event = t._parse_session_update({
            "sessionId": "s1",
            "update": {
                "kind": "permissionRequest",
                "permissionId": "p1",
                "tool": {"name": "shell", "params": {"cmd": "rm -rf /"}},
                "reason": "need shell access",
            },
        })
        assert event.event_type == "permission_request"
        assert event.data["permission_id"] == "p1"
        assert event.data["tool_name"] == "shell"

    @pytest.mark.asyncio
    async def test_parse_session_update_tool_call(self):
        t = ACPTransport()
        event = t._parse_session_update({
            "sessionId": "s1",
            "update": {"kind": "toolCall", "tool": {"name": "read_file"}},
        })
        assert event.event_type == "tool_use"
        assert event.data["tool_name"] == "read_file"

    @pytest.mark.asyncio
    async def test_parse_session_update_error(self):
        t = ACPTransport()
        event = t._parse_session_update({
            "sessionId": "s1",
            "update": {"kind": "error", "message": "something failed"},
        })
        assert event.event_type == "error"
        assert "something failed" in event.raw_text


class TestACPAutoApprove:
    @pytest.mark.asyncio
    async def test_permission_request_auto_approved(self):
        """For MVP, permission requests should be auto-approved."""
        init_responses = [
            {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": 1, "agentInfo": {}, "agentCapabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess-001"}},
        ]
        prompt_responses = [
            {"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-001",
                "update": {
                    "kind": "permissionRequest",
                    "permissionId": "perm-abc",
                    "tool": {"name": "Bash", "params": {"cmd": "ls"}},
                    "reason": "needs shell",
                },
            }},
            {"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": "sess-001",
                "update": {"kind": "text", "text": "Done after approval"},
            }},
            {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "endTurn"}},
        ]
        mock_proc = MockACPProcess(init_responses + prompt_responses)
        t = ACPTransport()
        with patch("agent_os.agent.transports.acp_transport.subprocess.Popen", return_value=mock_proc):
            await t.start("agent", [], ".")
        result = await t.send("do something")
        # Verify auto-approve was sent
        approval_sent = False
        for line in mock_proc._stdin_lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            msg = json.loads(line)
            if msg.get("method") == "session/permissionResponse":
                assert msg["params"]["granted"] is True
                assert msg["params"]["permissionId"] == "perm-abc"
                approval_sent = True
        assert approval_sent, "Permission auto-approve should have been sent"
        assert "Done after approval" in result
