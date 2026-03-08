# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for PipeTransport and pipe-mode wiring.

Tests cover: PipeTransport.send(), _parse_output(), session resume,
timeout handling, start/stop lifecycle, read_stream no-op, and
SubAgentManager wiring through CLIAdapter -> PipeTransport.
"""

import asyncio
import json
import os
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.pipe_transport import (
    PipeTransport,
    PipeTransportConfig,
    CLAUDE_CODE_PIPE_CONFIG,
)


# ---------------------------------------------------------------------------
# PipeTransport core tests
# ---------------------------------------------------------------------------


class TestPipeTransportSend:
    """Test PipeTransport.send() subprocess invocation."""

    @pytest.mark.asyncio
    async def test_send_builds_correct_command(self):
        config = PipeTransportConfig(
            prompt_flag="-p",
            output_format_args=["--output-format", "stream-json"],
        )
        transport = PipeTransport(config=config)
        await transport.start("claude", ["--verbose"], "/tmp")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello!"}]},
        }).encode() + b"\n"
        mock_result.stderr = b""

        with patch("agent_os.agent.transports.pipe_transport.subprocess.run", return_value=mock_result) as mock_run:
            response = await transport.send("say hello")

        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == "claude"
        assert "--verbose" in called_cmd
        assert "-p" in called_cmd
        assert "say hello" in called_cmd
        assert "Hello!" in response

    @pytest.mark.asyncio
    async def test_send_extracts_session_id(self):
        config = PipeTransportConfig(
            prompt_flag="-p",
            session_id_pattern=r'"session_id":\s*"([^"]+)"',
        )
        transport = PipeTransport(config=config)
        await transport.start("claude", [], "/tmp")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Hi"}]}}\n'
            '{"type":"result","session_id":"sess-xyz-123"}\n'
        ).encode()
        mock_result.stderr = b""

        with patch("agent_os.agent.transports.pipe_transport.subprocess.run", return_value=mock_result):
            await transport.send("hi")

        assert transport.session_id == "sess-xyz-123"

    @pytest.mark.asyncio
    async def test_send_resumes_with_session_id_and_resume_flag(self):
        config = PipeTransportConfig(
            prompt_flag="-p",
            resume_flag="--resume",
            session_id_pattern=r'"session_id":\s*"([^"]+)"',
        )
        transport = PipeTransport(config=config)
        await transport.start("claude", ["--output-format", "stream-json"], "/tmp")
        transport._session_id = "prev-session-456"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"type":"assistant","message":{"content":[{"type":"text","text":"Resumed"}]}}\n'.encode()
        mock_result.stderr = b""

        with patch("agent_os.agent.transports.pipe_transport.subprocess.run", return_value=mock_result) as mock_run:
            await transport.send("follow up")

        called_cmd = mock_run.call_args[0][0]
        assert "--resume" in called_cmd
        assert "prev-session-456" in called_cmd

    @pytest.mark.asyncio
    async def test_send_timeout_returns_error_string(self):
        transport = PipeTransport(config=PipeTransportConfig(prompt_flag="-p"))
        await transport.start("claude", [], "/tmp")

        with patch("agent_os.agent.transports.pipe_transport.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("claude", 300)):
            response = await transport.send("slow task")

        assert "timed out" in response.lower()

    @pytest.mark.asyncio
    async def test_send_nonzero_exit_returns_error_with_stderr(self):
        transport = PipeTransport(config=PipeTransportConfig(prompt_flag="-p"))
        await transport.start("claude", [], "/tmp")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b""
        mock_result.stderr = b"command not found"

        with patch("agent_os.agent.transports.pipe_transport.subprocess.run", return_value=mock_result):
            response = await transport.send("test")

        assert "Error" in response
        assert "command not found" in response


class TestPipeTransportParseOutput:
    """Test PipeTransport._parse_output parsing of stream-json lines."""

    def _make_transport(self):
        return PipeTransport()

    def test_parse_output_assistant_text_blocks(self):
        transport = self._make_transport()
        raw = '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello!"}]}}\n'
        result = transport._parse_output(raw)
        assert result == "Hello!"

    def test_parse_output_multiple_text_blocks(self):
        transport = self._make_transport()
        raw = '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"},{"type":"text","text":" World"}]}}\n'
        result = transport._parse_output(raw)
        assert "Hello" in result
        assert "World" in result

    def test_parse_output_tool_use_markers(self):
        transport = self._make_transport()
        raw = '{"type":"tool_use","name":"read_file"}\n'
        result = transport._parse_output(raw)
        assert "[Using tool: read_file]" in result

    def test_parse_output_error_result(self):
        transport = self._make_transport()
        raw = '{"type":"result","subtype":"error","error":"something broke"}\n'
        result = transport._parse_output(raw)
        assert "Error" in result
        assert "something broke" in result

    def test_parse_output_success_result_ignored(self):
        transport = self._make_transport()
        raw = '{"type":"result","subtype":"success"}\n'
        result = transport._parse_output(raw)
        assert result == "(no response)"

    def test_parse_output_empty_input(self):
        transport = self._make_transport()
        result = transport._parse_output("")
        assert result == "(no response)"

    def test_parse_output_skips_ansi_lines(self):
        transport = self._make_transport()
        raw = "\x1b[0msome ansi\n"
        result = transport._parse_output(raw)
        assert result == "(no response)"

    def test_parse_output_non_json_lines(self):
        transport = self._make_transport()
        raw = "plain text output\n"
        result = transport._parse_output(raw)
        assert result == "plain text output"

    def test_parse_output_mixed(self):
        transport = self._make_transport()
        raw = (
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Hello!"}]}}\n'
            '{"type":"tool_use","name":"shell"}\n'
            '{"type":"assistant","message":{"content":[{"type":"text","text":"Done."}]}}\n'
            '{"type":"result","subtype":"success"}\n'
        )
        result = transport._parse_output(raw)
        assert "Hello!" in result
        assert "[Using tool: shell]" in result
        assert "Done." in result


class TestPipeTransportLifecycle:
    """Test start/stop/is_alive/read_stream lifecycle."""

    @pytest.mark.asyncio
    async def test_start_is_noop(self):
        transport = PipeTransport()
        await transport.start("claude", ["--verbose"], "/tmp", env={"FOO": "bar"})
        assert transport._command == "claude"
        assert transport._args == ["--verbose"]
        assert transport._workspace == "/tmp"
        assert transport._env == {"FOO": "bar"}

    def test_is_alive_always_true(self):
        transport = PipeTransport()
        assert transport.is_alive() is True

    @pytest.mark.asyncio
    async def test_stop_clears_session_id(self):
        transport = PipeTransport()
        transport._session_id = "abc123"
        await transport.stop()
        assert transport.session_id is None

    @pytest.mark.asyncio
    async def test_read_stream_is_empty(self):
        transport = PipeTransport()
        events = []
        async for event in transport.read_stream():
            events.append(event)
        assert events == []


class TestPipeTransportConfig:
    """Test PipeTransportConfig and CLAUDE_CODE_PIPE_CONFIG."""

    def test_default_config(self):
        config = PipeTransportConfig()
        assert config.prompt_flag == "-p"
        assert config.resume_flag == "--resume"
        assert config.session_id_pattern == ""
        assert config.output_format_args == []

    def test_claude_code_config(self):
        assert CLAUDE_CODE_PIPE_CONFIG.prompt_flag == "-p"
        assert CLAUDE_CODE_PIPE_CONFIG.resume_flag == "--resume"
        assert CLAUDE_CODE_PIPE_CONFIG.session_id_pattern != ""
        assert "--output-format" in CLAUDE_CODE_PIPE_CONFIG.output_format_args
        assert "stream-json" in CLAUDE_CODE_PIPE_CONFIG.output_format_args
        assert "--verbose" in CLAUDE_CODE_PIPE_CONFIG.output_format_args


# ---------------------------------------------------------------------------
# Manifest pipe-mode fields
# ---------------------------------------------------------------------------


class TestManifestPipeFields:
    """Test that ManifestRuntime has pipe-mode fields."""

    def test_default_values(self):
        from agent_os.agents.manifest import ManifestRuntime
        rt = ManifestRuntime(adapter="cli")
        assert rt.mode == "interactive"
        assert rt.prompt_flag == "-p"
        assert rt.resume_flag == "--resume"
        assert rt.session_id_pattern == ""
        assert rt.transport == "auto"

    def test_pipe_mode_values(self):
        from agent_os.agents.manifest import ManifestRuntime
        rt = ManifestRuntime(
            adapter="cli", mode="pipe", transport="pipe",
            prompt_flag="-p", resume_flag="--resume",
            session_id_pattern=r'"session_id":\s*"([^"]+)"',
        )
        assert rt.mode == "pipe"
        assert rt.transport == "pipe"
        assert rt.session_id_pattern == r'"session_id":\s*"([^"]+)"'

    def test_claude_code_yaml_has_pipe_fallback_fields(self):
        from agent_os.agents.manifest import ManifestLoader
        path = os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir,
            "agent_os", "agents", "manifests", "claude_code.yaml",
        )
        path = os.path.normpath(path)
        m = ManifestLoader.load(path)
        assert m.runtime.mode == "pipe"
        assert m.runtime.transport == "sdk"  # primary is SDK
        assert m.runtime.prompt_flag == "-p"  # pipe fallback fields preserved
        assert m.runtime.resume_flag == "--resume"
        assert m.runtime.session_id_pattern != ""
        assert "--output-format" in m.runtime.args
        assert "stream-json" in m.runtime.args
        assert "--verbose" in m.runtime.args


# ---------------------------------------------------------------------------
# Transport resolution tests
# ---------------------------------------------------------------------------


class TestPipeTransportResolution:
    """Test _resolve_transport() returns PipeTransport for pipe configs."""

    def test_transport_pipe_gets_pipe_transport(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="pipe", mode="pipe"),
        )
        t = mgr._resolve_transport(manifest, {})
        assert isinstance(t, PipeTransport)

    def test_auto_pipe_mode_gets_pipe_transport_without_sdk(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="T", slug="t", description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="t", transport="auto", mode="pipe"),
        )
        # Patch SDK as unavailable to test pipe fallback
        import agent_os.agent.transports.sdk_transport as sdk_mod
        original_has_sdk = sdk_mod.HAS_SDK
        sdk_mod.HAS_SDK = False
        try:
            t = mgr._resolve_transport(manifest, {})
        finally:
            sdk_mod.HAS_SDK = original_has_sdk
        assert isinstance(t, PipeTransport)

    def test_claude_code_slug_gets_claude_config(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime
        mgr = SubAgentManager(process_manager=MagicMock())
        manifest = AgentManifest(
            manifest_version="1", name="Claude Code", slug="claude-code",
            description="", author="", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="claude", transport="pipe", mode="pipe"),
        )
        t = mgr._resolve_transport(manifest, {})
        assert isinstance(t, PipeTransport)
        assert t._config.session_id_pattern != ""
        assert t._config.output_format_args == ["--output-format", "stream-json", "--verbose"]


# ---------------------------------------------------------------------------
# SubAgentManager wiring tests
# ---------------------------------------------------------------------------


class TestSubAgentManagerPipeWiring:
    """Test SubAgentManager wiring through CLIAdapter -> PipeTransport."""

    @pytest.mark.asyncio
    async def test_pipe_mode_skips_process_manager(self):
        """In pipe mode, process_manager.start() should NOT be called."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.registry import AgentRegistry
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime

        registry = AgentRegistry()
        manifest = AgentManifest(
            manifest_version="1", name="Test Agent", slug="test-agent",
            description="A test agent", author="test", version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="test-cmd", mode="pipe", transport="pipe"),
        )
        registry.register(manifest)

        setup_engine = MagicMock()
        setup_engine.get_adapter_config.return_value = {
            "command": "/usr/bin/test-cmd",
            "args": [], "workspace": "/tmp",
            "approval_patterns": [], "env": {}, "network_domains": [],
        }

        pm = MagicMock()
        pm.start = AsyncMock()

        mgr = SubAgentManager(process_manager=pm, registry=registry, setup_engine=setup_engine)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance
            await mgr.start("proj_1", "test-agent")

        pm.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_returns_pipe_response(self):
        """SubAgentManager.send() should return actual response for pipe transport."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        mock_adapter = AsyncMock()
        mock_adapter._last_response = "Hello from sub-agent!"
        mgr._adapters["proj_1"] = {"test-agent": mock_adapter}

        result = await mgr.send("proj_1", "test-agent", "hi")

        mock_adapter.send.assert_called_once_with("hi")
        assert result == "Hello from sub-agent!"

    @pytest.mark.asyncio
    async def test_send_returns_generic_for_interactive(self):
        """SubAgentManager.send() should return generic message when no _last_response."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        mock_adapter = AsyncMock()
        mock_adapter._last_response = None
        mgr._adapters["proj_1"] = {"test-agent": mock_adapter}

        result = await mgr.send("proj_1", "test-agent", "hi")

        assert result == "Message sent to test-agent"

    @pytest.mark.asyncio
    async def test_send_returns_error_for_unknown_agent(self):
        """SubAgentManager.send() should return error for unknown agent."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)

        result = await mgr.send("proj_1", "nonexistent", "hi")

        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_agent_message_send_returns_response_through_full_chain(self):
        """Mock PipeTransport.send() to return 'Echo: hello', verify full chain."""
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agent.adapters.cli_adapter import CLIAdapter
        from agent_os.agent.adapters.base import AdapterConfig

        mock_transport = AsyncMock()
        mock_transport.send = AsyncMock(return_value="Echo: hello")
        mock_transport.is_alive.return_value = True

        adapter = CLIAdapter(
            handle="test-agent", display_name="Test",
            mode="pipe", transport=mock_transport,
        )
        config = AdapterConfig(command="echo", workspace="/tmp", approval_patterns=[])
        await adapter.start(config)

        pm = MagicMock()
        mgr = SubAgentManager(process_manager=pm)
        mgr._adapters["proj_1"] = {"test-agent": adapter}

        result = await mgr.send("proj_1", "test-agent", "hello")

        mock_transport.send.assert_called_once_with("hello")
        assert result == "Echo: hello"
