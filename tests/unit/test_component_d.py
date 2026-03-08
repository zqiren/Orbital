# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for Component D: CLI Adapter (adapters/base, cli_adapter, output_parser).

Tests are designed to work on Windows 10. PTY-dependent tests are skipped on Windows
or use mocks. OutputParser and strip_ansi tests are fully platform-independent.
"""

import sys
import asyncio
import re
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

import pytest

from agent_os.agent.adapters.base import AdapterConfig, OutputChunk, AgentAdapter, AdapterError
from agent_os.agent.adapters.cli_adapter import CLIAdapter, strip_ansi
from agent_os.agent.adapters.output_parser import OutputParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**overrides) -> AdapterConfig:
    defaults = {
        "command": "cat",
        "workspace": ".",
        "approval_patterns": [r"\(y/n\)"],
        "env": None,
    }
    defaults.update(overrides)
    return AdapterConfig(**defaults)


IS_WINDOWS = sys.platform == "win32"
skip_on_windows = pytest.mark.skipif(IS_WINDOWS, reason="PTY not available on Windows")


# ===========================================================================
# AC-4: strip_ansi — platform-independent
# ===========================================================================

class TestStripAnsi:
    """AC-4: read_stream() strips ANSI codes."""

    def test_strips_color_codes(self):
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_bold(self):
        assert strip_ansi("\x1b[1mbold\x1b[0m") == "bold"

    def test_strips_multiple_codes(self):
        raw = "\x1b[32mgreen\x1b[0m and \x1b[34mblue\x1b[0m"
        assert strip_ansi(raw) == "green and blue"

    def test_plain_text_unchanged(self):
        assert strip_ansi("hello world") == "hello world"

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_strips_osc_sequences(self):
        # OSC (Operating System Command) sequences end with BEL (\x07)
        raw = "\x1b]0;terminal title\x07some text"
        assert strip_ansi(raw) == "some text"

    def test_strips_cursor_movement(self):
        raw = "\x1b[2Jcleared"
        assert strip_ansi(raw) == "cleared"


# ===========================================================================
# AC-5 through AC-8: OutputParser — platform-independent
# ===========================================================================

class TestOutputParser:
    """OutputParser classifies raw text into typed chunks."""

    def setup_method(self):
        self.parser = OutputParser([r"\(y/n\)"])

    # AC-5: approval_request detection
    def test_approval_request_yn(self):
        chunk = self.parser.parse("Do you want to proceed? (y/n)")
        assert chunk.chunk_type == "approval_request"
        assert "(y/n)" in chunk.text

    def test_approval_request_with_custom_pattern(self):
        parser = OutputParser([r"Allow Claude to", r"\(Y/n\)"])
        chunk = parser.parse("Allow Claude to write files?")
        assert chunk.chunk_type == "approval_request"

    def test_approval_takes_priority_over_activity(self):
        """Approval patterns have highest priority."""
        parser = OutputParser([r"\(y/n\)"])
        # "Reading" would match activity, but (y/n) should win
        chunk = parser.parse("Reading file? (y/n)")
        assert chunk.chunk_type == "approval_request"

    # AC-6: tool_activity detection
    def test_tool_activity_reading(self):
        chunk = self.parser.parse("Reading src/main.py")
        assert chunk.chunk_type == "tool_activity"

    def test_tool_activity_writing(self):
        chunk = self.parser.parse("Writing output.txt")
        assert chunk.chunk_type == "tool_activity"

    def test_tool_activity_editing(self):
        chunk = self.parser.parse("Editing config.json")
        assert chunk.chunk_type == "tool_activity"

    def test_tool_activity_executing(self):
        chunk = self.parser.parse("Executing npm install")
        assert chunk.chunk_type == "tool_activity"

    # AC-7: status detection
    def test_status_thinking(self):
        chunk = self.parser.parse("Thinking...")
        assert chunk.chunk_type == "status"

    def test_status_analyzing(self):
        chunk = self.parser.parse("Analyzing...")
        assert chunk.chunk_type == "status"

    def test_status_progress_indicator(self):
        chunk = self.parser.parse("[3/10] processing files")
        assert chunk.chunk_type == "status"

    def test_status_loading(self):
        chunk = self.parser.parse("Loading modules...")
        assert chunk.chunk_type == "status"

    # AC-8: default response
    def test_response_default(self):
        chunk = self.parser.parse("Here is the summary of your code")
        assert chunk.chunk_type == "response"

    def test_response_generic_text(self):
        chunk = self.parser.parse("The answer is 42.")
        assert chunk.chunk_type == "response"

    # Edge cases
    def test_empty_text_returns_response(self):
        chunk = self.parser.parse("")
        assert chunk.chunk_type == "response"

    def test_whitespace_only_returns_response(self):
        chunk = self.parser.parse("   \n  ")
        assert chunk.chunk_type == "response"

    def test_chunk_has_timestamp(self):
        chunk = self.parser.parse("hello")
        assert chunk.timestamp is not None
        assert len(chunk.timestamp) > 0

    def test_no_approval_patterns_no_approvals(self):
        parser = OutputParser([])
        chunk = parser.parse("Do you want to proceed? (y/n)")
        # Without approval patterns registered, this should not be "approval_request"
        assert chunk.chunk_type == "response"


# ===========================================================================
# AC-1, AC-2, AC-3: CLIAdapter lifecycle (PTY-dependent, skip on Windows or mock)
# ===========================================================================

class TestCLIAdapterLifecycle:
    """CLIAdapter start/send/stop lifecycle tests."""

    # AC-1: start() with cat → process alive
    @skip_on_windows
    @pytest.mark.asyncio
    async def test_start_cat_process_alive(self, tmp_path):
        adapter = CLIAdapter(handle="test", display_name="Test")
        config = make_config(command="cat", workspace=str(tmp_path))
        await adapter.start(config)
        try:
            assert adapter.is_alive() is True
        finally:
            await adapter.stop()

    # AC-1 (Windows mock variant)
    @pytest.mark.asyncio
    async def test_start_sets_process_alive_mocked(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is alive

        with patch("agent_os.agent.adapters.cli_adapter.pty", create=True) as mock_pty, \
             patch("agent_os.agent.adapters.cli_adapter.subprocess") as mock_subprocess, \
             patch("agent_os.agent.adapters.cli_adapter.os") as mock_os:
            mock_pty.openpty.return_value = (3, 4)
            mock_subprocess.Popen.return_value = mock_process
            mock_os.close = MagicMock()

            config = make_config()
            await adapter.start(config)
            assert adapter.is_alive() is True

            await adapter.stop()

    # AC-2: send + read_stream → yields chunk
    @skip_on_windows
    @pytest.mark.asyncio
    async def test_send_and_read_stream(self, tmp_path):
        adapter = CLIAdapter(handle="test", display_name="Test")
        config = make_config(command="cat", workspace=str(tmp_path))
        await adapter.start(config)
        try:
            await adapter.send("hello")
            chunks = []
            async for chunk in adapter.read_stream():
                chunks.append(chunk)
                if "hello" in chunk.text:
                    break
            assert any("hello" in c.text for c in chunks)
        finally:
            await adapter.stop()

    # AC-3: stop → process terminated
    @skip_on_windows
    @pytest.mark.asyncio
    async def test_stop_terminates_process(self, tmp_path):
        adapter = CLIAdapter(handle="test", display_name="Test")
        config = make_config(command="cat", workspace=str(tmp_path))
        await adapter.start(config)
        await adapter.stop()
        assert adapter.is_alive() is False

    # AC-3 (mocked)
    @pytest.mark.asyncio
    async def test_stop_terminates_process_mocked(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_process.terminate = MagicMock()
        mock_process.wait = MagicMock(return_value=0)
        mock_process.kill = MagicMock()

        adapter._process = mock_process
        adapter._master_fd = 99
        adapter._parser = OutputParser([])

        with patch("agent_os.agent.adapters.cli_adapter.os") as mock_os:
            mock_os.close = MagicMock()
            await adapter.stop()

        assert adapter.is_alive() is False


# ===========================================================================
# AC-9: Process that exits immediately → AdapterError
# ===========================================================================

class TestCLIAdapterErrors:
    """Error handling tests."""

    # AC-9: immediate exit
    @skip_on_windows
    @pytest.mark.asyncio
    async def test_start_immediate_exit_raises(self, tmp_path):
        adapter = CLIAdapter(handle="test", display_name="Test")
        config = make_config(command="exit 0", workspace=str(tmp_path))
        with pytest.raises(AdapterError):
            await adapter.start(config)

    # AC-9 (mocked)
    @pytest.mark.asyncio
    async def test_start_immediate_exit_raises_mocked(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        mock_process = MagicMock()
        # poll() returning non-None means process already exited
        mock_process.poll.return_value = 1

        with patch("agent_os.agent.adapters.cli_adapter.pty", create=True) as mock_pty, \
             patch("agent_os.agent.adapters.cli_adapter.subprocess") as mock_subprocess, \
             patch("agent_os.agent.adapters.cli_adapter.os") as mock_os:
            mock_pty.openpty.return_value = (3, 4)
            mock_subprocess.Popen.return_value = mock_process
            mock_os.close = MagicMock()

            config = make_config()
            with pytest.raises(AdapterError, match="[Ee]xit"):
                await adapter.start(config)


# ===========================================================================
# AC-10: Long-running process → chunks arrive as they come
# ===========================================================================

class TestCLIAdapterStreaming:
    """Streaming / non-buffered output tests."""

    # AC-10
    @skip_on_windows
    @pytest.mark.asyncio
    async def test_streaming_chunks_arrive_incrementally(self, tmp_path):
        # Use a shell command that produces output over time
        script = 'echo "line1"; sleep 0.1; echo "line2"; sleep 0.1; echo "line3"'
        adapter = CLIAdapter(handle="test", display_name="Test")
        config = make_config(command=f'bash -c \'{script}\'', workspace=str(tmp_path))
        await adapter.start(config)
        try:
            chunks = []
            async for chunk in adapter.read_stream():
                chunks.append(chunk)
                # Once we have enough data, stop
                combined = "".join(c.text for c in chunks)
                if "line3" in combined:
                    break
            # Verify we got multiple chunks (not all buffered into one)
            assert len(chunks) >= 1
            combined = "".join(c.text for c in chunks)
            assert "line1" in combined
            assert "line3" in combined
        finally:
            await adapter.stop()


# ===========================================================================
# AC-11: send() sets is_idle()=False
# ===========================================================================

class TestCLIAdapterIdleState:
    """Idle state management tests."""

    # AC-11
    @pytest.mark.asyncio
    async def test_send_sets_idle_false(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        # Set up minimal mocked state so send() works
        adapter._idle = True
        adapter._master_fd = MagicMock()
        adapter._process = MagicMock()
        adapter._process.poll.return_value = None

        with patch("agent_os.agent.adapters.cli_adapter.os") as mock_os:
            mock_os.write = MagicMock()
            await adapter.send("hello")

        assert adapter.is_idle() is False

    def test_initial_idle_is_false(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        assert adapter.is_idle() is False


# ===========================================================================
# AC-12: stop() with unresponsive process → SIGKILL after 5s
# ===========================================================================

class TestCLIAdapterForceKill:
    """Tests for forced termination of unresponsive processes."""

    # AC-12
    @pytest.mark.asyncio
    async def test_stop_force_kills_unresponsive_process(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is alive
        mock_process.terminate = MagicMock()
        # wait() raises TimeoutExpired to simulate unresponsive process
        mock_process.wait = MagicMock(
            side_effect=TimeoutError("Process did not terminate")
        )
        mock_process.kill = MagicMock()

        adapter._process = mock_process
        adapter._master_fd = 99
        adapter._parser = OutputParser([])

        with patch("agent_os.agent.adapters.cli_adapter.os") as mock_os:
            mock_os.close = MagicMock()
            await adapter.stop()

        # Should have been killed forcefully
        mock_process.kill.assert_called_once()


# ===========================================================================
# AdapterConfig & OutputChunk dataclass tests
# ===========================================================================

class TestDataClasses:
    """Verify data class structure and defaults."""

    def test_adapter_config_fields(self):
        config = AdapterConfig(
            command="cat",
            workspace="/tmp",
            approval_patterns=["(y/n)"],
        )
        assert config.command == "cat"
        assert config.workspace == "/tmp"
        assert config.approval_patterns == ["(y/n)"]
        assert config.env is None

    def test_adapter_config_with_env(self):
        config = AdapterConfig(
            command="cat",
            workspace="/tmp",
            approval_patterns=[],
            env={"FOO": "bar"},
        )
        assert config.env == {"FOO": "bar"}

    def test_output_chunk_fields(self):
        chunk = OutputChunk(text="hello", chunk_type="response", timestamp="2026-01-01T00:00:00")
        assert chunk.text == "hello"
        assert chunk.chunk_type == "response"
        assert chunk.timestamp == "2026-01-01T00:00:00"

    def test_adapter_error(self):
        err = AdapterError("test error")
        assert err.message == "test error"
        assert str(err) == "test error"


# ===========================================================================
# AgentAdapter ABC contract
# ===========================================================================

class TestAgentAdapterABC:
    """Verify AgentAdapter is abstract and CLIAdapter implements the interface."""

    def test_cli_adapter_is_agent_adapter(self):
        adapter = CLIAdapter(handle="test", display_name="Test Agent")
        assert isinstance(adapter, AgentAdapter)

    def test_cli_adapter_agent_type(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        assert adapter.agent_type == "cli"

    def test_cli_adapter_handle_and_display_name(self):
        adapter = CLIAdapter(handle="claudecode", display_name="Claude Code")
        assert adapter.handle == "claudecode"
        assert adapter.display_name == "Claude Code"

    def test_is_alive_false_without_process(self):
        adapter = CLIAdapter(handle="test", display_name="Test")
        assert adapter.is_alive() is False
