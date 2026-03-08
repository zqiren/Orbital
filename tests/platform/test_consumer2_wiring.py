# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-wiring contract tests for Consumer 2: ShellTool -> PlatformProvider.run_command().

These tests define what the Wirer must implement. They verify:
1. Unit: ShellTool calls provider.run_command() when provider is present
2. Unit: ShellTool falls back to subprocess.run() when provider is None
3. Unit: ShellTool correctly maps its params to run_command() params
4. Integration: execute 'echo hello' through ShellTool with NullProvider
5. E2E: execute 'whoami' with WindowsPlatformProvider, assert output contains 'AgentOS-Worker'
6. E2E: attempt 'dir C:\\Users\\{real_user}\\Desktop' — must fail

Note: ShellTool.execute() is synchronous but calls asyncio.run() internally
when using the provider. Tests that check return values must be non-async
so asyncio.run() can create its own event loop.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.tools.base import ToolResult
from agent_os.platform.types import CommandResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_shell_tool(workspace="/tmp/ws", os_type="windows",
                     platform_provider=None, project_id="proj_1"):
    """Create a ShellTool with optional platform_provider and project_id."""
    from agent_os.agent.tools.shell import ShellTool

    return ShellTool(
        workspace=workspace,
        os_type=os_type,
        platform_provider=platform_provider,
        project_id=project_id,
    )


def _make_mock_provider(return_value=None, side_effect=None):
    """Create a mock provider with a proper async run_command.

    Uses a real coroutine function to be compatible with asyncio.run().
    """
    provider = MagicMock()
    _return_value = return_value or CommandResult(
        exit_code=0, stdout="", stderr="", timed_out=False
    )
    _calls = []

    async def mock_run_command(**kwargs):
        _calls.append(kwargs)
        if side_effect:
            raise side_effect
        return _return_value

    async def mock_run_command_positional(project_id, command, args,
                                          working_dir, timeout_sec=300,
                                          extra_env=None):
        call_info = {
            "project_id": project_id,
            "command": command,
            "args": args,
            "working_dir": working_dir,
            "timeout_sec": timeout_sec,
            "extra_env": extra_env,
        }
        _calls.append(call_info)
        if side_effect:
            raise side_effect
        return _return_value

    provider.run_command = mock_run_command_positional
    provider._calls = _calls
    return provider


# ---------------------------------------------------------------------------
# Unit tests: ShellTool calls provider.run_command() when provider is present
# ---------------------------------------------------------------------------


class TestShellToolCallsProvider:
    """When platform_provider is set, ShellTool delegates to provider.run_command()."""

    def test_execute_calls_run_command(self):
        """ShellTool.execute() calls provider.run_command() when provider is present."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="hello\n", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(platform_provider=provider, project_id="proj_1")
        result = tool.execute(command="echo hello")

        assert len(provider._calls) == 1

    def test_execute_returns_stdout_from_provider(self):
        """ShellTool returns the stdout from provider.run_command() in content."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="provider_output\n", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(platform_provider=provider)
        result = tool.execute(command="echo test")

        assert "provider_output" in result.content

    def test_execute_includes_exit_code_from_provider(self):
        """ShellTool includes exit code from provider result."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=42, stdout="", stderr="some error", timed_out=False)
        )

        tool = _make_shell_tool(platform_provider=provider)
        result = tool.execute(command="failing_cmd")

        assert "42" in result.content

    def test_execute_handles_timeout_from_provider(self):
        """ShellTool handles timed_out=True from provider.run_command()."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=-1, stdout="partial output", stderr="", timed_out=True)
        )

        tool = _make_shell_tool(platform_provider=provider)
        result = tool.execute(command="long_running_cmd")

        assert "partial output" in result.content or "timed out" in result.content.lower()

    def test_execute_wraps_provider_error(self):
        """ShellTool wraps RuntimeError from provider in ToolResult (tools never raise)."""
        provider = _make_mock_provider(side_effect=RuntimeError("sandbox not set up"))

        tool = _make_shell_tool(platform_provider=provider)
        result = tool.execute(command="echo hello")

        assert isinstance(result, ToolResult)
        assert "Error" in result.content or "error" in result.content


# ---------------------------------------------------------------------------
# Unit tests: ShellTool falls back to subprocess.run() when provider is None
# ---------------------------------------------------------------------------


class TestShellToolFallback:
    """When platform_provider is None, ShellTool uses subprocess.run() (legacy mode)."""

    def test_execute_uses_subprocess_when_no_provider(self):
        """ShellTool falls back to subprocess.run() when provider is None."""
        tool = _make_shell_tool(platform_provider=None, os_type="windows")

        with patch("agent_os.agent.tools.shell.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="subprocess_output\n", stderr=""
            )
            result = tool.execute(command="echo hello")

            mock_run.assert_called_once()
            assert "subprocess_output" in result.content

    def test_execute_uses_subprocess_on_existing_behavior(self):
        """Legacy behavior: without provider, subprocess.run is called with correct shell."""
        tool = _make_shell_tool(platform_provider=None, os_type="linux")

        with patch("agent_os.agent.tools.shell.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="linux output\n", stderr=""
            )
            result = tool.execute(command="ls -la")

            # Should use bash for linux
            call_args = mock_run.call_args
            cmd = call_args[0][0] if call_args[0] else call_args[1].get("args")
            assert "bash" in cmd


# ---------------------------------------------------------------------------
# Unit tests: ShellTool correctly maps params to run_command()
# ---------------------------------------------------------------------------


class TestShellToolParamMapping:
    """ShellTool must correctly map its params to provider.run_command() params."""

    def test_maps_project_id(self):
        """run_command() receives the project_id from ShellTool."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(
            platform_provider=provider,
            project_id="my_project",
        )
        tool.execute(command="echo hello")

        assert len(provider._calls) == 1
        assert provider._calls[0]["project_id"] == "my_project"

    def test_maps_working_dir_to_workspace(self):
        """run_command() receives workspace as working_dir."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(
            workspace="/my/workspace",
            platform_provider=provider,
        )
        tool.execute(command="echo hello")

        assert len(provider._calls) == 1
        assert provider._calls[0]["working_dir"] == "/my/workspace"

    def test_maps_command_for_windows(self):
        """On windows, run_command() receives 'powershell' as command with args."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(
            os_type="windows",
            platform_provider=provider,
        )
        tool.execute(command="npm install")

        assert len(provider._calls) == 1
        assert provider._calls[0]["command"] == "powershell"

    def test_maps_command_for_linux(self):
        """On linux, run_command() receives 'bash' as command with -c arg."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(
            os_type="linux",
            platform_provider=provider,
        )
        tool.execute(command="ls -la")

        assert len(provider._calls) == 1
        assert provider._calls[0]["command"] == "bash"

    def test_passes_timeout(self):
        """run_command() receives timeout_sec."""
        provider = _make_mock_provider(
            return_value=CommandResult(exit_code=0, stdout="", stderr="", timed_out=False)
        )

        tool = _make_shell_tool(platform_provider=provider)
        tool.execute(command="echo hello")

        assert len(provider._calls) == 1
        assert provider._calls[0]["timeout_sec"] == 120


# ---------------------------------------------------------------------------
# Integration: execute 'echo hello' through ShellTool with NullProvider
# ---------------------------------------------------------------------------


class TestShellToolNullProviderIntegration:
    """Integration test: ShellTool with NullProvider."""

    def test_echo_with_null_provider(self):
        """Execute 'echo hello' through ShellTool backed by NullProvider."""
        from agent_os.platform.null import NullProvider

        provider = NullProvider()
        tool = _make_shell_tool(
            platform_provider=provider,
            project_id="int_test",
        )

        result = tool.execute(command="echo hello")
        assert isinstance(result, ToolResult)
        # NullProvider.run_command returns CommandResult(exit_code=0, stdout="", stderr="")
        # So the result should show exit code 0
        assert "0" in result.content


# ---------------------------------------------------------------------------
# E2E: WindowsPlatformProvider tests (skipped on non-Windows)
# ---------------------------------------------------------------------------

from tests.platform.conftest import skip_not_windows, skip_no_sandbox, HAS_SANDBOX_USER


@skip_not_windows
@skip_no_sandbox
class TestShellToolE2E:
    """E2E tests with real WindowsPlatformProvider. Require sandbox user."""

    def test_whoami_returns_sandbox_user(self):
        """Execute 'whoami' via ShellTool+WindowsPlatformProvider, assert AgentOS-Worker."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()
        caps = provider.get_capabilities()
        if not caps.setup_complete:
            pytest.skip("Platform setup not complete")

        workspace = os.environ.get("TEMP", "C:\\Temp")
        # Mirror production: grant sandbox user access before launching
        provider.grant_folder_access(workspace, "read_write")

        tool = _make_shell_tool(
            workspace=workspace,
            os_type="windows",
            platform_provider=provider,
            project_id="e2e_whoami",
        )

        result = tool.execute(command="whoami")
        assert isinstance(result, ToolResult)
        # Output should contain the sandbox user name
        assert "agentos-worker" in result.content.lower() or \
               "agentosworker" in result.content.lower()

    def test_desktop_access_denied(self):
        """Attempt to access real user's Desktop — must fail."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()
        caps = provider.get_capabilities()
        if not caps.setup_complete:
            pytest.skip("Platform setup not complete")

        real_user = os.environ.get("USERNAME", "")
        desktop_path = f"C:\\Users\\{real_user}\\Desktop"
        if not os.path.exists(desktop_path):
            pytest.skip(f"Desktop path does not exist: {desktop_path}")

        workspace = os.environ.get("TEMP", "C:\\Temp")
        # Mirror production: grant sandbox user access to workspace (not Desktop)
        provider.grant_folder_access(workspace, "read_write")

        tool = _make_shell_tool(
            workspace=workspace,
            os_type="windows",
            platform_provider=provider,
            project_id="e2e_escape",
        )

        result = tool.execute(command=f'dir "{desktop_path}"')
        assert isinstance(result, ToolResult)
        content_lower = result.content.lower()
        # Must indicate access denied or error
        assert "access is denied" in content_lower or \
               "not find" in content_lower or \
               "error" in content_lower or \
               "denied" in content_lower
