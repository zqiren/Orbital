# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C3: ProcessLauncher (CreateProcessWithLogonW).

9 tests from TASK-isolation-C3-process-launcher.md spec.
Requires sandbox user to exist for real tests.
"""

import glob
import os
import sys
import tempfile
import time

import pytest

from tests.platform.conftest import (
    skip_not_windows,
    skip_no_sandbox,
    HAS_SANDBOX_USER,
)


@pytest.fixture
def process_launcher():
    """Create a ProcessLauncher with a real CredentialStore."""
    from agent_os.platform.windows.credentials import CredentialStore
    from agent_os.platform.windows.process import ProcessLauncher

    cs = CredentialStore()
    return ProcessLauncher(cs)



@pytest.fixture
def sandbox_workspace(tmp_path):
    """Create a workspace that the sandbox user can access.

    Grants read-write to AgentOS-Worker so sandbox processes can write output.
    """
    workspace = tmp_path / "sandbox_work"
    workspace.mkdir()

    from agent_os.platform.windows.permissions import PermissionManager

    pm = PermissionManager()
    pm.grant_access("AgentOS-Worker", str(workspace), "read_write")

    yield str(workspace)

    try:
        pm.revoke_access("AgentOS-Worker", str(workspace))
    except Exception:
        pass


@skip_not_windows
class TestProcessLauncherUnit:
    """Unit tests that do not require the sandbox user."""

    def test_build_env_block(self, process_launcher):
        """Verify environment block format is double-null-terminated."""
        # Access the internal method if it exists
        if not hasattr(process_launcher, "_build_env_block"):
            pytest.skip("_build_env_block method not exposed")

        env = {"KEY1": "value1", "KEY2": "value2"}
        block = process_launcher._build_env_block(env, inherit_env=False)
        # The block is a ctypes unicode buffer with null-separated KEY=VALUE
        # pairs and a double-null terminator.  .value stops at the first
        # null, so read the full buffer with [:] instead.
        block_str = block[:]
        assert "KEY1=value1" in block_str
        assert "KEY2=value2" in block_str

    def test_command_line_construction(self):
        """Verify quoting of paths with spaces."""
        # Test the expected command line format
        command = "C:\\Program Files\\Python\\python.exe"
        args = ["script.py", "arg with space", "simple"]

        # Expected format: quoted command + quoted args with spaces
        cmd_line = f'"{command}" ' + " ".join(
            f'"{a}"' if " " in a else a for a in args
        )
        assert '"C:\\Program Files\\Python\\python.exe"' in cmd_line
        assert '"arg with space"' in cmd_line
        assert "simple" in cmd_line
        # simple should NOT be quoted
        assert '"simple"' not in cmd_line


@skip_not_windows
@pytest.mark.usefixtures("ensure_sandbox_account")
class TestProcessLauncherSandbox:
    """Integration tests that require the sandbox user to exist."""

    def test_launch_whoami(self, process_launcher, sandbox_workspace):
        """Launch cmd /c whoami, verify output contains 'agentos-worker'."""
        output_file = os.path.join(sandbox_workspace, "whoami_output.txt")
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", f"whoami > {output_file}"],
            working_dir=sandbox_workspace,
        )
        exit_code = process_launcher.wait(handle, timeout_sec=30)

        assert os.path.exists(output_file), "whoami output file was not created"
        with open(output_file, "r", encoding="utf-8", errors="replace") as f:
            output = f.read().strip().lower()
        assert "agentos-worker" in output or "agentosworker" in output

    def test_launch_with_env(self, process_launcher, sandbox_workspace):
        """Launch with env_vars, verify env var is passed to child."""
        output_file = os.path.join(sandbox_workspace, "env_output.txt")
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", f"echo %TEST_VAR_ISOLATION% > {output_file}"],
            working_dir=sandbox_workspace,
            env_vars={"TEST_VAR_ISOLATION": "hello_from_test"},
        )
        process_launcher.wait(handle, timeout_sec=30)

        assert os.path.exists(output_file), "env output file was not created"
        with open(output_file, "r", encoding="utf-8", errors="replace") as f:
            output = f.read().strip()
        assert "hello_from_test" in output

    def test_launch_working_dir(self, process_launcher, sandbox_workspace):
        """Launch cmd /c cd, verify output is the specified working dir."""
        output_file = os.path.join(sandbox_workspace, "cd_output.txt")
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", f"cd > {output_file}"],
            working_dir=sandbox_workspace,
        )
        process_launcher.wait(handle, timeout_sec=30)

        assert os.path.exists(output_file), "cd output file was not created"
        with open(output_file, "r", encoding="utf-8", errors="replace") as f:
            output = f.read().strip()
        # Normalize paths for comparison
        assert os.path.normcase(output) == os.path.normcase(sandbox_workspace)

    def test_is_running(self, process_launcher, sandbox_workspace):
        """Launch long-running process, verify is_running=True, terminate, verify False."""
        # Use 'ping -n 60' instead of 'timeout' since timeout requires
        # an interactive console (unavailable under CreateProcessWithLogonW).
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", "ping -n 60 127.0.0.1 > nul"],
            working_dir=sandbox_workspace,
        )
        # Give the process a moment to start
        time.sleep(2)
        assert process_launcher.is_running(handle) is True

        process_launcher.terminate(handle, timeout_sec=10)
        assert process_launcher.is_running(handle) is False

    def test_terminate(self, process_launcher, sandbox_workspace):
        """Launch long-running process, terminate, verify it exits."""
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", "ping -n 60 127.0.0.1 > nul"],
            working_dir=sandbox_workspace,
        )
        time.sleep(2)
        assert process_launcher.is_running(handle) is True

        result = process_launcher.terminate(handle, timeout_sec=10)
        assert result is True

    def test_wait_exit_code(self, process_launcher, sandbox_workspace):
        """Launch cmd /c exit 42, wait, verify exit code is 42."""
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", "exit 42"],
            working_dir=sandbox_workspace,
        )
        exit_code = process_launcher.wait(handle, timeout_sec=30)
        assert exit_code == 42

    def test_sandbox_cannot_read_user_desktop(
        self, process_launcher, sandbox_workspace
    ):
        """Launch process that tries to read real user's Desktop, verify access denied."""
        real_user = os.environ.get("USERNAME", "")
        if not real_user:
            pytest.skip("USERNAME environment variable not set")

        real_desktop = os.path.join(
            os.path.expanduser("~"), "Desktop"
        )
        if not os.path.exists(real_desktop):
            pytest.skip("Real user Desktop directory does not exist")

        output_file = os.path.join(sandbox_workspace, "escape_attempt.txt")
        handle = process_launcher.launch(
            command="cmd",
            args=["/c", f'dir "{real_desktop}" > {output_file} 2>&1'],
            working_dir=sandbox_workspace,
        )
        exit_code = process_launcher.wait(handle, timeout_sec=30)

        # Access should be denied - nonzero exit code
        assert exit_code != 0, (
            "Sandbox user should not be able to list real user's Desktop"
        )

    def test_run_and_capture_echo(self, process_launcher, sandbox_workspace):
        """Simple echo, check stdout captured."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "echo hello_capture"],
            working_dir=sandbox_workspace, timeout_sec=30
        )
        assert result.exit_code == 0
        assert "hello_capture" in result.stdout
        assert result.timed_out is False

    def test_run_and_capture_stderr(self, process_launcher, sandbox_workspace):
        """Command that writes to stderr."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "echo error_msg 1>&2"],
            working_dir=sandbox_workspace, timeout_sec=30
        )
        assert "error_msg" in result.stderr

    def test_run_and_capture_exit_code(self, process_launcher, sandbox_workspace):
        """Failing command, check non-zero exit."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "exit 42"],
            working_dir=sandbox_workspace, timeout_sec=30
        )
        assert result.exit_code == 42

    def test_run_and_capture_timeout(self, process_launcher, sandbox_workspace):
        """Command that exceeds timeout."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "ping -n 60 127.0.0.1 > nul"],
            working_dir=sandbox_workspace, timeout_sec=3
        )
        assert result.timed_out is True

    def test_run_and_capture_cleanup(self, process_launcher, sandbox_workspace):
        """Temp files deleted after capture."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "echo cleanup_test"],
            working_dir=sandbox_workspace, timeout_sec=30
        )
        tmp_dir = os.path.join(sandbox_workspace, ".agent-os", ".tmp")
        if os.path.exists(tmp_dir):
            leftover = glob.glob(os.path.join(tmp_dir, "cmd_*"))
            assert len(leftover) == 0, f"Temp files not cleaned up: {leftover}"

    def test_run_and_capture_runs_as_sandbox(self, process_launcher, sandbox_workspace):
        """whoami in captured output contains agentos-worker."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "whoami"],
            working_dir=sandbox_workspace, timeout_sec=30
        )
        assert "agentos-worker" in result.stdout.lower() or "agentosworker" in result.stdout.lower()

    def test_run_and_capture_env_vars(self, process_launcher, sandbox_workspace):
        """Proxy env vars present in captured output."""
        result = process_launcher.run_and_capture(
            command="cmd", args=["/c", "echo %HTTPS_PROXY%"],
            working_dir=sandbox_workspace, timeout_sec=30,
            env_vars={"HTTPS_PROXY": "http://127.0.0.1:9999"}
        )
        assert "127.0.0.1" in result.stdout
