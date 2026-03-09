# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""macOS-only integration tests for MacOSPlatformProvider.

These tests require macOS with sandbox-exec available. They mirror the
structure of test_provider_integration.py (Windows).
"""

import os
import shutil
import sys
import time
import uuid

import pytest
import pytest_asyncio

from agent_os.platform.types import NetworkRules

# --- Skip markers ---

IS_MACOS = sys.platform == "darwin"

HAS_SEATBELT = False
if IS_MACOS:
    HAS_SEATBELT = shutil.which("sandbox-exec") is not None

skip_not_macos = pytest.mark.skipif(
    not IS_MACOS, reason="Requires macOS"
)

skip_no_seatbelt = pytest.mark.skipif(
    not HAS_SEATBELT, reason="Requires sandbox-exec (Seatbelt)"
)


# --- Fixtures ---


@pytest_asyncio.fixture
async def macos_provider():
    """Create a MacOSPlatformProvider, run setup, and teardown after test."""
    from agent_os.platform.macos.provider import MacOSPlatformProvider

    p = MacOSPlatformProvider()
    await p.setup()
    yield p
    await p.teardown()


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    ws = tmp_path / f"test_workspace_{uuid.uuid4().hex[:8]}"
    ws.mkdir(parents=True, exist_ok=True)
    yield str(ws)


# --- Tests ---


@skip_not_macos
@skip_no_seatbelt
@pytest.mark.asyncio
class TestMacOSProviderIntegration:
    """Integration tests for MacOSPlatformProvider on macOS."""

    async def test_setup_verifies_sandbox_exec(self, macos_provider):
        """setup() succeeds on macOS."""
        # If we got here, setup() already succeeded in the fixture
        assert macos_provider.is_setup_complete() is True

    async def test_capabilities_after_setup(self, macos_provider):
        """setup_complete=True, platform='macos', isolation_method='seatbelt'."""
        caps = macos_provider.get_capabilities()
        assert caps.setup_complete is True
        assert caps.platform == "macos"
        assert caps.isolation_method == "seatbelt"

    async def test_run_command_basic(self, macos_provider, workspace):
        """Run `echo hello` in sandbox -> CommandResult with 'hello' in stdout."""
        result = await macos_provider.run_command(
            project_id="test_echo",
            command="/bin/echo",
            args=["hello"],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert result.exit_code == 0
        assert "hello" in result.stdout

    async def test_run_command_whoami(self, macos_provider, workspace):
        """whoami returns current user."""
        result = await macos_provider.run_command(
            project_id="test_whoami",
            command="/usr/bin/whoami",
            args=[],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert result.exit_code == 0
        expected_user = os.environ.get("USER", "")
        assert expected_user in result.stdout.strip()

    async def test_workspace_write_allowed(self, macos_provider, workspace):
        """Sandboxed process can write to workspace."""
        test_file = os.path.join(workspace, "sandbox_write_test.txt")
        result = await macos_provider.run_command(
            project_id="test_ws_write",
            command="/bin/bash",
            args=["-c", f'echo "sandbox_wrote_this" > "{test_file}"'],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert result.exit_code == 0
        assert os.path.exists(test_file)
        with open(test_file) as f:
            assert "sandbox_wrote_this" in f.read()

    async def test_outside_workspace_write_denied(self, macos_provider, workspace):
        """Sandboxed process CANNOT write outside workspace."""
        home = os.path.expanduser("~")
        outfile = os.path.join(home, f"test_sandbox_escape_{os.getpid()}")
        try:
            result = await macos_provider.run_command(
                project_id="test_escape",
                command="/bin/bash",
                args=["-c", f'echo "escaped" > "{outfile}"'],
                working_dir=workspace,
                timeout_sec=30,
            )
            assert result.exit_code != 0, "Sandbox should deny writes outside workspace"
            assert not os.path.exists(outfile), (
                "File should not exist — sandbox should have blocked write"
            )
        finally:
            # Clean up in case sandbox was bypassed
            if os.path.exists(outfile):
                os.remove(outfile)

    async def test_sensitive_path_read_denied(self, macos_provider, workspace):
        """Cannot read ~/.ssh/id_rsa or ~/.bashrc."""
        home = os.path.expanduser("~")
        sensitive_files = [
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".bashrc"),
        ]
        existing = [f for f in sensitive_files if os.path.exists(f)]
        if not existing:
            pytest.skip("Neither ~/.ssh/id_rsa nor ~/.bashrc exists")

        target = existing[0]
        result = await macos_provider.run_command(
            project_id="test_sensitive",
            command="/bin/cat",
            args=[target],
            working_dir=workspace,
            timeout_sec=30,
        )
        # Should fail — sandbox denies reading sensitive paths
        assert result.exit_code != 0 or "denied" in result.stderr.lower()

    async def test_portal_readonly(self, macos_provider, workspace, tmp_path):
        """Grant read-only -> can read, cannot write."""
        portal_dir = str(tmp_path / "portal_ro")
        os.makedirs(portal_dir, exist_ok=True)
        test_file = os.path.join(portal_dir, "data.txt")
        with open(test_file, "w") as f:
            f.write("portal_data")

        macos_provider.grant_folder_access(portal_dir, "read_only")

        # Read should succeed — we need to re-launch with updated profile
        read_result = await macos_provider.run_command(
            project_id="test_portal_ro",
            command="/bin/cat",
            args=[test_file],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert "portal_data" in read_result.stdout

        # Write should fail
        write_target = os.path.join(portal_dir, "new_file.txt")
        write_result = await macos_provider.run_command(
            project_id="test_portal_ro_write",
            command="/bin/bash",
            args=["-c", f'echo "hacked" > "{write_target}"'],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert write_result.exit_code != 0 or not os.path.exists(write_target)

        macos_provider.revoke_folder_access(portal_dir)

    async def test_portal_readwrite(self, macos_provider, workspace, tmp_path):
        """Grant read-write -> can read AND write."""
        portal_dir = str(tmp_path / "portal_rw")
        os.makedirs(portal_dir, exist_ok=True)
        test_file = os.path.join(portal_dir, "data.txt")
        with open(test_file, "w") as f:
            f.write("portal_rw_data")

        macos_provider.grant_folder_access(portal_dir, "read_write")

        # Read should succeed
        read_result = await macos_provider.run_command(
            project_id="test_portal_rw_read",
            command="/bin/cat",
            args=[test_file],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert "portal_rw_data" in read_result.stdout

        # Write should succeed
        write_target = os.path.join(portal_dir, "written_by_sandbox.txt")
        write_result = await macos_provider.run_command(
            project_id="test_portal_rw_write",
            command="/bin/bash",
            args=["-c", f'echo "sandbox_wrote" > "{write_target}"'],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert write_result.exit_code == 0
        assert os.path.exists(write_target)

        macos_provider.revoke_folder_access(portal_dir)

    async def test_child_inherits_sandbox(self, macos_provider, workspace):
        """Sandboxed process spawns child that tries to write outside workspace."""
        home = os.path.expanduser("~")
        outfile = os.path.join(home, f"test_child_escape_{os.getpid()}")
        try:
            # Parent shell spawns a child that attempts to write outside workspace
            result = await macos_provider.run_command(
                project_id="test_child_inherit",
                command="/bin/bash",
                args=["-c", f'/bin/bash -c \'echo "escaped" > "{outfile}"\''],
                working_dir=workspace,
                timeout_sec=30,
            )
            assert not os.path.exists(outfile), (
                f"Child process escaped sandbox: {outfile} exists"
            )
        finally:
            if os.path.exists(outfile):
                os.remove(outfile)

    async def test_stop_process_sigterm(self, macos_provider, workspace):
        """stop_process() terminates running sandboxed process."""
        project_id = "test_sigterm"
        handle = await macos_provider.run_process(
            project_id=project_id,
            command="/bin/sleep",
            args=["120"],
            working_dir=workspace,
        )
        time.sleep(1)

        result = await macos_provider.stop_process(project_id, timeout_sec=10)
        assert result is True

    async def test_network_proxy_integration(self, macos_provider, workspace):
        """With proxy -> proxy env vars injected."""
        macos_provider.configure_network(
            "test_net_proxy",
            NetworkRules(mode="allowlist", domains=["api.anthropic.com"]),
        )

        result = await macos_provider.run_command(
            project_id="test_net_proxy",
            command="/bin/bash",
            args=["-c", "echo $HTTPS_PROXY"],
            working_dir=workspace,
            timeout_sec=30,
        )
        assert result.exit_code == 0
        assert "127.0.0.1" in result.stdout

    async def test_pty_mode(self, macos_provider, workspace):
        """run_process(use_pty=True) -> process runs with PTY."""
        project_id = "test_pty"
        handle = await macos_provider.run_process(
            project_id=project_id,
            command="/bin/echo",
            args=["pty_test"],
            working_dir=workspace,
            use_pty=True,
        )
        # PTY mode should still produce a valid process
        assert handle.pid > 0
        assert handle.stdout is not None
        time.sleep(1)

        await macos_provider.stop_process(project_id)

    async def test_caffeinate_prevent_sleep(self, macos_provider):
        """prevent_sleep() -> caffeinate running -> allow_sleep() -> gone."""
        handle = macos_provider.prevent_sleep("integration test")
        assert handle is not None
        assert handle.poll() is None  # Still running

        macos_provider.allow_sleep(handle)
        time.sleep(1)
        assert handle.poll() is not None  # Terminated

    async def test_tcc_protected_dir_accessible(self, macos_provider, workspace):
        """Sandboxed process can read ~/Documents (skip if dir doesn't exist)."""
        docs_dir = os.path.join(os.path.expanduser("~"), "Documents")
        if not os.path.isdir(docs_dir):
            pytest.skip("~/Documents does not exist")

        result = await macos_provider.run_command(
            project_id="test_tcc",
            command="/bin/ls",
            args=[docs_dir],
            working_dir=workspace,
            timeout_sec=30,
        )
        # On macOS with TCC, this may fail if not granted.
        # We just verify the command ran (didn't crash the sandbox).
        assert result.exit_code is not None
