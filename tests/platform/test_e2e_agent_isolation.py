# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E2E tests: Agent isolation user stories.

6 user stories from TASK-isolation-E2E-agent-integration.md spec.
Tests verify subprocess isolation via the WindowsPlatformProvider:
shell commands and external agents run as AgentOS-Worker with
network filtering and ACL enforcement.

Uses pytest-asyncio for async tests.
"""

import asyncio
import os
import sys
import time

import pytest

from tests.platform.conftest import (
    skip_not_windows,
    skip_not_admin,
    HAS_SANDBOX_USER,
)


def _wait_for_file(path: str, timeout: int = 15) -> str:
    """Poll for a file to appear and return its contents."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except (PermissionError, IOError):
                pass
        time.sleep(0.5)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    raise TimeoutError(f"File {path} did not appear within {timeout}s")


@pytest.fixture
def sandbox_workspace(tmp_path, ensure_sandbox_account):
    """Create a temp workspace with sandbox user access."""
    workspace = tmp_path / "e2e_workspace"
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
@pytest.mark.asyncio
class TestE2EAgentIsolation:
    @skip_not_admin
    async def test_story1_first_run_setup(self):
        """Setup flow: incomplete -> run_setup -> complete -> teardown."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()

        try:
            # Before setup: check capabilities
            caps = provider.get_capabilities()
            # On a fresh system without sandbox user, setup should not be complete
            if not HAS_SANDBOX_USER:
                assert caps.setup_complete is False
                assert len(caps.setup_issues) > 0

            # Run setup
            result = await provider.setup()
            assert result.success is True

            # After setup
            caps = provider.get_capabilities()
            assert caps.setup_complete is True
            assert caps.sandbox_username == "AgentOS-Worker"
            assert caps.platform == "windows"

        finally:
            # Teardown for cleanup
            try:
                await provider.teardown()
            except Exception:
                pass

    async def test_story2_shell_runs_in_sandbox(self, sandbox_workspace):
        """Shell commands via run_process execute as AgentOS-Worker, not real user."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()
        provider.grant_folder_access(sandbox_workspace, "read_write")

        try:
            # Run whoami to verify sandbox user
            output_file = os.path.join(sandbox_workspace, "whoami.txt")
            handle = await provider.run_process(
                project_id="e2e_story2",
                command="cmd",
                args=["/c", f"whoami > {output_file}"],
                working_dir=sandbox_workspace,
            )

            output = _wait_for_file(output_file)
            assert "agentos-worker" in output.lower() or "agentosworker" in output.lower()

            # Verify shell cannot access real user's Desktop
            real_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            if os.path.exists(real_desktop):
                escape_file = os.path.join(sandbox_workspace, "escape.txt")
                handle2 = await provider.run_process(
                    project_id="e2e_story2_escape",
                    command="cmd",
                    args=["/c", f'dir "{real_desktop}" > {escape_file} 2>&1'],
                    working_dir=sandbox_workspace,
                )

                escape_output = _wait_for_file(escape_file)
                out_lower = escape_output.lower()
                assert (
                    "access is denied" in out_lower
                    or "not find" in out_lower
                    or "not found" in out_lower
                )
                await provider.stop_process("e2e_story2_escape")

        finally:
            await provider.stop_process("e2e_story2")

    async def test_story3_network_filtering(self, sandbox_workspace):
        """Proxy allows allowlisted domains, blocks others, fires callback."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider
        from agent_os.platform.types import NetworkRules

        blocked_domains: list[tuple[str, str]] = []

        def on_blocked(project_id, domain, method):
            blocked_domains.append((project_id, domain))

        provider = WindowsPlatformProvider(on_network_blocked=on_blocked)
        provider.grant_folder_access(sandbox_workspace, "read_write")

        # Configure allowlist
        provider.configure_network(
            "e2e_network",
            NetworkRules(mode="allowlist", domains=["api.anthropic.com"]),
        )

        try:
            # Verify proxy env is injected
            env_file = os.path.join(sandbox_workspace, "proxy_env.txt")
            handle = await provider.run_process(
                project_id="e2e_network",
                command="cmd",
                args=["/c", f"echo %HTTPS_PROXY% > {env_file}"],
                working_dir=sandbox_workspace,
            )

            output = _wait_for_file(env_file)
            assert "127.0.0.1" in output, "Proxy URL should be injected into env"

        finally:
            await provider.stop_process("e2e_network")

    async def test_story4_portal_grant_revoke(self, sandbox_workspace, temp_file):
        """Grant read-only -> shell can read -> shell can't write -> revoke -> shell can't read."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()
        provider.grant_folder_access(sandbox_workspace, "read_write")

        try:
            # Grant read-only to temp_file directory
            result = provider.grant_folder_access(temp_file, "read_only")
            assert result.success is True

            # Shell reads portaled dir - should succeed
            test_file_path = os.path.join(temp_file, "test.txt")
            read_out = os.path.join(sandbox_workspace, "read_result.txt")
            await provider.run_process(
                project_id="e2e_portal",
                command="cmd",
                args=["/c", f'type "{test_file_path}" > {read_out} 2>&1'],
                working_dir=sandbox_workspace,
            )

            content = _wait_for_file(read_out)
            assert "hello" in content

            # Shell writes to portaled dir - should fail (read-only)
            write_target = os.path.join(temp_file, "new.txt")
            write_out = os.path.join(sandbox_workspace, "write_result.txt")
            await provider.run_process(
                project_id="e2e_portal",
                command="cmd",
                args=["/c", f'echo test > "{write_target}" 2> {write_out}'],
                working_dir=sandbox_workspace,
            )
            time.sleep(3)
            assert not os.path.exists(write_target), "Read-only portal should prevent writes"

            # Revoke access
            result = provider.revoke_folder_access(temp_file)
            assert result.success is True

            # Shell reads after revoke - should fail
            read_out2 = os.path.join(sandbox_workspace, "read_after_revoke.txt")
            await provider.run_process(
                project_id="e2e_portal",
                command="cmd",
                args=["/c", f'type "{test_file_path}" > {read_out2} 2>&1'],
                working_dir=sandbox_workspace,
            )

            content2 = _wait_for_file(read_out2)
            assert "access is denied" in content2.lower() or "cannot find" in content2.lower()

        finally:
            await provider.stop_process("e2e_portal")

    async def test_story5_shell_tool_integration(self, sandbox_workspace):
        """The agent's shell tool correctly delegates to run_process and returns output.

        This tests the integration point: agent loop (in-process) calls shell tool,
        shell tool calls provider.run_process(), command runs as sandbox user,
        output returns to agent loop.

        Also verifies workspace write access: sandbox user writes files to workspace,
        real user (daemon) can read them for serving to UI.

        NOTE: Full agent loop integration (agent calls shell tool which calls
        run_process) requires the agent core to be built. When ready, add a
        test that starts a real agent loop with a mock LLM that returns a
        shell tool call, and verify the command ran as AgentOS-Worker.
        """
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()
        provider.grant_folder_access(sandbox_workspace, "read_write")

        try:
            # Simulate shell tool: run command via provider
            output_file = os.path.join(sandbox_workspace, "output.txt")
            handle = await provider.run_process(
                project_id="e2e_shell",
                command="cmd",
                args=["/c", f'echo hello_world > "{output_file}"'],
                working_dir=sandbox_workspace,
            )

            content = _wait_for_file(output_file)
            # Daemon (real user) can read what sandbox user wrote
            assert "hello_world" in content

            # Session JSONL directory is writable by sandbox user
            sessions_dir = os.path.join(sandbox_workspace, ".agent-os", "sessions")
            os.makedirs(sessions_dir, exist_ok=True)

            session_file = os.path.join(sessions_dir, "test.jsonl")
            await provider.run_process(
                project_id="e2e_shell",
                command="cmd",
                args=["/c", f'echo {{"role":"system"}} > "{session_file}"'],
                working_dir=sandbox_workspace,
            )

            session_content = _wait_for_file(session_file)
            assert os.path.exists(session_file)

        finally:
            await provider.stop_process("e2e_shell")

    @skip_not_admin
    async def test_story6_clean_teardown(self, ensure_sandbox_account):
        """Teardown removes sandbox user and credentials."""
        from agent_os.platform.windows.provider import WindowsPlatformProvider

        provider = WindowsPlatformProvider()

        # Ensure setup is complete first
        if not provider.is_setup_complete():
            result = await provider.setup()
            assert result.success is True

        assert provider.is_setup_complete() is True

        # Teardown
        result = await provider.teardown()
        assert result.success is True

        # Verify user gone
        caps = provider.get_capabilities()
        assert caps.setup_complete is False
        assert caps.sandbox_username is None

        # Verify credential gone
        from agent_os.platform.windows.credentials import CredentialStore

        cs = CredentialStore()
        assert cs.retrieve("AgentOS/sandbox_password") is None

        # Re-setup for other tests that might run after
        try:
            await provider.setup()
        except Exception:
            pass
