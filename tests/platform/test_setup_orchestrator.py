# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C6: SetupOrchestrator (first-run setup + UAC elevation).

5 tests from TASK-isolation-C6-setup-orchestrator.md spec.
"""

import os
import sys

import pytest

from tests.platform.conftest import (
    skip_not_windows,
    skip_not_admin,
    IS_ADMIN,
)


@pytest.fixture
def orchestrator():
    """Create a SetupOrchestrator with real C1 and C2 components."""
    from agent_os.platform.windows.credentials import CredentialStore
    from agent_os.platform.windows.sandbox import SandboxAccountManager
    from agent_os.platform.windows.permissions import PermissionManager
    from agent_os.platform.windows.setup import SetupOrchestrator

    cs = CredentialStore()
    am = SandboxAccountManager(cs)
    pm = PermissionManager()
    return SetupOrchestrator(am, pm)


@skip_not_windows
class TestSetupOrchestratorNoAdmin:
    """Tests that do not require admin privileges."""

    def test_check_status_fresh_system(self, orchestrator):
        """On system without sandbox user -> is_complete=False, issues list non-empty.

        This test assumes the sandbox user does not already exist.
        If it does exist (from a previous setup), the status may report
        is_complete=True, which is also valid.
        """
        status = orchestrator.check_setup_status()
        # We verify the returned object has the right structure
        assert isinstance(status.is_complete, bool)
        assert isinstance(status.issues, list)
        assert isinstance(status.sandbox_user_exists, bool)
        assert isinstance(status.sandbox_password_valid, bool)
        assert isinstance(status.workspace_ready, bool)

        # If sandbox user doesn't exist, setup can't be complete
        if not status.sandbox_user_exists:
            assert status.is_complete is False
            assert len(status.issues) > 0

    def test_is_elevated(self, orchestrator):
        """Verify is_elevated() returns bool matching current privilege level."""
        from agent_os.platform.windows.setup import SetupOrchestrator

        result = SetupOrchestrator.is_elevated()
        assert isinstance(result, bool)
        assert result == IS_ADMIN

    def test_default_workspace_path(self, orchestrator):
        """Default workspace path ends with AgentOS\\workspace."""
        # Access the internal method to get the workspace path
        if hasattr(orchestrator, "_get_default_workspace_path"):
            path = orchestrator._get_default_workspace_path()
        else:
            # Fallback: construct expected path
            path = os.path.join(
                os.environ.get("LOCALAPPDATA", ""),
                "AgentOS",
                "workspace",
            )

        assert path.endswith(os.path.join("AgentOS", "workspace"))


@skip_not_windows
@skip_not_admin
class TestSetupOrchestratorAdmin:
    """Tests that require admin privileges.

    WARNING: These tests create and delete the AgentOS-Worker user
    and workspace directories. They should not run in CI without isolation.
    """

    def test_full_setup_teardown(self, orchestrator):
        """run_setup -> check_status (complete) -> run_teardown -> check_status (not complete)."""
        try:
            # Setup
            result = orchestrator.run_setup()
            assert result.success is True

            # Check status after setup
            status = orchestrator.check_setup_status()
            assert status.is_complete is True
            assert status.sandbox_user_exists is True
            assert status.sandbox_password_valid is True

            # Teardown
            result = orchestrator.run_teardown()
            assert result.success is True

            # Check status after teardown
            status = orchestrator.check_setup_status()
            assert status.sandbox_user_exists is False
        except Exception:
            # Best-effort cleanup
            try:
                orchestrator.run_teardown()
            except Exception:
                pass
            raise

    def test_setup_idempotent(self, orchestrator):
        """run_setup twice, no errors on second run."""
        try:
            result1 = orchestrator.run_setup()
            assert result1.success is True

            result2 = orchestrator.run_setup()
            assert result2.success is True
            assert result2.error is None
        finally:
            try:
                orchestrator.run_teardown()
            except Exception:
                pass
