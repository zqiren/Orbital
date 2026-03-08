# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: run_teardown() must revoke ACL entries BEFORE deleting the account.

Bug: run_teardown() deleted the sandbox account without revoking its ACL entries,
causing orphaned SID accumulation that eventually hit the 64KB DACL limit.
See REPORT-sandbox-lifecycle-findings.md.
"""

from unittest.mock import MagicMock, call

from agent_os.platform.types import SetupResult
from agent_os.platform.windows.setup import SetupOrchestrator


class TestTeardownRevokesACL:
    """Teardown must revoke ACL entries before deleting the sandbox account."""

    def _make_orchestrator(self):
        """Create a SetupOrchestrator with mocked dependencies."""
        account_mgr = MagicMock()
        perm_mgr = MagicMock()

        account_mgr.get_username.return_value = "AgentOS-Worker"
        account_mgr.delete_account.return_value = None

        perm_mgr.revoke_access.return_value = MagicMock(success=True)

        orch = SetupOrchestrator(
            account_manager=account_mgr,
            permission_manager=perm_mgr,
        )
        return orch, account_mgr, perm_mgr

    def test_teardown_revokes_before_delete(self):
        """revoke_access() is called BEFORE delete_account()."""
        orch, account_mgr, perm_mgr = self._make_orchestrator()

        # Track call order across both mocks
        call_order = []
        perm_mgr.revoke_access.side_effect = lambda *a, **kw: call_order.append("revoke")
        account_mgr.delete_account.side_effect = lambda *a, **kw: call_order.append("delete")

        result = orch.run_teardown()

        assert result.success is True

        # revoke_access was called with the username and workspace path
        perm_mgr.revoke_access.assert_called_once()
        args = perm_mgr.revoke_access.call_args[0]
        assert args[0] == "AgentOS-Worker"
        # Second arg is the workspace path (ends with workspace)
        assert args[1].endswith("workspace")

        # delete_account was called
        account_mgr.delete_account.assert_called_once()

        # Order: revoke BEFORE delete
        assert call_order == ["revoke", "delete"], (
            f"Expected revoke before delete, got: {call_order}"
        )

    def test_teardown_deletes_even_if_revoke_fails(self):
        """If revoke_access() raises, delete_account() still runs."""
        orch, account_mgr, perm_mgr = self._make_orchestrator()

        perm_mgr.revoke_access.side_effect = RuntimeError("icacls failed")

        result = orch.run_teardown()

        # Teardown still succeeds (revoke failure is non-fatal)
        assert result.success is True

        # delete_account was still called despite revoke failure
        account_mgr.delete_account.assert_called_once()
