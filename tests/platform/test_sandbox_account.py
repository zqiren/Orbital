# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C1: SandboxAccountManager (Windows user creation).

5 tests from TASK-isolation-C1-sandbox-account.md spec.
Split into tests that need admin and tests that don't.
"""

import sys

import pytest

from tests.platform.conftest import (
    skip_not_windows,
    skip_not_admin,
    skip_no_sandbox,
    IS_ADMIN,
    HAS_SANDBOX_USER,
)


@pytest.fixture
def account_manager():
    """Create a SandboxAccountManager with a real CredentialStore."""
    from agent_os.platform.windows.credentials import CredentialStore
    from agent_os.platform.windows.sandbox import SandboxAccountManager

    cs = CredentialStore()
    return SandboxAccountManager(cs)


@pytest.fixture
def credential_store():
    """Create a CredentialStore instance."""
    from agent_os.platform.windows.credentials import CredentialStore

    return CredentialStore()


@skip_not_windows
class TestSandboxAccountNoAdmin:
    """Tests that do not require admin privileges."""

    def test_get_username(self, account_manager):
        """get_username() returns 'AgentOS-Worker'."""
        assert account_manager.get_username() == "AgentOS-Worker"

    def test_validate_nonexistent_account(self, account_manager):
        """If sandbox user doesn't exist, validate returns exists=False.

        This test only makes sense if the user does NOT exist.
        Skip if the user already exists (e.g. from a previous setup).
        """
        if HAS_SANDBOX_USER:
            pytest.skip("AgentOS-Worker user already exists")
        status = account_manager.validate_account()
        assert status.exists is False


@skip_not_windows
@skip_not_admin
class TestSandboxAccountAdmin:
    """Tests that require admin privileges.

    WARNING: These tests create and delete the AgentOS-Worker user.
    They should not run in CI without proper isolation.
    """

    def test_create_and_delete_lifecycle(self, account_manager, credential_store):
        """ensure_account_exists -> validate -> delete -> validate (not exists)."""
        from agent_os.platform.types import SANDBOX_PASSWORD_KEY

        # Create
        status = account_manager.ensure_account_exists()
        assert status.exists is True
        assert status.password_valid is True
        assert status.username == "AgentOS-Worker"

        # Validate
        status = account_manager.validate_account()
        assert status.exists is True
        assert status.password_valid is True

        # Delete
        account_manager.delete_account()

        # Validate after delete
        status = account_manager.validate_account()
        assert status.exists is False

        # Credential should be cleaned up
        assert credential_store.retrieve(SANDBOX_PASSWORD_KEY) is None

    def test_idempotent_creation(self, account_manager):
        """Call ensure_account_exists twice, second call succeeds without error."""
        try:
            status1 = account_manager.ensure_account_exists()
            assert status1.exists is True

            status2 = account_manager.ensure_account_exists()
            assert status2.exists is True
            assert status2.error is None
        finally:
            # Cleanup
            try:
                account_manager.delete_account()
            except Exception:
                pass

    def test_password_stored_in_credential_store(
        self, account_manager, credential_store
    ):
        """After creation, credential store has the password key."""
        from agent_os.platform.types import SANDBOX_PASSWORD_KEY

        try:
            account_manager.ensure_account_exists()
            assert credential_store.exists(SANDBOX_PASSWORD_KEY) is True
            password = credential_store.retrieve(SANDBOX_PASSWORD_KEY)
            assert password is not None
            assert len(password) > 0
        finally:
            try:
                account_manager.delete_account()
            except Exception:
                pass
