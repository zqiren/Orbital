# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C5: CredentialStore (Windows Credential Manager).

6 tests from TASK-isolation-C5-credential-store.md spec.
All tests should pass on Windows, skip on other platforms.
Uses test-specific key prefix and cleans up in teardown.
"""

import sys
import uuid

import pytest

from tests.platform.conftest import skip_not_windows

# Test-specific key prefix to avoid colliding with real credentials
TEST_KEY_PREFIX = f"AgentOS-Test/{uuid.uuid4().hex[:8]}"


@pytest.fixture
def credential_store():
    """Create a CredentialStore instance and clean up test keys after."""
    from agent_os.platform.windows.credentials import CredentialStore

    store = CredentialStore()
    created_keys: list[str] = []

    class TrackedStore:
        """Wrapper that tracks keys for cleanup."""

        def store(self, key: str, value: str) -> None:
            created_keys.append(key)
            return store.store(key, value)

        def retrieve(self, key: str) -> str | None:
            return store.retrieve(key)

        def delete(self, key: str) -> bool:
            return store.delete(key)

        def exists(self, key: str) -> bool:
            return store.exists(key)

    tracked = TrackedStore()
    yield tracked

    # Cleanup: delete all test keys
    for key in created_keys:
        try:
            store.delete(key)
        except Exception:
            pass


def _test_key(name: str) -> str:
    """Generate a unique test key."""
    return f"{TEST_KEY_PREFIX}/{name}"


@skip_not_windows
class TestCredentialStore:
    def test_store_and_retrieve(self, credential_store):
        """Store a value, retrieve it, assert equal."""
        key = _test_key("store_retrieve")
        credential_store.store(key, "test_value_123")
        result = credential_store.retrieve(key)
        assert result == "test_value_123"

    def test_retrieve_missing(self, credential_store):
        """Retrieve nonexistent key, assert None."""
        key = _test_key("nonexistent_key_" + uuid.uuid4().hex[:8])
        result = credential_store.retrieve(key)
        assert result is None

    def test_delete_existing(self, credential_store):
        """Store, delete, assert returns True, retrieve returns None."""
        key = _test_key("delete_existing")
        credential_store.store(key, "to_be_deleted")
        deleted = credential_store.delete(key)
        assert deleted is True
        result = credential_store.retrieve(key)
        assert result is None

    def test_delete_missing(self, credential_store):
        """Delete nonexistent key, assert returns False."""
        key = _test_key("delete_missing_" + uuid.uuid4().hex[:8])
        deleted = credential_store.delete(key)
        assert deleted is False

    def test_exists(self, credential_store):
        """Store, assert exists=True, delete, assert exists=False."""
        key = _test_key("exists_check")
        credential_store.store(key, "existence_test")
        assert credential_store.exists(key) is True
        credential_store.delete(key)
        assert credential_store.exists(key) is False

    def test_overwrite(self, credential_store):
        """Store value, store different value same key, retrieve returns new value."""
        key = _test_key("overwrite")
        credential_store.store(key, "original_value")
        credential_store.store(key, "updated_value")
        result = credential_store.retrieve(key)
        assert result == "updated_value"
