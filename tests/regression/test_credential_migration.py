# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: credential migration from settings.json to OS keychain.

Root cause: the one-time migration in create_app() (lines ~114-122) could
silently clear the API key from settings.json even when the keychain write
failed, leaving the user with no working key.

Fix: migration must verify the keychain write before clearing settings.
These tests exercise every edge of the migration and the ApiKeyStore
input-validation hardening.
"""

import json
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from agent_os.daemon_v2.credential_store import ApiKeyStore
from agent_os.daemon_v2.settings_store import GlobalSettings, SettingsStore


# ---------------------------------------------------------------------------
# Helper: reproduce the migration logic from app.py create_app()
# ---------------------------------------------------------------------------

def _run_migration(settings_store: SettingsStore, credential_store: ApiKeyStore,
                   logger: logging.Logger | None = None):
    """Reproduce the one-time migration block from create_app().

    This mirrors the exact logic at agent_os/api/app.py lines ~114-122:

        _legacy = settings_store.get()
        if _legacy.llm.api_key:
            try:
                credential_store.set_api_key(_legacy.llm.api_key)
                _legacy.llm.api_key = None
                settings_store.update(_legacy)
            except Exception:
                pass  # keep legacy key if keychain unavailable
    """
    _legacy = settings_store.get()
    if _legacy.llm.api_key:
        try:
            credential_store.set_api_key(_legacy.llm.api_key)
            _legacy.llm.api_key = None
            settings_store.update(_legacy)
        except Exception:
            if logger:
                logger.warning("Credential migration failed", exc_info=True)
            # keep legacy key if keychain unavailable


def _write_settings(tmp_path, api_key: str | None):
    """Write a settings.json with the given llm.api_key and return a SettingsStore."""
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    settings = GlobalSettings()
    settings.llm.api_key = api_key
    path = os.path.join(data_dir, "settings.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)
    return SettingsStore(data_dir=data_dir)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_env():
    """Ensure AGENT_OS_API_KEY is not set during tests."""
    old = os.environ.pop("AGENT_OS_API_KEY", None)
    yield
    if old is not None:
        os.environ["AGENT_OS_API_KEY"] = old
    else:
        os.environ.pop("AGENT_OS_API_KEY", None)


# ---------------------------------------------------------------------------
# Test 1: migration does not clear settings on keychain failure
# ---------------------------------------------------------------------------

class TestMigrationDoesNotClearOnKeychainFailure:
    """If keyring.set_password raises, settings.json must keep the key."""

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_migration_does_not_clear_settings_on_keychain_failure(
        self, mock_kr, tmp_path, caplog
    ):
        mock_kr.set_password.side_effect = RuntimeError("Keychain locked")

        ss = _write_settings(tmp_path, api_key="sk-test-123")
        cred = ApiKeyStore()

        with caplog.at_level(logging.WARNING):
            _run_migration(ss, cred, logger=logging.getLogger("test"))

        # Settings must still contain the original key
        after = ss.get()
        assert after.llm.api_key == "sk-test-123"


# ---------------------------------------------------------------------------
# Test 2: migration clears settings on verified keychain success
# ---------------------------------------------------------------------------

class TestMigrationClearsOnVerifiedSuccess:
    """If keyring write + readback succeed, settings.json key is cleared."""

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_migration_clears_settings_on_verified_success(
        self, mock_kr, tmp_path
    ):
        mock_kr.set_password.return_value = None  # success
        mock_kr.get_password.return_value = "sk-test-456"  # verified readback

        ss = _write_settings(tmp_path, api_key="sk-test-456")
        cred = ApiKeyStore()

        _run_migration(ss, cred)

        # Settings key must be cleared
        after = ss.get()
        assert after.llm.api_key is None

        # Keychain must have been called with the correct key
        mock_kr.set_password.assert_called_once_with(
            "agent-os", "llm-api-key", "sk-test-456"
        )


# ---------------------------------------------------------------------------
# Test 3: migration does not clear settings on readback mismatch
# ---------------------------------------------------------------------------

class TestMigrationDoesNotClearOnReadbackMismatch:
    """If keyring.set_password succeeds but readback returns None (silent
    write failure), settings.json must keep the key."""

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_migration_does_not_clear_settings_on_readback_mismatch(
        self, mock_kr, tmp_path, caplog
    ):
        mock_kr.set_password.return_value = None  # no exception
        mock_kr.get_password.return_value = None   # silent write failure

        ss = _write_settings(tmp_path, api_key="sk-test-789")
        cred = ApiKeyStore()

        with caplog.at_level(logging.WARNING):
            _run_migration(ss, cred, logger=logging.getLogger("test"))

        # Settings must still contain the original key
        after = ss.get()
        assert after.llm.api_key == "sk-test-789"


# ---------------------------------------------------------------------------
# Test 4: set_api_key rejects empty string
# ---------------------------------------------------------------------------

class TestSetApiKeyRejectsEmpty:
    """ApiKeyStore.set_api_key('') must raise ValueError without touching
    the keychain."""

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_set_api_key_rejects_empty_string(self, mock_kr):
        store = ApiKeyStore()
        with pytest.raises(ValueError, match="non-empty"):
            store.set_api_key("")
        mock_kr.set_password.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: set_api_key rejects whitespace-only string
# ---------------------------------------------------------------------------

class TestSetApiKeyRejectsWhitespace:
    """ApiKeyStore.set_api_key('   ') must raise ValueError."""

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_set_api_key_rejects_whitespace(self, mock_kr):
        store = ApiKeyStore()
        with pytest.raises(ValueError, match="non-empty"):
            store.set_api_key("   ")


# ---------------------------------------------------------------------------
# Test 6: migration skips when no key present in settings
# ---------------------------------------------------------------------------

class TestMigrationSkipsWhenNoKey:
    """If settings.json has no api_key, migration must not touch the keychain."""

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_migration_skips_when_no_key(self, mock_kr, tmp_path):
        ss = _write_settings(tmp_path, api_key=None)
        cred = ApiKeyStore()

        _run_migration(ss, cred)

        # Keychain must NOT have been called
        mock_kr.set_password.assert_not_called()

        # Settings must be unchanged
        after = ss.get()
        assert after.llm.api_key is None
