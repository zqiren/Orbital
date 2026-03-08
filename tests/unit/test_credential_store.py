# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for ApiKeyStore (OS keychain API key storage)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent_os.daemon_v2.credential_store import ApiKeyStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    return ApiKeyStore()


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure AGENT_OS_API_KEY is not set for each test."""
    old = os.environ.pop("AGENT_OS_API_KEY", None)
    yield
    if old is not None:
        os.environ["AGENT_OS_API_KEY"] = old
    else:
        os.environ.pop("AGENT_OS_API_KEY", None)


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------

class TestEnvVarOverride:
    def test_get_api_key_returns_env_var(self, store):
        os.environ["AGENT_OS_API_KEY"] = "env-key-123"
        assert store.get_api_key() == "env-key-123"

    def test_get_source_returns_environment(self, store):
        os.environ["AGENT_OS_API_KEY"] = "env-key-123"
        assert store.get_source() == "environment"

    def test_set_api_key_noop_when_env_set(self, store):
        os.environ["AGENT_OS_API_KEY"] = "env-key-123"
        result = store.set_api_key("ignored")
        assert result == {"source": "environment"}

    def test_delete_api_key_noop_when_env_set(self, store):
        os.environ["AGENT_OS_API_KEY"] = "env-key-123"
        result = store.delete_api_key()
        assert result == {"source": "environment"}


# ---------------------------------------------------------------------------
# Keyring happy path (mocked)
# ---------------------------------------------------------------------------

class TestKeyringHappyPath:
    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_get_api_key_from_keyring(self, mock_kr, store):
        mock_kr.get_password.return_value = "kr-key-456"
        assert store.get_api_key() == "kr-key-456"
        mock_kr.get_password.assert_called_once_with("agent-os", "llm-api-key")

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_set_api_key_stores_to_keyring(self, mock_kr, store):
        result = store.set_api_key("new-key")
        mock_kr.set_password.assert_called_once_with("agent-os", "llm-api-key", "new-key")
        assert result == {"source": "keychain"}

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_delete_api_key_removes_from_keyring(self, mock_kr, store):
        result = store.delete_api_key()
        mock_kr.delete_password.assert_called_once_with("agent-os", "llm-api-key")
        assert result == {"source": "none"}

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_get_source_keychain(self, mock_kr, store):
        mock_kr.get_password.return_value = "some-key"
        assert store.get_source() == "keychain"

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_get_source_none_when_no_key(self, mock_kr, store):
        mock_kr.get_password.return_value = None
        assert store.get_source() == "none"

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_get_api_key_none_when_not_stored(self, mock_kr, store):
        mock_kr.get_password.return_value = None
        assert store.get_api_key() is None


# ---------------------------------------------------------------------------
# Keyring unavailable (headless)
# ---------------------------------------------------------------------------

class TestKeyringUnavailable:
    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", False)
    def test_get_api_key_returns_none(self, store):
        assert store.get_api_key() is None

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", False)
    def test_set_api_key_raises(self, store):
        with pytest.raises(RuntimeError, match="keyring package not available"):
            store.set_api_key("key")

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", False)
    def test_delete_api_key_returns_none_source(self, store):
        result = store.delete_api_key()
        assert result == {"source": "none"}

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", False)
    def test_get_source_returns_none(self, store):
        assert store.get_source() == "none"


# ---------------------------------------------------------------------------
# Keyring errors (graceful handling)
# ---------------------------------------------------------------------------

class TestKeyringErrors:
    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_get_api_key_handles_exception(self, mock_kr, store):
        mock_kr.get_password.side_effect = Exception("locked")
        assert store.get_api_key() is None

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_set_api_key_raises_runtime_error(self, mock_kr, store):
        mock_kr.set_password.side_effect = Exception("access denied")
        with pytest.raises(RuntimeError, match="keyring.set_password failed"):
            store.set_api_key("key")

    @patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True)
    @patch("agent_os.daemon_v2.credential_store.keyring")
    def test_delete_handles_exception(self, mock_kr, store):
        mock_kr.delete_password.side_effect = Exception("not found")
        result = store.delete_api_key()
        assert result == {"source": "none"}


# ---------------------------------------------------------------------------
# SettingsStore integration
# ---------------------------------------------------------------------------

class TestSettingsStoreIntegration:
    def test_get_masked_uses_credential_store(self, tmp_path):
        from agent_os.daemon_v2.settings_store import SettingsStore

        cred = MagicMock()
        cred.get_api_key.return_value = "sk-test1234567890"
        cred.get_source.return_value = "keychain"

        ss = SettingsStore(data_dir=str(tmp_path), credential_store=cred)
        masked = ss.get_masked()
        assert masked["llm"]["api_key_set"] is True
        assert masked["llm"]["api_key_masked"] == "sk-t...7890"
        assert masked["llm"]["api_key_source"] == "keychain"
        assert "api_key" not in masked["llm"]

    def test_get_masked_no_credential_store_fallback(self, tmp_path):
        from agent_os.daemon_v2.settings_store import SettingsStore

        ss = SettingsStore(data_dir=str(tmp_path))
        masked = ss.get_masked()
        assert masked["llm"]["api_key_set"] is False
        assert masked["llm"]["api_key_masked"] == ""
