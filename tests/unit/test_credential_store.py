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
        mock_kr.get_password.return_value = "new-key"
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
    def test_get_masked_derives_the_llm_block_from_the_default_card(self, tmp_path):
        """Spec 082 §3.8: the ``llm`` block survives, DERIVED from the default card.

        It used to read the one global key slot. Cards replaced that slot, but
        the block is kept for a release so App.tsx, SubAgentSettings.tsx and any
        older SPA pointed at a newer daemon keep reading provider/model/key
        status where they always did. The masking contract is unchanged: a
        prefix, an ellipsis, a suffix, and never the raw key.
        """
        from agent_os.daemon_v2.settings_store import SettingsStore

        ss = SettingsStore(data_dir=str(tmp_path), credential_store=MagicMock())
        card = ss.create_card(
            provider="anthropic", model="claude-x", api_key="sk-test1234567890",
        )
        ss.set_default_card(card.id)

        masked = ss.get_masked()
        assert masked["llm"]["api_key_set"] is True
        assert masked["llm"]["api_key_masked"] == "sk-t...7890"
        assert masked["llm"]["provider"] == "anthropic"
        assert masked["llm"]["model"] == "claude-x"
        assert "api_key" not in masked["llm"], "the raw key must never be serialized"
        # And the card itself is exposed alongside, equally masked.
        assert masked["default_card_id"] == card.id
        assert all("api_key" not in c for c in masked["credential_cards"])

    def test_get_masked_with_no_cards_reports_no_key(self, tmp_path):
        """A store with no cards is the wizard-incomplete state, not an error."""
        from agent_os.daemon_v2.settings_store import SettingsStore

        ss = SettingsStore(data_dir=str(tmp_path), credential_store=MagicMock())
        masked = ss.get_masked()
        assert masked["llm"]["api_key_set"] is False
        assert masked["credential_cards"] == []
        assert masked["default_card_id"] is None

    def test_get_masked_no_credential_store_fallback(self, tmp_path):
        from agent_os.daemon_v2.settings_store import SettingsStore

        ss = SettingsStore(data_dir=str(tmp_path))
        masked = ss.get_masked()
        assert masked["llm"]["api_key_set"] is False
        assert masked["llm"]["api_key_masked"] == ""
