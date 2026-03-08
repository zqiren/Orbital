# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for UserCredentialStore -- keychain storage, domain matching, metadata."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent_os.daemon_v2.credential_store import UserCredentialStore


@pytest.fixture
def tmp_meta(tmp_path):
    return str(tmp_path / "credential-meta.json")


@pytest.fixture
def store(tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.get_password = MagicMock(return_value=None)
        mock_kr.set_password = MagicMock()
        mock_kr.delete_password = MagicMock()
        s = UserCredentialStore(meta_path=tmp_meta)
        yield s


# --- Storage ---

def test_store_saves_to_keyring_and_metadata(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"username": "john", "password": "secret123"})
        mock_kr.set_password.assert_any_call("agent-os-creds", "twitter.username", "john")
        mock_kr.set_password.assert_any_call("agent-os-creds", "twitter.password", "secret123")

    # Metadata file created
    assert os.path.exists(tmp_meta)
    with open(tmp_meta) as f:
        meta = json.load(f)
    assert "twitter" in meta
    assert meta["twitter"]["domain"] == "twitter.com"
    assert meta["twitter"]["fields"] == ["password", "username"]
    assert meta["twitter"]["use_count"] == 0


def test_get_value_reads_from_keyring(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.get_password = MagicMock(return_value="secret123")
        val = store.get_value("twitter", "password")
        mock_kr.get_password.assert_called_with("agent-os-creds", "twitter.password")
        assert val == "secret123"


def test_get_value_returns_none_if_missing(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.get_password = MagicMock(return_value=None)
        assert store.get_value("twitter", "password") is None


# --- Domain Matching ---

def test_domain_exact_match(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    assert store.check_domain("twitter", "https://twitter.com/login") is True


def test_domain_subdomain_match(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    assert store.check_domain("twitter", "https://mobile.twitter.com/login") is True


def test_domain_mismatch_evil_prefix(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    assert store.check_domain("twitter", "https://evil-twitter.com/login") is False


def test_domain_mismatch_suffix_attack(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    assert store.check_domain("twitter", "https://twitter.com.evil.com/login") is False


def test_domain_check_unknown_credential(store):
    assert store.check_domain("nonexistent", "https://example.com") is False


# --- Metadata ---

def test_get_metadata(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("github", "github.com", {"token": "ghp_xxx"})
    meta = store.get_metadata("github")
    assert meta is not None
    assert meta["domain"] == "github.com"
    assert meta["fields"] == ["token"]
    assert "created" in meta


def test_get_metadata_returns_none_for_unknown(store):
    assert store.get_metadata("nonexistent") is None


def test_list_all(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
        store.store("github", "github.com", {"token": "y"})
    items = store.list_all()
    assert len(items) == 2
    names = {item["name"] for item in items}
    assert names == {"twitter", "github"}


def test_record_use_increments_counter(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    store.record_use("twitter")
    store.record_use("twitter")
    meta = store.get_metadata("twitter")
    assert meta["use_count"] == 2
    assert "last_used" in meta


# --- Delete ---

def test_delete_removes_from_keyring_and_metadata(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"username": "j", "password": "p"})
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.delete_password = MagicMock()
        store.delete("twitter")
        mock_kr.delete_password.assert_any_call("agent-os-creds", "twitter.password")
        mock_kr.delete_password.assert_any_call("agent-os-creds", "twitter.username")
    assert store.get_metadata("twitter") is None
    assert store.list_all() == []


def test_delete_nonexistent_is_noop(store):
    store.delete("nonexistent")  # Should not raise
