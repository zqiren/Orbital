# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Security tests for credential store — attack scenarios."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agent_os.daemon_v2.credential_store import UserCredentialStore
from agent_os.agent.tools.request_credential import RequestCredentialTool


@pytest.fixture
def tmp_meta(tmp_path):
    return str(tmp_path / "credential-meta.json")


@pytest.fixture
def store(tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        mock_kr.get_password = MagicMock(return_value=None)
        mock_kr.delete_password = MagicMock()
        s = UserCredentialStore(meta_path=tmp_meta)
        yield s


# ATTACK-02: LLM extracts value from tool result
def test_attack_02_tool_result_has_token_not_value(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"username": "john", "password": "secret123"})
    tool = RequestCredentialTool(credential_store=store)
    result = tool.execute(
        name="twitter", domain="twitter.com",
        fields=["username", "password"],
        reason="Login"
    )
    assert "secret123" not in result.content
    assert "john" not in result.content
    assert "<secret:twitter.username>" in result.content
    assert "<secret:twitter.password>" in result.content


# ATTACK-03: Value in session JSONL (tool result content)
def test_attack_03_no_value_in_tool_output(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("bank", "mybank.com", {"password": "hunter2"})
    tool = RequestCredentialTool(credential_store=store)
    result = tool.execute(name="bank", domain="mybank.com", fields=["password"], reason="Login")
    assert "hunter2" not in result.content
    assert "hunter2" not in json.dumps(result.meta or {})


# ATTACK-05: Direct file read of metadata
def test_attack_05_metadata_has_no_values(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "secret123"})
    with open(tmp_meta) as f:
        content = f.read()
    assert "secret123" not in content


# ATTACK-07: Secret used on wrong domain
def test_attack_07_wrong_domain_blocked(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    assert store.check_domain("twitter", "https://evil.com/twitter") is False


# ATTACK-08: Subdomain spoofing
def test_attack_08_subdomain_spoofing_blocked(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"password": "x"})
    assert store.check_domain("twitter", "https://evil-twitter.com") is False
    assert store.check_domain("twitter", "https://twitter.com.evil.com") is False


# ATTACK-10: URL parsing edge cases
def test_attack_10_url_parsing_edge_cases(store):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("bank", "mybank.com", {"password": "x"})
    assert store.check_domain("bank", "") is False
    assert store.check_domain("bank", "not-a-url") is False
    assert store.check_domain("bank", "javascript:alert(1)") is False
    assert store.check_domain("bank", "https://mybank.com.attacker.com") is False


# ATTACK-20: Credential persistence after deletion
def test_attack_20_full_cleanup_on_delete(store, tmp_meta):
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.set_password = MagicMock()
        store.store("twitter", "twitter.com", {"username": "j", "password": "p"})
    with patch("agent_os.daemon_v2.credential_store.keyring") as mock_kr:
        mock_kr.delete_password = MagicMock()
        store.delete("twitter")
    assert store.get_metadata("twitter") is None
    assert store.list_all() == []
    with open(tmp_meta) as f:
        content = f.read()
    assert "twitter" not in content
