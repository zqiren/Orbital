# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for RequestCredentialTool."""

import json
from unittest.mock import MagicMock

import pytest

from agent_os.agent.tools.request_credential import RequestCredentialTool


@pytest.fixture
def mock_cred_store():
    store = MagicMock()
    store.get_metadata.return_value = None
    return store


@pytest.fixture
def tool(mock_cred_store):
    return RequestCredentialTool(credential_store=mock_cred_store)


def test_schema_has_required_fields(tool):
    schema = tool.schema()
    fn = schema["function"]
    assert fn["name"] == "request_credential"
    props = fn["parameters"]["properties"]
    assert "name" in props
    assert "domain" in props
    assert "fields" in props
    assert "reason" in props
    required = fn["parameters"]["required"]
    assert set(required) == {"name", "domain", "fields", "reason"}


def test_new_credential_returns_pending(tool, mock_cred_store):
    mock_cred_store.get_metadata.return_value = None
    result = tool.execute(
        name="twitter", domain="twitter.com",
        fields=["username", "password"],
        reason="Log into Twitter"
    )
    data = json.loads(result.content)
    assert data["status"] == "pending"
    assert data["name"] == "twitter"
    assert data["fields"] == ["username", "password"]
    assert result.meta is not None
    assert result.meta["credential_request"] is True


def test_existing_credential_returns_tokens(tool, mock_cred_store):
    mock_cred_store.get_metadata.return_value = {
        "domain": "twitter.com",
        "fields": ["username", "password"],
        "use_count": 3,
    }
    result = tool.execute(
        name="twitter", domain="twitter.com",
        fields=["username", "password"],
        reason="Log into Twitter"
    )
    data = json.loads(result.content)
    assert data["status"] == "ready"
    assert "<secret:twitter.username>" in data["tokens"]["username"]
    assert "<secret:twitter.password>" in data["tokens"]["password"]
    assert result.meta is None or result.meta.get("credential_request") is not True


def test_error_returns_error_content(tool, mock_cred_store):
    mock_cred_store.get_metadata.side_effect = RuntimeError("kaboom")
    result = tool.execute(name="x", domain="x.com", fields=["p"], reason="r")
    assert "Error" in result.content
