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


# ---------------------------------------------------------------------------
# TASK-credential-contract-fixes 1a: the PENDING branch must hand the agent
# usable token names up front (so resume needs no redundant second call),
# and must never embed stored values.
# ---------------------------------------------------------------------------

_SENTINEL = "SENTINEL-value-zV9q3xKfPLEASE-NEVER-APPEAR"


class _FakeStoreWithValues:
    """First-time request (no metadata) against a store that WOULD hand out a
    sentinel value if asked — proving the pending result is built from field
    names only, never from stored values."""

    def get_metadata(self, name):
        return None

    def get_value(self, name, field):
        return _SENTINEL


def test_pending_result_contains_placeholder_tokens_for_every_field():
    tool = RequestCredentialTool(credential_store=_FakeStoreWithValues())
    result = tool.execute(
        name="dify", domain="cloud.dify.ai",
        fields=["email", "password", "code"],
        reason="login",
    )
    data = json.loads(result.content)
    assert data["status"] == "pending"
    assert data["tokens"] == {
        "email": "<secret:dify.email>",
        "password": "<secret:dify.password>",
        "code": "<secret:dify.code>",
    }


def test_pending_message_states_tokens_usable_after_submit():
    tool = RequestCredentialTool(credential_store=_FakeStoreWithValues())
    result = tool.execute(
        name="dify", domain="cloud.dify.ai", fields=["email"], reason="login",
    )
    data = json.loads(result.content)
    msg = data["message"].lower()
    assert "token" in msg
    assert "submit" in msg
    assert "after" in msg


def test_pending_result_contains_no_stored_values():
    tool = RequestCredentialTool(credential_store=_FakeStoreWithValues())
    result = tool.execute(
        name="dify", domain="cloud.dify.ai",
        fields=["email", "password"],
        reason="login",
    )
    assert _SENTINEL not in result.content
    assert result.meta is not None
    assert _SENTINEL not in json.dumps(result.meta)


def test_fields_guidance_is_field_adaptive_not_username_biased():
    """The schema must teach 'request exactly the fields the form shows' with
    multi-shape examples, not a lone ['username', 'password'] example that
    biases agents into requesting fields the page doesn't have."""
    tool = RequestCredentialTool(credential_store=_FakeStoreWithValues())
    desc = tool.parameters["properties"]["fields"]["description"]
    assert "exactly the fields" in desc
    assert "'email'" in desc
    assert "'code'" in desc
