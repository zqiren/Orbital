# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for credential submit flow (BUG-LOG-4470).

Root cause: submitting credentials through CredentialCard left the agent stuck
because the credential endpoint was tangled with broken session lifecycle code.

Fix (two-part):
  1. credentials.py endpoint now ONLY stores credentials (removed broken
     session lifecycle code — resume is handled by the separate /approve
     endpoint).
  2. CredentialCard.tsx now calls both POST /credentials AND POST /approve
     so the agent is unblocked after credentials are stored.

These tests verify:
  - The credential endpoint stores without touching session lifecycle
  - Errors from the credential store propagate as 500
  - Missing credential store returns 501
  - The approve flow resolves a pending request_credential interception
"""

from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes.credentials import router, configure, _credential_store
from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_store():
    """Mock credential store with store/list_all/delete/get_metadata."""
    store = MagicMock()
    store.list_all.return_value = []
    store.get_metadata.return_value = None
    return store


@pytest.fixture
def mock_agent_manager():
    return MagicMock()


@pytest.fixture
def client(mock_store, mock_agent_manager):
    """TestClient wired to a minimal FastAPI app with credential routes."""
    app = FastAPI()
    configure(mock_store, agent_manager=mock_agent_manager)
    app.include_router(router)
    return TestClient(app)


CREDENTIAL_PAYLOAD = {
    "name": "xiaohongshu",
    "domain": "xiaohongshu.com",
    "fields": {"username": "test", "password": "secret"},
    "project_id": "proj_123",
}


# ======================================================================
# Test 1: credential endpoint stores without session calls
# ======================================================================

class TestCredentialEndpointStoresWithoutSessionCalls:
    """POST /api/v2/credentials must store credentials and do nothing else."""

    def test_credential_endpoint_stores_without_session_calls(
        self, client, mock_store, mock_agent_manager,
    ):
        """Credential endpoint stores and returns 200 — no session lifecycle calls."""
        resp = client.post("/api/v2/credentials", json=CREDENTIAL_PAYLOAD)

        # Must return 200 with correct shape
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "stored", "name": "xiaohongshu"}

        # Must have called store() with the correct positional args
        mock_store.store.assert_called_once_with(
            "xiaohongshu",
            "xiaohongshu.com",
            {"username": "test", "password": "secret"},
        )

        # Must NOT call any session-related methods on agent_manager.
        # The old buggy code tried to look up sessions and resume them
        # directly from the credential endpoint; this must never happen.
        mock_agent_manager.assert_not_called()
        # Also verify no method calls on agent_manager (including
        # get_pending_credential_tc, resume, approve, etc.)
        assert mock_agent_manager.method_calls == [], (
            "Credential endpoint must not call any method on agent_manager. "
            f"Got: {mock_agent_manager.method_calls}"
        )


# ======================================================================
# Test 2: credential endpoint errors propagate
# ======================================================================

class TestCredentialEndpointErrorsPropagateAsHTTP500:
    """Errors from credential_store.store() must surface as HTTP 500."""

    def test_credential_endpoint_errors_propagate(
        self, client, mock_store,
    ):
        """RuntimeError from store() → HTTP 500, not silently swallowed."""
        mock_store.store.side_effect = RuntimeError("keyring unavailable")

        resp = client.post("/api/v2/credentials", json=CREDENTIAL_PAYLOAD)

        assert resp.status_code == 500
        detail = resp.json().get("detail", "")
        assert "keyring unavailable" in detail


# ======================================================================
# Test 3: credential endpoint returns 501 without store
# ======================================================================

class TestCredentialEndpointReturns501WithoutStore:
    """When no credential store is configured, the endpoint must return 501."""

    def test_credential_endpoint_returns_501_without_store(self):
        """Set _credential_store = None → POST returns 501."""
        import agent_os.api.routes.credentials as creds_module

        # Save original so we can restore after test
        original_store = creds_module._credential_store
        original_manager = creds_module._agent_manager
        try:
            # Simulate no credential store configured
            creds_module._credential_store = None
            creds_module._agent_manager = None

            app = FastAPI()
            app.include_router(router)
            no_store_client = TestClient(app)

            resp = no_store_client.post("/api/v2/credentials", json=CREDENTIAL_PAYLOAD)
            assert resp.status_code == 501
            assert "not available" in resp.json().get("detail", "").lower()
        finally:
            # Restore originals so other tests are not affected
            creds_module._credential_store = original_store
            creds_module._agent_manager = original_manager


# ======================================================================
# Test 4: approve resolves a pending request_credential interception
# ======================================================================

class TestApproveResolvesRequestCredential:
    """Simulates the full intercept → approve flow for request_credential."""

    def test_approve_resolves_request_credential(self):
        """Intercept a request_credential tool call, then approve it,
        verifying the pending approval is removed."""
        ws_manager = MagicMock()

        interceptor = AutonomyInterceptor(
            preset=Autonomy.HANDS_OFF,
            ws_manager=ws_manager,
            project_id="test-project",
        )

        # Build a request_credential tool call
        tool_call = {
            "id": "tc-cred-42",
            "name": "request_credential",
            "arguments": {
                "name": "xiaohongshu",
                "domain": "xiaohongshu.com",
                "fields": ["username", "password"],
                "reason": "Login to Xiaohongshu",
            },
        }

        # Step 1: should_intercept must return True
        assert interceptor.should_intercept(tool_call) is True

        # Step 2: on_intercept stores the pending approval and broadcasts
        interceptor.on_intercept(tool_call, recent_context=[])

        assert "tc-cred-42" in interceptor._pending_approvals
        pending = interceptor.get_pending("tc-cred-42")
        assert pending is not None
        assert pending["tool_name"] == "request_credential"
        assert pending["tool_args"]["domain"] == "xiaohongshu.com"

        # Verify WebSocket broadcast happened
        ws_manager.broadcast.assert_called_once()
        broadcast_args = ws_manager.broadcast.call_args
        payload = broadcast_args[0][1]  # positional arg: (project_id, payload)
        assert payload["type"] == "approval.request"
        assert payload["tool_call_id"] == "tc-cred-42"
        assert payload["tool_name"] == "request_credential"

        # Step 3: Simulate the approve flow — agent_manager.approve()
        # calls interceptor.remove_pending() after executing the tool.
        # We test the interceptor's own removal method directly, which
        # is what agent_manager.approve() calls at line 924.
        interceptor.record_approval(
            pending["tool_name"], pending["tool_args"],
        )
        interceptor.remove_pending("tc-cred-42")

        # Step 4: Verify the tool_call_id is no longer pending
        assert interceptor.get_pending("tc-cred-42") is None
        assert "tc-cred-42" not in interceptor._pending_approvals
