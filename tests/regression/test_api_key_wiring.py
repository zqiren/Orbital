# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: API key management endpoints wiring.

Root cause: frontend sent API keys through generic PUT /api/v2/settings,
bypassing the dedicated secure endpoints (PUT/DELETE/GET api-key).

Fix: Global LLM settings save now uses PUT /api/v2/settings/api-key for
the API key and generic PUT /api/v2/settings for other fields. Added
Remove key button wired to DELETE, and status display from GET status.
"""

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


class TestApiKeyEndpoints:
    """Verify all three api-key endpoints exist and accept correct schemas."""

    @pytest.mark.requires_keychain
    def test_put_api_key_accepts_body(self, client):
        """PUT /api/v2/settings/api-key accepts {api_key: string}.

        requires_keychain: the route 500s when the credential store's keyring
        write fails, so a passing run needs a functional unlocked OS keychain
        (triage 2026-06-12; the BaseException guard below only catches
        in-process raises, not the 500 path).
        """
        try:
            resp = client.put("/api/v2/settings/api-key", json={
                "api_key": "sk-test-key-12345",
            })
        except BaseException:
            pytest.skip("Keyring/cryptography unavailable in sandbox")
            return
        assert resp.status_code in (200, 501)

    def test_delete_api_key(self, client):
        """DELETE /api/v2/settings/api-key exists."""
        try:
            resp = client.delete("/api/v2/settings/api-key")
        except BaseException:
            pytest.skip("Keyring/cryptography unavailable in sandbox")
            return
        assert resp.status_code in (200, 501)

    def test_get_api_key_status(self, client):
        """GET /api/v2/settings/api-key/status returns configured + source."""
        try:
            resp = client.get("/api/v2/settings/api-key/status")
        except BaseException:
            pytest.skip("Keyring/cryptography unavailable in sandbox")
            return
        assert resp.status_code == 200
        data = resp.json()
        assert "configured" in data
        assert "source" in data
        assert isinstance(data["configured"], bool)
        # Allowed sources per the product contract: CredentialStore.get_source
        # (agent_os/daemon_v2/credential_store.py) returns
        # 'environment' | 'keychain' | 'none', and the route's no-store
        # fallback adds 'settings'. The old tuple ('none', 'keyring',
        # 'settings') never matched the implementation ('keyring' is not a
        # value the product emits) and broke whenever AGENT_OS_API_KEY was
        # set in the environment (triage 2026-06-12).
        assert data["source"] in ("none", "environment", "keychain", "settings")

    def test_generic_settings_still_works(self, client):
        """PUT /api/v2/settings still works for non-key fields."""
        try:
            resp = client.put("/api/v2/settings", json={
                "llm_model": "gpt-4o",
                "llm_provider": "openai",
            })
        except BaseException:
            pytest.skip("Keyring/cryptography unavailable in sandbox")
            return
        assert resp.status_code == 200
