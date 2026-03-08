# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pytest
from fastapi.testclient import TestClient
from agent_os.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


class TestGlobalSettingsJourney:
    def test_set_and_get_user_preferences(self, client, tmp_path):
        prefs_path = str(tmp_path / "user_preferences.md")
        resp = client.put("/api/v2/settings", json={
            "user_preferences_content": "I'm a senior Python dev\nPrefer concise responses",
            "user_preferences_path": prefs_path,
        })
        assert resp.status_code == 200

        resp = client.get("/api/v2/settings")
        data = resp.json()
        assert data.get("user_preferences_content") == "I'm a senior Python dev\nPrefer concise responses"

    def test_set_scratch_workspace(self, client, tmp_path):
        scratch_dir = str(tmp_path / "my-scratch")
        resp = client.put("/api/v2/settings", json={
            "scratch_workspace": scratch_dir,
        })
        assert resp.status_code == 200
        resp = client.get("/api/v2/settings")
        assert resp.json().get("scratch_workspace") == scratch_dir

    def test_preferences_default_path(self, client, tmp_path):
        """User preferences are saved even without explicit path."""
        resp = client.put("/api/v2/settings", json={
            "user_preferences_content": "Test preference",
        })
        assert resp.status_code == 200
        resp = client.get("/api/v2/settings")
        assert "Test preference" in resp.json().get("user_preferences_content", "")
