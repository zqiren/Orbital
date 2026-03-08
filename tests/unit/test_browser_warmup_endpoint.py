# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for POST /api/v2/platform/browser/warmup endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from agent_os.api.routes.platform import router, configure


@pytest.fixture
def client():
    mock_provider = MagicMock()
    mock_provider.get_capabilities.return_value = MagicMock(platform="windows", setup_complete=True)
    mock_bm = MagicMock()
    mock_bm.launch_warmup = AsyncMock()
    configure(mock_provider, browser_manager=mock_bm)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), mock_bm


class TestBrowserWarmupEndpoint:

    def test_warmup_calls_launch_warmup(self, client):
        """POST /api/v2/platform/browser/warmup calls BrowserManager.launch_warmup."""
        test_client, mock_bm = client
        resp = test_client.post("/api/v2/platform/browser/warmup")
        assert resp.status_code == 200
        mock_bm.launch_warmup.assert_awaited_once_with("https://accounts.google.com")

    def test_warmup_custom_url(self, client):
        """POST /api/v2/platform/browser/warmup with custom URL."""
        test_client, mock_bm = client
        resp = test_client.post("/api/v2/platform/browser/warmup", json={"url": "https://github.com"})
        assert resp.status_code == 200
        mock_bm.launch_warmup.assert_awaited_once_with("https://github.com")

    def test_warmup_error_returns_500(self, client):
        """POST /api/v2/platform/browser/warmup returns 500 on browser error."""
        test_client, mock_bm = client
        mock_bm.launch_warmup = AsyncMock(side_effect=RuntimeError("No browser available"))
        resp = test_client.post("/api/v2/platform/browser/warmup")
        assert resp.status_code == 500
        assert "No browser" in resp.json()["detail"]
