# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for POST /api/v2/platform/browser/warmup endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from agent_os.api.routes.platform import router, configure


def _make_client(warmup_active_sequence):
    """Create test client with a browser manager whose warmup_active follows the given sequence."""
    mock_provider = MagicMock()
    mock_provider.get_capabilities.return_value = MagicMock(platform="windows", setup_complete=True)
    mock_bm = MagicMock()
    mock_bm.launch_warmup = AsyncMock()
    type(mock_bm).warmup_active = PropertyMock(side_effect=list(warmup_active_sequence))
    configure(mock_provider, browser_manager=mock_bm)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), mock_bm


class TestBrowserWarmupEndpoint:

    def test_warmup_calls_launch_warmup(self):
        """POST /api/v2/platform/browser/warmup launches warmup via background task."""
        test_client, mock_bm = _make_client([False, True])
        resp = test_client.post("/api/v2/platform/browser/warmup")
        assert resp.status_code == 200
        assert resp.json()["status"] == "launched"

    def test_warmup_custom_url(self):
        """POST /api/v2/platform/browser/warmup with custom URL returns launched."""
        test_client, mock_bm = _make_client([False, True])
        resp = test_client.post("/api/v2/platform/browser/warmup", json={"url": "https://github.com"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "launched"

    def test_warmup_error_returns_500(self):
        """POST /api/v2/platform/browser/warmup returns 500 when browser fails to start."""
        # warmup_active: False on guard, False after sleep → browser failed
        test_client, mock_bm = _make_client([False, False])
        resp = test_client.post("/api/v2/platform/browser/warmup")
        assert resp.status_code == 500
        assert "failed to launch" in resp.json()["detail"].lower()
