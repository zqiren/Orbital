# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: daemon-served index.html must contain __AGENT_OS_LOCAL__=true.

When the daemon serves the SPA, it injects a script tag setting
window.__AGENT_OS_LOCAL__=true so the frontend can detect it's running
in LAN/localhost mode (no relay auth needed).  Static assets like .js
files must NOT receive the injection.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def spa_client(tmp_path, monkeypatch):
    """Create a test client whose SPA dir contains a minimal index.html."""
    spa_dir = tmp_path / "spa"
    spa_dir.mkdir()
    (spa_dir / "index.html").write_text(
        "<html><head></head><body></body></html>", encoding="utf-8"
    )
    monkeypatch.setenv("AGENT_OS_SPA_DIR", str(spa_dir))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    app = create_app(data_dir=str(data_dir))
    return TestClient(app), spa_dir


def test_index_html_contains_local_flag(spa_client):
    """GET / must return index.html with window.__AGENT_OS_LOCAL__=true injected."""
    client, _ = spa_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert "window.__AGENT_OS_LOCAL__=true" in resp.text
    assert "text/html" in resp.headers.get("content-type", "")


def test_static_asset_not_injected(spa_client):
    """Static JS files must be served verbatim without __AGENT_OS_LOCAL__ injection."""
    client, spa_dir = spa_client
    (spa_dir / "test.js").write_text("console.log('hello');", encoding="utf-8")
    resp = client.get("/test.js")
    assert resp.status_code == 200
    assert "__AGENT_OS_LOCAL__" not in resp.text
