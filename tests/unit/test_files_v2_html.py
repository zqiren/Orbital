# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the HTML branch of the files_v2 content classifier (spec 003).

`.html` / `.htm` files must be classified as ``type: "html"`` (rendered in a
sandboxed iframe by the frontend), while `.svg` stays on the existing image
branch and ordinary text keeps ``type: "text"``.
"""
import base64
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.api.routes import files_v2

# Minimal valid 1x1 PNG (shared with test_files_v2.py).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

SELF_CONTAINED_HTML = (
    "<!doctype html><html><head><title>Report</title>"
    "<style>body{color:red}</style></head>"
    "<body><h1>Dashboard</h1><script>console.log('hi')</script></body></html>"
)


@pytest.fixture
def workspace(tmp_path):
    """Temp workspace holding one of each interesting extension."""
    (tmp_path / "report.html").write_text(SELF_CONTAINED_HTML, encoding="utf-8")
    (tmp_path / "page.htm").write_text("<html><body>old-school</body></html>", encoding="utf-8")
    (tmp_path / "logo.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>',
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("plain text", encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(workspace):
    app = FastAPI()
    mock_store = MagicMock()
    mock_store.get_project.return_value = {
        "project_id": "proj_1",
        "workspace": str(workspace),
    }
    files_v2.configure(mock_store)
    app.include_router(files_v2.router)
    return TestClient(app)


class TestHtmlClassification:
    def test_html_file_is_html_type(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=report.html")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "html"
        assert data["mime"] == "text/html"
        # Raw HTML string is returned verbatim (not base64) so the iframe can srcDoc it.
        assert data["content"] == SELF_CONTAINED_HTML
        assert data["size"] == len(SELF_CONTAINED_HTML.encode("utf-8"))
        assert data["truncated"] is False
        assert data["path"] == "report.html"

    def test_htm_extension_is_also_html(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=page.htm")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "html"
        assert data["mime"] == "text/html"

    def test_svg_stays_on_image_branch(self, client):
        """`.svg` must NOT be routed to the html branch — it stays an image."""
        resp = client.get("/api/v2/projects/proj_1/files/content?path=logo.svg")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "image"
        assert data["mime"] == "image/svg+xml"
        # base64-encoded body, not raw text.
        assert base64.b64decode(data["content"]).startswith(b"<svg")

    def test_plain_text_unaffected(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=notes.txt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "text"
        assert data["content"] == "plain text"

    def test_html_respects_truncation_cap(self, client, workspace):
        """An HTML file over MAX_PREVIEW_BYTES truncates and flags it."""
        big = "<html><body>" + ("a" * (files_v2.MAX_PREVIEW_BYTES + 100)) + "</body></html>"
        (workspace / "big.html").write_text(big, encoding="utf-8")
        resp = client.get("/api/v2/projects/proj_1/files/content?path=big.html")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "html"
        assert data["truncated"] is True
        assert len(data["content"]) == files_v2.MAX_PREVIEW_BYTES
        assert data["size"] == os.path.getsize(workspace / "big.html")
