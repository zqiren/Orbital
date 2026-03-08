# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for files_v2 endpoints."""
import base64
import os
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from agent_os.api.routes import files_v2

# Minimal valid 1x1 PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def workspace(tmp_path):
    """Create a temp workspace with known files."""
    # Text file
    (tmp_path / "readme.txt").write_text("hello world", encoding="utf-8")

    # Image file
    (tmp_path / "icon.png").write_bytes(TINY_PNG)

    # Binary file (PDF-like, with actual non-UTF-8 bytes)
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4\x00\x80\x81\xff\xfe binary")

    # Subdirectory with a file
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested content", encoding="utf-8")

    return tmp_path


@pytest.fixture
def client(workspace):
    """Create a TestClient with files_v2 routes configured."""
    app = FastAPI()

    mock_store = MagicMock()
    mock_store.get_project.return_value = {
        "project_id": "proj_1",
        "workspace": str(workspace),
    }

    files_v2.configure(mock_store)
    app.include_router(files_v2.router)

    return TestClient(app)


@pytest.fixture
def client_no_project(workspace):
    """Client where project lookup returns None."""
    app = FastAPI()

    mock_store = MagicMock()
    mock_store.get_project.return_value = None

    files_v2.configure(mock_store)
    app.include_router(files_v2.router)

    return TestClient(app)


class TestListFiles:
    def test_list_directory(self, client):
        resp = client.get("/api/v2/projects/proj_1/files")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["name"] for e in data["entries"]]
        assert "readme.txt" in names
        assert "icon.png" in names
        assert "doc.pdf" in names
        assert "subdir" in names
        # Verify directory type
        subdir_entry = next(e for e in data["entries"] if e["name"] == "subdir")
        assert subdir_entry["type"] == "directory"
        # Verify file type
        txt_entry = next(e for e in data["entries"] if e["name"] == "readme.txt")
        assert txt_entry["type"] == "file"
        assert txt_entry["size"] > 0

    def test_list_subdirectory(self, client):
        resp = client.get("/api/v2/projects/proj_1/files?path=subdir")
        assert resp.status_code == 200
        data = resp.json()
        names = [e["name"] for e in data["entries"]]
        assert "nested.txt" in names

    def test_list_nonexistent_directory(self, client):
        resp = client.get("/api/v2/projects/proj_1/files?path=nonexistent")
        assert resp.status_code == 404

    def test_list_path_traversal(self, client):
        resp = client.get("/api/v2/projects/proj_1/files?path=../../etc/passwd")
        assert resp.status_code == 400
        assert "outside workspace" in resp.json()["detail"].lower()


class TestGetContent:
    def test_text_file_content(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=readme.txt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "text"
        assert data["content"] == "hello world"
        assert data["size"] == 11
        assert data["path"] == "readme.txt"

    def test_image_file_content(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=icon.png")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "image"
        assert data["mime"] == "image/png"
        assert data["size"] == len(TINY_PNG)
        # Verify base64 roundtrips
        decoded = base64.b64decode(data["content"])
        assert decoded == TINY_PNG

    def test_binary_file_content(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=doc.pdf")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "binary"
        assert data["mime"] == "application/pdf"
        assert "download_url" in data
        assert "content" in data
        # Verify content is valid base64 of the binary file
        import base64
        decoded = base64.b64decode(data["content"])
        assert len(decoded) > 0

    def test_content_nonexistent_file(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=missing.txt")
        assert resp.status_code == 404

    def test_content_path_traversal(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/content?path=../../etc/passwd")
        assert resp.status_code == 400
        assert "outside workspace" in resp.json()["detail"].lower()


class TestUpload:
    def test_upload_file(self, client, workspace):
        content = b"uploaded data"
        resp = client.post(
            "/api/v2/projects/proj_1/files/upload",
            files={"file": ("test.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["size"] == len(content)
        assert "test.txt" in data["path"]
        # Verify file was written
        uploaded = os.path.join(str(workspace), "uploads", "test.txt")
        assert os.path.isfile(uploaded)
        with open(uploaded, "rb") as f:
            assert f.read() == content

    def test_upload_too_large(self, client):
        # Create data just over 10MB
        content = b"x" * (10 * 1024 * 1024 + 1)
        resp = client.post(
            "/api/v2/projects/proj_1/files/upload",
            files={"file": ("big.bin", content, "application/octet-stream")},
        )
        assert resp.status_code == 413

    def test_upload_path_traversal(self, client):
        resp = client.post(
            "/api/v2/projects/proj_1/files/upload?path=../../etc",
            files={"file": ("evil.txt", b"hack", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_custom_path(self, client, workspace):
        content = b"custom path data"
        resp = client.post(
            "/api/v2/projects/proj_1/files/upload?path=subdir",
            files={"file": ("custom.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        uploaded = os.path.join(str(workspace), "subdir", "custom.txt")
        assert os.path.isfile(uploaded)


class TestDownload:
    def test_download_file(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/download?path=readme.txt")
        assert resp.status_code == 200
        assert resp.content == b"hello world"

    def test_download_binary(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/download?path=icon.png")
        assert resp.status_code == 200
        assert resp.content == TINY_PNG

    def test_download_nonexistent(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/download?path=missing.bin")
        assert resp.status_code == 404

    def test_download_path_traversal(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/download?path=../../etc/passwd")
        assert resp.status_code == 400


class TestProjectNotFound:
    def test_list_project_not_found(self, client_no_project):
        resp = client_no_project.get("/api/v2/projects/bad_id/files")
        assert resp.status_code == 404
        assert "project" in resp.json()["detail"].lower()

    def test_content_project_not_found(self, client_no_project):
        resp = client_no_project.get("/api/v2/projects/bad_id/files/content?path=x")
        assert resp.status_code == 404

    def test_upload_project_not_found(self, client_no_project):
        resp = client_no_project.post(
            "/api/v2/projects/bad_id/files/upload",
            files={"file": ("f.txt", b"data", "text/plain")},
        )
        assert resp.status_code == 404

    def test_download_project_not_found(self, client_no_project):
        resp = client_no_project.get("/api/v2/projects/bad_id/files/download?path=x")
        assert resp.status_code == 404
