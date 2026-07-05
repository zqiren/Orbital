# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the files_v2 write endpoint (editable markdown).

``PUT /projects/{id}/files/content`` writes UTF-8 text back to an EXISTING
file in the workspace. It is edit-only (never creates), refuses image
extensions (415), caps content at ``MAX_UPLOAD_BYTES`` (413), and is guarded
by the same path-traversal check as reads (400/404 for an escape path).
"""
import base64
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.api.routes import files_v2

# Minimal valid 1x1 PNG (shared with test_files_v2.py / test_files_v2_html.py).
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def workspace(tmp_path):
    """Temp workspace holding a markdown file and an image."""
    (tmp_path / "notes.md").write_text("# Original\n", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(TINY_PNG)
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


class TestWriteFileContent:
    def test_writes_markdown_to_disk(self, client, workspace):
        new_body = "# Edited\n\nHello, world.\n"
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": "notes.md", "content": new_body},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "notes.md"
        assert data["size"] == len(new_body.encode("utf-8"))
        # The bytes actually landed on disk verbatim.
        assert (workspace / "notes.md").read_text(encoding="utf-8") == new_body

    def test_404_for_nonexistent_file(self, client):
        """Edit-only: writing a file that does not exist must NOT create it."""
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": "does-not-exist.md", "content": "nope"},
        )
        assert resp.status_code == 404

    def test_415_for_image_extension(self, client):
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": "logo.png", "content": "not really an image"},
        )
        assert resp.status_code == 415

    def test_413_for_content_over_cap(self, client):
        too_big = "a" * (files_v2.MAX_UPLOAD_BYTES + 100)
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": "notes.md", "content": too_big},
        )
        assert resp.status_code == 413

    def test_traversal_escape_rejected(self, client, workspace):
        """A ``../escape`` path must be rejected by the traversal guard."""
        # Seed a file OUTSIDE the workspace we must never be able to clobber.
        outside = workspace.parent / "secret.md"
        outside.write_text("SECRET\n", encoding="utf-8")
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": "../secret.md", "content": "hijacked"},
        )
        assert resp.status_code in (400, 404)
        # And the outside file is untouched.
        assert outside.read_text(encoding="utf-8") == "SECRET\n"

    def test_symlink_escape_rejected(self, client, workspace):
        """A workspace file that is a SYMLINK pointing OUTSIDE the workspace
        must NOT be written through. This is the high-severity escape: the
        daemon runs unsandboxed, so writing through e.g. notes.md ->
        ~/.ssh/authorized_keys would be an arbitrary out-of-sandbox write.
        Regression for the realpath-containment guard in _resolve_path.
        """
        outside = workspace.parent / "outside_secret.txt"
        outside.write_text("SECRET\n", encoding="utf-8")
        os.symlink(outside, workspace / "evil.md")
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": "evil.md", "content": "pwned-via-symlink"},
        )
        assert resp.status_code == 400
        # The out-of-workspace target is untouched.
        assert outside.read_text(encoding="utf-8") == "SECRET\n"

    def test_sibling_prefix_escape_rejected(self, client, workspace):
        """A sibling directory that merely shares the workspace's NAME PREFIX
        (workspace ``/a/proj`` vs ``/a/proj-evil``) must not be reachable via
        ``../`` — the startswith-without-separator bypass. Regression for the
        trailing-os.sep boundary in _resolve_path.
        """
        sibling = workspace.parent / (workspace.name + "-evil")
        sibling.mkdir()
        victim = sibling / "x.md"
        victim.write_text("SIBLING\n", encoding="utf-8")
        resp = client.put(
            "/api/v2/projects/proj_1/files/content",
            json={"path": f"../{workspace.name}-evil/x.md", "content": "hijacked"},
        )
        assert resp.status_code == 400
        assert victim.read_text(encoding="utf-8") == "SIBLING\n"
