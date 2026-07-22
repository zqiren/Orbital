# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for POST /api/v2/platform/mkdir endpoint.

Backing the "New folder" affordance in the workspace picker (backlog #25):
creates the LEAF only inside an existing parent — never `parents=True` — so a
typo'd parent path fails loudly instead of materializing a wrong deep tree.
Name validation is cross-platform (Windows-safe names enforced on every
platform, since a workspace folder may later sync to Windows) because these
tests also run on real Windows in CI (windows-latest job) — no POSIX-only
assumptions (no os.chmod-based permission tests; PermissionError is exercised
via monkeypatch instead).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.api.routes import platform as platform_routes


@pytest.fixture
def client():
    app = FastAPI()
    platform_routes.configure(MagicMock())
    app.include_router(platform_routes.router)
    return TestClient(app)


class TestMkdirValid:
    def test_creates_leaf_folder(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "my-new-project",
        })
        assert resp.status_code == 200
        data = resp.json()
        expected = str(tmp_path / "my-new-project")
        assert data["path"] == expected
        assert (tmp_path / "my-new-project").is_dir()

    def test_leaf_only_does_not_create_missing_parent(self, client, tmp_path):
        """Never `parents=True` — a nonexistent parent must fail, not be
        silently created along with the leaf."""
        missing_parent = tmp_path / "does" / "not" / "exist"
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(missing_parent),
            "name": "leaf",
        })
        assert resp.status_code == 400
        assert not missing_parent.exists()


class TestMkdirInvalidNames:
    """Cross-platform name validation — enforced on ALL platforms."""

    @pytest.mark.parametrize("bad_char", list('<>:"/\\|?*'))
    def test_rejects_invalid_characters(self, client, tmp_path, bad_char):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": f"bad{bad_char}name",
        })
        assert resp.status_code == 400

    def test_rejects_control_characters(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "bad\x01name",
        })
        assert resp.status_code == 400

    @pytest.mark.parametrize("reserved", [
        "CON", "con", "PRN", "AUX", "NUL",
        "COM1", "com3", "COM9", "LPT1", "lpt9",
        "CON.txt", "nul.log",
    ])
    def test_rejects_reserved_device_names(self, client, tmp_path, reserved):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": reserved,
        })
        assert resp.status_code == 400

    def test_rejects_leading_dot(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": ".hidden",
        })
        assert resp.status_code == 400

    def test_rejects_trailing_dot(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "myfolder.",
        })
        assert resp.status_code == 400

    def test_rejects_leading_space(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": " myfolder",
        })
        assert resp.status_code == 400

    def test_rejects_trailing_space(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "myfolder ",
        })
        assert resp.status_code == 400

    def test_rejects_empty_name(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "",
        })
        assert resp.status_code == 400

    def test_rejects_whitespace_only_name(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "   ",
        })
        assert resp.status_code == 400

    def test_rejects_path_exceeding_max_length(self, client, tmp_path):
        long_name = "a" * 250
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": long_name,
        })
        assert resp.status_code == 400
        assert not (tmp_path / long_name).exists()

    def test_valid_name_with_dot_in_middle_is_allowed(self, client, tmp_path):
        """A dot that isn't leading/trailing (e.g. a versioned folder name)
        must not be rejected — only reserved stems and edge dots are banned."""
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "my.project.v2",
        })
        assert resp.status_code == 200


class TestMkdirErrorMapping:
    def test_missing_parent_returns_400(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path / "nope"),
            "name": "leaf",
        })
        assert resp.status_code == 400
        assert "detail" in resp.json()

    def test_existing_folder_returns_409(self, client, tmp_path):
        (tmp_path / "already-here").mkdir()
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "already-here",
        })
        assert resp.status_code == 409

    def test_permission_error_returns_403(self, client, tmp_path, monkeypatch):
        from pathlib import Path as PathlibPath

        def raise_permission_error(self, *args, **kwargs):
            raise PermissionError("Operation not permitted")

        monkeypatch.setattr(PathlibPath, "mkdir", raise_permission_error)
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "blocked",
        })
        assert resp.status_code == 403
        assert "detail" in resp.json()


class TestMkdirResponseShape:
    def test_returns_created_path(self, client, tmp_path):
        resp = client.post("/api/v2/platform/mkdir", json={
            "parent": str(tmp_path),
            "name": "shape-check",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == str(tmp_path / "shape-check")
