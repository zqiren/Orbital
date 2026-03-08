# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GET /api/v2/platform/browse endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.api.routes import platform as platform_routes


@pytest.fixture
def fake_home(tmp_path):
    """Create a fake home directory structure for testing."""
    # Directories
    (tmp_path / "Documents").mkdir()
    (tmp_path / "Documents" / "Projects").mkdir()
    (tmp_path / "Documents" / "Projects" / "MyApp").mkdir()
    (tmp_path / "Documents" / "Notes").mkdir()
    (tmp_path / "Downloads").mkdir()
    (tmp_path / "EmptyDir").mkdir()

    # Hidden directory (should be excluded)
    (tmp_path / ".config").mkdir()

    # Files (should be excluded from entries)
    (tmp_path / "readme.txt").write_text("hello")
    (tmp_path / "Documents" / "report.pdf").write_text("pdf content")

    return tmp_path


@pytest.fixture
def client(fake_home, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    app = FastAPI()
    platform_routes.configure(MagicMock())
    app.include_router(platform_routes.router)
    return TestClient(app)


class TestBrowseNoPath:
    """GET /api/v2/platform/browse with no path param returns home directory."""

    def test_returns_home_contents(self, client, fake_home):
        resp = client.get("/api/v2/platform/browse")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == str(fake_home)
        assert data["parent"] == str(fake_home.parent)
        names = [e["name"] for e in data["entries"]]
        assert "Documents" in names
        assert "Downloads" in names
        assert "EmptyDir" in names

    def test_excludes_hidden_dirs(self, client):
        resp = client.get("/api/v2/platform/browse")
        names = [e["name"] for e in resp.json()["entries"]]
        assert ".config" not in names

    def test_excludes_files(self, client):
        resp = client.get("/api/v2/platform/browse")
        names = [e["name"] for e in resp.json()["entries"]]
        assert "readme.txt" not in names

    def test_parent_at_home_points_to_parent_dir(self, client, fake_home):
        resp = client.get("/api/v2/platform/browse")
        # Home is inside tmp_path, so parent should be its actual parent
        assert resp.json()["parent"] == str(fake_home.parent)


class TestBrowseWithPath:
    """GET /api/v2/platform/browse?path=<subdir> returns subdirectory contents."""

    def test_returns_subdir_contents(self, client, fake_home):
        path = str(fake_home / "Documents")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == path
        assert data["display_name"] == "Documents"
        names = [e["name"] for e in data["entries"]]
        assert "Projects" in names
        assert "Notes" in names
        # Files excluded
        assert "report.pdf" not in names

    def test_parent_points_to_parent_dir(self, client, fake_home):
        path = str(fake_home / "Documents")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        assert resp.json()["parent"] == str(fake_home)

    def test_has_children_true_when_has_subdirs(self, client, fake_home):
        path = str(fake_home / "Documents")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        entries = {e["name"]: e for e in resp.json()["entries"]}
        # Projects has subdirectory MyApp
        assert entries["Projects"]["has_children"] is True

    def test_has_children_false_when_no_subdirs(self, client, fake_home):
        path = str(fake_home / "Documents")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        entries = {e["name"]: e for e in resp.json()["entries"]}
        # Notes has no subdirectories
        assert entries["Notes"]["has_children"] is False

    def test_empty_dir(self, client, fake_home):
        path = str(fake_home / "EmptyDir")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        assert resp.status_code == 200
        assert resp.json()["entries"] == []

    def test_entry_path_is_absolute(self, client, fake_home):
        resp = client.get("/api/v2/platform/browse")
        for entry in resp.json()["entries"]:
            assert entry["path"].startswith(str(fake_home))

    def test_parent_is_null_at_root(self, client):
        resp = client.get("/api/v2/platform/browse", params={"path": "/"})
        assert resp.status_code == 200
        assert resp.json()["parent"] is None


class TestBrowseErrors:
    """Error cases for the browse endpoint."""

    def test_path_outside_home_is_allowed(self, client, fake_home):
        """After removing home-only restriction, any accessible dir should work."""
        other = fake_home.parent  # parent of fake_home, outside "home"
        resp = client.get("/api/v2/platform/browse", params={"path": str(other)})
        assert resp.status_code == 200

    def test_nonexistent_path_returns_404(self, client, fake_home):
        path = str(fake_home / "NoSuchDir")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        assert resp.status_code == 404

    def test_file_path_returns_400(self, client, fake_home):
        path = str(fake_home / "readme.txt")
        resp = client.get("/api/v2/platform/browse", params={"path": path})
        assert resp.status_code == 400


class TestBrowseResponseShape:
    """Verify the exact response shape matches the spec."""

    def test_top_level_keys(self, client):
        resp = client.get("/api/v2/platform/browse")
        data = resp.json()
        assert set(data.keys()) == {"path", "parent", "display_name", "entries"}

    def test_entry_keys(self, client):
        resp = client.get("/api/v2/platform/browse")
        for entry in resp.json()["entries"]:
            assert set(entry.keys()) == {"name", "path", "has_children"}

    def test_entries_sorted_by_name(self, client):
        resp = client.get("/api/v2/platform/browse")
        names = [e["name"] for e in resp.json()["entries"]]
        assert names == sorted(names)
