# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the files_v2 resolve endpoint (spec 002 follow-up).

Agents abbreviate workspace paths in chat replies ('drafts/x.md' for
'content/drafts/x.md', bare 'DECISIONS.md' for 'orbital/DECISIONS.md').
The resolve endpoint lets the client recover: given an abbreviated path it
returns every workspace file whose path ends with that suffix on a segment
boundary.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from agent_os.api.routes import files_v2


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "content" / "drafts").mkdir(parents=True)
    (tmp_path / "content" / "drafts" / "003-async.md").write_text("a", encoding="utf-8")
    (tmp_path / "orbital").mkdir()
    (tmp_path / "orbital" / "DECISIONS.md").write_text("d", encoding="utf-8")
    (tmp_path / "notes.md").write_text("n", encoding="utf-8")
    # Same basename in two places (neither at the queried path) -> ambiguous
    # for bare-basename queries.
    (tmp_path / "content" / "todo.md").write_text("t1", encoding="utf-8")
    (tmp_path / "orbital" / "todo.md").write_text("t2", encoding="utf-8")
    # Files under skip-listed directories must never match.
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "DECISIONS.md").write_text("x", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "DECISIONS.md").write_text("x", encoding="utf-8")
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


class TestResolveFiles:
    def test_bare_basename_resolves_to_nested_file(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "DECISIONS.md"})
        assert resp.status_code == 200
        assert resp.json()["matches"] == ["orbital/DECISIONS.md"]

    def test_multi_segment_suffix_resolves(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "drafts/003-async.md"})
        assert resp.status_code == 200
        assert resp.json()["matches"] == ["content/drafts/003-async.md"]

    def test_exact_existing_path_returns_itself(self, client):
        resp = client.get(
            "/api/v2/projects/proj_1/files/resolve",
            params={"path": "content/drafts/003-async.md"},
        )
        assert resp.status_code == 200
        assert resp.json()["matches"] == ["content/drafts/003-async.md"]

    def test_ambiguous_basename_returns_all_matches(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "todo.md"})
        assert resp.status_code == 200
        assert sorted(resp.json()["matches"]) == ["content/todo.md", "orbital/todo.md"]

    def test_no_match_returns_empty(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "missing.md"})
        assert resp.status_code == 200
        assert resp.json()["matches"] == []

    def test_suffix_must_align_on_segment_boundary(self, client):
        # 'sync.md' is a tail substring of '003-async.md' but not a whole
        # filename -> must not match.
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "sync.md"})
        assert resp.status_code == 200
        assert resp.json()["matches"] == []

    def test_skips_hidden_and_dependency_dirs(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "DECISIONS.md"})
        matches = resp.json()["matches"]
        assert not any(".git" in m or "node_modules" in m for m in matches)

    def test_traversal_rejected(self, client):
        resp = client.get("/api/v2/projects/proj_1/files/resolve", params={"path": "../etc/passwd"})
        assert resp.status_code == 400

    def test_project_not_found(self, workspace):
        app = FastAPI()
        mock_store = MagicMock()
        mock_store.get_project.return_value = None
        files_v2.configure(mock_store)
        app.include_router(files_v2.router)
        c = TestClient(app)
        resp = c.get("/api/v2/projects/nope/files/resolve", params={"path": "notes.md"})
        assert resp.status_code == 404
