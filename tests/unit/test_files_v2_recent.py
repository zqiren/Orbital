# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 187: ``GET /projects/{pid}/files/recent`` — the project's latest
modified files for the right panel's 最近修改 group.

Membership is file history's rule (``history.rules.list_tracked``: rg with
gitignore semantics minus the deny list), minus dot-paths, plus the one
deliverable folder inside ``orbital/`` (``ProjectPaths.pdfs_dir``). Order is
the disk's mtime, newest first, capped at 50. The route writes nothing.
"""
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.agent.project_paths import ProjectPaths
from agent_os.api.routes import files_v2
from agent_os.history import rules as history_rules

# Far apart on purpose: Windows' clock and some filesystems are coarse.
BASE_MTIME = 1_700_000_000


def _write(root, rel, text="x", mtime=None):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _snapshot(root):
    """Every path under root with its mtime: proof that a call wrote nothing."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root)] = os.lstat(full).st_mtime_ns
    return out


@pytest.fixture
def projects():
    """pid -> project record; the store mock reads it live."""
    return {}


@pytest.fixture
def client(projects):
    app = FastAPI()
    store = MagicMock()
    store.get_project.side_effect = lambda pid: projects.get(pid)
    store.list_projects.side_effect = lambda: list(projects.values())
    files_v2.configure(store)
    app.include_router(files_v2.router)
    return TestClient(app)


@pytest.fixture
def ws(tmp_path, projects):
    root = tmp_path / "proj"
    root.mkdir()
    projects["p1"] = {"project_id": "p1", "workspace": str(root)}
    return str(root)


def _recent(client, pid="p1"):
    resp = client.get(f"/api/v2/projects/{pid}/files/recent")
    assert resp.status_code == 200, resp.text
    return resp.json()["files"]


def _paths(files):
    return [f["path"] for f in files]


class TestOrderAndCap:
    def test_newest_first_by_mtime(self, client, ws):
        _write(ws, "old.md", mtime=BASE_MTIME)
        _write(ws, "docs/new.md", mtime=BASE_MTIME + 300)
        _write(ws, "mid.txt", mtime=BASE_MTIME + 150)
        files = _recent(client)
        assert _paths(files) == ["docs/new.md", "mid.txt", "old.md"]
        assert files[0]["modified_at"] == pytest.approx(BASE_MTIME + 300)

    def test_ties_break_by_path(self, client, ws):
        _write(ws, "b.md", mtime=BASE_MTIME)
        _write(ws, "a.md", mtime=BASE_MTIME)
        assert _paths(_recent(client)) == ["a.md", "b.md"]

    def test_capped_at_exactly_fifty(self, client, ws):
        for i in range(60):
            _write(ws, f"f{i:02d}.md", mtime=BASE_MTIME + i * 10)
        files = _recent(client)
        assert len(files) == 50
        assert files[0]["path"] == "f59.md"
        assert files[-1]["path"] == "f10.md"

    def test_only_path_and_modified_at(self, client, ws):
        _write(ws, "a.md", mtime=BASE_MTIME)
        assert set(_recent(client)[0]) == {"path", "modified_at"}

    def test_empty_project_has_no_files(self, client, ws):
        assert _recent(client) == []


class TestMembership:
    def test_dot_paths_are_dropped(self, client, ws):
        _write(ws, ".DS_Store", mtime=BASE_MTIME + 100)
        _write(ws, ".env", mtime=BASE_MTIME + 100)
        _write(ws, ".vscode/settings.json", mtime=BASE_MTIME + 100)
        _write(ws, "src/.hidden/x.py", mtime=BASE_MTIME + 100)
        _write(ws, "keep.md", mtime=BASE_MTIME)
        assert _paths(_recent(client)) == ["keep.md"]

    def test_orbital_store_is_excluded(self, client, ws):
        _write(ws, "orbital/sessions/s_1.jsonl", mtime=BASE_MTIME + 100)
        _write(ws, "orbital/PROJECT_STATE.md", mtime=BASE_MTIME + 100)
        _write(ws, "orbital/output/screenshots/shot.png", mtime=BASE_MTIME + 100)
        _write(ws, "orbital/output/shell-output/out.txt", mtime=BASE_MTIME + 100)
        _write(ws, "report.md", mtime=BASE_MTIME)
        assert _paths(_recent(client)) == ["report.md"]

    def test_saved_pdfs_are_included(self, client, ws):
        pdf_dir = ProjectPaths(ws).pdfs_dir
        rel_dir = os.path.relpath(pdf_dir, ws).replace(os.sep, "/")
        _write(ws, f"{rel_dir}/page.pdf", mtime=BASE_MTIME + 200)
        _write(ws, f"{rel_dir}/.DS_Store", mtime=BASE_MTIME + 300)
        _write(ws, "orbital/output/screenshots/shot.png", mtime=BASE_MTIME + 400)
        _write(ws, "orbital/sessions/s_1.jsonl", mtime=BASE_MTIME + 400)
        _write(ws, "notes.md", mtime=BASE_MTIME)
        assert _paths(_recent(client)) == [f"{rel_dir}/page.pdf", "notes.md"]

    def test_gitignored_files_are_excluded_without_git(self, client, ws):
        _write(ws, ".gitignore", "build/\n*.log\n")
        _write(ws, "build/out.js", mtime=BASE_MTIME + 100)
        _write(ws, "debug.log", mtime=BASE_MTIME + 100)
        _write(ws, "node_modules/pkg/index.js", mtime=BASE_MTIME + 100)
        _write(ws, "src/app.py", mtime=BASE_MTIME)
        assert not os.path.isdir(os.path.join(ws, ".git"))
        assert _paths(_recent(client)) == ["src/app.py"]

    def test_nested_project_files_belong_to_that_project(self, client, ws, projects):
        _write(ws, "inner/child.md", mtime=BASE_MTIME + 100)
        _write(ws, "outer.md", mtime=BASE_MTIME)
        projects["p2"] = {"project_id": "p2", "workspace": os.path.join(ws, "inner")}
        assert _paths(_recent(client)) == ["outer.md"]
        assert _paths(_recent(client, "p2")) == ["child.md"]

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlink_out_of_the_workspace_is_not_followed(self, client, ws, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(str(outside), "secret.txt", mtime=BASE_MTIME + 500)
        os.symlink(str(outside), os.path.join(ws, "linkdir"))
        os.symlink(str(outside / "secret.txt"), os.path.join(ws, "link.txt"))
        _write(ws, "mine.md", mtime=BASE_MTIME)
        files = _recent(client)
        assert _paths(files) == ["mine.md"]
        assert all("secret" not in f["path"] for f in files)


class TestFailures:
    def test_file_gone_between_listing_and_stat_is_skipped(self, client, ws, monkeypatch):
        _write(ws, "here.md", mtime=BASE_MTIME)
        monkeypatch.setattr(
            history_rules, "list_tracked",
            lambda workspace, nested=(): ["gone.md", "here.md"],
        )
        assert _paths(_recent(client)) == ["here.md"]

    def test_ripgrep_missing_is_503(self, client, ws, monkeypatch):
        _write(ws, "a.md")
        monkeypatch.setattr(history_rules, "find_ripgrep", lambda: None)
        resp = client.get("/api/v2/projects/p1/files/recent")
        assert resp.status_code == 503
        assert "ripgrep" in resp.json()["detail"]

    def test_unknown_project_is_404(self, client, ws):
        resp = client.get("/api/v2/projects/nope/files/recent")
        assert resp.status_code == 404


class TestReadOnly:
    def test_a_fresh_folder_gets_no_orbital_dir(self, client, ws):
        _write(ws, "mine.docx", mtime=BASE_MTIME)
        _write(ws, "photos/cat.png", mtime=BASE_MTIME + 10)
        before = _snapshot(ws)
        files = _recent(client)
        assert _paths(files) == ["photos/cat.png", "mine.docx"]
        assert not os.path.exists(os.path.join(ws, "orbital"))
        assert _snapshot(ws) == before
