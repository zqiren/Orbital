# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Targeted tests for folder expansion bug.

Reproduces the exact API flow the frontend performs when a user clicks
a folder in the FileExplorer:
  1. Root listing  → GET /api/v2/projects/{pid}/files
  2. Folder click  → GET /api/v2/projects/{pid}/files?path=<folder>
  3. Nested folder  → GET /api/v2/projects/{pid}/files?path=<folder>/<sub>

Also includes pure-function tests for the tree manipulation helpers
(findNode, updateNode, setNodeChildren) extracted from FileExplorer.tsx
to verify they handle all edge cases correctly.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes import files_v2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deep_workspace(tmp_path):
    """Create a workspace with nested directories mimicking a real project."""
    # Root files
    (tmp_path / "README.md").write_text("# Project", encoding="utf-8")
    (tmp_path / "setup.py").write_text("setup()", encoding="utf-8")

    # Level-1 directory
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')", encoding="utf-8")

    # Level-2 directory
    utils = src / "utils"
    utils.mkdir()
    (utils / "helpers.py").write_text("def helper(): pass", encoding="utf-8")

    # Empty directory
    empty = tmp_path / "empty_dir"
    empty.mkdir()

    # agent_output directory (pinned to top by sortEntries)
    ao = tmp_path / "agent_output"
    ao.mkdir()
    (ao / "result.json").write_text("{}", encoding="utf-8")

    return tmp_path


@pytest.fixture
def client(deep_workspace):
    app = FastAPI()
    mock_store = MagicMock()
    mock_store.get_project.return_value = {
        "project_id": "proj_1",
        "workspace": str(deep_workspace),
    }
    files_v2.configure(mock_store)
    app.include_router(files_v2.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Backend: exact folder-expansion API flow
# ---------------------------------------------------------------------------

class TestFolderExpansionFlow:
    """Simulates the exact sequence the frontend executes when expanding
    folders in the FileExplorer tree."""

    def test_root_listing_returns_directories(self, client):
        """Step 1: initial page load fetches the root directory."""
        resp = client.get("/api/v2/projects/proj_1/files")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == ""
        dirs = [e for e in data["entries"] if e["type"] == "directory"]
        dir_names = {d["name"] for d in dirs}
        assert "src" in dir_names
        assert "empty_dir" in dir_names
        assert "agent_output" in dir_names

    def test_expand_first_level_folder(self, client):
        """Step 2: user clicks 'src' folder → fetch its children."""
        resp = client.get("/api/v2/projects/proj_1/files?path=src")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "src"
        names = [e["name"] for e in data["entries"]]
        assert "main.py" in names
        assert "utils" in names

    def test_expand_nested_folder(self, client):
        """Step 3: user clicks 'src/utils' folder → fetch its children."""
        resp = client.get("/api/v2/projects/proj_1/files?path=src/utils")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "src/utils"
        names = [e["name"] for e in data["entries"]]
        assert "helpers.py" in names

    def test_expand_empty_directory(self, client):
        """Edge case: expanding an empty directory returns empty entries."""
        resp = client.get("/api/v2/projects/proj_1/files?path=empty_dir")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entries"] == []

    def test_full_expansion_sequence(self, client):
        """End-to-end: root → src → src/utils in one flow, exactly as
        the frontend would perform it."""
        # 1. Root listing
        root = client.get("/api/v2/projects/proj_1/files").json()
        src_entry = next(e for e in root["entries"] if e["name"] == "src")
        assert src_entry["type"] == "directory"

        # 2. Expand src (frontend constructs path = "src")
        src_data = client.get("/api/v2/projects/proj_1/files?path=src").json()
        utils_entry = next(e for e in src_data["entries"] if e["name"] == "utils")
        assert utils_entry["type"] == "directory"

        # 3. Expand src/utils (frontend constructs path = "src/utils")
        utils_data = client.get(
            "/api/v2/projects/proj_1/files?path=src/utils"
        ).json()
        assert any(e["name"] == "helpers.py" for e in utils_data["entries"])


# ---------------------------------------------------------------------------
# Frontend helper functions (extracted from FileExplorer.tsx for testing)
# ---------------------------------------------------------------------------

class TreeNode:
    """Python mirror of the TypeScript TreeNode interface."""
    def __init__(self, name, node_type, parent_path=""):
        self.entry = {"name": name, "type": node_type}
        self.path = f"{parent_path}/{name}" if parent_path else name
        self.children = None  # None = not loaded
        self.expanded = False
        self.loading = False

    def to_dict(self):
        return {
            "entry": self.entry,
            "path": self.path,
            "children": [c.to_dict() for c in self.children] if self.children is not None else None,
            "expanded": self.expanded,
            "loading": self.loading,
        }


def find_node(nodes, path):
    """Python port of findNode from FileExplorer.tsx."""
    for node in nodes:
        if node.path == path:
            return node
        if node.children:
            found = find_node(node.children, path)
            if found:
                return found
    return None


def update_node(nodes, path, updates):
    """Python port of updateNode from FileExplorer.tsx."""
    result = []
    for node in nodes:
        if node.path == path:
            new_node = TreeNode.__new__(TreeNode)
            new_node.__dict__ = {**node.__dict__, **updates}
            result.append(new_node)
        elif node.children:
            new_node = TreeNode.__new__(TreeNode)
            new_node.__dict__ = {**node.__dict__, "children": update_node(node.children, path, updates)}
            result.append(new_node)
        else:
            result.append(node)
    return result


def set_node_children(nodes, path, children):
    """Python port of setNodeChildren from FileExplorer.tsx."""
    result = []
    for node in nodes:
        if node.path == path:
            new_node = TreeNode.__new__(TreeNode)
            new_node.__dict__ = {**node.__dict__, "children": children, "loading": False}
            result.append(new_node)
        elif node.children:
            new_node = TreeNode.__new__(TreeNode)
            new_node.__dict__ = {**node.__dict__, "children": set_node_children(node.children, path, children)}
            result.append(new_node)
        else:
            result.append(node)
    return result


class TestTreeHelpers:
    """Tests for the pure tree manipulation functions ported from
    FileExplorer.tsx.  These verify the logic that drives folder expansion."""

    def _make_root(self):
        """Build a sample tree: [agent_output/, src/, README.md]."""
        ao = TreeNode("agent_output", "directory")
        src = TreeNode("src", "directory")
        readme = TreeNode("README.md", "file")
        return [ao, src, readme]

    def test_find_node_at_root(self):
        nodes = self._make_root()
        found = find_node(nodes, "src")
        assert found is not None
        assert found.path == "src"

    def test_find_node_not_found(self):
        nodes = self._make_root()
        assert find_node(nodes, "nonexistent") is None

    def test_find_node_in_children(self):
        nodes = self._make_root()
        # Simulate "src" being expanded with children
        main_py = TreeNode("main.py", "file", "src")
        utils = TreeNode("utils", "directory", "src")
        nodes[1].children = [main_py, utils]
        nodes[1].expanded = True
        found = find_node(nodes, "src/utils")
        assert found is not None
        assert found.path == "src/utils"

    def test_update_node_sets_loading(self):
        nodes = self._make_root()
        updated = update_node(nodes, "src", {"loading": True, "expanded": True})
        target = find_node(updated, "src")
        assert target.loading is True
        assert target.expanded is True
        # Other nodes unaffected
        assert find_node(updated, "agent_output").loading is False

    def test_set_node_children_on_root_node(self):
        nodes = self._make_root()
        # Mark src as loading (simulating first setRootNodes call)
        nodes = update_node(nodes, "src", {"loading": True, "expanded": True})
        # Now set children (simulating second setRootNodes call after fetch)
        child1 = TreeNode("main.py", "file", "src")
        child2 = TreeNode("utils", "directory", "src")
        updated = set_node_children(nodes, "src", [child1, child2])
        target = find_node(updated, "src")
        assert target.loading is False
        assert target.children is not None
        assert len(target.children) == 2
        assert target.expanded is True  # expanded preserved from spread

    def test_set_node_children_on_nested_node(self):
        """Tests expanding a second-level folder (src/utils)."""
        nodes = self._make_root()
        # First expand src
        child_main = TreeNode("main.py", "file", "src")
        child_utils = TreeNode("utils", "directory", "src")
        nodes = update_node(nodes, "src", {"expanded": True})
        nodes = set_node_children(nodes, "src", [child_main, child_utils])
        # Now click utils → set loading
        nodes = update_node(nodes, "src/utils", {"loading": True, "expanded": True})
        target = find_node(nodes, "src/utils")
        assert target.loading is True
        # Fetch returns helpers.py
        helper = TreeNode("helpers.py", "file", "src/utils")
        nodes = set_node_children(nodes, "src/utils", [helper])
        target = find_node(nodes, "src/utils")
        assert target.loading is False
        assert len(target.children) == 1
        assert target.children[0].path == "src/utils/helpers.py"

    def test_set_node_children_empty_directory(self):
        nodes = self._make_root()
        nodes = update_node(nodes, "agent_output", {"loading": True, "expanded": True})
        nodes = set_node_children(nodes, "agent_output", [])
        target = find_node(nodes, "agent_output")
        assert target.loading is False
        assert target.children == []
        assert target.expanded is True


class TestNeedsFetchRaceCondition:
    """Demonstrates the core bug: the `needsFetch` closure variable pattern
    used in toggleDirectory is unreliable with React 19's batched updates.

    In React 19 (createRoot + StrictMode), setState updater functions may
    be deferred to the render phase rather than called eagerly.  When that
    happens, the local `needsFetch` variable is still `false` when the
    `if (needsFetch)` check executes, so the fetch never fires.

    This test simulates both the working and broken scenarios.
    """

    def test_synchronous_updater_works(self):
        """When the updater runs synchronously (eager computation),
        needsFetch is correctly set to True."""
        nodes = [TreeNode("src", "directory")]
        need_fetch = False

        # Simulate synchronous updater execution
        def updater(prev):
            nonlocal need_fetch
            target = find_node(prev, "src")
            if not target:
                return prev
            if target.expanded:
                return update_node(prev, "src", {"expanded": False})
            if target.children is not None:
                return update_node(prev, "src", {"expanded": True})
            need_fetch = True
            return update_node(prev, "src", {"loading": True, "expanded": True})

        # Simulate React calling updater SYNCHRONOUSLY (eager path)
        new_nodes = updater(nodes)
        # needsFetch should be True → fetch proceeds
        assert need_fetch is True

    def test_deferred_updater_breaks(self):
        """When the updater is deferred (React 19 batching), needsFetch
        is still False when the if-check runs.  This is the BUG."""
        nodes = [TreeNode("src", "directory")]
        need_fetch = False

        def updater(prev):
            nonlocal need_fetch
            target = find_node(prev, "src")
            if not target:
                return prev
            if target.expanded:
                return update_node(prev, "src", {"expanded": False})
            if target.children is not None:
                return update_node(prev, "src", {"expanded": True})
            need_fetch = True
            return update_node(prev, "src", {"loading": True, "expanded": True})

        # Simulate React DEFERRING the updater (queued for render phase)
        # The updater is NOT called yet — it's just stored.
        enqueued_updater = updater  # saved but not called

        # At this point, toggleDirectory checks needsFetch:
        assert need_fetch is False  # BUG: fetch is skipped!

        # Later, React calls the updater during render:
        new_nodes = enqueued_updater(nodes)
        # Now needsFetch is True, but toggleDirectory already returned
        assert need_fetch is True  # too late — the fetch was skipped

    def test_set_node_children_unreachable_when_no_fetch(self):
        """When needsFetch is False (deferred updater), setNodeChildren
        is never called, so the node stays loading forever."""
        nodes = [TreeNode("src", "directory")]

        # Simulate the deferred scenario: updater runs during render
        nodes = update_node(nodes, "src", {"loading": True, "expanded": True})
        target = find_node(nodes, "src")

        # The node is stuck: loading=True, children=None
        assert target.loading is True
        assert target.children is None
        assert target.expanded is True

        # This state causes the spinner to show but children to never render:
        # - node.loading → spinner shows
        # - node.expanded && node.children → False (children is None)
        # Result: infinite loading spinner
