# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the files_v2 reveal endpoint (spec 093).

``POST /projects/{id}/files/reveal`` shows a workspace path in the OS file
manager: ``open -R`` on macOS, ``explorer /select,`` on Windows, and the
project folder itself (opened, not selected in its parent) for the root. The
target goes through ``_resolve_path``'s realpath containment, so nothing
outside the workspace (``..``, absolute paths, symlinks pointing out, sibling
prefixes) can ever be handed to the OS. A relayed request (the phone) is
refused: Finder would open on the desktop, not in the user's hand.

Nothing real is spawned here: the route tests replace ``_spawn_reveal``, and
the spawner tests replace ``subprocess.run`` / ``os.startfile``.
"""
import os
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.api.routes import files_v2
from agent_os.utils.subprocess_flags import win_no_window_flags


URL = "/api/v2/projects/proj_1/files/reveal"


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "notes.md").write_text("# notes\n", encoding="utf-8")
    (ws / "docs").mkdir()
    (ws / "docs" / "plan.md").write_text("plan\n", encoding="utf-8")
    return ws


@pytest.fixture
def spawned(monkeypatch):
    """Record every reveal instead of opening a real Finder window."""
    calls = []
    monkeypatch.setattr(
        files_v2, "_spawn_reveal",
        lambda target, is_root, platform: calls.append((target, is_root, platform)),
    )
    return calls


@pytest.fixture
def client(workspace, spawned, monkeypatch):
    monkeypatch.setattr(files_v2.sys, "platform", "darwin")
    app = FastAPI()
    mock_store = MagicMock()
    mock_store.get_project.side_effect = lambda pid: (
        {"project_id": "proj_1", "workspace": str(workspace)} if pid == "proj_1" else None
    )
    files_v2.configure(mock_store)
    app.include_router(files_v2.router)
    return TestClient(app)


# ── the argv each platform needs ──────────────────────────────────────────


class TestRevealArgv:
    def test_macos_file_is_selected_in_its_folder(self):
        assert files_v2._reveal_argv("/ws/a.md", False, "darwin") == ["open", "-R", "/ws/a.md"]

    def test_macos_root_opens_inside_the_project_folder(self):
        assert files_v2._reveal_argv("/ws", True, "darwin") == ["open", "/ws"]

    def test_windows_file_uses_explorer_select(self):
        assert files_v2._reveal_argv(r"C:\ws\a b.md", False, "win32") == [
            "explorer", r"/select,C:\ws\a b.md",
        ]

    def test_windows_root_is_left_to_startfile(self):
        assert files_v2._reveal_argv(r"C:\ws", True, "win32") is None

    def test_other_platforms_are_refused(self):
        with pytest.raises(ValueError):
            files_v2._reveal_argv("/ws/a.md", False, "linux")


# ── the spawner ───────────────────────────────────────────────────────────


class TestSpawnReveal:
    def _capture_run(self, monkeypatch, returncode=0):
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return subprocess.CompletedProcess(argv, returncode, b"", b"")

        monkeypatch.setattr(files_v2.subprocess, "run", fake_run)
        return captured

    def test_runs_the_argv_with_a_timeout_and_the_no_window_flag(self, monkeypatch):
        captured = self._capture_run(monkeypatch)
        files_v2._spawn_reveal("/ws/a.md", False, "darwin")
        assert captured["argv"] == ["open", "-R", "/ws/a.md"]
        assert captured["kwargs"]["creationflags"] == win_no_window_flags()
        assert captured["kwargs"]["timeout"] == 10
        assert captured["kwargs"].get("shell") in (None, False)

    def test_windows_ignores_explorers_exit_code_one(self, monkeypatch):
        """explorer.exe exits 1 even when the window opened fine (spec R1)."""
        captured = self._capture_run(monkeypatch, returncode=1)
        files_v2._spawn_reveal(r"C:\ws\a.md", False, "win32")  # must not raise
        assert captured["argv"] == ["explorer", r"/select,C:\ws\a.md"]

    def test_windows_root_goes_through_startfile(self, monkeypatch):
        opened = []
        monkeypatch.setattr(files_v2.os, "startfile", opened.append, raising=False)
        monkeypatch.setattr(
            files_v2.subprocess, "run",
            lambda *a, **k: pytest.fail("the root must not spawn explorer"),
        )
        files_v2._spawn_reveal(r"C:\ws", True, "win32")
        assert opened == [r"C:\ws"]


# ── the route ─────────────────────────────────────────────────────────────


class TestRevealRoute:
    def test_reveals_a_file(self, client, spawned, workspace):
        resp = client.post(URL, json={"path": "notes.md"})
        assert resp.status_code == 200
        assert resp.json() == {"revealed": True, "path": "notes.md"}
        assert spawned == [(str(workspace / "notes.md"), False, "darwin")]

    def test_reveals_a_directory_selected_in_its_parent(self, client, spawned, workspace):
        resp = client.post(URL, json={"path": "docs"})
        assert resp.status_code == 200
        assert spawned == [(str(workspace / "docs"), False, "darwin")]

    def test_reveals_a_nested_file(self, client, spawned, workspace):
        resp = client.post(URL, json={"path": "docs/plan.md"})
        assert resp.status_code == 200
        assert spawned == [(str(workspace / "docs" / "plan.md"), False, "darwin")]

    @pytest.mark.parametrize("body", [{"path": ""}, {"path": "."}, {}])
    def test_root_opens_the_project_folder(self, client, spawned, workspace, body):
        resp = client.post(URL, json=body)
        assert resp.status_code == 200
        assert len(spawned) == 1
        target, is_root, _platform = spawned[0]
        assert is_root is True
        assert os.path.realpath(target) == os.path.realpath(workspace)

    def test_missing_path_is_404(self, client, spawned):
        resp = client.post(URL, json={"path": "nope.md"})
        assert resp.status_code == 404
        assert spawned == []

    def test_unknown_project_is_404(self, client, spawned):
        resp = client.post("/api/v2/projects/nope/files/reveal", json={"path": "notes.md"})
        assert resp.status_code == 404
        assert spawned == []

    def test_traversal_is_400(self, client, spawned, workspace):
        (workspace.parent / "secret.md").write_text("SECRET\n", encoding="utf-8")
        for path in ("..", "../", "../secret.md", "docs/../../secret.md"):
            resp = client.post(URL, json={"path": path})
            assert resp.status_code == 400, path
        assert spawned == []

    def test_absolute_path_outside_is_400(self, client, spawned, workspace):
        outside = workspace.parent / "elsewhere.md"
        outside.write_text("x\n", encoding="utf-8")
        for path in (str(outside), "/etc", "/"):
            resp = client.post(URL, json={"path": path})
            assert resp.status_code == 400, path
        assert spawned == []

    def test_symlink_pointing_outside_is_400(self, client, spawned, workspace):
        outside_file = workspace.parent / "outside_secret.txt"
        outside_file.write_text("SECRET\n", encoding="utf-8")
        outside_dir = workspace.parent / "outside_dir"
        outside_dir.mkdir()
        os.symlink(outside_file, workspace / "evil.md")
        os.symlink(outside_dir, workspace / "evil_dir")
        for path in ("evil.md", "evil_dir"):
            resp = client.post(URL, json={"path": path})
            assert resp.status_code == 400, path
        assert spawned == []

    def test_sibling_prefix_is_400(self, client, spawned, workspace):
        sibling = workspace.parent / (workspace.name + "-evil")
        sibling.mkdir()
        (sibling / "x.md").write_text("x\n", encoding="utf-8")
        resp = client.post(URL, json={"path": f"../{workspace.name}-evil/x.md"})
        assert resp.status_code == 400
        assert spawned == []

    def test_relayed_request_is_refused(self, client, spawned):
        """The phone reaches the daemon through the relay; Finder would open on
        the desktop, so the daemon refuses rather than acting remotely."""
        resp = client.post(URL, json={"path": "notes.md"}, headers={"X-Via-Relay": "true"})
        assert resp.status_code == 403
        assert spawned == []

    def test_relay_refusal_survives_a_second_forged_header(self, client, spawned):
        """The relay client ADDS X-Via-Relay: true to the phone's headers; a
        phone that also sends its own value must not mask it."""
        resp = client.post(
            URL,
            json={"path": "notes.md"},
            headers=[("x-via-relay", "false"), ("X-Via-Relay", "true")],
        )
        assert resp.status_code == 403
        assert spawned == []

    def test_unsupported_platform_is_501(self, client, spawned, monkeypatch):
        monkeypatch.setattr(files_v2.sys, "platform", "linux")
        resp = client.post(URL, json={"path": "notes.md"})
        assert resp.status_code == 501
        assert spawned == []

    def test_windows_platform_is_passed_to_the_spawner(self, client, spawned, monkeypatch):
        monkeypatch.setattr(files_v2.sys, "platform", "win32")
        resp = client.post(URL, json={"path": "notes.md"})
        assert resp.status_code == 200
        assert spawned[0][2] == "win32"

    def test_spawn_failure_is_500(self, client, monkeypatch):
        def boom(target, is_root, platform):
            raise FileNotFoundError("open")

        monkeypatch.setattr(files_v2, "_spawn_reveal", boom)
        resp = client.post(URL, json={"path": "notes.md"})
        assert resp.status_code == 500
