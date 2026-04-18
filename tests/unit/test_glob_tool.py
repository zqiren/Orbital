# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GlobTool."""

import os

import pytest

from agent_os.agent.tools.glob_tool import GlobTool


class TestGlobToolBasics:
    def test_basic_pattern_python_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("x")
        (tmp_path / "c.md").write_text("x")

        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.py")

        lines = sorted(result.content.splitlines())
        assert lines == ["a.py", "b.py"]

    def test_recursive_pattern(self, tmp_path):
        (tmp_path / "top.py").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("x")
        (sub / "deeper").mkdir()
        (sub / "deeper" / "x.py").write_text("x")

        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="**/*.py")

        lines = result.content.splitlines()
        # pathlib rglob-equivalent via ** should find all three
        assert "top.py" in lines
        assert "sub/deep.py" in lines
        assert "sub/deeper/x.py" in lines

    def test_empty_results(self, tmp_path):
        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.nonexistent")
        assert "no matches" in result.content.lower()

    def test_missing_pattern_argument(self, tmp_path):
        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute()
        assert result.content.startswith("Error")
        assert "pattern" in result.content.lower()

    def test_results_are_sorted(self, tmp_path):
        for name in ["zeta.py", "alpha.py", "mu.py"]:
            (tmp_path / name).write_text("x")

        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.py")
        lines = result.content.splitlines()
        assert lines == sorted(lines)

    def test_path_subdirectory(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "main.py").write_text("x")
        (tmp_path / "outside.py").write_text("x")

        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.py", path="src")
        lines = result.content.splitlines()
        assert "src/main.py" in lines
        assert "outside.py" not in lines


class TestGlobWorkspaceScoping:
    def test_rejects_parent_escape(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("nope")

        tool = GlobTool(workspace=str(workspace))
        result = tool.execute(pattern="*.txt", path="../outside")
        assert result.content.startswith("Error")
        assert "outside workspace" in result.content

    def test_rejects_absolute_path_outside(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "other"
        outside.mkdir()

        tool = GlobTool(workspace=str(workspace))
        result = tool.execute(pattern="*", path=str(outside))
        assert result.content.startswith("Error")

    def test_rejects_nonexistent_path(self, tmp_path):
        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.py", path="does/not/exist")
        assert result.content.startswith("Error")
        assert "not found" in result.content

    def test_rejects_path_is_file_not_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.py", path="file.txt")
        assert result.content.startswith("Error")


class TestGlobTruncation:
    def test_truncation_at_1000(self, tmp_path):
        # Create 1100 matching files
        for i in range(1100):
            (tmp_path / f"f{i:04d}.dat").write_text("")

        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern="*.dat")
        lines = result.content.splitlines()

        # One line is the truncation marker
        assert any("more paths" in line for line in lines)
        # 1000 result lines + 1 marker line
        assert len(lines) == 1001


class TestGlobNeverRaises:
    def test_invalid_pattern_returns_error_not_raises(self, tmp_path):
        tool = GlobTool(workspace=str(tmp_path))
        # An arguably-invalid path-like pattern; should not raise
        try:
            result = tool.execute(pattern="[")
        except Exception as e:
            pytest.fail(f"GlobTool raised instead of returning error: {e}")
        # Either an Error or (no matches) — the point is: no raise
        assert isinstance(result.content, str)

    def test_none_pattern_returns_error(self, tmp_path):
        tool = GlobTool(workspace=str(tmp_path))
        result = tool.execute(pattern=None)
        assert result.content.startswith("Error")
