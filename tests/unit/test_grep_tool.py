# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for GrepTool.

If ripgrep is not on PATH and there is no bundled binary, most tests skip —
they do not fail — per the MVP policy of bundling being deferred.
"""

import shutil

import pytest

from agent_os.agent.tools.grep_tool import GrepTool, _find_ripgrep


def _rg_available():
    return _find_ripgrep() is not None


requires_rg = pytest.mark.skipif(not _rg_available(), reason="ripgrep not available")


class TestGrepToolBasics:
    @requires_rg
    def test_basic_literal_match(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    return 42\n")
        (tmp_path / "b.py").write_text("def goodbye():\n    pass\n")

        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="hello", fixed_strings=True)
        assert "a.py:" in result.content
        assert "hello" in result.content
        assert "b.py" not in result.content

    @requires_rg
    def test_regex_match(self, tmp_path):
        (tmp_path / "main.py").write_text(
            "def auth_login():\n    pass\ndef auth_logout():\n    pass\ndef other():\n    pass\n"
        )

        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern=r"auth_\w+")
        assert "auth_login" in result.content
        assert "auth_logout" in result.content
        assert "other" not in result.content

    @requires_rg
    def test_no_matches_returns_friendly_message(self, tmp_path):
        (tmp_path / "a.py").write_text("foo\n")
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="bananaphone_xyz")
        assert result.content.strip() == "No matches found"

    @requires_rg
    def test_fixed_strings_flag(self, tmp_path):
        # The content contains a literal '.*' sequence — without -F, regex would
        # match anything; with -F, only the literal chars match.
        (tmp_path / "a.txt").write_text("a.*b\nxxxxxx\n")
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="a.*b", fixed_strings=True)
        assert "a.*b" in result.content

    @requires_rg
    def test_glob_filter(self, tmp_path):
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "a.md").write_text("target\n")
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="target", glob_filter="*.py")
        assert "a.py" in result.content
        assert "a.md" not in result.content

    @requires_rg
    def test_paths_are_relative_to_workspace(self, tmp_path):
        sub = tmp_path / "src"
        sub.mkdir()
        (sub / "x.py").write_text("needle\n")

        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="needle")
        # Must not leak absolute path of tmp_path
        assert str(tmp_path) not in result.content
        assert "src/x.py" in result.content

    def test_missing_pattern_argument(self, tmp_path):
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute()
        assert result.content.startswith("Error")


class TestGrepWorkspaceScoping:
    def test_rejects_parent_escape(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("supersecret")

        tool = GrepTool(workspace=str(workspace))
        result = tool.execute(pattern="supersecret", path="../outside")
        assert result.content.startswith("Error")
        assert "outside workspace" in result.content

    def test_rejects_absolute_path_outside(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        outside = tmp_path / "out"
        outside.mkdir()
        (outside / "f.txt").write_text("zzz")

        tool = GrepTool(workspace=str(workspace))
        result = tool.execute(pattern="zzz", path=str(outside))
        assert result.content.startswith("Error")

    def test_rejects_nonexistent_path(self, tmp_path):
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="x", path="missing")
        assert result.content.startswith("Error")
        assert "not found" in result.content


class TestGrepTruncation:
    @requires_rg
    def test_truncation_at_100(self, tmp_path):
        # 150 matches, split across multiple files so --max-count 50 per file
        # doesn't hide them all. 3 files x 50 hits each = 150 total hits.
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text("\n".join(["needle"] * 60))

        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="needle", fixed_strings=True)
        lines = result.content.splitlines()
        assert any("truncated" in line for line in lines)
        # Results should be capped near 100 matches
        match_lines = [ln for ln in lines if "needle" in ln and ".txt:" in ln]
        assert len(match_lines) <= 100

    @requires_rg
    def test_max_columns_truncates_long_lines(self, tmp_path):
        # 2000-char line with a match
        (tmp_path / "wide.txt").write_text("needle" + "x" * 2000 + "\n")
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="needle", fixed_strings=True)
        # ripgrep emits an "omitted" marker for over-length lines — either way,
        # our returned lines should not exceed ~500 chars plus path prefix
        for line in result.content.splitlines():
            # Allow some slack for path prefix and ripgrep's own markers
            assert len(line) < 700


class TestGrepNeverRaises:
    def test_none_pattern(self, tmp_path):
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern=None)
        assert result.content.startswith("Error")

    def test_ripgrep_not_found_graceful(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "agent_os.agent.tools.grep_tool._find_ripgrep",
            lambda: None,
        )
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="anything")
        assert result.content.startswith("Error")
        assert "ripgrep not found" in result.content

    @requires_rg
    def test_timeout_returns_error_not_raises(self, tmp_path, monkeypatch):
        # Force subprocess.run to raise TimeoutExpired
        import subprocess as _sp

        def fake_run(*args, **kwargs):
            raise _sp.TimeoutExpired(cmd="rg", timeout=kwargs.get("timeout", 10))

        monkeypatch.setattr("agent_os.agent.tools.grep_tool.subprocess.run", fake_run)
        tool = GrepTool(workspace=str(tmp_path))
        result = tool.execute(pattern="x")
        assert result.content.startswith("Error")
        assert "timed out" in result.content


class TestGrepLineParsing:
    def test_split_rg_line_windows_path(self):
        from agent_os.agent.tools.grep_tool import _split_rg_line

        # Windows-style absolute path with drive colon
        line = r"C:\work\ws\src\main.py:42:    def foo():"
        parsed = _split_rg_line(line)
        assert parsed is not None
        path, lineno, content = parsed
        assert path == r"C:\work\ws\src\main.py"
        assert lineno == "42"
        assert content == "    def foo():"

    def test_split_rg_line_posix_path(self):
        from agent_os.agent.tools.grep_tool import _split_rg_line

        line = "/home/ws/src/main.py:42:    def foo():"
        parsed = _split_rg_line(line)
        assert parsed is not None
        path, lineno, content = parsed
        assert path == "/home/ws/src/main.py"
        assert lineno == "42"
        assert content == "    def foo():"
