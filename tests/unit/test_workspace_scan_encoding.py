# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the Windows cold-start-scan encoding crash.

On Windows, ``subprocess.run(text=True)`` without ``encoding=`` decodes child
output with the ANSI code page (cp936/cp1252), but ripgrep emits UTF-8
filenames. A workspace containing e.g. 丁字路口.txt (U+4E01 = E4 B8 81; 0x81 is
undefined in cp1252) made the decode fail. Two observed failure shapes:

- POSIX / predicted shape: ``UnicodeDecodeError`` raised straight out of
  ``subprocess.run`` — a ``ValueError``, so the old
  ``except (OSError, subprocess.SubprocessError)`` let it escape.
- Actual Windows shape (``capture_output=True`` uses reader threads): the
  decode error kills the reader thread silently, ``proc.stdout`` comes back
  ``None``, and ``.splitlines()`` raised ``AttributeError``.

Either way the exception escaped ``_list_files`` → HTTP 500 → the frontend
mislabeled it "provider error". The fix passes ``encoding="utf-8",
errors="replace"`` (matching grep_tool) and degrades ANY listing failure to
the os.walk fallback.
"""
import subprocess

import agent_os.agent.workspace_scan as workspace_scan
from agent_os.agent.workspace_scan import scan_workspace


def _cp1252_decode_error() -> UnicodeDecodeError:
    # The real error a cp1252 box produces for 丁 (E4 B8 81): 0x81 is undefined.
    return UnicodeDecodeError(
        "charmap", b"\xe4\xb8\x81", 2, 3, "character maps to <undefined>"
    )


def test_decode_error_degrades_to_walk_fallback(tmp_path, monkeypatch):
    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")

    def raise_decode_error(*args, **kwargs):
        raise _cp1252_decode_error()

    monkeypatch.setattr(workspace_scan.subprocess, "run", raise_decode_error)
    out = scan_workspace(str(tmp_path))
    assert "kept.py" in out


def test_none_stdout_degrades_to_walk_fallback(tmp_path, monkeypatch):
    # The shape actually observed on Windows: the reader thread dies on the
    # decode error and subprocess.run returns with stdout=None.
    (tmp_path / "kept.py").write_text("x = 1\n", encoding="utf-8")

    def return_none_stdout(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=kwargs.get("args", []), returncode=0, stdout=None, stderr=None
        )

    monkeypatch.setattr(workspace_scan.subprocess, "run", return_none_stdout)
    out = scan_workspace(str(tmp_path))
    assert "kept.py" in out


def test_scan_succeeds_with_utf8_filenames(tmp_path):
    # End-to-end through the real ripgrep: red on Windows before the fix
    # (ANSI decode of UTF-8 filenames), green everywhere after.
    (tmp_path / "项目笔记.md").write_text("notes\n", encoding="utf-8")
    (tmp_path / "丁字路口.txt").write_text("junction\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text("plain\n", encoding="utf-8")
    out = scan_workspace(str(tmp_path))
    assert "readme.md" in out
    assert "项目笔记.md" in out
    assert "丁字路口.txt" in out
