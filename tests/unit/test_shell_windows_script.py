# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The PowerShell script an agent's command runs in, against real PowerShell.

Found on a Windows smoke of the sandbox worker. PowerShell's location starts at
``C:\\`` when the worker cannot read the workspace's parents (any workspace under
a user profile), and ``Set-Location`` is refused the same way, so ``git init``
in a safe-mode project created ``C:\\.git``. The process's own directory is right
throughout, which is how a test can reproduce the mismatch without an account:
``Set-Location`` moves only PowerShell's location.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agent_os.agent.tools.shell import build_windows_script

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="PowerShell on Windows")


def _run(tmp_path, command: str, *, lost: bool = False) -> subprocess.CompletedProcess:
    script = tmp_path / "s.ps1"
    # Stand-in for the worker's start: PowerShell's location is the drive root
    # while the process's directory is the workspace.
    prefix = "Set-Location -LiteralPath ($env:SystemDrive + '\\')\n" if lost else ""
    body = build_windows_script(command)
    script.write_text(prefix + body, encoding="utf-8-sig", newline="\r\n")
    workspace = tmp_path / "中文 目录"
    workspace.mkdir(exist_ok=True)
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(script)],
        cwd=workspace, capture_output=True, timeout=60)


def _out(proc) -> str:
    return proc.stdout.decode("utf-8", "replace")


def test_a_lost_location_comes_back_to_the_workspace(tmp_path):
    proc = _run(tmp_path, "cmd /c cd", lost=True)
    assert proc.returncode == 0, proc.stderr
    assert _out(proc).strip().endswith("中文 目录")


def test_a_relative_file_lands_in_the_workspace(tmp_path):
    proc = _run(tmp_path, "New-Item probe.txt -Force | Out-Null", lost=True)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "中文 目录" / "probe.txt").exists()


def test_a_normal_start_keeps_its_ordinary_location(tmp_path):
    proc = _run(tmp_path, "(Get-Location).Path")
    assert _out(proc).strip().endswith("中文 目录")


def test_recovering_the_location_reports_no_error(tmp_path):
    proc = _run(tmp_path, "$Error.Count", lost=True)
    assert _out(proc).strip() == "0"


def test_python_prints_chinese_into_the_redirected_output(tmp_path):
    code = "print('\\u4f60\\u597d')"
    proc = _run(tmp_path, f'& "{sys.executable}" -c "{code}"')
    assert proc.returncode == 0, proc.stderr
    assert "你好" in _out(proc)
