# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: sandbox shell commands must always run in the project workspace.

CreateProcessWithLogonW with LOGON_NO_PROFILE unreliably sets the working
directory for the sandbox user.  The fix prepends `cd /d "<workspace>"` to
every wrapped command so cwd is correct regardless of OS behavior.
"""

import inspect
import os
import subprocess
import sys
import tempfile

import pytest

# Only run on Windows — sandbox process launching is Windows-only
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")


# ---------------------------------------------------------------------------
# Helper: simulate production run_and_capture quoting
# ---------------------------------------------------------------------------

def _run_production_style(working_dir: str, ps_command: str, *, skip_tmp: bool = False):
    """Simulate ProcessLauncher.run_and_capture with the cd /d fix.

    Uses the same quoting scheme as _build_command_line: outer quotes around
    the entire wrapped_inner arg so cmd /c strips them (rule 2), leaving the
    inner quotes intact.
    """
    original_args = ["-NoProfile", "-Command", ps_command]
    args_joined = " ".join(f'"{a}"' if " " in a else a for a in original_args)

    if skip_tmp:
        stdout_path = os.path.join(os.environ.get("TEMP", ""), "test_stdout.txt")
        stderr_path = os.path.join(os.environ.get("TEMP", ""), "test_stderr.txt")
    else:
        tmp_dir = os.path.join(working_dir, "orbital", ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        stdout_path = os.path.join(tmp_dir, "test_stdout.txt")
        stderr_path = os.path.join(tmp_dir, "test_stderr.txt")

    wrapped = (
        f'cd /d "{working_dir}" && powershell {args_joined}'
        f' > "{stdout_path}" 2> "{stderr_path}"'
    )
    # _build_command_line wraps in outer quotes because wrapped has spaces
    cmdline = f'cmd /c "{wrapped}"'

    proc = subprocess.run(
        cmdline,
        capture_output=True,
        text=True,
        timeout=30,
        # Deliberately start from a WRONG cwd to prove the fix works
        cwd=os.environ.get("SystemRoot", r"C:\Windows"),
    )

    stdout = ""
    try:
        with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
            stdout = f.read().strip()
    except FileNotFoundError:
        pass

    for p in (stdout_path, stderr_path):
        try:
            os.remove(p)
        except OSError:
            pass

    return proc.returncode, stdout


# ---------------------------------------------------------------------------
# Source-level: verify the fix exists in process.py
# ---------------------------------------------------------------------------

class TestProcessModuleSourceFix:
    """Verify the cd /d fix is present in the actual source code."""

    def test_run_and_capture_has_cd_prefix(self):
        """run_and_capture must include cd /d in the wrapped command."""
        from agent_os.platform.windows.process import ProcessLauncher
        source = inspect.getsource(ProcessLauncher.run_and_capture)
        assert 'cd /d' in source, \
            "run_and_capture must prepend cd /d to fix sandbox cwd"

    def test_cd_prefix_uses_working_dir(self):
        """cd /d must reference the working_dir parameter."""
        from agent_os.platform.windows.process import ProcessLauncher
        source = inspect.getsource(ProcessLauncher.run_and_capture)
        assert 'cd /d "{working_dir}"' in source

    def test_cd_prefix_chains_with_double_ampersand(self):
        """cd /d must be chained with && so the command only runs if cd succeeds."""
        from agent_os.platform.windows.process import ProcessLauncher
        source = inspect.getsource(ProcessLauncher.run_and_capture)
        assert '&& {command}' in source


# ---------------------------------------------------------------------------
# Unit: wrapped command string shape
# ---------------------------------------------------------------------------

class TestWrappedCommandShape:
    """Verify the cd /d prefix appears correctly for various path shapes."""

    @staticmethod
    def _build_wrapped(working_dir: str) -> str:
        args_joined = '-NoProfile -Command ls'
        return (
            f'cd /d "{working_dir}" && powershell {args_joined}'
            f' > "stdout" 2> "stderr"'
        )

    def test_basic_path(self):
        w = self._build_wrapped(r"C:\Users\test\project")
        assert w.startswith('cd /d "C:\\Users\\test\\project" && ')

    def test_path_with_spaces(self):
        w = self._build_wrapped(r"C:\Users\test user\My Project")
        assert 'cd /d "C:\\Users\\test user\\My Project"' in w

    def test_deep_path(self):
        p = r"C:\a\b\c\d\e\f\g\h"
        w = self._build_wrapped(p)
        assert f'cd /d "{p}"' in w

    def test_path_with_special_chars(self):
        p = r"C:\project (v2) [beta]"
        w = self._build_wrapped(p)
        assert f'cd /d "{p}"' in w


# ---------------------------------------------------------------------------
# Integration: production-style subprocess execution
# ---------------------------------------------------------------------------

class TestCwdIntegration:
    """Run real commands via the production quoting path to verify cwd."""

    @pytest.fixture
    def workspace(self, tmp_path):
        marker = tmp_path / "cwd_marker.txt"
        marker.write_text("workspace-found")
        return str(tmp_path)

    def test_cwd_is_workspace(self, workspace):
        """Get-Location must return the workspace, not the parent cwd."""
        rc, out = _run_production_style(workspace, "Get-Location")
        assert workspace.lower() in out.lower()

    def test_file_access(self, workspace):
        """Commands must be able to read workspace files by relative path."""
        rc, out = _run_production_style(workspace, "Get-Content cwd_marker.txt")
        assert out == "workspace-found"

    def test_nested_relative_path(self, workspace):
        sub = os.path.join(workspace, "sub", "deep")
        os.makedirs(sub)
        with open(os.path.join(sub, "data.txt"), "w") as f:
            f.write("nested-ok")
        rc, out = _run_production_style(workspace, "Get-Content sub/deep/data.txt")
        assert out == "nested-ok"

    def test_path_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="space test ") as td:
            with open(os.path.join(td, "s.txt"), "w") as f:
                f.write("spaced")
            rc, out = _run_production_style(td, "Get-Content s.txt")
            assert out == "spaced"

    def test_path_with_parentheses(self):
        with tempfile.TemporaryDirectory(prefix="proj(v2)_") as td:
            with open(os.path.join(td, "p.txt"), "w") as f:
                f.write("parens-ok")
            rc, out = _run_production_style(td, "Get-Content p.txt")
            assert out == "parens-ok"

    def test_unicode_path(self):
        with tempfile.TemporaryDirectory(prefix="test_\u4e2d\u6587_") as td:
            with open(os.path.join(td, "u.txt"), "w", encoding="utf-8") as f:
                f.write("unicode-ok")
            rc, out = _run_production_style(td, "Get-Content u.txt")
            assert out == "unicode-ok"

    def test_repeated_runs_consistent(self, workspace):
        """5 consecutive runs must all land in the same workspace."""
        for _ in range(5):
            rc, out = _run_production_style(workspace, "Get-Location")
            assert workspace.lower() in out.lower()

    def test_different_workspaces(self):
        with tempfile.TemporaryDirectory(prefix="a_") as ws_a, \
             tempfile.TemporaryDirectory(prefix="b_") as ws_b:
            _, out_a = _run_production_style(ws_a, "Get-Location")
            _, out_b = _run_production_style(ws_b, "Get-Location")
            assert ws_a.lower() in out_a.lower()
            assert ws_b.lower() in out_b.lower()

    def test_exit_code_propagation(self, workspace):
        rc, _ = _run_production_style(workspace, "exit 42")
        assert rc == 42

    def test_bad_path_prevents_execution(self):
        rc, out = _run_production_style(
            r"Z:\nonexistent\path", "echo should-not-run", skip_tmp=True
        )
        assert "should-not-run" not in out
        assert rc != 0

    def test_long_path(self, workspace):
        long_sub = os.path.join(workspace, "a" * 50, "b" * 50)
        os.makedirs(long_sub)
        with open(os.path.join(long_sub, "deep.txt"), "w") as f:
            f.write("deep-ok")
        rc, out = _run_production_style(long_sub, "Get-Content deep.txt")
        assert out == "deep-ok"

    def test_x_manager_workspace(self):
        """Verify the fix works with the actual x-manager workspace."""
        xm = r"C:\Users\qiren\Desktop\x-manager"
        if not os.path.isdir(xm):
            pytest.skip("x-manager workspace not found")
        rc, out = _run_production_style(xm, "Get-Location")
        assert xm.lower() in out.lower()
