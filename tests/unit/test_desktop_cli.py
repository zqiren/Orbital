# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test --setup-sandbox and --teardown-sandbox CLI flag handling."""

import subprocess
import sys

import pytest


class TestSetupSandboxFlag:
    """Tests for --setup-sandbox CLI entry point."""

    def test_flag_exits_cleanly(self):
        """--setup-sandbox exits 0 (NullProvider on non-admin Windows skips setup)."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_os.desktop.main", "--setup-sandbox"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "AGENT_OS_NO_SANDBOX": "1"},
        )
        assert result.returncode == 0

    def test_no_window_opens(self):
        """--setup-sandbox should NOT start daemon or open pywebview window.

        The subprocess should exit quickly (< 10 seconds).
        If it hangs, it means daemon/window started incorrectly.
        """
        result = subprocess.run(
            [sys.executable, "-m", "agent_os.desktop.main", "--setup-sandbox"],
            capture_output=True, text=True, timeout=10,
            env={**__import__("os").environ, "AGENT_OS_NO_SANDBOX": "1"},
        )
        assert result.returncode == 0

    def test_idempotent(self):
        """Running --setup-sandbox twice should succeed both times."""
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-m", "agent_os.desktop.main", "--setup-sandbox"],
                capture_output=True, text=True, timeout=30,
                env={**__import__("os").environ, "AGENT_OS_NO_SANDBOX": "1"},
            )
            assert result.returncode == 0


class TestTeardownSandboxFlag:
    """Tests for --teardown-sandbox CLI entry point."""

    def test_flag_exits_cleanly(self):
        """--teardown-sandbox exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_os.desktop.main", "--teardown-sandbox"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "AGENT_OS_NO_SANDBOX": "1"},
        )
        assert result.returncode == 0

    def test_never_fails(self):
        """Teardown must NEVER return non-zero, even if sandbox doesn't exist."""
        result = subprocess.run(
            [sys.executable, "-m", "agent_os.desktop.main", "--teardown-sandbox"],
            capture_output=True, text=True, timeout=30,
            env={**__import__("os").environ, "AGENT_OS_NO_SANDBOX": "1"},
        )
        assert result.returncode == 0


class TestNormalLaunch:
    """Verify normal launch (no flags) doesn't accidentally run sandbox setup."""

    def test_no_flag_does_not_run_setup(self):
        """Without --setup-sandbox, the entry point should NOT print sandbox messages.

        It will start the daemon (which we kill after a few seconds) — that's expected.
        """
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent_os.desktop.main"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        combined = (stdout or "") + (stderr or "")
        assert "Running sandbox setup" not in combined
        assert "Sandbox setup completed" not in combined
