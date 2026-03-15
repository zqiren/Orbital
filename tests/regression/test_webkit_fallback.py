# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Source-level regression tests for WebKit/Safari fallback on macOS.

These tests verify that browser_manager.py contains the expected WebKit
fallback code paths without requiring a running browser.
"""

import inspect
import textwrap

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager


def _get_source(method_name: str) -> str:
    method = getattr(BrowserManager, method_name)
    return textwrap.dedent(inspect.getsource(method))


class TestLaunchWebKitFallback:
    """Verify _launch() contains WebKit fallback code."""

    def test_launch_uses_playwright_webkit(self):
        src = _get_source("_launch")
        assert "self._playwright.webkit" in src

    def test_launch_logs_webkit(self):
        src = _get_source("_launch")
        assert "WebKit" in src

    def test_launch_checks_darwin(self):
        src = _get_source("_launch")
        assert 'sys.platform == "darwin"' in src


class TestWarmupWebKitFallback:
    """Verify launch_warmup() contains WebKit fallback code."""

    def test_warmup_uses_pw_webkit(self):
        src = _get_source("_launch_warmup_impl")
        assert "pw.webkit" in src

    def test_warmup_logs_webkit(self):
        src = _get_source("_launch_warmup_impl")
        assert "WebKit" in src

    def test_warmup_checks_darwin(self):
        src = _get_source("_launch_warmup_impl")
        assert 'sys.platform == "darwin"' in src
