# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Source-level regression tests for the browser launch fallback chain.

The chain is Chrome → Edge → bundled Chromium, in both the automation
launch and the sign-in warmup. There is deliberately NO WebKit tier: it
required a Playwright WebKit build that nothing ever downloads (the
migration only runs ``install chromium``), so it could only fail — while
its error copy claimed Safari support the product doesn't have. These
tests keep the dead tier from coming back and keep the terminal error
honest for users who cannot install Chrome (mainland China: lead with
Edge, never suggest the CDN download that just failed).
"""

import inspect
import textwrap

from agent_os.daemon_v2.browser_manager import BrowserManager


def _get_source(method_name: str) -> str:
    method = getattr(BrowserManager, method_name)
    return textwrap.dedent(inspect.getsource(method))


class TestLaunchFallbackChain:
    def test_launch_has_no_webkit_tier(self):
        src = _get_source("_launch")
        assert ".webkit" not in src

    def test_launch_tries_chrome_then_edge_then_bundled(self):
        src = _get_source("_launch")
        assert '"chrome"' in src and '"msedge"' in src

    def test_launch_terminal_error_is_honest(self):
        src = _get_source("_launch")
        assert "Safari" not in src
        assert "patchright install" not in src
        assert "Microsoft Edge" in src

    def test_launch_diagnoses_profile_lock_before_generic_error(self):
        src = _get_source("_launch")
        assert "_profile_lock_holder" in src


class TestWarmupFallbackChain:
    def test_warmup_has_no_webkit_tier(self):
        src = _get_source("_launch_warmup_impl")
        assert ".webkit" not in src

    def test_warmup_terminal_error_is_honest(self):
        src = _get_source("_launch_warmup_impl")
        assert "Safari" not in src
        assert "Microsoft Edge" in src

    def test_warmup_diagnoses_profile_lock_before_generic_error(self):
        src = _get_source("_launch_warmup_impl")
        assert "_profile_lock_holder" in src


class TestWorkerBrowserError:
    def test_worker_terminal_error_is_honest(self):
        src = _get_source("_ensure_worker_browser_unlocked")
        assert "patchright install" not in src
        assert "Microsoft Edge" in src
