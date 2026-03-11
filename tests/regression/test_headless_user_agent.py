# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: headless browser must use a realistic User-Agent.

The BrowserManager._get_clean_user_agent() method reads the real browser UA,
strips 'HeadlessChrome' → 'Chrome', and passes it as user_agent override in
headless mode.  In headed mode no override is applied (user_agent=None).
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager, _STEALTH_JS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REALISTIC_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)


def _make_mock_page(ua_string: str = REALISTIC_UA):
    """Create a mock page whose evaluate('navigator.userAgent') returns *ua_string*."""
    page = MagicMock()
    page.is_closed = MagicMock(return_value=False)
    page.close = AsyncMock()
    page.goto = AsyncMock()
    page.on = MagicMock()
    page.screenshot = AsyncMock()
    page.evaluate = AsyncMock(return_value=ua_string)
    return page


def _make_mock_context(pages=None):
    ctx = AsyncMock()
    _pages = list(pages or [])
    ctx.pages = _pages
    ctx.close = AsyncMock()
    ctx.new_page = AsyncMock(return_value=_make_mock_page())

    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    ctx.browser = browser
    return ctx


def _make_mock_playwright(context):
    pw = MagicMock()
    pw.chromium.launch_persistent_context = AsyncMock(return_value=context)
    pw.chromium.launch = AsyncMock()
    pw.stop = AsyncMock()
    return pw


# ---------------------------------------------------------------------------
# Test 1: headless launch sets a realistic (non-Headless) UA
# ---------------------------------------------------------------------------

class TestHeadlessUserAgent:

    @pytest.mark.asyncio
    async def test_headless_launch_sets_realistic_ua(self, tmp_path):
        """In headless mode, launch_persistent_context must receive a user_agent
        kwarg that contains 'Chrome/' but NOT 'Headless'."""
        page = _make_mock_page(REALISTIC_UA)
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            # Mock _get_clean_user_agent so it returns a clean UA without
            # actually launching a browser.
            with patch.object(mgr, "_get_clean_user_agent", new=AsyncMock(return_value=REALISTIC_UA)):
                await mgr.ensure_browser()

        # Inspect the first successful call to launch_persistent_context
        call_kwargs = pw.chromium.launch_persistent_context.call_args
        ua_arg = call_kwargs[1].get("user_agent")

        assert ua_arg is not None, "user_agent kwarg must be set in headless mode"
        assert "Chrome/" in ua_arg, f"UA should contain 'Chrome/': {ua_arg}"
        assert "Headless" not in ua_arg, f"UA must NOT contain 'Headless': {ua_arg}"

    # ------------------------------------------------------------------
    # Test 2: headed launch uses default UA (None)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_headed_launch_uses_default_ua(self, tmp_path):
        """In headed mode, launch_persistent_context must receive user_agent=None
        so the browser uses its built-in default."""
        page = _make_mock_page(REALISTIC_UA)
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=False)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        call_kwargs = pw.chromium.launch_persistent_context.call_args
        ua_arg = call_kwargs[1].get("user_agent")

        assert ua_arg is None, (
            f"headed mode should pass user_agent=None, got: {ua_arg!r}"
        )

    # ------------------------------------------------------------------
    # Test 3: UA warning NOT emitted when the fix is active
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ua_warning_not_emitted_after_fix(self, tmp_path, caplog):
        """When _get_clean_user_agent returns a clean UA, the warning
        'User-Agent contains Headless' must NOT appear in the logs."""
        page = _make_mock_page(REALISTIC_UA)
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            with patch.object(mgr, "_get_clean_user_agent", new=AsyncMock(return_value=REALISTIC_UA)):
                with caplog.at_level(logging.WARNING, logger="agent_os.daemon_v2.browser_manager"):
                    await mgr.ensure_browser()

        headless_warnings = [
            rec for rec in caplog.records
            if "User-Agent contains" in rec.message and "Headless" in rec.message
        ]
        assert headless_warnings == [], (
            f"Expected no 'Headless' UA warning, but got: "
            f"{[r.message for r in headless_warnings]}"
        )


# ---------------------------------------------------------------------------
# Test 4: stealth JS does NOT modify userAgent (that's the UA override's job)
# ---------------------------------------------------------------------------

class TestStealthJsIntegrity:

    def test_stealth_js_not_modified(self):
        """_STEALTH_JS must NOT touch navigator.userAgent — UA spoofing is
        handled by the launch-time user_agent kwarg.  It MUST still patch
        navigator.webdriver."""
        assert "userAgent" not in _STEALTH_JS, (
            "_STEALTH_JS should not override navigator.userAgent; "
            "that is handled by the user_agent launch kwarg"
        )
        assert "webdriver" in _STEALTH_JS, (
            "_STEALTH_JS must still patch navigator.webdriver"
        )
