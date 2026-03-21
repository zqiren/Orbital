# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: _get_clean_user_agent() must not crash when bundled chromium
is missing but system Chrome is available.

Bug: On a fresh machine, `_get_clean_user_agent()` calls
`self._playwright.chromium.launch(headless=True)` WITHOUT a channel parameter.
This requires bundled chromium (chromium_headless_shell), which hasn't been
downloaded yet.  The browser tool crashes even though system Chrome is installed
and the warmup (headed mode) works fine.

Fix: `_get_clean_user_agent()` should try system browsers (channel="chrome",
"msedge") first before falling back to bare chromium.launch(), and return None
gracefully if nothing works.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager


REALISTIC_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)


def _make_mock_context():
    ctx = AsyncMock()
    ctx.pages = []
    ctx.close = AsyncMock()
    ctx.new_page = AsyncMock()
    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    ctx.browser = browser
    return ctx


def _make_ua_browser():
    """Mock browser that returns a realistic UA string."""
    page = MagicMock()
    page.evaluate = AsyncMock(return_value=REALISTIC_UA)
    browser = AsyncMock()
    browser.new_page = AsyncMock(return_value=page)
    browser.close = AsyncMock()
    return browser


class TestUaProbeNoBundledChromium:
    """_get_clean_user_agent() must work when bundled chromium is missing."""

    @pytest.mark.asyncio
    async def test_ua_probe_falls_back_to_system_chrome(self, tmp_path):
        """When bare chromium.launch() fails (no bundled chromium),
        _get_clean_user_agent() should try channel='chrome' and succeed."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        pw.stop = AsyncMock()

        # Simulate: bare launch() fails, but launch(channel="chrome") works
        ua_browser = _make_ua_browser()

        def _launch_side_effect(**kwargs):
            if "channel" not in kwargs:
                raise FileNotFoundError(
                    "Executable doesn't exist at "
                    "C:\\Users\\Lyne\\AppData\\Roaming\\Orbital\\browsers\\"
                    "chromium_headless_shell-1208\\..."
                )
            return ua_browser

        pw.chromium.launch = AsyncMock(side_effect=_launch_side_effect)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # Should have succeeded — verify UA was obtained
        assert hasattr(mgr, "_cached_clean_ua")
        assert "Chrome/" in mgr._cached_clean_ua
        assert "HeadlessChrome" not in mgr._cached_clean_ua

    @pytest.mark.asyncio
    async def test_ua_probe_returns_none_when_all_browsers_fail(self, tmp_path):
        """When ALL browser launches fail for UA probe, _get_clean_user_agent()
        should return None instead of crashing."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        pw.stop = AsyncMock()

        # ALL launch attempts fail
        pw.chromium.launch = AsyncMock(
            side_effect=FileNotFoundError("No browser available")
        )

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            # Should not crash — should gracefully degrade
            await mgr.ensure_browser()

        # launch_persistent_context should still have been called (main launch)
        pw.chromium.launch_persistent_context.assert_awaited()

    @pytest.mark.asyncio
    async def test_ua_probe_tries_channels_before_bare_launch(self, tmp_path):
        """_get_clean_user_agent() should try system browsers (with channel)
        before attempting a bare chromium.launch()."""
        pw = MagicMock()
        ctx = _make_mock_context()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        pw.stop = AsyncMock()

        ua_browser = _make_ua_browser()
        launch_calls = []

        async def _record_launch(**kwargs):
            launch_calls.append(kwargs)
            # First call with channel succeeds
            if kwargs.get("channel") == "chrome":
                return ua_browser
            raise FileNotFoundError("not available")

        pw.chromium.launch = AsyncMock(side_effect=_record_launch)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # Verify a channel-based launch was attempted
        channel_calls = [c for c in launch_calls if "channel" in c]
        assert len(channel_calls) > 0, (
            "Expected at least one launch attempt with a channel parameter, "
            f"but got: {launch_calls}"
        )
        # And the first attempt should be channel="chrome"
        assert channel_calls[0]["channel"] == "chrome"
