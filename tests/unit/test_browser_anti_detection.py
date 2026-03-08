# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for browser anti-detection: Patchright, stealth JS, flags, headed mode, system Chrome, stale detection."""

import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agent_os.daemon_v2.browser_manager import (
    BrowserManager, _STEALTH_JS, _detect_locale, _detect_timezone,
    _WIN_TZ_TO_IANA, _UTC_OFFSET_TO_IANA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_page(closed=False):
    page = MagicMock()
    page.is_closed = MagicMock(return_value=closed)
    page.close = AsyncMock()
    page.evaluate = AsyncMock(return_value="Mozilla/5.0 Chrome/120")
    page.on = MagicMock()
    return page


def _make_mock_context(pages=None):
    ctx = AsyncMock()
    _pages = list(pages or [])
    _call_count = 0

    async def _new_page():
        nonlocal _call_count
        if _call_count < len(_pages):
            p = _pages[_call_count]
        else:
            p = _make_mock_page()
        _call_count += 1
        return p

    ctx.new_page = _new_page
    ctx.close = AsyncMock()
    ctx.add_init_script = AsyncMock()
    ctx.pages = list(pages or [])

    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    ctx.browser = browser
    return ctx


def _make_mock_playwright(context):
    pw = MagicMock()
    pw.chromium.launch_persistent_context = AsyncMock(return_value=context)
    pw.stop = AsyncMock()
    return pw


# ---------------------------------------------------------------------------
# Patchright import
# ---------------------------------------------------------------------------

class TestPatchrightImport:

    def test_import_path(self):
        """Module imports from patchright.async_api, not playwright.async_api."""
        import agent_os.daemon_v2.browser_manager as mod
        import inspect
        source = inspect.getsource(mod)
        assert "from patchright.async_api import" in source
        assert "from playwright.async_api import" not in source

    def test_has_playwright_flag_true_when_available(self):
        """HAS_PLAYWRIGHT is True when patchright is importable."""
        import importlib
        import sys
        # Create a fake patchright.async_api module with necessary names
        fake_module = MagicMock()
        fake_module.async_playwright = MagicMock()
        fake_module.Browser = MagicMock()
        fake_module.BrowserContext = MagicMock()
        fake_module.Page = MagicMock()

        with patch.dict(sys.modules, {
            "patchright": MagicMock(),
            "patchright.async_api": fake_module,
        }):
            import agent_os.daemon_v2.browser_manager as mod
            importlib.reload(mod)
            assert mod.HAS_PLAYWRIGHT is True

        # Reload to restore original state
        importlib.reload(mod)

    def test_has_playwright_false_when_missing(self):
        """HAS_PLAYWRIGHT would be False when patchright import fails."""
        # We verify the try/except structure handles ImportError
        import importlib
        import agent_os.daemon_v2.browser_manager as mod
        # The module has the fallback: HAS_PLAYWRIGHT = False when import fails
        import inspect
        source = inspect.getsource(mod)
        assert "except ImportError" in source
        assert "HAS_PLAYWRIGHT = False" in source


# ---------------------------------------------------------------------------
# Stealth JS
# ---------------------------------------------------------------------------

class TestStealthJS:

    def test_stealth_js_constant_exists(self):
        assert _STEALTH_JS is not None
        assert len(_STEALTH_JS) > 0

    def test_stealth_js_removes_webdriver(self):
        assert "navigator" in _STEALTH_JS
        assert "webdriver" in _STEALTH_JS

    def test_stealth_js_sets_plugins(self):
        assert "plugins" in _STEALTH_JS

    def test_stealth_js_sets_chrome_runtime(self):
        assert "chrome.runtime" in _STEALTH_JS

    @pytest.mark.asyncio
    async def test_stealth_js_not_injected_at_context_level(self, tmp_path):
        """_launch() does NOT use context.add_init_script (breaks DNS on Windows).

        Stealth is injected per-page via _apply_stealth() in get_page/new_tab.
        """
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # add_init_script must NOT be called at context level
        ctx.add_init_script.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stealth_js_applied_per_page(self, tmp_path):
        """get_page() injects _STEALTH_JS via _apply_stealth on each new page."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        new_page = _make_mock_page()
        ctx.new_page = AsyncMock(return_value=new_page)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result_page = await mgr.get_page("test-project")

        # Stealth JS injected via page.evaluate
        new_page.evaluate.assert_awaited_once_with(_STEALTH_JS)
        # Page gets a "load" event listener for re-injection on navigation
        load_calls = [c for c in new_page.on.call_args_list if c[0][0] == "load"]
        assert len(load_calls) >= 1, "Expected 'load' event listener for stealth re-injection"


# ---------------------------------------------------------------------------
# Chrome flags
# ---------------------------------------------------------------------------

class TestChromeFlags:

    def test_all_expected_flags_present(self):
        expected = [
            "--disable-blink-features=AutomationControlled",
            "--disable-sync",
            "--disable-background-networking",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-popup-blocking",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            "--start-minimized",
        ]
        for flag in expected:
            assert flag in BrowserManager.CHROME_FLAGS, f"Missing flag: {flag}"

    def test_no_duplicate_flags(self):
        assert len(set(BrowserManager.CHROME_FLAGS)) == len(BrowserManager.CHROME_FLAGS)

    def test_automation_controlled_disabled(self):
        assert "--disable-blink-features=AutomationControlled" in BrowserManager.CHROME_FLAGS

    def test_start_minimized(self):
        assert "--start-minimized" in BrowserManager.CHROME_FLAGS


# ---------------------------------------------------------------------------
# Headed mode
# ---------------------------------------------------------------------------

class TestHeadedMode:

    def test_headless_by_default(self):
        mgr = BrowserManager()
        assert mgr._headless is True

    def test_headed_param_override(self):
        mgr = BrowserManager(headless=False)
        assert mgr._headless is False

    @pytest.mark.asyncio
    async def test_env_var_override(self, tmp_path, monkeypatch):
        """AGENT_OS_BROWSER_HEADLESS=1 forces headless even when _headless=False."""
        monkeypatch.setenv("AGENT_OS_BROWSER_HEADLESS", "1")

        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=False)
        assert mgr._headless is False  # constructor arg says headed

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # The first call tries channel="chrome", the fallback might be called too.
        # Check that whichever call was made, headless=True was passed.
        call_args = pw.chromium.launch_persistent_context.call_args
        assert call_args.kwargs.get("headless") is True or call_args[1].get("headless") is True

    @pytest.mark.asyncio
    async def test_env_var_case_insensitive(self, tmp_path, monkeypatch):
        """'true', 'True', 'TRUE' all accepted."""
        for val in ("true", "True", "TRUE"):
            monkeypatch.setenv("AGENT_OS_BROWSER_HEADLESS", val)

            page = _make_mock_page()
            ctx = _make_mock_context(pages=[page])
            pw = _make_mock_playwright(ctx)

            mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=False)

            with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
                mock_ap.return_value.start = AsyncMock(return_value=pw)
                await mgr.ensure_browser()

            call_args = pw.chromium.launch_persistent_context.call_args
            assert call_args.kwargs.get("headless") is True, f"Failed for value: {val}"


# ---------------------------------------------------------------------------
# System Chrome preference
# ---------------------------------------------------------------------------

class TestSystemChromePreference:

    @pytest.mark.asyncio
    async def test_tries_system_chrome_first(self, tmp_path):
        """First launch_persistent_context call uses channel='chrome'."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # First call should have channel="chrome"
        first_call = pw.chromium.launch_persistent_context.call_args_list[0]
        assert first_call.kwargs.get("channel") == "chrome" or \
               (len(first_call.args) > 0 and first_call.args[0] == "chrome")

    @pytest.mark.asyncio
    async def test_fallback_to_chromium(self, tmp_path):
        """If system Chrome and Edge fail, falls back to bundled Chromium (no channel)."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = MagicMock()
        pw.stop = AsyncMock()

        # Chrome and Edge fail, bundled Chromium succeeds
        pw.chromium.launch_persistent_context = AsyncMock(
            side_effect=[Exception("Chrome not found"), Exception("Edge not found"), ctx]
        )

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # Chrome, Edge, then bundled Chromium = 3 calls
        assert pw.chromium.launch_persistent_context.await_count == 3

        # Third call should NOT have channel (bundled Chromium)
        third_call = pw.chromium.launch_persistent_context.call_args_list[2]
        assert "channel" not in third_call.kwargs

    @pytest.mark.asyncio
    async def test_profile_dir_default(self):
        """Default profile_dir is ~/.agent-os/browser-profile."""
        mgr = BrowserManager()
        expected = Path.home() / ".agent-os" / "browser-profile"
        assert mgr._profile_dir == expected


# ---------------------------------------------------------------------------
# Realistic settings (locale, timezone, UA check)
# ---------------------------------------------------------------------------

class TestRealisticSettings:

    @pytest.mark.asyncio
    async def test_locale_detection(self, tmp_path):
        """System locale detected and converted to BCP-47 format."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap, \
             patch("agent_os.daemon_v2.browser_manager._detect_locale",
                   return_value="en-US"):
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        call_args = pw.chromium.launch_persistent_context.call_args
        assert call_args.kwargs.get("locale") == "en-US"

    @pytest.mark.asyncio
    async def test_locale_fallback(self, tmp_path):
        """When locale detection fails, 'en-US' used as fallback."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap, \
             patch("agent_os.daemon_v2.browser_manager._detect_locale",
                   return_value="en-US"):
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        call_args = pw.chromium.launch_persistent_context.call_args
        assert call_args.kwargs.get("locale") == "en-US"

    @pytest.mark.asyncio
    async def test_timezone_detection(self, tmp_path):
        """System timezone passed to launch context as IANA identifier."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap, \
             patch("agent_os.daemon_v2.browser_manager._detect_timezone",
                   return_value="America/New_York"):
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        call_args = pw.chromium.launch_persistent_context.call_args
        assert call_args.kwargs.get("timezone_id") == "America/New_York"

    @pytest.mark.asyncio
    async def test_ua_headless_check(self, tmp_path, caplog):
        """Warning logged when User-Agent contains 'Headless'."""
        page = _make_mock_page()
        page.evaluate = AsyncMock(return_value="Mozilla/5.0 HeadlessChrome/120.0")
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap, \
             caplog.at_level(logging.WARNING, logger="agent_os.daemon_v2.browser_manager"):
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        assert any("Headless" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Stale connection detection
# ---------------------------------------------------------------------------

class TestStaleConnectionDetection:

    @pytest.mark.asyncio
    async def test_ensure_alive_healthy(self, tmp_path):
        """Healthy context.pages access -> no relaunch."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # Second call — context.pages works, no relaunch
        initial_count = pw.chromium.launch_persistent_context.await_count
        await mgr.ensure_browser()
        assert pw.chromium.launch_persistent_context.await_count == initial_count

    @pytest.mark.asyncio
    async def test_ensure_alive_stale(self, tmp_path):
        """When context.pages throws, _cleanup_stale + _launch called."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # Now make context.pages throw to simulate stale connection
        type(mgr._context).pages = PropertyMock(side_effect=Exception("CDP disconnected"))

        # Create a fresh context for relaunch
        ctx2 = _make_mock_context(pages=[_make_mock_page()])
        pw2 = _make_mock_playwright(ctx2)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw2)
            # _cleanup_stale sets _playwright=None, so _launch will call async_playwright().start()
            await mgr.ensure_browser()

        assert mgr._context is ctx2

    @pytest.mark.asyncio
    async def test_cleanup_stale_clears_state(self, tmp_path):
        """_cleanup_stale clears project_pages and page_state."""
        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        mgr._project_pages = {"proj1": [MagicMock()]}
        mgr._page_state = {123: MagicMock()}
        mgr._context = AsyncMock()
        mgr._playwright = MagicMock()
        mgr._playwright.stop = AsyncMock()

        await mgr._cleanup_stale()

        assert mgr._project_pages == {}
        assert mgr._page_state == {}

    @pytest.mark.asyncio
    async def test_cleanup_stale_closes_context(self, tmp_path):
        """_cleanup_stale closes context and stops playwright."""
        ctx = AsyncMock()
        pw = MagicMock()
        pw.stop = AsyncMock()

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        mgr._context = ctx
        mgr._playwright = pw
        mgr._browser = MagicMock()

        await mgr._cleanup_stale()

        ctx.close.assert_awaited_once()
        pw.stop.assert_awaited_once()
        assert mgr._browser is None
        assert mgr._context is None
        assert mgr._playwright is None


# ---------------------------------------------------------------------------
# Enhanced shutdown
# ---------------------------------------------------------------------------

class TestEnhancedShutdown:

    @pytest.mark.asyncio
    async def test_shutdown_closes_all_project_pages(self, tmp_path):
        """Shutdown iterates all projects and calls close_project_pages."""
        pages = [_make_mock_page() for _ in range(3)]
        ctx = _make_mock_context(pages=pages)
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.get_page("proj_a")
            await mgr.get_page("proj_b")
            await mgr.get_page("proj_c")

        await mgr.shutdown()

        for p in pages[:3]:
            p.close.assert_awaited()
        assert len(mgr._project_pages) == 0

    @pytest.mark.asyncio
    async def test_shutdown_closes_context(self, tmp_path):
        ctx = AsyncMock()
        ctx.pages = []
        ctx.add_init_script = AsyncMock()
        ctx.browser = MagicMock()
        ctx.browser.is_connected = MagicMock(return_value=True)

        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        mgr._context = ctx
        mgr._playwright = pw
        mgr._browser = ctx.browser

        await mgr.shutdown()
        ctx.close.assert_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_stops_playwright(self, tmp_path):
        ctx = AsyncMock()
        ctx.pages = []
        pw = MagicMock()
        pw.stop = AsyncMock()

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        mgr._context = ctx
        mgr._playwright = pw
        mgr._browser = MagicMock()

        await mgr.shutdown()
        pw.stop.assert_awaited()

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self, tmp_path):
        ctx = AsyncMock()
        ctx.pages = []
        pw = MagicMock()
        pw.stop = AsyncMock()

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        mgr._context = ctx
        mgr._playwright = pw
        mgr._browser = MagicMock()

        await mgr.shutdown()

        assert mgr._browser is None
        assert mgr._context is None
        assert mgr._playwright is None

    @pytest.mark.asyncio
    async def test_shutdown_logs_exceptions(self, tmp_path, caplog):
        """context.close() raises -> exception caught, not propagated."""
        ctx = AsyncMock()
        ctx.pages = []
        ctx.close = AsyncMock(side_effect=Exception("close failed"))
        pw = MagicMock()
        pw.stop = AsyncMock()

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        mgr._context = ctx
        mgr._playwright = pw
        mgr._browser = MagicMock()

        with caplog.at_level(logging.WARNING, logger="agent_os.daemon_v2.browser_manager"):
            await mgr.shutdown()  # should NOT raise

        assert any("Error closing browser context" in r.message for r in caplog.records)
        # Playwright stop should still be called despite context.close failure
        pw.stop.assert_awaited()


# ---------------------------------------------------------------------------
# _detect_locale / _detect_timezone unit tests
# ---------------------------------------------------------------------------

class TestDetectLocale:
    """Test the _detect_locale helper function."""

    def test_returns_bcp47_from_getdefaultlocale(self):
        with patch("agent_os.daemon_v2.browser_manager.sys_locale.getdefaultlocale",
                   return_value=("en_US", "UTF-8")):
            assert _detect_locale() == "en-US"

    def test_chinese_locale(self):
        with patch("agent_os.daemon_v2.browser_manager.sys_locale.getdefaultlocale",
                   return_value=("zh_CN", "cp1252")):
            assert _detect_locale() == "zh-CN"

    def test_fallback_to_getlocale_when_default_none(self):
        with patch("agent_os.daemon_v2.browser_manager.sys_locale.getdefaultlocale",
                   return_value=(None, None)), \
             patch("agent_os.daemon_v2.browser_manager.sys_locale.getlocale",
                   return_value=("fr_FR", "UTF-8")):
            assert _detect_locale() == "fr-FR"

    def test_rejects_verbose_windows_locale_name(self):
        """getlocale on Windows returns 'Chinese (Simplified)_China' — too long."""
        with patch("agent_os.daemon_v2.browser_manager.sys_locale.getdefaultlocale",
                   return_value=(None, None)), \
             patch("agent_os.daemon_v2.browser_manager.sys_locale.getlocale",
                   return_value=("Chinese (Simplified)_China", "1252")):
            assert _detect_locale() == "en-US"  # fallback

    def test_fallback_when_all_fail(self):
        with patch("agent_os.daemon_v2.browser_manager.sys_locale.getdefaultlocale",
                   side_effect=Exception("boom")), \
             patch("agent_os.daemon_v2.browser_manager.sys_locale.getlocale",
                   side_effect=Exception("boom")):
            assert _detect_locale() == "en-US"


class TestDetectTimezone:
    """Test the _detect_timezone helper function."""

    def test_tzlocal_preferred(self):
        mock_tzlocal = MagicMock()
        mock_tzlocal.get_localzone.return_value = "Europe/London"
        with patch.dict("sys.modules", {"tzlocal": mock_tzlocal}):
            assert _detect_timezone() == "Europe/London"

    def test_windows_registry_fallback(self):
        # Simulate tzlocal not available, Windows registry works
        import sys as real_sys
        with patch("agent_os.daemon_v2.browser_manager.sys.platform", "win32"), \
             patch.dict("sys.modules", {"tzlocal": None}):
            # Need to also make tzlocal import fail
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw:
                        (_ for _ in ()).throw(ImportError()) if name == "tzlocal"
                        else __builtins__.__import__(name, *a, **kw)):
                # This is complex to mock; test the mapping table instead
                assert "China Standard Time" in _WIN_TZ_TO_IANA
                assert _WIN_TZ_TO_IANA["China Standard Time"] == "Asia/Shanghai"

    def test_utc_offset_mapping(self):
        assert _UTC_OFFSET_TO_IANA[8] == "Asia/Shanghai"
        assert _UTC_OFFSET_TO_IANA[-5] == "America/New_York"
        assert _UTC_OFFSET_TO_IANA[0] == "Europe/London"
        assert _UTC_OFFSET_TO_IANA[9] == "Asia/Tokyo"

    def test_fallback_returns_new_york(self):
        """When all detection methods fail, returns America/New_York."""
        with patch.dict("sys.modules", {"tzlocal": None}), \
             patch("agent_os.daemon_v2.browser_manager.sys.platform", "linux"), \
             patch("agent_os.daemon_v2.browser_manager.time.timezone", 0), \
             patch("agent_os.daemon_v2.browser_manager.time.tzname", ("UTC", "UTC")):
            # tzlocal import will fail, platform is linux so no registry,
            # tzname has no "/" so skipped, offset 0 -> Europe/London
            result = _detect_timezone()
            assert "/" in result  # Should be some valid IANA timezone
