# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for BrowserManager — all Playwright interactions mocked."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager, _PageState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_page(closed=False):
    """Create a mock Playwright Page with sync is_closed()."""
    page = MagicMock()
    page.is_closed = MagicMock(return_value=closed)
    page.close = AsyncMock()
    page.screenshot = AsyncMock()
    page.goto = AsyncMock()
    page.on = MagicMock()
    return page


def _make_mock_context(pages=None):
    """Create a mock BrowserContext that produces pages on new_page()."""
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
    ctx.pages = list(_pages)

    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    ctx.browser = browser
    return ctx


def _make_mock_playwright(context):
    """Create a mock Playwright instance."""
    pw = MagicMock()
    pw.chromium.launch_persistent_context = AsyncMock(return_value=context)
    pw.stop = AsyncMock()
    # Mock chromium.launch() for _get_clean_user_agent() UA probe
    ua_page = MagicMock()
    ua_page.evaluate = AsyncMock(return_value="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36")
    ua_browser = AsyncMock()
    ua_browser.new_page = AsyncMock(return_value=ua_page)
    ua_browser.close = AsyncMock()
    pw.chromium.launch = AsyncMock(return_value=ua_browser)
    return pw


def _add_ua_probe_mock(pw):
    """Add chromium.launch mock for _get_clean_user_agent() UA probe."""
    ua_page = MagicMock()
    ua_page.evaluate = AsyncMock(return_value="Mozilla/5.0 Chrome/145.0.0.0 Safari/537.36")
    ua_browser = AsyncMock()
    ua_browser.new_page = AsyncMock(return_value=ua_page)
    ua_browser.close = AsyncMock()
    pw.chromium.launch = AsyncMock(return_value=ua_browser)


async def _setup_mgr(tmp_path, pw, ctx=None):
    """Bootstrap a BrowserManager with mocked Playwright already wired in."""
    mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        await mgr.ensure_browser()
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBrowserManager:

    @pytest.mark.asyncio
    async def test_lazy_launch(self, tmp_path):
        """BrowserManager created -> no browser launched. ensure_browser() -> launch called."""
        ctx = _make_mock_context()
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        assert mgr._context is None
        assert mgr._browser is None

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result = await mgr.ensure_browser()

        assert result is ctx
        pw.chromium.launch_persistent_context.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reuse_existing(self, tmp_path):
        """ensure_browser() twice -> launch called once."""
        ctx = _make_mock_context()
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()
            await mgr.ensure_browser()

        assert pw.chromium.launch_persistent_context.await_count == 1

    @pytest.mark.asyncio
    async def test_get_page_creates_new(self, tmp_path):
        """get_page('proj1') -> new page created."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result = await mgr.get_page("proj1")

        assert result is page
        assert "proj1" in mgr._project_pages
        assert len(mgr._project_pages["proj1"]) == 1

    @pytest.mark.asyncio
    async def test_get_page_reuses_existing(self, tmp_path):
        """get_page('proj1') twice -> same page returned."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            p1 = await mgr.get_page("proj1")
            p2 = await mgr.get_page("proj1")

        assert p1 is p2
        assert len(mgr._project_pages["proj1"]) == 1

    @pytest.mark.asyncio
    async def test_page_cap_eviction(self, tmp_path):
        """Create 10 pages -> 11th triggers eviction of oldest."""
        pages = [_make_mock_page() for _ in range(11)]
        ctx = _make_mock_context(pages=pages)
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            for i in range(10):
                await mgr.get_page(f"proj{i}")

            total_before = sum(len(ps) for ps in mgr._project_pages.values())
            assert total_before == 10

            # 11th page triggers eviction
            await mgr.get_page("proj_extra")

        total_after = sum(len(ps) for ps in mgr._project_pages.values())
        assert total_after == 10
        pages[0].close.assert_awaited()

    @pytest.mark.asyncio
    async def test_close_project_pages(self, tmp_path):
        """Close pages for one project -> others unaffected."""
        pages = [_make_mock_page() for _ in range(3)]
        ctx = _make_mock_context(pages=pages)
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.get_page("proj_a")
            await mgr.get_page("proj_b")
            await mgr.get_page("proj_c")

        await mgr.close_project_pages("proj_a")

        assert "proj_a" not in mgr._project_pages
        assert "proj_b" in mgr._project_pages
        assert "proj_c" in mgr._project_pages
        pages[0].close.assert_awaited()

    @pytest.mark.asyncio
    async def test_crash_recovery_relaunches(self, tmp_path):
        """Set browser.is_connected() to False -> ensure_browser re-launches."""
        ctx1 = _make_mock_context()
        ctx2 = _make_mock_context()
        pw = _make_mock_playwright(ctx1)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        assert mgr._context is ctx1

        # Simulate crash
        mgr._browser.is_connected.return_value = False

        # _handle_crash calls _launch which reuses self._playwright (already set)
        # so we update the launch return value
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx2)

        result = await mgr.ensure_browser()
        assert result is ctx2

    @pytest.mark.asyncio
    async def test_crash_rate_limit(self, tmp_path):
        """3 crashes in < 5min -> RuntimeError raised."""
        ctx = _make_mock_context()
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # Pre-fill restart timestamps to simulate 2 prior crashes within window
        now = time.monotonic()
        mgr._restart_timestamps.append(now - 60)
        mgr._restart_timestamps.append(now - 30)

        # Simulate crash
        mgr._browser.is_connected.return_value = False

        with pytest.raises(RuntimeError, match="Browser keeps crashing"):
            await mgr.ensure_browser()

    def test_ref_map_store_get(self):
        """Store RefMap -> get returns it."""
        mgr = BrowserManager(headless=True)
        page_id = 12345
        mgr._page_state[page_id] = _PageState()

        ref_map = {"e1": {"role": "link", "name": "Home"}}
        mgr.store_ref_map("proj1", page_id, ref_map)

        result = mgr.get_ref_map("proj1", page_id)
        assert result == ref_map

    def test_ref_map_clear(self):
        """Clear -> get returns None."""
        mgr = BrowserManager(headless=True)
        page_id = 12345
        mgr._page_state[page_id] = _PageState()

        mgr.store_ref_map("proj1", page_id, {"e1": {}})
        mgr.clear_ref_map("proj1", page_id)

        assert mgr.get_ref_map("proj1", page_id) is None

    @pytest.mark.asyncio
    async def test_screenshot_retention(self, tmp_path):
        """51 captures -> oldest deleted, max 50 remain."""
        from agent_os.agent.project_paths import ProjectPaths
        workspace = str(tmp_path / "workspace")
        session_id = "sess_001"
        screenshot_dir = Path(ProjectPaths(workspace).screenshots_dir) / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create 49 screenshot files (step_0001 through step_0049)
        for i in range(1, 50):
            (screenshot_dir / f"step_{i:04d}.png").write_bytes(b"fake")

        # Mock page whose screenshot writes the file to disk
        page = _make_mock_page()

        async def _write_screenshot(**kwargs):
            Path(kwargs["path"]).write_bytes(b"fake")

        page.screenshot = _write_screenshot

        mgr = BrowserManager(headless=True)
        mgr._screenshot_counters[session_id] = 49

        # 50th capture — 49 existing + 1 new = 50, no eviction
        await mgr.capture_screenshot(page, workspace, session_id)
        existing = sorted(screenshot_dir.glob("step_*.png"))
        assert len(existing) == 50

        # 51st capture — 50 existing >= 50, evicts oldest, then writes = 50
        await mgr.capture_screenshot(page, workspace, session_id)
        existing = sorted(screenshot_dir.glob("step_*.png"))
        assert len(existing) == 50
        assert not (screenshot_dir / "step_0001.png").exists()

    def test_page_state_caps(self):
        """Add 600 console messages -> only 500 retained."""
        state = _PageState()
        for i in range(600):
            state.add_console(f"msg_{i}")

        assert len(state.console_log) == 500
        assert state.console_log[0] == "msg_100"
        assert state.console_log[-1] == "msg_599"

        # Errors capped at 200
        for i in range(300):
            state.add_error(f"err_{i}")
        assert len(state.errors) == 200
        assert state.errors[0] == "err_100"
        assert state.errors[-1] == "err_299"

    @pytest.mark.asyncio
    async def test_shutdown_closes_all(self, tmp_path):
        """Shutdown -> all pages and context closed."""
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
        ctx.close.assert_awaited()
        pw.stop.assert_awaited()
        assert mgr._browser is None
        assert mgr._context is None
        assert mgr._playwright is None

    # ------------------------------------------------------------------
    # Filechooser interceptor tests
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_filechooser_handler_registered_on_get_page(self, tmp_path):
        """When get_page creates a new page, page.on is called with 'filechooser'."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.get_page("proj_fc")

        # page.on should have been called with "filechooser" among other handlers
        on_calls = [call for call in page.on.call_args_list if call[0][0] == "filechooser"]
        assert len(on_calls) == 1, "Expected exactly one 'filechooser' handler registered"

    @pytest.mark.asyncio
    async def test_filechooser_handler_registered_on_new_tab(self, tmp_path):
        """When new_tab creates a page, page.on is called with 'filechooser'."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result_page = await mgr.new_tab("proj_nt")

        # The page returned by new_tab should have filechooser handler registered
        on_calls = [call for call in result_page.on.call_args_list if call[0][0] == "filechooser"]
        assert len(on_calls) == 1, "Expected 'filechooser' handler on new_tab page"

    def test_consume_file_chooser_returns_and_removes(self):
        """Store a chooser, consume returns it, second consume returns None."""
        mgr = BrowserManager(headless=True)
        mock_chooser = MagicMock()
        mgr._pending_file_choosers["proj1"] = mock_chooser

        result = mgr.consume_file_chooser("proj1")
        assert result is mock_chooser

        # Second consume should return None (was removed)
        result2 = mgr.consume_file_chooser("proj1")
        assert result2 is None

    def test_consume_file_chooser_empty(self):
        """Consume with no pending chooser returns None."""
        mgr = BrowserManager(headless=True)
        result = mgr.consume_file_chooser("nonexistent_project")
        assert result is None


# ---------------------------------------------------------------------------
# Browser channel fallback chain tests
# ---------------------------------------------------------------------------

class TestBrowserChannelFallback:
    """Tests for Chrome → Edge → bundled Chromium fallback chain."""

    @pytest.mark.asyncio
    async def test_chrome_succeeds_no_fallback(self, tmp_path):
        """When system Chrome works, Edge and bundled Chromium are not tried."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        pw.stop = AsyncMock()
        _add_ua_probe_mock(pw)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result = await mgr.ensure_browser()

        assert result is ctx
        # launch_persistent_context called once with channel="chrome"
        assert pw.chromium.launch_persistent_context.await_count == 1
        call_kwargs = pw.chromium.launch_persistent_context.call_args[1]
        assert call_kwargs.get("channel") == "chrome"

    @pytest.mark.asyncio
    async def test_chrome_fails_edge_succeeds(self, tmp_path):
        """When Chrome is missing, falls back to Edge."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.stop = AsyncMock()
        _add_ua_probe_mock(pw)

        call_count = 0

        async def _launch_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            channel = kwargs.get("channel")
            if channel == "chrome":
                raise FileNotFoundError("chrome not found")
            if channel == "msedge":
                return ctx
            raise FileNotFoundError("no bundled chromium")

        pw.chromium.launch_persistent_context = AsyncMock(side_effect=_launch_side_effect)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result = await mgr.ensure_browser()

        assert result is ctx
        assert call_count == 2  # Chrome tried, then Edge

    @pytest.mark.asyncio
    async def test_chrome_and_edge_fail_bundled_succeeds(self, tmp_path):
        """When Chrome and Edge both missing, falls back to bundled Chromium."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.stop = AsyncMock()
        _add_ua_probe_mock(pw)

        channels_tried = []

        async def _launch_side_effect(*args, **kwargs):
            channel = kwargs.get("channel")
            channels_tried.append(channel)
            if channel == "chrome":
                raise FileNotFoundError("chrome not found")
            if channel == "msedge":
                raise FileNotFoundError("edge not found")
            # No channel = bundled Chromium
            return ctx

        pw.chromium.launch_persistent_context = AsyncMock(side_effect=_launch_side_effect)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            result = await mgr.ensure_browser()

        assert result is ctx
        assert channels_tried == ["chrome", "msedge", None]

    @pytest.mark.asyncio
    async def test_all_browsers_fail_clear_error(self, tmp_path):
        """When all browser channels fail, raises RuntimeError with actionable message."""
        pw = MagicMock()
        pw.stop = AsyncMock()
        _add_ua_probe_mock(pw)
        pw.chromium.launch_persistent_context = AsyncMock(
            side_effect=FileNotFoundError("not found")
        )

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            with pytest.raises(RuntimeError, match="No browser.*available"):
                await mgr.ensure_browser()

    @pytest.mark.asyncio
    async def test_playwright_server_missing_clear_error(self, tmp_path):
        """When patchright driver (node.exe) is missing, raises RuntimeError not raw WinError."""
        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(
                side_effect=FileNotFoundError("[WinError 2] The system cannot find the file specified")
            )
            with pytest.raises(RuntimeError, match="[Pp]atchright.*driver"):
                await mgr.ensure_browser()


# ---------------------------------------------------------------------------
# Browser warmup tests
# ---------------------------------------------------------------------------

class TestBrowserWarmup:
    """Tests for browser warmup (headed session for cookie warmup)."""

    @pytest.mark.asyncio
    async def test_warmup_launches_headed_with_same_profile(self, tmp_path):
        """Warmup launches a HEADED browser using the same profile directory."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        pw.stop = AsyncMock()

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.launch_warmup("https://accounts.google.com")

        # Verify launched with headless=False
        call_kwargs = pw.chromium.launch_persistent_context.call_args[1]
        assert call_kwargs["headless"] is False
        # Verify same profile dir
        assert call_kwargs["user_data_dir"] == str(tmp_path / "profile")
        # Verify playwright was stopped after warmup
        pw.stop.assert_awaited()

    @pytest.mark.asyncio
    async def test_warmup_navigates_to_url(self, tmp_path):
        """Warmup navigates to the provided URL."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = MagicMock()
        pw.chromium.launch_persistent_context = AsyncMock(return_value=ctx)
        pw.stop = AsyncMock()

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.launch_warmup("https://accounts.google.com")

        page.goto.assert_awaited_once_with("https://accounts.google.com")

    @pytest.mark.asyncio
    async def test_warmup_does_not_interfere_with_daemon_browser(self, tmp_path):
        """Warmup uses a separate playwright instance — doesn't touch self._playwright."""
        ctx_daemon = _make_mock_context()
        pw_daemon = _make_mock_playwright(ctx_daemon)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        # First, set up the daemon's own browser
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw_daemon)
            await mgr.ensure_browser()

        # Now run warmup — should NOT affect mgr._playwright or mgr._context
        ctx_warmup = _make_mock_context()
        pw_warmup = MagicMock()
        pw_warmup.chromium.launch_persistent_context = AsyncMock(return_value=ctx_warmup)
        pw_warmup.stop = AsyncMock()

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap2:
            mock_ap2.return_value.start = AsyncMock(return_value=pw_warmup)
            await mgr.launch_warmup("https://accounts.google.com")

        # Daemon browser should be untouched
        assert mgr._playwright is pw_daemon
        assert mgr._context is ctx_daemon

    @pytest.mark.asyncio
    async def test_warmup_uses_channel_fallback(self, tmp_path):
        """Warmup uses the same Chrome -> Edge -> Chromium fallback."""
        ctx = _make_mock_context()
        pw = MagicMock()
        pw.stop = AsyncMock()

        channels_tried = []

        async def _launch_side_effect(*args, **kwargs):
            channel = kwargs.get("channel")
            channels_tried.append(channel)
            if channel == "chrome":
                raise FileNotFoundError("no chrome")
            if channel == "msedge":
                return ctx
            raise FileNotFoundError("no chromium")

        pw.chromium.launch_persistent_context = AsyncMock(side_effect=_launch_side_effect)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.launch_warmup("https://example.com")

        assert channels_tried == ["chrome", "msedge"]
