# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for browser resilience: headless default, action timeout, dialog handler.

Task 1 of the browser resilience spec. Tests are written TDD-first — they fail
until the corresponding production code changes are made.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager, _PageState
from agent_os.agent.tools.browser import BrowserTool
from agent_os.agent.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_page():
    page = MagicMock()
    page.is_closed = MagicMock(return_value=False)
    page.close = AsyncMock()
    page.screenshot = AsyncMock()
    page.goto = AsyncMock()
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
    ctx.pages = []

    browser = MagicMock()
    browser.is_connected = MagicMock(return_value=True)
    ctx.browser = browser
    return ctx


def _make_mock_playwright(context):
    pw = MagicMock()
    pw.chromium.launch_persistent_context = AsyncMock(return_value=context)
    pw.stop = AsyncMock()
    return pw


def _make_browser_manager(page=None):
    bm = MagicMock()
    if page is None:
        page = AsyncMock()
        page.url = "https://example.com"
        page.title = AsyncMock(return_value="Example")
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(return_value="/tmp/screenshot.png")
    bm.get_ref_map = MagicMock(return_value=None)
    bm.clear_ref_map = MagicMock()
    bm.store_ref_map = MagicMock()
    return bm


def _make_tool(bm=None):
    if bm is None:
        bm = _make_browser_manager()
    return BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace="/workspace",
        autonomy_preset="hands_off",
        session_id="default",
    )


# ---------------------------------------------------------------------------
# Section 1: Headless default tests
# ---------------------------------------------------------------------------

class TestHeadlessDefault:

    def test_headless_by_default(self):
        """BrowserManager() with no args has _headless == True."""
        mgr = BrowserManager()
        assert mgr._headless is True

    @pytest.mark.asyncio
    async def test_headed_env_override(self, tmp_path):
        """With AGENT_OS_BROWSER_HEADED=1, _launch() uses headless=False."""
        ctx = _make_mock_context()
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"))
        # Default is headless=True now

        env = {
            "AGENT_OS_BROWSER_HEADED": "1",
            "AGENT_OS_BROWSER_HEADLESS": "",
        }
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap, \
             patch.dict(os.environ, env, clear=False):
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        # The launch call should have headless=False
        call_kwargs = pw.chromium.launch_persistent_context.call_args
        assert call_kwargs[1]["headless"] is False

    @pytest.mark.asyncio
    async def test_headless_env_override(self, tmp_path):
        """With AGENT_OS_BROWSER_HEADLESS=1, headless stays True (existing behavior)."""
        ctx = _make_mock_context()
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"))

        env = {
            "AGENT_OS_BROWSER_HEADLESS": "1",
            "AGENT_OS_BROWSER_HEADED": "",
        }
        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap, \
             patch.dict(os.environ, env, clear=False):
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.ensure_browser()

        call_kwargs = pw.chromium.launch_persistent_context.call_args
        assert call_kwargs[1]["headless"] is True


# ---------------------------------------------------------------------------
# Section 2: Universal operation timeout tests
# ---------------------------------------------------------------------------

class TestDispatchTimeout:

    @pytest.mark.asyncio
    async def test_dispatch_timeout_returns_error_result(self):
        """A handler that hangs indefinitely is interrupted by the timeout
        and returns a ToolResult with timeout error info."""
        import agent_os.agent.tools.browser as browser_mod

        bm = _make_browser_manager()
        tool = _make_tool(bm=bm)

        # Monkeypatch the timeout to something tiny for test speed
        original_timeout = browser_mod.BROWSER_ACTION_TIMEOUT
        browser_mod.BROWSER_ACTION_TIMEOUT = 0.1

        # Add a handler that sleeps forever
        async def _action_hang(args):
            await asyncio.sleep(9999)

        tool._action_hang = _action_hang

        try:
            result = await tool._dispatch("hang", {})
        finally:
            browser_mod.BROWSER_ACTION_TIMEOUT = original_timeout

        assert "timed out" in result.content.lower()
        assert "hang" in result.content.lower()
        assert result.meta is not None
        assert result.meta.get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_dispatch_normal_action_succeeds(self):
        """A fast handler returns normally through _dispatch."""
        bm = _make_browser_manager()
        tool = _make_tool(bm=bm)

        # done action is always fast
        result = await tool._dispatch("done", {"text": "All good"})
        assert result.content == "All good"


# ---------------------------------------------------------------------------
# Section 3: Dialog auto-handler tests
# ---------------------------------------------------------------------------

class TestDialogHandler:

    def test_page_state_has_last_dialog_field(self):
        """_PageState() has last_dialog attribute, initially None."""
        state = _PageState()
        assert hasattr(state, "last_dialog")
        assert state.last_dialog is None

    @pytest.mark.asyncio
    async def test_dialog_handler_registered(self, tmp_path):
        """After _setup_page_handlers(), a 'dialog' handler is registered on the page mock."""
        page = _make_mock_page()
        ctx = _make_mock_context(pages=[page])
        pw = _make_mock_playwright(ctx)

        mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)

        with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=pw)
            await mgr.get_page("proj_dialog")

        # Check that page.on was called with "dialog"
        on_calls = [call for call in page.on.call_args_list if call[0][0] == "dialog"]
        assert len(on_calls) == 1, (
            f"Expected exactly one 'dialog' handler registered, "
            f"got {len(on_calls)}. All page.on calls: "
            f"{[c[0][0] for c in page.on.call_args_list]}"
        )
