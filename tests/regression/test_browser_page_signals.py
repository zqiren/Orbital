# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for page signal collection after navigation/snapshot/screenshot.

Task 2 of the browser resilience spec. After every navigation or snapshot action,
BrowserTool collects objective page signals and appends them to
ToolResult.meta["page_signals"].
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.tools.browser import BrowserTool
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tools.browser_refs import RefEntry


# ---------------------------------------------------------------------------
# Helpers (mirror patterns from test_browser_tool.py)
# ---------------------------------------------------------------------------

def _make_mock_page(url="https://example.com", title="Example"):
    page = AsyncMock()
    page.url = url
    page.title = AsyncMock(return_value=title)
    page.is_closed = MagicMock(return_value=False)
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value=None)
    page.wait_for_load_state = AsyncMock()
    page.screenshot = AsyncMock()
    page.bring_to_front = AsyncMock()
    page.close = AsyncMock()
    page.inner_text = AsyncMock(return_value="Hello world")

    # CDP session mock for _get_ax_tree
    cdp_session = AsyncMock()
    cdp_session.send = AsyncMock(return_value={"nodes": []})
    cdp_session.detach = AsyncMock()
    context = MagicMock()
    context.new_cdp_session = AsyncMock(return_value=cdp_session)
    page.context = context

    return page


def _make_browser_manager(page=None):
    bm = MagicMock()
    if page is None:
        page = _make_mock_page()
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(
        return_value="/workspace/orbital/output/screenshots/step_0001.png"
    )
    bm.store_ref_map = MagicMock()
    bm.get_ref_map = MagicMock(return_value=None)
    bm.clear_ref_map = MagicMock()
    return bm


def _make_tool(bm=None, page=None):
    if bm is None:
        bm = _make_browser_manager(page)
    return BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace="/workspace",
        autonomy_preset="hands_off",
        session_id="default",
    )


def _setup_evaluate_side_effects(page, *, has_password=False, captcha=None,
                                  visible_text="Hello world", input_count=2,
                                  form_count=1):
    """Configure page.evaluate to return appropriate values for each JS snippet.

    _collect_page_signals calls page.evaluate multiple times with different JS.
    We use side_effect to return the right value based on call order:
      1. has_password_field query
      2. captcha iframe query
      3. visible_text_snippet query
      4. input_count query
      5. form_count query
    """
    side_effects = [
        has_password,
        captcha,
        visible_text,
        input_count,
        form_count,
    ]
    page.evaluate = AsyncMock(side_effect=side_effects)


# ---------------------------------------------------------------------------
# 1. Signal dict has correct keys
# ---------------------------------------------------------------------------

class TestCollectPageSignals:

    @pytest.mark.asyncio
    async def test_signals_dict_has_correct_keys(self):
        """_collect_page_signals returns dict with all 8 expected keys."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        expected_keys = {
            "has_password_field",
            "has_captcha_iframe",
            "visible_text_snippet",
            "input_count",
            "form_count",
            "was_redirected",
            "redirected_from",
            "http_status",
        }
        assert set(signals.keys()) == expected_keys

    # -----------------------------------------------------------------------
    # 2. Password field detection
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_password_field_detected(self):
        """Mock page with password input returns has_password_field == True."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page, has_password=True)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert signals["has_password_field"] is True

    @pytest.mark.asyncio
    async def test_no_password_field(self):
        """Page without password input returns has_password_field == False."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page, has_password=False)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert signals["has_password_field"] is False

    # -----------------------------------------------------------------------
    # 3. Captcha detection
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_captcha_detected(self):
        """Page with reCAPTCHA iframe returns has_captcha_iframe == 'recaptcha'."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page, captcha="recaptcha")
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert signals["has_captcha_iframe"] == "recaptcha"

    @pytest.mark.asyncio
    async def test_no_captcha(self):
        """Page without captcha returns has_captcha_iframe == None."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page, captcha=None)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert signals["has_captcha_iframe"] is None

    # -----------------------------------------------------------------------
    # 4. Visible text snippet truncation
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_visible_text_snippet_truncated(self):
        """visible_text_snippet is the first 200 chars (truncation done in JS)."""
        long_text = "x" * 500
        page = _make_mock_page()
        # The JS in production returns substring(0, 200), so we simulate what
        # the JS would return: first 200 chars
        _setup_evaluate_side_effects(page, visible_text=long_text[:200])
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert len(signals["visible_text_snippet"]) == 200

    @pytest.mark.asyncio
    async def test_visible_text_snippet_short(self):
        """Short visible text is returned as-is."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page, visible_text="Hello world")
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert signals["visible_text_snippet"] == "Hello world"

    # -----------------------------------------------------------------------
    # 5. Redirect detection
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_redirect_detected(self):
        """requested_url differs from page.url -> was_redirected == True."""
        page = _make_mock_page(url="https://login.example.com/sso")
        _setup_evaluate_side_effects(page)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(
            page, requested_url="https://example.com/dashboard"
        )

        assert signals["was_redirected"] is True
        assert signals["redirected_from"] == "https://example.com/dashboard"

    # -----------------------------------------------------------------------
    # 6. No redirect when URL matches
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_no_redirect_same_url(self):
        """requested_url matches page.url -> was_redirected == False."""
        page = _make_mock_page(url="https://example.com/page")
        _setup_evaluate_side_effects(page)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(
            page, requested_url="https://example.com/page"
        )

        assert signals["was_redirected"] is False
        assert signals["redirected_from"] is None

    @pytest.mark.asyncio
    async def test_no_redirect_when_no_requested_url(self):
        """When no requested_url is provided, was_redirected == False."""
        page = _make_mock_page()
        _setup_evaluate_side_effects(page)
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        assert signals["was_redirected"] is False
        assert signals["redirected_from"] is None

    # -----------------------------------------------------------------------
    # Resilience: partial signals on exception
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_partial_signals_on_exception(self):
        """If page.evaluate throws, returns partial signals dict (no crash)."""
        page = _make_mock_page()
        page.evaluate = AsyncMock(side_effect=Exception("Page crashed"))
        tool = _make_tool(page=page)

        signals = await tool._collect_page_signals(page)

        # Should return a dict (possibly partial) without raising
        assert isinstance(signals, dict)


# ---------------------------------------------------------------------------
# 7. Signals in navigate result
# ---------------------------------------------------------------------------

class TestSignalsInActionResults:

    @pytest.mark.asyncio
    async def test_signals_in_navigate_result(self):
        """After navigate, result.meta contains page_signals dict."""
        page = _make_mock_page(url="https://example.com", title="Example Domain")
        bm = _make_browser_manager(page)
        tool = _make_tool(bm=bm)

        _setup_evaluate_side_effects(page, has_password=True, captcha=None,
                                      visible_text="Welcome to Example",
                                      input_count=3, form_count=1)

        with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
             patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
            result = await tool.execute_async(action="navigate", url="https://example.com")

        assert "Navigated to" in result.content
        assert result.meta is not None
        assert "page_signals" in result.meta
        signals = result.meta["page_signals"]
        assert isinstance(signals, dict)
        assert "has_password_field" in signals
        assert "visible_text_snippet" in signals

    # -----------------------------------------------------------------------
    # 8. Signals in snapshot result
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_signals_in_snapshot_result(self):
        """After snapshot, result.meta contains page_signals dict."""
        page = _make_mock_page()
        bm = _make_browser_manager(page)
        tool = _make_tool(bm=bm)

        _setup_evaluate_side_effects(page, has_password=False, captcha="hcaptcha",
                                      visible_text="Please verify",
                                      input_count=0, form_count=0)

        mock_tree = {
            "role": "WebArea",
            "name": "Example",
            "children": [{"role": "heading", "name": "Test", "level": 1}],
        }
        with patch("agent_os.agent.tools.browser._get_ax_tree",
                    new_callable=AsyncMock, return_value=mock_tree):
            result = await tool.execute_async(action="snapshot")

        assert result.meta is not None
        assert "page_signals" in result.meta
        signals = result.meta["page_signals"]
        assert isinstance(signals, dict)
        assert signals["has_captcha_iframe"] == "hcaptcha"

    # -----------------------------------------------------------------------
    # 9. Signals in screenshot result
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_signals_in_screenshot_result(self):
        """After screenshot, result.meta contains page_signals dict."""
        page = _make_mock_page()
        bm = _make_browser_manager(page)
        tool = _make_tool(bm=bm)

        _setup_evaluate_side_effects(page, has_password=False, captcha=None,
                                      visible_text="Dashboard content",
                                      input_count=1, form_count=0)

        result = await tool.execute_async(action="screenshot")

        assert result.meta is not None
        assert "page_signals" in result.meta
        signals = result.meta["page_signals"]
        assert isinstance(signals, dict)
        assert "visible_text_snippet" in signals
