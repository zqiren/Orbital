# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for MFA relay: blocker detection, resolution polling, and approve endpoint.

Tests cover:
- Blocker detection in navigate (password field, captcha, normal page)
- Resolution polling (URL change, timeout)
- ApproveRequest model extension with response_payload
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.tools.browser import BrowserTool
from agent_os.daemon_v2.browser_resolution import poll_for_resolution


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_mock_page(url="https://example.com", title="Example"):
    """Create a mock Playwright Page object."""
    page = AsyncMock()
    page.url = url
    page.title = AsyncMock(return_value=title)
    page.is_closed = MagicMock(return_value=False)
    page.goto = AsyncMock()
    page.evaluate = AsyncMock(return_value=None)
    page.wait_for_load_state = AsyncMock()
    page.screenshot = AsyncMock()
    page.keyboard = AsyncMock()
    page.mouse = AsyncMock()
    page.accessibility = MagicMock()
    page.accessibility.snapshot = AsyncMock(return_value=None)
    return page


def _make_browser_manager(page=None):
    """Create a mock BrowserManager."""
    bm = MagicMock()
    if page is None:
        page = _make_mock_page()
    bm.get_page = AsyncMock(return_value=page)
    bm.capture_screenshot = AsyncMock(return_value="/workspace/orbital-output/default/screenshots/step_0001.png")
    bm.clear_ref_map = MagicMock()
    bm.get_ref_map = MagicMock(return_value=None)
    bm.store_ref_map = MagicMock()
    return bm


def _make_tool(bm=None, page=None):
    """Create a BrowserTool with mocked dependencies."""
    if bm is None:
        bm = _make_browser_manager(page)
    return BrowserTool(
        browser_manager=bm,
        project_id="test-project",
        workspace="/workspace",
        autonomy_preset="hands_off",
        session_id="default",
    )


def _configure_page_signals(page, has_password=False, captcha=None, text_snippet=""):
    """Configure page.evaluate to return specific signal values.

    The evaluate calls in _collect_page_signals follow a specific order:
    1. has_password_field
    2. captcha detection (JS function)
    3. visible_text_snippet
    4. input_count
    5. form_count
    """
    call_count = 0

    async def _eval_side_effect(script, *args):
        nonlocal call_count
        call_count += 1
        if call_count == 1:  # has_password_field
            return has_password
        elif call_count == 2:  # captcha detection
            return captcha
        elif call_count == 3:  # visible_text_snippet
            return text_snippet
        elif call_count == 4:  # input_count
            return 1 if has_password else 0
        elif call_count == 5:  # form_count
            return 1 if has_password else 0
        return None

    page.evaluate = AsyncMock(side_effect=_eval_side_effect)


# ---------------------------------------------------------------------------
# 1. Blocker detected with password field
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocker_detected_with_password_field():
    """Navigate to a page with a password field sets blocker_detected in meta."""
    page = _make_mock_page(url="https://accounts.google.com/login", title="Sign in")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)
    _configure_page_signals(page, has_password=True)

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(action="navigate", url="https://accounts.google.com/login")

    assert result.meta.get("blocker_detected") is True
    assert "authentication" in result.content.lower() or "verification" in result.content.lower()


# ---------------------------------------------------------------------------
# 2. Blocker detected with captcha
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blocker_detected_with_captcha():
    """Navigate to a page with a captcha iframe sets blocker_detected in meta."""
    page = _make_mock_page(url="https://example.com/verify", title="Verify")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)
    _configure_page_signals(page, captcha="recaptcha")

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(action="navigate", url="https://example.com/verify")

    assert result.meta.get("blocker_detected") is True


# ---------------------------------------------------------------------------
# 3. No blocker on normal page
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_blocker_normal_page():
    """Navigate to a normal page does NOT set blocker_detected."""
    page = _make_mock_page(url="https://example.com", title="Example")
    bm = _make_browser_manager(page)
    tool = _make_tool(bm=bm)
    _configure_page_signals(page, has_password=False, captcha=None)

    with patch("agent_os.agent.tools.browser.validate_url_pre_navigation", return_value=None), \
         patch("agent_os.agent.tools.browser.validate_url_post_navigation", return_value=None):
        result = await tool.execute_async(action="navigate", url="https://example.com")

    assert result.meta.get("blocker_detected") is not True


# ---------------------------------------------------------------------------
# 4. Resolution polling: URL change resolves
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolution_polling_url_change():
    """poll_for_resolution returns True when page URL changes."""
    page = AsyncMock()
    # First check: URL unchanged. Second check: URL changed.
    type(page).url = property(lambda self: "https://new-url.com")

    resolved = await poll_for_resolution(
        page,
        original_url="https://login.example.com",
        original_signals={"has_password_field": True},
        interval=0.01,
        timeout=0.1,
    )
    assert resolved is True


# ---------------------------------------------------------------------------
# 5. Resolution polling: timeout returns False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolution_polling_timeout():
    """poll_for_resolution returns False when nothing changes before timeout."""
    page = AsyncMock()
    type(page).url = property(lambda self: "https://login.example.com")
    # has_password_field stays True, text stays the same
    page.evaluate = AsyncMock(side_effect=[
        True,   # has_pw check 1
        "Sign in to your account",  # text check 1
        True,   # has_pw check 2
        "Sign in to your account",  # text check 2
        True,   # has_pw check 3
        "Sign in to your account",  # text check 3
        True,   # has_pw check 4
        "Sign in to your account",  # text check 4
        True,   # has_pw check 5
        "Sign in to your account",  # text check 5
    ])

    resolved = await poll_for_resolution(
        page,
        original_url="https://login.example.com",
        original_signals={
            "has_password_field": True,
            "visible_text_snippet": "Sign in to your account",
        },
        interval=0.01,
        timeout=0.05,
    )
    assert resolved is False


# ---------------------------------------------------------------------------
# 6. ApproveRequest accepts response_payload field
# ---------------------------------------------------------------------------

def test_approve_request_accepts_response_payload():
    """ApproveRequest model validates with response_payload field."""
    from agent_os.api.routes.agents_v2 import ApproveRequest

    req = ApproveRequest(
        tool_call_id="tc_123",
        reply_text="Go ahead",
        response_payload="123456",
    )
    assert req.response_payload == "123456"
    assert req.tool_call_id == "tc_123"


def test_approve_request_response_payload_defaults_none():
    """ApproveRequest.response_payload defaults to None when not provided."""
    from agent_os.api.routes.agents_v2 import ApproveRequest

    req = ApproveRequest(tool_call_id="tc_456")
    assert req.response_payload is None
