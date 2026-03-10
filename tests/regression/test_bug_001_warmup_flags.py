# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for BUG-001: browser warmup must use full anti-detection flags.

Verifies that launch_warmup() passes the full CHROME_FLAGS list (including
--disable-blink-features=AutomationControlled) and applies stealth JS.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.browser_manager import BrowserManager


@pytest.fixture
def browser_manager(tmp_path):
    return BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)


def _mock_playwright_context():
    """Build a mock playwright + context that resolves launch_persistent_context."""
    mock_page = MagicMock()
    mock_page.evaluate = AsyncMock()
    mock_page.on = MagicMock()
    mock_page.goto = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.new_page = AsyncMock(return_value=mock_page)
    mock_ctx.wait_for_event = AsyncMock(side_effect=Exception("closed"))
    mock_ctx.close = AsyncMock()

    mock_chromium = MagicMock()
    mock_chromium.launch_persistent_context = AsyncMock(return_value=mock_ctx)

    mock_pw = MagicMock()
    mock_pw.chromium = mock_chromium
    mock_pw.stop = AsyncMock()

    return mock_pw, mock_ctx, mock_page


@pytest.mark.asyncio
async def test_warmup_uses_full_chrome_flags(browser_manager):
    """launch_warmup() must include --disable-blink-features=AutomationControlled."""
    mock_pw, mock_ctx, mock_page = _mock_playwright_context()

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=mock_pw)

        await browser_manager.launch_warmup("https://accounts.google.com")

        # Get the args from the first successful launch call
        call_kwargs = mock_pw.chromium.launch_persistent_context.call_args
        args = call_kwargs.kwargs.get("args", call_kwargs[1].get("args", []))

        # Critical anti-detection flag must be present
        assert "--disable-blink-features=AutomationControlled" in args
        # All CHROME_FLAGS should be included
        for flag in BrowserManager.CHROME_FLAGS:
            assert flag in args, f"Missing flag: {flag}"


@pytest.mark.asyncio
async def test_warmup_applies_stealth(browser_manager):
    """launch_warmup() must call _apply_stealth on the page before goto."""
    mock_pw, mock_ctx, mock_page = _mock_playwright_context()

    call_order = []
    original_evaluate = mock_page.evaluate

    async def track_evaluate(*args, **kwargs):
        call_order.append("stealth")
        return await original_evaluate(*args, **kwargs)

    async def track_goto(*args, **kwargs):
        call_order.append("goto")

    mock_page.evaluate = track_evaluate
    mock_page.goto = track_goto

    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=mock_pw)

        await browser_manager.launch_warmup("https://accounts.google.com")

        # Stealth JS must be injected before navigation
        assert "stealth" in call_order, "_apply_stealth was not called"
        assert call_order.index("stealth") < call_order.index("goto"), \
            "_apply_stealth must be called before page.goto()"
