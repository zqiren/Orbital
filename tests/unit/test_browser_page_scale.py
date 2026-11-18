# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The agent's browser renders at 2x for a sharp live view, while what the
agent itself gets stays CSS-pixel sized (2026-11-18).

The real-Chrome side of this (frames at 2x, the tool's screenshot at the CSS
viewport, ``devicePixelRatio`` constant) is in
``tests/platform/test_browser_live_chrome.py``.
"""

import io
from unittest.mock import AsyncMock, patch

import pytest

from agent_os.daemon_v2 import browser_manager as bm_module
from agent_os.daemon_v2.browser_manager import BrowserManager
from tests.unit.test_browser_anti_detection import (
    _make_mock_context,
    _make_mock_page,
    _make_mock_playwright,
)

PIL = pytest.importorskip("PIL.Image")


@pytest.mark.asyncio
async def test_the_agents_browser_launches_at_the_page_device_scale(tmp_path):
    page = _make_mock_page()
    pw = _make_mock_playwright(_make_mock_context(pages=[page]))
    mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    with patch("agent_os.daemon_v2.browser_manager.async_playwright") as mock_ap:
        mock_ap.return_value.start = AsyncMock(return_value=pw)
        await mgr.ensure_browser()
    kwargs = pw.chromium.launch_persistent_context.call_args.kwargs
    assert bm_module.PAGE_DEVICE_SCALE == 2
    # Both halves, or the screencast stays 1x and every tool screenshot
    # re-emulates the page (see PAGE_DEVICE_SCALE).
    assert kwargs["device_scale_factor"] == 2
    assert "--force-device-scale-factor=2" in kwargs["args"]
    # The class-level flag list is not mutated (the sign-in browser and the
    # worker browser use it as is).
    assert not any(f.startswith("--force-device-scale-factor")
                   for f in BrowserManager.CHROME_FLAGS)


def _png(width, height) -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (width, height), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_a_2x_tool_screenshot_is_saved_at_the_css_viewport(tmp_path):
    """The token cost of the agent's screenshot does not double: a device-
    scale capture of a 690x820 page (1380x1640 px) is saved at 690x820."""
    page = _make_mock_page()
    page.viewport_size = {"width": 690, "height": 820}
    calls = []

    async def screenshot(**kwargs):
        calls.append(kwargs)
        with open(kwargs["path"], "wb") as f:
            f.write(_png(1380, 1640))
    page.screenshot = screenshot
    mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    path = await mgr.capture_screenshot(page, str(tmp_path / "ws"), "default")
    with PIL.open(path) as im:
        assert im.size == (690, 820)
    # Taken at device scale: a css-scale capture makes Chrome re-emulate the
    # page for the shot, which a watching live view sees as a flash.
    assert "scale" not in calls[0]


@pytest.mark.asyncio
async def test_a_1x_screenshot_is_left_byte_identical(tmp_path):
    """Worker browsers (1x) and anything already at CSS size: untouched."""
    page = _make_mock_page()
    page.viewport_size = {"width": 800, "height": 600}
    data = _png(800, 600)

    async def screenshot(**kwargs):
        with open(kwargs["path"], "wb") as f:
            f.write(data)
    page.screenshot = screenshot
    mgr = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    path = await mgr.capture_screenshot(page, str(tmp_path / "ws"), "default")
    with open(path, "rb") as f:
        assert f.read() == data
