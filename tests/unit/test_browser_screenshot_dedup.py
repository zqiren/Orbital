# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 066 phase 1a — identical screenshots are stored once.

Real projects capture the same unchanged page over and over (one worker
stored one image 43 times). When a capture is byte-identical to one this
process already stored in the same screenshot folder, the new copy is dropped
and the result points at the stored one. Different bytes are kept as always;
files that existed before are never touched.
"""

from __future__ import annotations

import base64
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.tools import browser as browser_mod
from agent_os.agent.tools.browser import BrowserTool


PNG_A = b"\x89PNG\r\n\x1a\n" + b"A" * 1024
PNG_B = b"\x89PNG\r\n\x1a\n" + b"B" * 1024


class _FakeCapture:
    """capture_screenshot stand-in that writes real step files."""

    def __init__(self, directory):
        self.directory = directory
        self.frames: list[bytes] = []
        self.step = 0

    async def __call__(self, page, workspace, namespace):
        self.step += 1
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"step_{self.step:04d}.png")
        with open(path, "wb") as f:
            f.write(self.frames.pop(0))
        return path


def _page():
    page = MagicMock()
    page.url = "https://example.com"
    page.title = AsyncMock(return_value="Example")
    page.evaluate = AsyncMock(return_value={})
    return page


@pytest.fixture(autouse=True)
def _fresh_dedup_state():
    browser_mod._SEEN_SCREENSHOTS.clear()
    yield
    browser_mod._SEEN_SCREENSHOTS.clear()


def _tool(tmp_path, *, vision=False):
    shots = tmp_path / "ws" / "orbital" / "output" / "screenshots" / "default"
    capture = _FakeCapture(str(shots))
    bm = MagicMock()
    bm.warmup_active = False
    bm.get_page = AsyncMock(return_value=_page())
    bm.capture_screenshot = capture
    tool = BrowserTool(
        browser_manager=bm, project_id="p", workspace=str(tmp_path / "ws"),
        autonomy_preset="hands_off", screenshot_namespace="default",
        vision_enabled=vision,
    )
    tool._collect_page_signals = AsyncMock(return_value={})
    return tool, capture, shots


@pytest.mark.asyncio
async def test_identical_screenshot_is_stored_once(tmp_path):
    tool, capture, shots = _tool(tmp_path)
    capture.frames = [PNG_A, PNG_A]

    first = await tool._action_screenshot({})
    second = await tool._action_screenshot({})

    assert second.meta["screenshot_path"] == first.meta["screenshot_path"]
    assert sorted(os.listdir(shots)) == ["step_0001.png"]


@pytest.mark.asyncio
async def test_different_screenshots_are_all_kept(tmp_path):
    tool, capture, shots = _tool(tmp_path)
    capture.frames = [PNG_A, PNG_B, PNG_A]

    paths = [(await tool._action_screenshot({})).meta["screenshot_path"] for _ in range(3)]

    assert paths[0] != paths[1]
    assert paths[2] == paths[0]
    assert sorted(os.listdir(shots)) == ["step_0001.png", "step_0002.png"]


@pytest.mark.asyncio
async def test_preexisting_identical_file_is_never_touched(tmp_path):
    """Only captures made by this process are dedup candidates — a file from
    before is neither reused nor removed."""
    tool, capture, shots = _tool(tmp_path)
    os.makedirs(shots)
    old = shots / "step_0900.png"
    old.write_bytes(PNG_A)
    capture.frames = [PNG_A]

    result = await tool._action_screenshot({})

    assert result.meta["screenshot_path"].endswith("step_0001.png")
    assert old.read_bytes() == PNG_A
    assert sorted(os.listdir(shots)) == ["step_0001.png", "step_0900.png"]


@pytest.mark.asyncio
async def test_reused_file_that_changed_is_not_reused(tmp_path):
    tool, capture, shots = _tool(tmp_path)
    capture.frames = [PNG_A, PNG_A]
    first = await tool._action_screenshot({})
    with open(first.meta["screenshot_path"], "wb") as f:
        f.write(PNG_B)

    second = await tool._action_screenshot({})

    assert second.meta["screenshot_path"] != first.meta["screenshot_path"]
    with open(second.meta["screenshot_path"], "rb") as f:
        assert f.read() == PNG_A


@pytest.mark.asyncio
async def test_vision_screenshot_still_returns_the_image(tmp_path):
    tool, capture, _shots = _tool(tmp_path, vision=True)
    capture.frames = [PNG_A, PNG_A]
    await tool._action_screenshot({})
    result = await tool._action_screenshot({})

    image = [b for b in result.content if b.get("type") == "image_url"][0]
    assert image["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(PNG_A).decode("ascii")
    )


@pytest.mark.asyncio
async def test_missing_capture_file_passes_through(tmp_path):
    """A mocked or failed capture that left no file is returned unchanged."""
    tool, _capture, _shots = _tool(tmp_path)
    tool._bm.capture_screenshot = AsyncMock(return_value="/nowhere/step_0001.png")
    result = await tool._action_screenshot({})
    assert result.meta["screenshot_path"] == "/nowhere/step_0001.png"
