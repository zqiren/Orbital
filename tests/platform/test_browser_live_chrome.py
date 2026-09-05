# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The live browser route against real Chrome (2026-09-05).

The panel is a mirror of the agent's page: the route must never change the
page's size, whatever the client sends, and its frames must describe the
page's own size. Everything in ``tests/unit/test_browser_live.py`` runs on
fakes; this drives the real route through the real BrowserManager (temp
profile, headless) and repeats what the browser tool does around it — the
side tab its ``fetch`` opens and closes, and its screenshot after every
action. Skipped where no Chrome can be launched. No network: pages are
``set_content``.
"""

import asyncio

import pytest
from fastapi import WebSocketDisconnect

from agent_os.api.routes import browser_live
from agent_os.daemon_v2.browser_manager import BrowserManager

OWN_SIZE = (1280, 720)  # the context default the agent browses at


class ScriptedSocket:
    """A client that sends what the test pushes and records what it gets."""

    def __init__(self):
        self.sent: list[dict] = []
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.closed = asyncio.Event()

    async def receive_json(self):
        getter = asyncio.ensure_future(self._incoming.get())
        closer = asyncio.ensure_future(self.closed.wait())
        done, pending = await asyncio.wait(
            {getter, closer}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if getter in done:
            return getter.result()
        raise WebSocketDisconnect(1000)

    async def send_json(self, payload):
        self.sent.append(payload)

    def push(self, msg):
        self._incoming.put_nowait(msg)

    def frames(self):
        return [(m["width"], m["height"]) for m in self.sent if m.get("type") == "frame"]

    def errors(self):
        return [m["message"] for m in self.sent if m.get("type") == "error"]


async def _until(predicate, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def _size(page):
    m = await page.evaluate("({w: innerWidth, h: innerHeight})")
    return m["w"], m["h"]


@pytest.mark.asyncio
async def test_the_route_mirrors_the_page_and_never_resizes_it(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.5)
    bm = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    try:
        page = await bm.get_page("proj_live")
    except Exception as exc:  # no Chrome / Edge / bundled Chromium here
        pytest.skip(f"no browser to launch: {exc}")
    try:
        await page.set_content("<div style='height:3000px'>agent page</div>")
        assert await _size(page) == OWN_SIZE
        ws = ScriptedSocket()
        task = asyncio.ensure_future(
            browser_live.serve(ws, "proj_live", None, browser_manager=bm)
        )
        assert await _until(lambda: ws.frames())
        assert ws.frames()[-1] == OWN_SIZE

        # A client asking for a size (the retired protocol) is refused.
        ws.push({"type": "viewport", "width": 690, "height": 820, "dpr": 2})
        assert await _until(lambda: ws.errors())
        assert "viewport" in ws.errors()[0]
        assert await _size(page) == OWN_SIZE

        # What ``BrowserTool._action_fetch`` does from a non-blank page, and
        # the screenshot the tool takes after each action.
        side = await page.context.new_page()
        await side.set_content("<p>side tab</p>")
        await side.close()
        await page.bring_to_front()
        await bm.capture_screenshot(page, str(tmp_path), "t")
        await asyncio.sleep(0.3)
        assert await _size(page) == OWN_SIZE

        # Input still lands in the page. The side tab leaves the screencast
        # capturing 87 px less than the page (a Chrome quirk; the page and
        # the tool's screenshots are untouched): the route repairs the
        # capture area, so frames settle at the page's own size.
        before = len(ws.frames())
        ws.push({"type": "mouse", "action": "wheel", "x": 100, "y": 100,
                 "deltaX": 0, "deltaY": 300})
        assert await _until(lambda: len(ws.frames()) > before)
        assert await _until(lambda: ws.frames()[-1] == OWN_SIZE, timeout=5.0), ws.frames()[-3:]
        scrolled = await page.evaluate("scrollY")
        assert scrolled > 0
        assert await _size(page) == OWN_SIZE

        ws.closed.set()
        await asyncio.wait_for(task, timeout=10)
        assert await _size(page) == OWN_SIZE
        shot = await page.screenshot()
        import struct
        assert struct.unpack(">II", shot[16:24]) == OWN_SIZE
    finally:
        await bm.shutdown()
