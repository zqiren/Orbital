# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The live browser route against real Chrome (2026-11-18).

The page takes the watching viewer's size and keeps it — through the
browser tool's own screenshots and the side tab its ``fetch`` opens — without
the client asking again, and goes back to the context default when the last
viewer leaves. These are the conditions that made the 2026-09-03 version of
this feature blink (a size fight between the route, Chrome's clipped-capture
emulation restore and Playwright's same-size dedupe), so each is repeated
here against a real browser. Everything in ``tests/unit/test_browser_live.py``
runs on fakes; this drives the real route through the real BrowserManager
(temp profile, headless). Skipped where no Chrome can be launched. No
network: pages are ``set_content``.
"""

import asyncio
import base64
import io
import struct

import pytest
from fastapi import WebSocketDisconnect

from agent_os.api.routes import browser_live
from agent_os.daemon_v2 import browser_manager as bm_module
from agent_os.daemon_v2.browser_manager import BrowserManager

HOME = browser_live.HOME_SIZE  # (1280, 720), the context default
SCALE = bm_module.PAGE_DEVICE_SCALE

PAGE_HTML = """<body style='margin:0;font:16px sans-serif'>
<a id=small href='#clicked' style='position:absolute;left:%dpx;top:40px;font-size:11px'>tiny</a>
<input id=field style='position:absolute;left:10px;top:80px;width:200px'>
<div id=tick style='position:fixed;right:0;bottom:0;width:6px;height:6px;background:red'></div>
<div style='height:3000px'>agent page</div></body>"""


class ScriptedSocket:
    """A client that sends what the test pushes and records what it gets."""

    def __init__(self):
        self.sent: list[dict] = []
        self.pushed: list[dict] = []
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
        self.pushed.append(msg)
        self._incoming.put_nowait(msg)

    def frames(self):
        return [(m["width"], m["height"]) for m in self.sent if m.get("type") == "frame"]

    def last_frame_pixels(self):
        frame = [m for m in self.sent if m.get("type") == "frame"][-1]
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(frame["jpeg"]))).size

    def owner(self):
        rows = [m for m in self.sent if m.get("type") == "viewport"]
        return rows[-1]["owner"] if rows else None

    def errors(self):
        return [m["message"] for m in self.sent if m.get("type") == "error"]


async def _until(predicate, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return True
        await asyncio.sleep(0.05)
    return False


async def _size(page):
    m = await page.evaluate("({w: innerWidth, h: innerHeight})")
    return m["w"], m["h"]


async def _launch(tmp_path):
    bm = BrowserManager(profile_dir=str(tmp_path / "profile"), headless=True)
    try:
        page = await bm.get_page("proj_live")
    except Exception as exc:  # no Chrome / Edge / bundled Chromium here
        pytest.skip(f"no browser to launch: {exc}")
    return bm, page


def _connect(bm, dpr=2):
    ws = ScriptedSocket()
    task = asyncio.ensure_future(
        browser_live.serve(ws, "proj_live", None, browser_manager=bm, dpr=dpr))
    return ws, task


def _count_screencast_starts(monkeypatch):
    starts = []
    original = browser_live._LiveStream._start_screencast

    async def counting(self, cdp):
        starts.append(self)
        await original(self, cdp)
    monkeypatch.setattr(browser_live._LiveStream, "_start_screencast", counting)
    return starts


def _tick_frames(page):
    """Repaint a corner of the page so Chrome emits a screencast frame."""
    return page.evaluate("(() => { const t = document.getElementById('tick');"
                         " t.style.background = t.style.background === 'red' ? 'blue' : 'red'; })()")


@pytest.mark.asyncio
async def test_the_page_takes_the_viewers_size_and_keeps_it(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.5)
    starts = _count_screencast_starts(monkeypatch)
    bm, page = await _launch(tmp_path)
    try:
        await page.set_content(PAGE_HTML % 600)
        assert await _size(page) == HOME
        assert await page.evaluate("devicePixelRatio") == SCALE

        ws, task = _connect(bm)
        assert await _until(lambda: ws.frames())
        assert ws.frames()[-1] == HOME          # sized by nobody yet

        want = (690, 820)
        ws.push({"type": "viewport", "width": want[0], "height": want[1], "dpr": 2})

        async def page_is_wanted():
            return await _size(page) == want
        assert await _until(page_is_wanted)
        await _tick_frames(page)
        assert await _until(lambda: ws.frames()[-1] == want), ws.frames()[-3:]
        assert ws.owner() is True
        # Frames arrive at the page's device pixels (2x), not upscaled 1x.
        await _tick_frames(page)
        await asyncio.sleep(0.3)
        assert ws.last_frame_pixels() == (want[0] * SCALE, want[1] * SCALE)

        # What the browser tool does after every action (its screenshot) and
        # what its ``fetch`` does from a non-blank page (a side tab, closed,
        # then back to the agent's page) — with no further client message.
        settled_from = len(ws.frames())
        stream = starts[0]
        repairs, dropped = stream.repair_count, stream.dropped_total
        for _ in range(3):
            path = await bm.capture_screenshot(page, str(tmp_path), "t")
            with open(path, "rb") as f:
                png = f.read()
            # The agent's screenshot is its CSS viewport, at 1x pixels.
            assert struct.unpack(">II", png[16:24]) == want
            await _tick_frames(page)
            await asyncio.sleep(0.1)
        # The tool's screenshots disturb nothing: no off-size frame to drop,
        # no capture area to repair (a css-scale capture at 2x would).
        assert (stream.repair_count, stream.dropped_total) == (repairs, dropped)
        side = await page.context.new_page()
        await side.set_content("<p>side tab</p>")
        await side.close()
        await page.bring_to_front()
        for _ in range(4):
            await _tick_frames(page)
            await asyncio.sleep(0.15)
        assert await _size(page) == want
        assert await _until(lambda: ws.frames()[-1] == want, timeout=5.0), ws.frames()[-3:]
        # No frame of another size ever reached the client: the side tab's
        # cropped capture area is repaired, not forwarded.
        assert set(ws.frames()[settled_from:]) == {want}, set(ws.frames()[settled_from:])
        assert [m for m in ws.pushed if m["type"] == "viewport"] == [
            {"type": "viewport", "width": 690, "height": 820, "dpr": 2}]
        assert len(starts) == 1
        # The side tab costs a bounded repair, not a loop.
        assert stream.repair_count - repairs <= 3, stream.repair_count - repairs

        # Input lands at the size the client sees.
        await page.evaluate("location.hash = ''")
        box = await page.locator("#small").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        ws.push({"type": "mouse", "action": "down", "x": cx, "y": cy, "button": "left"})
        ws.push({"type": "mouse", "action": "up", "x": cx, "y": cy, "button": "left"})
        assert await _until(lambda: page.url.endswith("#clicked"))

        ws.closed.set()
        await asyncio.wait_for(task, timeout=10)
        # The last viewer left: back to the agent's own size.
        assert await _size(page) == HOME
    finally:
        await bm.shutdown()


@pytest.mark.asyncio
async def test_a_drag_resizes_without_restarting_the_screencast(monkeypatch, tmp_path):
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.5)
    starts = _count_screencast_starts(monkeypatch)
    bm, page = await _launch(tmp_path)
    try:
        await page.set_content(PAGE_HTML % 300)
        ws, task = _connect(bm)
        assert await _until(lambda: ws.frames())
        # A client that (wrongly) sends every step of a drag: the route
        # still lands on the last size, one screencast throughout.
        widths = list(range(400, 1000, 40))
        for w in widths:
            ws.push({"type": "viewport", "width": w, "height": 700, "dpr": 2})
        final = (widths[-1], 700)

        async def page_is_final():
            return await _size(page) == final
        assert await _until(page_is_final)
        await _tick_frames(page)
        assert await _until(lambda: ws.frames()[-1] == final, timeout=5.0)
        assert len(starts) == 1
        sizer = browser_live._SIZERS[id(page)]
        assert sizer.apply_count <= len(widths)

        ws.closed.set()
        await asyncio.wait_for(task, timeout=10)
        assert await _size(page) == HOME
    finally:
        await bm.shutdown()


@pytest.mark.asyncio
async def test_the_newest_viewer_owns_the_size_and_the_page_goes_home_last(
        monkeypatch, tmp_path):
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.5)
    bm, page = await _launch(tmp_path)
    try:
        await page.set_content(PAGE_HTML % 100)
        desk, desk_task = _connect(bm)
        desk.push({"type": "viewport", "width": 900, "height": 760, "dpr": 2})

        async def size_is(target):
            return await _size(page) == target
        assert await _until(lambda: size_is((900, 760)))

        phone, phone_task = _connect(bm, dpr=3)
        phone.push({"type": "viewport", "width": 390, "height": 700, "dpr": 3})
        assert await _until(lambda: size_is((390, 700)))
        await _tick_frames(page)
        assert await _until(lambda: desk.owner() is False and phone.owner() is True)
        # The desktop keeps watching, letterboxed: its frames are the phone's size.
        async def desk_sees_phone_size():
            await _tick_frames(page)
            return bool(desk.frames()) and desk.frames()[-1] == (390, 700)
        assert await _until(desk_sees_phone_size, timeout=5.0), desk.frames()[-3:]

        phone.closed.set()
        await asyncio.wait_for(phone_task, timeout=10)
        assert await _until(lambda: size_is((900, 760)))
        assert await _until(lambda: desk.owner() is True)

        desk.closed.set()
        await asyncio.wait_for(desk_task, timeout=10)
        assert await _size(page) == HOME
        assert id(page) not in browser_live._SIZERS
    finally:
        await bm.shutdown()


class _LocalSite:
    """A loopback HTTP page (the idle close only remembers web URLs)."""

    BODY = (b"<!doctype html><title>Reopen probe</title>"
            b"<body style='margin:0'><div id=w></div>"
            b"<script>document.getElementById('w').textContent="
            b"innerWidth+'x'+innerHeight</script></body>")

    def __enter__(self):
        import http.server
        import threading

        body = self.BODY

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/docs"
        return self

    def __exit__(self, *exc):
        self.server.shutdown()


async def _idle_close(bm):
    import time
    bm._last_use = time.monotonic() - bm.IDLE_CLOSE_AFTER - 1
    assert await bm.close_if_idle() is True
    assert bm._context is None


@pytest.mark.asyncio
async def test_opening_the_view_after_the_idle_close_brings_the_page_back(monkeypatch, tmp_path):
    """2026-11-18: the view reopens the idle-closed page live, at the view's
    size, once; a second idle close under the same untouched view stays
    closed until the user clicks."""
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.2)
    bm, page = await _launch(tmp_path)
    try:
        with _LocalSite() as site:
            await page.goto(site.url)
            await _idle_close(bm)
            assert bm.idle_closed_page("proj_live") == ("proj_live", site.url)

            want = (700, 800)
            ws, task = _connect(bm)
            ws.push({"type": "viewport", "width": want[0], "height": want[1], "dpr": 2})
            assert await _until(lambda: ws.frames(), timeout=20.0), ws.sent[-3:]
            # Straight to the page, loaded, at the view's size: never a
            # "no browser", never about:blank, never the default size.
            states = [m for m in ws.sent if m.get("type") == "state"]
            assert states and all(s["status"] == "open" for s in states)
            assert states[0]["url"] == site.url
            assert ws.frames()[0] == want
            assert ws.errors() == []
            reopened = (await bm.get_all_pages("proj_live"))[-1]
            assert await _size(reopened) == want
            # It loaded at that size (the page measured itself on load).
            assert await reopened.evaluate("document.getElementById('w').textContent") == "700x800"
            # Same persistent profile as any launch.
            assert bm._profile_dir == tmp_path / "profile"
            # A use: a fresh idle period.
            assert await bm.close_if_idle() is False

            # Idle again under the same, untouched view: it falls back and
            # stays (no relaunch loop).
            await _idle_close(bm)
            assert await _until(lambda: [m for m in ws.sent if m.get("type") == "state"][-1]["status"] != "open")
            await asyncio.sleep(1.0)
            assert bm._context is None

            # A click on the picture brings it back.
            frames_before = len(ws.frames())
            ws.push({"type": "nav", "action": "reopen"})
            assert await _until(lambda: len(ws.frames()) > frames_before, timeout=20.0)
            assert [m for m in ws.sent if m.get("type") == "state"][-1]["status"] == "open"
            assert ws.frames()[-1] == want
            ws.closed.set()
            await asyncio.wait_for(task, timeout=10)
    finally:
        await bm.shutdown()
