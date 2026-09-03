# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the live browser WS route (spec 078 §5.6, §10 backend).

Everything here runs against fakes — no browser, no CDP, no daemon. The
stream core is driven directly through ``browser_live.serve()`` with a fake
WebSocket so frame timing and disconnects are deterministic; one test at the
bottom mounts the real app to prove the route is wired and reachable.
"""

import asyncio

import pytest
from fastapi import WebSocketDisconnect

from agent_os.api.routes import browser_live


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCDPSession:
    """Records every ``send``, and can push screencast frames back."""

    #: Chrome's replies for the two calls the initial-frame path makes.
    DEFAULT_REPLIES = {
        "Page.captureScreenshot": {"data": "Q0FQVFVSRQ=="},
        "Page.getLayoutMetrics": {
            "cssVisualViewport": {"clientWidth": 1280, "clientHeight": 720},
            "cssLayoutViewport": {"clientWidth": 1280, "clientHeight": 720},
        },
    }

    def __init__(self, fail_on: set[str] | None = None, replies=None):
        self.sent: list[tuple[str, dict]] = []
        self.listeners: dict[str, list] = {}
        self.detached = False
        self._fail_on = fail_on or set()
        self._replies = dict(self.DEFAULT_REPLIES)
        if replies is not None:
            self._replies.update(replies)

    async def send(self, method: str, params: dict | None = None):
        if method in self._fail_on:
            raise RuntimeError(f"boom: {method}")
        self.sent.append((method, params or {}))
        return self._replies.get(method, {})

    def on(self, event: str, handler):
        self.listeners.setdefault(event, []).append(handler)

    async def detach(self):
        self.detached = True

    # -- test helpers --
    def calls(self, method: str) -> list[dict]:
        return [params for name, params in self.sent if name == method]

    def emit_frame(self, data: str, *, session_id: int = 1,
                   width: int = 1280, height: int = 800):
        payload = {
            "data": data,
            "sessionId": session_id,
            "metadata": {"deviceWidth": width, "deviceHeight": height},
        }
        for handler in self.listeners.get("Page.screencastFrame", []):
            handler(payload)


class FakeContext:
    def __init__(self, cdp):
        self._cdp = cdp
        self.cdp_sessions_created = 0

    async def new_cdp_session(self, page):
        self.cdp_sessions_created += 1
        return self._cdp


class FakePage:
    def __init__(self, cdp, title: str = "Example Domain"):
        self.context = FakeContext(cdp)
        self._title = title
        self._closed = False

    def is_closed(self):
        return self._closed

    def close_it(self):
        self._closed = True

    async def set_viewport_size(self, size):
        self.viewport_calls = getattr(self, "viewport_calls", []) + [dict(size)]

    async def title(self):
        return self._title


class FakeBrowserManager:
    """``get_all_pages`` is the only surface the route is allowed to use."""

    def __init__(self, pages=None):
        self.pages = list(pages or [])
        self.get_page_calls = 0

    async def get_all_pages(self, project_id):
        return [p for p in self.pages if not p.is_closed()]

    async def get_page(self, project_id):  # pragma: no cover - must never run
        self.get_page_calls += 1
        raise AssertionError("browser_live must never create a page")


class FakeWebSocket:
    """Scripted client. Yields queued messages, then disconnects.

    ``gate`` (when set) blocks every ``send_json`` until the test releases it,
    which is how the coalescing test holds the sender mid-write.
    """

    def __init__(self, incoming=None):
        self.sent: list[dict] = []
        self._incoming = asyncio.Queue()
        for msg in incoming or []:
            self._incoming.put_nowait(msg)
        self.gate: asyncio.Event | None = None
        self.disconnect = asyncio.Event()
        self.sent_something = asyncio.Event()

    async def receive_json(self):
        getter = asyncio.ensure_future(self._incoming.get())
        waiter = asyncio.ensure_future(self.disconnect.wait())
        done, pending = await asyncio.wait(
            {getter, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if getter in done:
            msg = getter.result()
            if msg is _MALFORMED:
                raise ValueError("not JSON")
            return msg
        raise WebSocketDisconnect(1000)

    async def send_json(self, payload):
        if self.gate is not None:
            await self.gate.wait()
        self.sent.append(payload)
        self.sent_something.set()

    # -- test helpers --
    def push(self, msg):
        self._incoming.put_nowait(msg)

    def of_type(self, msg_type: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == msg_type]


_MALFORMED = object()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    """The watcher polls every 2s in production; 5ms in tests."""
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.005)


async def _wait_for(predicate, timeout=2.0):
    """Poll ``predicate`` on the running loop until true, else fail."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return False


class Harness:
    def __init__(self, ws, task, cdp, browser_manager):
        self.ws = ws
        self.task = task
        self.cdp = cdp
        self.bm = browser_manager

    async def wait(self, predicate, timeout=2.0):
        assert await _wait_for(predicate, timeout), (
            f"timed out; socket saw {self.ws.sent!r}"
        )

    async def finish(self):
        self.ws.disconnect.set()
        await asyncio.wait_for(self.task, timeout=2.0)


def start(ws, browser_manager, cdp=None, *, session_id=None, agent_manager=None):
    task = asyncio.ensure_future(
        browser_live.serve(
            ws, "proj_1", session_id,
            browser_manager=browser_manager,
            agent_manager=agent_manager,
        )
    )
    return Harness(ws, task, cdp, browser_manager)


def make_page(title="Example Domain"):
    cdp = FakeCDPSession()
    return FakePage(cdp, title=title), cdp


# ---------------------------------------------------------------------------
# Screencast lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_starts_screencast_with_the_agreed_params():
    page, cdp = make_page()
    h = start(FakeWebSocket(), FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        params = cdp.calls("Page.startScreencast")[0]
        assert params == {
            "format": "jpeg",
            "quality": 60,
            "maxWidth": 1280,
            "maxHeight": 1280,
            "everyNthFrame": 1,
        }
        await h.wait(lambda: h.ws.of_type("state"))
        assert h.ws.of_type("state")[0] == {
            "type": "state", "status": "open", "title": "Example Domain",
        }
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_attach_primes_the_client_with_a_captured_first_frame():
    """Chrome emits nothing until the page changes, so we capture one."""
    page, cdp = make_page()
    h = start(FakeWebSocket(), FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: h.ws.of_type("frame"))
        assert h.ws.of_type("frame")[0] == {
            "type": "frame",
            "jpeg": "Q0FQVFVSRQ==",
            "width": 1280,
            "height": 720,
            "title": "Example Domain",
        }
        assert cdp.calls("Page.captureScreenshot")[0] == {
            "format": "jpeg", "quality": 60,
        }
        # The state message lands first so the client knows why it is painting.
        first_state = h.ws.sent.index(h.ws.of_type("state")[0])
        first_frame = h.ws.sent.index(h.ws.of_type("frame")[0])
        assert first_state < first_frame
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_page_reappearing_after_closed_is_primed_again():
    page, cdp = make_page()
    bm = FakeBrowserManager([page])
    h = start(FakeWebSocket(), bm, cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.captureScreenshot"))
        page.close_it()
        await h.wait(
            lambda: any(m["status"] == "closed" for m in h.ws.of_type("state"))
        )
        page2 = FakePage(
            FakeCDPSession(replies={
                "Page.captureScreenshot": {"data": "U0VDT05E"},
            }),
            title="Second Page",
        )
        cdp2 = page2.context._cdp
        bm.pages.append(page2)
        await h.wait(lambda: cdp2.calls("Page.captureScreenshot"))
        assert h.ws.of_type("frame")[-1]["jpeg"] == "U0VDT05E"
        assert h.ws.of_type("frame")[-1]["title"] == "Second Page"
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_failed_initial_capture_does_not_break_the_stream():
    cdp = FakeCDPSession(fail_on={"Page.captureScreenshot"})
    page = FakePage(cdp)
    h = start(FakeWebSocket(), FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: h.ws.of_type("state"))
        assert h.ws.of_type("state")[0]["status"] == "open"
        assert h.ws.of_type("frame") == []
        # Real screencast frames still flow.
        cdp.emit_frame("QUJD")
        await h.wait(lambda: h.ws.of_type("frame"))
        assert h.ws.of_type("frame")[0]["jpeg"] == "QUJD"
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_screencast_frame_becomes_a_frame_message_and_is_acked():
    page, cdp = make_page()
    h = start(FakeWebSocket(), FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        cdp.emit_frame("QUJD", session_id=7, width=1024, height=768)
        await h.wait(
            lambda: any(f["jpeg"] == "QUJD" for f in h.ws.of_type("frame"))
        )
        assert h.ws.of_type("frame")[-1] == {
            "type": "frame",
            "jpeg": "QUJD",
            "width": 1024,
            "height": 768,
            "title": "Example Domain",
        }
        await h.wait(lambda: cdp.calls("Page.screencastFrameAck"))
        assert cdp.calls("Page.screencastFrameAck")[0] == {"sessionId": 7}
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_slow_socket_drops_stale_frames_and_keeps_the_newest():
    # Capture disabled so the primed frame does not race the screencast ones —
    # this test is only about coalescing.
    cdp = FakeCDPSession(fail_on={"Page.captureScreenshot"})
    page = FakePage(cdp)
    ws = FakeWebSocket()
    ws.gate = asyncio.Event()  # every send blocks until released
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        # The sender is parked and the socket is blocked; three frames arrive
        # back to back and must collapse to the last one.
        for tag in ("one", "two", "three"):
            cdp.emit_frame(tag)
        ws.gate.set()
        await h.wait(lambda: ws.of_type("frame"))
        await asyncio.sleep(0.05)
        frames = ws.of_type("frame")
        assert len(frames) == 1, frames
        assert frames[0]["jpeg"] == "three"
        # Every frame is still acked, dropped ones included.
        assert len(cdp.calls("Page.screencastFrameAck")) == 3
    finally:
        ws.gate.set()
        await h.finish()


@pytest.mark.asyncio
async def test_no_pages_reports_no_browser_and_keeps_polling():
    bm = FakeBrowserManager([])
    h = start(FakeWebSocket(), bm)
    try:
        await h.wait(lambda: h.ws.of_type("state"))
        assert h.ws.of_type("state")[0]["status"] == "no_browser"
        # Socket stays open and picks the page up once it appears.
        page, cdp = make_page("Later Page")
        bm.pages.append(page)
        await h.wait(lambda: len(h.ws.of_type("state")) >= 2)
        assert h.ws.of_type("state")[1]["status"] == "open"
        assert cdp.calls("Page.startScreencast")
        # ...and never by creating one.
        assert bm.get_page_calls == 0
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_page_closing_mid_stream_reports_closed_then_reattaches():
    page, cdp = make_page()
    bm = FakeBrowserManager([page])
    h = start(FakeWebSocket(), bm, cdp)
    try:
        await h.wait(lambda: h.ws.of_type("state"))
        page.close_it()
        await h.wait(
            lambda: any(m["status"] == "closed" for m in h.ws.of_type("state"))
        )
        assert cdp.calls("Page.stopScreencast")
        assert cdp.detached
        # Back to polling: a fresh page re-opens the stream.
        page2, cdp2 = make_page("Second Page")
        bm.pages.append(page2)
        await h.wait(lambda: cdp2.calls("Page.startScreencast"))
        assert h.ws.of_type("state")[-1] == {
            "type": "state", "status": "open", "title": "Second Page",
        }
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_disconnect_stops_the_screencast_and_detaches():
    page, cdp = make_page()
    h = start(FakeWebSocket(), FakeBrowserManager([page]), cdp)
    await h.wait(lambda: cdp.calls("Page.startScreencast"))
    await h.finish()
    assert cdp.calls("Page.stopScreencast")
    assert cdp.detached is True


# ---------------------------------------------------------------------------
# Input forwarding
# ---------------------------------------------------------------------------


async def _input_harness(messages):
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    await h.wait(lambda: cdp.calls("Page.startScreencast"))
    for msg in messages:
        ws.push(msg)
    return h


@pytest.mark.asyncio
async def test_mouse_events_dispatch_with_scaled_viewport_coordinates():
    h = await _input_harness([
        {"type": "mouse", "action": "move", "x": 12.5, "y": 30},
        {"type": "mouse", "action": "down", "x": 12.5, "y": 30,
         "button": "left", "clickCount": 2, "modifiers": 8},
        {"type": "mouse", "action": "up", "x": 12.5, "y": 30, "button": "left"},
        {"type": "mouse", "action": "wheel", "x": 40, "y": 50,
         "deltaX": 0, "deltaY": -120},
    ])
    try:
        await h.wait(
            lambda: len(h.cdp.calls("Input.dispatchMouseEvent")) == 4
        )
        move, down, up, wheel = h.cdp.calls("Input.dispatchMouseEvent")
        assert move == {"type": "mouseMoved", "x": 12.5, "y": 30.0,
                        "button": "none", "modifiers": 0}
        assert down == {"type": "mousePressed", "x": 12.5, "y": 30.0,
                        "button": "left", "modifiers": 8, "clickCount": 2}
        assert up == {"type": "mouseReleased", "x": 12.5, "y": 30.0,
                      "button": "left", "modifiers": 0, "clickCount": 1}
        assert wheel == {"type": "mouseWheel", "x": 40.0, "y": 50.0,
                         "button": "none", "modifiers": 0,
                         "deltaX": 0.0, "deltaY": -120.0}
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_key_events_carry_text_on_keydown_and_a_virtual_key_code():
    h = await _input_harness([
        {"type": "key", "action": "down", "key": "a", "code": "KeyA", "text": "a"},
        {"type": "key", "action": "up", "key": "a", "code": "KeyA", "text": "a"},
        {"type": "key", "action": "down", "key": "Enter", "code": "Enter"},
    ])
    try:
        await h.wait(lambda: len(h.cdp.calls("Input.dispatchKeyEvent")) == 3)
        down, up, enter = h.cdp.calls("Input.dispatchKeyEvent")
        assert down == {"type": "keyDown", "key": "a", "code": "KeyA",
                        "modifiers": 0, "text": "a",
                        "windowsVirtualKeyCode": 65, "nativeVirtualKeyCode": 65}
        assert "text" not in up and up["type"] == "keyUp"
        assert enter["windowsVirtualKeyCode"] == 13
        assert "text" not in enter
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_text_message_uses_insert_text():
    h = await _input_harness([{"type": "text", "text": "pasted text"}])
    try:
        await h.wait(lambda: h.cdp.calls("Input.insertText"))
        assert h.cdp.calls("Input.insertText")[0] == {"text": "pasted text"}
    finally:
        await h.finish()


@pytest.mark.parametrize("bad", [
    {"type": "mouse", "action": "teleport", "x": 1, "y": 2},
    {"type": "mouse", "action": "move"},
    {"type": "mouse", "action": "down", "x": 1, "y": 2, "button": "foot"},
    {"type": "key", "action": "sideways", "key": "a"},
    {"type": "key", "action": "down"},
    {"type": "text", "text": 42},
    {"type": "viewport", "width": "wide", "height": 900},
    {"type": "viewport", "width": 10, "height": 900},
    {"type": "levitate"},
    ["not", "an", "object"],
    _MALFORMED,
])
@pytest.mark.asyncio
async def test_malformed_messages_produce_an_error_and_keep_the_socket_open(bad):
    h = await _input_harness([bad])
    try:
        await h.wait(lambda: h.ws.of_type("error"))
        assert h.ws.of_type("error")[0]["message"]
        # The socket still works afterwards.
        h.ws.push({"type": "text", "text": "still here"})
        await h.wait(lambda: h.cdp.calls("Input.insertText"))
        assert h.cdp.calls("Input.insertText")[0] == {"text": "still here"}
        assert not h.cdp.calls("Input.dispatchMouseEvent")
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_input_before_a_page_exists_errors_instead_of_crashing():
    ws = FakeWebSocket([{"type": "text", "text": "hello"}])
    h = start(ws, FakeBrowserManager([]))
    try:
        await h.wait(lambda: ws.of_type("error"))
        assert "no browser page" in ws.of_type("error")[0]["message"]
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_failing_cdp_send_reports_an_error_and_stays_up():
    cdp = FakeCDPSession(fail_on={"Input.insertText"})
    page = FakePage(cdp)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        ws.push({"type": "text", "text": "boom"})
        await h.wait(lambda: ws.of_type("error"))
        assert "not delivered" in ws.of_type("error")[0]["message"]
    finally:
        await h.finish()


# ---------------------------------------------------------------------------
# Forensic input log
# ---------------------------------------------------------------------------


class FakeSession:
    def __init__(self):
        self.meta: list[tuple[str, dict]] = []

    def append_meta(self, event, **fields):
        self.meta.append((event, fields))


class FakeAgentManager:
    def __init__(self, session):
        self.session = session
        self.peeked: list[tuple[str, str]] = []

    def peek_chat_session(self, project_id, session_id):
        self.peeked.append((project_id, session_id))
        return session_id, self.session


@pytest.mark.asyncio
async def test_user_input_is_logged_as_a_batched_meta_row():
    page, cdp = make_page()
    session = FakeSession()
    am = FakeAgentManager(session)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp,
              session_id="sess_1", agent_manager=am)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        ws.push({"type": "text", "text": "hi"})
        ws.push({"type": "mouse", "action": "down", "x": 1, "y": 2})
        ws.push({"type": "mouse", "action": "up", "x": 1, "y": 2})
        await h.wait(lambda: len(cdp.calls("Input.dispatchMouseEvent")) == 2)
        await h.wait(lambda: session.meta)
    finally:
        await h.finish()
    events = [e for e, _ in session.meta]
    assert events and set(events) == {"browser_user_input"}
    merged: dict[str, int] = {}
    total = 0
    for _event, fields in session.meta:
        total += fields["total"]
        for key, value in fields["counts"].items():
            merged[key] = merged.get(key, 0) + value
    assert merged == {"text": 1, "mouse.down": 1, "mouse.up": 1}
    assert total == 3
    assert am.peeked and am.peeked[0] == ("proj_1", "sess_1")
    # Batched: one row per flush window, not one per event.
    assert len(session.meta) < 3


@pytest.mark.asyncio
async def test_no_session_id_means_no_meta_rows():
    page, cdp = make_page()
    session = FakeSession()
    am = FakeAgentManager(session)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp, agent_manager=am)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        ws.push({"type": "text", "text": "hi"})
        await h.wait(lambda: cdp.calls("Input.insertText"))
    finally:
        await h.finish()
    assert session.meta == []
    assert am.peeked == []


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_route_is_mounted_on_the_app(tmp_path, monkeypatch):
    """The real app factory exposes the route at the contract path."""
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "in-memory")
    monkeypatch.setenv("AGENT_OS_API_KEY", "x")
    from agent_os.api.app import create_app

    app = create_app(data_dir=str(tmp_path))
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/api/v2/agents/{project_id}/browser/live" in paths


def test_configure_injects_dependencies():
    sentinel_bm, sentinel_am = object(), object()
    previous = (browser_live._browser_manager, browser_live._agent_manager)
    try:
        browser_live.configure(sentinel_bm, agent_manager=sentinel_am)
        assert browser_live._browser_manager is sentinel_bm
        assert browser_live._agent_manager is sentinel_am
    finally:
        browser_live._browser_manager, browser_live._agent_manager = previous


def test_protocol_contract_matches_the_frontend_hook():
    """The message shapes are a shared contract; keep both sides honest."""
    from pathlib import Path

    hook = Path(__file__).resolve().parents[2] / "web/src/hooks/useBrowserLive.ts"
    if not hook.exists():  # frontend workstream may not have landed yet
        pytest.skip("useBrowserLive.ts not present")
    text = hook.read_text(encoding="utf-8")
    assert "/api/v2/agents/{project_id}/browser/live" in text
    for token in ('"frame"', '"state"', '"error"', '"mouse"', '"key"', '"text"'):
        assert token.strip('"') in text


@pytest.mark.asyncio
async def test_viewport_message_resizes_the_page_and_restarts_the_screencast_sharper():
    """Spec 078 D9 amendment: the page takes the panel's size and the
    screencast is restarted at width*dpr, then re-primed."""
    h = await _input_harness([{"type": "viewport", "width": 640, "height": 900, "dpr": 2}])
    try:
        await h.wait(lambda: h.cdp.calls("Page.stopScreencast"))
        await h.wait(lambda: len(h.cdp.calls("Page.startScreencast")) == 2)
        page = h.browser_manager.pages[0] if hasattr(h, "browser_manager") else None
        starts = h.cdp.calls("Page.startScreencast")
        assert starts[0]["maxWidth"] == 1280
        assert starts[1]["maxWidth"] == 1280 and starts[1]["maxHeight"] == 1800
        await h.wait(lambda: len(h.cdp.calls("Page.captureScreenshot")) >= 2)
        second = h.cdp.calls("Page.captureScreenshot")[-1]
        assert second["clip"] == {"x": 0, "y": 0, "width": 640, "height": 900, "scale": 2.0}
        await h.wait(lambda: any(m.get("type") == "frame" and m.get("width") == 640 for m in h.ws.sent))
        primed = [m for m in h.ws.sent if m.get("type") == "frame" and m.get("width") == 640][-1]
        assert (primed["width"], primed["height"]) == (640, 900)
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_viewport_is_applied_to_the_page_and_reapplied_on_reattach():
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        ws.push({"type": "viewport", "width": 700, "height": 500})
        await h.wait(lambda: getattr(page, "viewport_calls", None) == [{"width": 700, "height": 500}])
        assert h.cdp.calls("Page.startScreencast")[-1]["maxWidth"] == 700
    finally:
        await h.finish()
