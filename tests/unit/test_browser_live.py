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

    def set_layout_size(self, width: int, height: int) -> None:
        """What ``Page.getLayoutMetrics`` reports from now on (CSS px)."""
        self._replies["Page.getLayoutMetrics"] = {
            "cssVisualViewport": {"clientWidth": width, "clientHeight": height},
            "cssLayoutViewport": {"clientWidth": width, "clientHeight": height},
        }

    def on(self, event: str, handler):
        self.listeners.setdefault(event, []).append(handler)

    async def detach(self):
        self.detached = True

    # -- test helpers --
    def calls(self, method: str) -> list[dict]:
        return [params for name, params in self.sent if name == method]

    def emit(self, event: str, params: dict):
        for handler in self.listeners.get(event, []):
            handler(params)

    def emit_frame(self, data: str, *, session_id: int = 1,
                   width: int = 1280, height: int = 720):
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
    """``viewport_size`` / ``set_viewport_size`` behave like Playwright's:
    the call records the size and Chrome's layout metrics follow it."""

    def __init__(self, cdp, title: str = "Example Domain"):
        self.context = FakeContext(cdp)
        self._cdp = cdp
        self._title = title
        self.url = "https://example.com/"
        self._closed = False
        self.viewport_size = {"width": 1280, "height": 720}
        self.viewport_calls: list[dict] = []

    def is_closed(self):
        return self._closed

    def close_it(self):
        self._closed = True

    async def set_viewport_size(self, size):
        if self._closed:
            raise RuntimeError("Target page has been closed")
        self.viewport_calls.append(dict(size))
        self.viewport_size = dict(size)
        self._cdp.set_layout_size(size["width"], size["height"])

    async def title(self):
        return self._title


class FakeBrowserManager:
    """``get_all_pages`` is what the route watches with.

    ``get_page`` (get-or-create — the same call the agent's browser tool
    makes) is reached only by a typed address; ``opens`` is the page it hands
    back, and leaving it None is a manager that cannot open one.
    """

    def __init__(self, pages=None, opens=None):
        self.pages = list(pages or [])
        self.get_page_calls = 0
        self._opens = opens

    async def get_all_pages(self, project_id):
        return [p for p in self.pages if not p.is_closed()]

    async def get_page(self, project_id):
        self.get_page_calls += 1
        alive = [p for p in self.pages if not p.is_closed()]
        if alive:
            return alive[-1]
        if self._opens is None:
            raise RuntimeError("no browser could be launched")
        self.pages.append(self._opens)
        return self._opens


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


#: Long enough for several watcher ticks at the test poll interval.
POLL_TICKS = 0.05


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    """The watcher polls every 2s in production; 5ms in tests. The repair's
    waits for a fresh frame shrink the same way."""
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 0.005)
    monkeypatch.setattr(browser_live, "REPAIR_PRIME_DELAYS_S", (0.005, 0.005))
    browser_live._SIZERS.clear()
    yield
    browser_live._SIZERS.clear()


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


def start(ws, browser_manager, cdp=None, *, session_id=None, agent_manager=None,
          dpr=None):
    task = asyncio.ensure_future(
        browser_live.serve(
            ws, "proj_1", session_id,
            browser_manager=browser_manager,
            agent_manager=agent_manager,
            dpr=dpr,
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
            "url": "https://example.com/",
            "loading": False, "canGoBack": False, "canGoForward": False,
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
            "url": "https://example.com/",
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
        # At the page's size: a frame of another size is a shrunken capture
        # area and gets repaired (its own test below).
        cdp.emit_frame("QUJD", session_id=7, width=1280, height=720)
        await h.wait(
            lambda: any(f["jpeg"] == "QUJD" for f in h.ws.of_type("frame"))
        )
        assert h.ws.of_type("frame")[-1] == {
            "type": "frame",
            "jpeg": "QUJD",
            "width": 1280,
            "height": 720,
            "title": "Example Domain",
            "url": "https://example.com/",
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
async def test_typed_address_with_nothing_open_opens_a_page_and_navigates():
    """The panel's header is always there, so the address field has to work in
    the state every fresh project is in: no browser yet (2026-09-05)."""
    page, cdp = make_page("Blank")
    page.url = "about:blank"
    bm = FakeBrowserManager([], opens=page)
    ws = FakeWebSocket([
        {"type": "nav", "action": "goto", "url": "https://example.com/"},
    ])
    h = start(ws, bm, cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.navigate"))
        assert bm.get_page_calls == 1
        assert cdp.calls("Page.navigate")[0] == {"url": "https://example.com/"}
        # Opened means attached and streaming, not just created.
        assert cdp.calls("Page.startScreencast")
        await h.wait(
            lambda: any(m["status"] == "open" for m in ws.of_type("state"))
        )
        # The watcher keeps polling on top of this; the page the user opened
        # must survive its ticks rather than being torn down as a stray.
        await asyncio.sleep(POLL_TICKS)
        assert not cdp.detached
        assert ws.of_type("state")[-1]["status"] == "open"
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_typed_address_says_so_when_no_browser_can_be_opened():
    bm = FakeBrowserManager([])  # cannot open one
    ws = FakeWebSocket([
        {"type": "nav", "action": "goto", "url": "https://example.com/"},
    ])
    h = start(ws, bm)
    try:
        await h.wait(lambda: ws.of_type("error"))
        assert "could not open a browser page" in ws.of_type("error")[0]["message"]
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
            "url": "https://example.com/",
            "loading": False, "canGoBack": False, "canGoForward": False,
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
    {"type": "viewport", "width": 0, "height": 900},
    {"type": "viewport"},
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
    import agent_os.api.app as app_module
    from agent_os.api.app import create_app

    # The daemon is a singleton keyed on ~/orbital/daemon.pid. When an earlier
    # test in the same process built an app, the factory sees its own live
    # PID, takes the "already running" branch and returns the stub app with
    # no API routes — which is what CI hit (only /docs, /ws, /openapi.json
    # were mounted). Neutralise the singleton like the other app tests do.
    monkeypatch.setattr(app_module, "acquire_pid_file", lambda *a, **k: None)

    app = create_app(data_dir=str(tmp_path))
    # Resolve by name: FastAPI >= 0.16x keeps an included router as one
    # wrapper entry in ``app.routes`` (no ``.path``), so scanning the list
    # missed every API route on CI.
    assert (app.url_path_for("browser_live", project_id="p1")
            == "/api/v2/agents/p1/browser/live")


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
    for token in ('"frame"', '"state"', '"error"', '"mouse"', '"key"', '"text"',
                  '"nav"', '"cursor"', "canGoBack", "loading"):
        assert token.strip('"') in text


def fake_jpeg(width: int, height: int) -> str:
    """A base64 "JPEG" that is only a frame header of the given pixel size —
    enough for the route's size check."""
    import base64
    sof = (b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08"
           + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x00" * 9)
    return base64.b64encode(b"\xff\xd8" + sof + b"\xff\xd9").decode()


def test_the_frame_header_reader_reads_chromes_jpeg_size():
    assert browser_live._jpeg_size(fake_jpeg(1380, 1640)) == (1380, 1640)
    assert browser_live._jpeg_size("not a jpeg") is None
    assert browser_live._same_aspect((1380, 1640), (690, 820))
    assert not browser_live._same_aspect((690, 733), (690, 820))
    assert browser_live._same_aspect(None, (690, 820))


@pytest.mark.parametrize("asked,got", [
    ((690.4, 820.6), (690, 821)),
    ((100, 50), (280, 240)),
    ((292, 828.5), (292, 828)),
    ((9000, 4000), (2560, 2560)),
])
def test_a_viewers_size_is_clamped_to_whole_sane_pixels(asked, got):
    assert browser_live._clamp_view_size(*asked) == got


@pytest.mark.asyncio
async def test_the_page_takes_the_viewers_size_through_playwright_only():
    """2026-11-18 — the page fills the panel: a viewer's size becomes the
    page's viewport, through Playwright's ``set_viewport_size`` (the session
    whose emulation Chrome restores after the tool's screenshots). This
    route's own CDP session never touches device metrics and never captures
    with a clip — that is how the 2026-09 size fight was armed."""
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: ws.of_type("frame"))
        ws.push({"type": "viewport", "width": 690, "height": 820, "dpr": 2})
        await h.wait(lambda: page.viewport_calls)
        assert page.viewport_calls == [{"width": 690, "height": 820}]
        await h.wait(lambda: ws.of_type("viewport"))
        assert ws.of_type("viewport")[-1] == {
            "type": "viewport", "width": 690, "height": 820, "owner": True}
        assert not cdp.calls("Emulation.setDeviceMetricsOverride")
        assert all(call == {"format": "jpeg", "quality": 60}
                   for call in cdp.calls("Page.captureScreenshot"))
        # The same size again is not a second resize.
        ws.push({"type": "viewport", "width": 690, "height": 820, "dpr": 2})
        ws.push({"type": "text", "text": "x"})
        await h.wait(lambda: cdp.calls("Input.insertText"))
        assert page.viewport_calls == [{"width": 690, "height": 820}]
    finally:
        await h.finish()
    # The last viewer left: the page goes back to the agent's own size.
    assert page.viewport_calls[-1] == {"width": 1280, "height": 720}
    assert page.viewport_size == {"width": 1280, "height": 720}
    assert browser_live._SIZERS == {}


@pytest.mark.asyncio
async def test_a_client_that_never_sends_a_size_leaves_the_page_alone():
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: ws.of_type("frame"))
        await asyncio.sleep(0.03)
    finally:
        await h.finish()
    assert page.viewport_calls == []


@pytest.mark.asyncio
async def test_a_size_sent_before_the_page_exists_applies_when_it_attaches():
    page, cdp = make_page()
    bm = FakeBrowserManager([])
    ws = FakeWebSocket([{"type": "viewport", "width": 500, "height": 600, "dpr": 1}])
    h = start(ws, bm, cdp)
    try:
        await asyncio.sleep(0.02)
        bm.pages.append(page)
        await h.wait(lambda: page.viewport_calls)
        assert page.viewport_calls == [{"width": 500, "height": 600}]
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_drag_never_restarts_the_screencast_and_lands_on_the_last_size():
    """A burst of sizes (a client that sends every step of a drag) merges:
    the newest wins, the screencast keeps running, and the page ends at the
    final size."""
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp, dpr=2)
    try:
        await h.wait(lambda: ws.of_type("frame"))
        for w in range(400, 1000, 20):
            ws.push({"type": "viewport", "width": w, "height": 700, "dpr": 2})
        await h.wait(lambda: page.viewport_size == {"width": 980, "height": 700})
        await asyncio.sleep(0.03)
        assert page.viewport_size == {"width": 980, "height": 700}
        assert len(cdp.calls("Page.startScreencast")) == 1
        assert not cdp.calls("Page.stopScreencast")
        assert len(page.viewport_calls) <= 30
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_retina_viewer_gets_2x_frames_and_only_a_density_change_restarts():
    page, cdp = make_page()
    ws = FakeWebSocket()
    task = asyncio.ensure_future(browser_live.serve(
        ws, "proj_1", None, browser_manager=FakeBrowserManager([page]), dpr=2))
    h = Harness(ws, task, cdp, None)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        assert cdp.calls("Page.startScreencast")[0]["maxWidth"] == 2560
        ws.push({"type": "viewport", "width": 700, "height": 800, "dpr": 2})
        await h.wait(lambda: page.viewport_calls)
        assert len(cdp.calls("Page.startScreencast")) == 1
        # The window moved to a 1x display.
        ws.push({"type": "viewport", "width": 700, "height": 800, "dpr": 1})
        await h.wait(lambda: len(cdp.calls("Page.startScreencast")) == 2)
        assert cdp.calls("Page.startScreencast")[1]["maxWidth"] == 1280
        assert page.viewport_calls == [{"width": 700, "height": 800}]
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_the_newest_viewer_owns_the_size_and_others_letterbox():
    """Desktop app plus a phone: whoever attached or resized last sizes the
    page; the other is told it does not own the size (and letterboxes).
    When the owner leaves the size goes to the next most recent viewer, and
    home when the last one leaves."""
    page, cdp = make_page()
    bm = FakeBrowserManager([page])
    desk = FakeWebSocket([{"type": "viewport", "width": 900, "height": 760, "dpr": 2}])
    hd = start(desk, bm, cdp)
    await hd.wait(lambda: page.viewport_size == {"width": 900, "height": 760})
    phone = FakeWebSocket([{"type": "viewport", "width": 390, "height": 700, "dpr": 3}])
    hp = start(phone, bm, cdp)
    try:
        await hp.wait(lambda: page.viewport_size == {"width": 390, "height": 700})
        await hd.wait(lambda: desk.of_type("viewport")[-1]["owner"] is False)
        assert phone.of_type("viewport")[-1] == {
            "type": "viewport", "width": 390, "height": 700, "owner": True}
        assert desk.of_type("viewport")[-1]["width"] == 390
        # The desktop resizes: it is the most recent again.
        desk.push({"type": "viewport", "width": 1000, "height": 760, "dpr": 2})
        await hd.wait(lambda: page.viewport_size == {"width": 1000, "height": 760})
        await hp.wait(lambda: phone.of_type("viewport")[-1]["owner"] is False)
        await hp.finish()
        assert page.viewport_size == {"width": 1000, "height": 760}
        phone2 = FakeWebSocket([{"type": "viewport", "width": 400, "height": 650}])
        hp2 = start(phone2, bm, cdp)
        await hp2.wait(lambda: page.viewport_size == {"width": 400, "height": 650})
        await hp2.finish()
        # The owner left: back to the remaining viewer's size.
        await hd.wait(lambda: page.viewport_size == {"width": 1000, "height": 760})
        await hd.wait(lambda: desk.of_type("viewport")[-1]["owner"] is True)
    finally:
        await hd.finish()
    assert page.viewport_size == {"width": 1280, "height": 720}


@pytest.mark.asyncio
async def test_a_size_waits_behind_a_click_sent_before_it():
    """A click aimed at the old layout lands before the page changes size."""
    page, cdp = make_page()
    order = []
    real_send = cdp.send

    async def send(method, params=None):
        if method == "Input.dispatchMouseEvent":
            order.append("click")
        return await real_send(method, params)
    cdp.send = send
    real_resize = page.set_viewport_size

    async def resize(size):
        order.append("resize")
        await real_resize(size)
    page.set_viewport_size = resize
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: ws.of_type("frame"))
        ws.push({"type": "mouse", "action": "down", "x": 5, "y": 5})
        ws.push({"type": "viewport", "width": 600, "height": 600})
        await h.wait(lambda: "resize" in order)
        assert order[:2] == ["click", "resize"]
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_the_client_is_never_asked_to_resend_and_an_off_size_frame_is_not_shown():
    """The September loop: a frame whose size disagreed made the client ask
    again, forever. Now the route alone repairs: an off-size frame (the
    capture area after a side tab) is withheld from the client, the capture
    area is put back with ``setVisibleSize``, and the page is NOT resized
    again — its size never depended on the frame."""
    page, cdp = make_page()
    ws = FakeWebSocket([{"type": "viewport", "width": 690, "height": 820, "dpr": 2}])
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: page.viewport_size == {"width": 690, "height": 820})
        await asyncio.sleep(0.03)
        calls = list(page.viewport_calls)
        visible = len(cdp.calls("Emulation.setVisibleSize"))
        shown = len(ws.of_type("frame"))
        cdp.emit_frame(fake_jpeg(1380, 1640), width=690, height=820)
        await h.wait(lambda: len(ws.of_type("frame")) == shown + 1)
        cdp.emit_frame(fake_jpeg(1280, 633), width=1280, height=633)
        await h.wait(lambda: len(cdp.calls("Emulation.setVisibleSize")) > visible)
        assert cdp.calls("Emulation.setVisibleSize")[-1] == {"width": 690, "height": 820}
        # Same metadata, cropped picture (Chrome mid-restore): also withheld.
        cdp.emit_frame(fake_jpeg(690, 733), width=690, height=820)
        await asyncio.sleep(0.03)
        sizes = {(f["width"], f["height"]) for f in ws.of_type("frame")[shown:]}
        assert sizes == {(690, 820)}
        assert all(f["jpeg"] != fake_jpeg(1280, 633) and f["jpeg"] != fake_jpeg(690, 733)
                   for f in ws.of_type("frame"))
        assert page.viewport_calls == calls
        assert not cdp.calls("Emulation.setDeviceMetricsOverride")
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_capture_area_that_will_not_repair_still_shows_frames():
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: ws.of_type("frame"))
        shown = len(ws.of_type("frame"))
        for _ in range(browser_live.MAX_DROPPED_FRAMES + 1):
            cdp.emit_frame("Q1JPUA==", width=1280, height=633)
        await h.wait(lambda: len(ws.of_type("frame")) > shown)
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_frame_shorter_than_the_page_repairs_the_capture_area_not_the_page():
    """After the browser tool's ``fetch`` side tab, Chrome's screencast
    captures 87 px less than the page while the page itself is untouched.
    A frame that disagrees with the layout size gets the capture area put
    back with ``Emulation.setVisibleSize`` and a fresh prime — and no
    device-metrics call, ever."""
    page, cdp = make_page()   # layout metrics: 1280x720
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: ws.of_type("frame"))
        primes = len(cdp.calls("Page.captureScreenshot"))
        cdp.emit_frame("RlVMTA==", width=1280, height=720)
        await h.wait(lambda: len(ws.of_type("frame")) >= 2)
        await asyncio.sleep(0.02)
        assert not cdp.calls("Emulation.setVisibleSize")   # a frame at the page's size is fine
        cdp.emit_frame("Q1JPUA==", width=1280, height=633)
        await h.wait(lambda: cdp.calls("Emulation.setVisibleSize"))
        assert cdp.calls("Emulation.setVisibleSize") == [{"width": 1280, "height": 720}]
        await h.wait(lambda: len(cdp.calls("Page.captureScreenshot")) > primes)
        assert not cdp.calls("Emulation.setDeviceMetricsOverride")
        assert not page.viewport_calls
        # The cropped frame itself never reached the client.
        assert all(f["jpeg"] != "Q1JPUA==" for f in ws.of_type("frame"))
        # Repaired once, not on every cropped frame in flight.
        cdp.emit_frame("Q1JPUA==", width=1280, height=633)
        await asyncio.sleep(0.03)
        assert len(cdp.calls("Emulation.setVisibleSize")) <= 2
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_state_and_frames_carry_the_page_url():
    """The client tells a blank page (about:blank) from a real one by url."""
    page, cdp = make_page()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: any(m.get("type") == "state" for m in ws.sent))
        assert [m for m in ws.sent if m.get("type") == "state"][0]["url"] == "https://example.com/"
        await h.wait(lambda: any(m.get("type") == "frame" for m in ws.sent))
        assert [m for m in ws.sent if m.get("type") == "frame"][0]["url"] == "https://example.com/"
    finally:
        await h.finish()


# ---------------------------------------------------------------------------
# Input never queues up (2026-09-04): merging moves / wheels, ordered clicks
# ---------------------------------------------------------------------------


class GatedCDPSession(FakeCDPSession):
    """``Input.dispatchMouseEvent`` blocks until the test releases it — the
    stand-in for Chrome taking ~20 ms to confirm a wheel."""

    def __init__(self):
        super().__init__()
        self.releases = asyncio.Queue()

    async def send(self, method, params=None):
        if method == "Input.dispatchMouseEvent":
            await self.releases.get()
        return await super().send(method, params)

    def release(self, n=1):
        for _ in range(n):
            self.releases.put_nowait(None)


def wheel(dy, x=40, y=50):
    return {"type": "mouse", "action": "wheel", "x": x, "y": y, "deltaX": 0, "deltaY": dy}


def move(x, y):
    return {"type": "mouse", "action": "move", "x": x, "y": y}


async def _gated_harness():
    cdp = GatedCDPSession()
    page = FakePage(cdp)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    await h.wait(lambda: cdp.calls("Page.startScreencast"))
    return h, cdp, ws


@pytest.mark.asyncio
async def test_a_wheel_burst_collapses_into_one_dispatch_per_round_trip():
    """120 Hz trackpad wheels while Chrome is busy confirming the previous one:
    the waiting deltas add up into ONE event, never a backlog."""
    h, cdp, ws = await _gated_harness()
    try:
        for _ in range(30):
            ws.push(wheel(8))
        # First wheel goes out at once and sits in flight (gated) …
        await h.wait(lambda: cdp.releases.qsize() == 0 and len(h.ws.sent) >= 0)
        await asyncio.sleep(0.02)
        cdp.release()
        await h.wait(lambda: len(cdp.calls("Input.dispatchMouseEvent")) == 1)
        # … the other 29 merged into a single second dispatch.
        cdp.release()
        await h.wait(lambda: len(cdp.calls("Input.dispatchMouseEvent")) == 2)
        await asyncio.sleep(0.02)
        assert len(cdp.calls("Input.dispatchMouseEvent")) == 2
        first, second = cdp.calls("Input.dispatchMouseEvent")
        assert first["deltaY"] == 8.0
        assert second["deltaY"] == 8.0 * 29
    finally:
        cdp.release(10)
        await h.finish()


@pytest.mark.asyncio
async def test_moves_waiting_behind_a_dispatch_keep_only_the_newest():
    h, cdp, ws = await _gated_harness()
    try:
        ws.push(move(1, 1))
        await asyncio.sleep(0.02)
        for i in range(2, 12):
            ws.push(move(i, i))
        await asyncio.sleep(0.02)          # let the receive loop ingest them
        cdp.release(2)
        await h.wait(lambda: len(cdp.calls("Input.dispatchMouseEvent")) == 2)
        await asyncio.sleep(0.02)
        moves = cdp.calls("Input.dispatchMouseEvent")
        assert len(moves) == 2
        assert (moves[1]["x"], moves[1]["y"]) == (11.0, 11.0)
    finally:
        cdp.release(10)
        await h.finish()


@pytest.mark.asyncio
async def test_clicks_keep_their_place_between_merged_wheels():
    h, cdp, ws = await _gated_harness()
    try:
        ws.push(wheel(10))
        await asyncio.sleep(0.02)          # in flight
        ws.push(wheel(20))
        ws.push({"type": "mouse", "action": "down", "x": 5, "y": 5})
        ws.push(wheel(5))
        ws.push(wheel(5))                  # merges with the previous 5
        await asyncio.sleep(0.02)          # let the receive loop ingest them
        cdp.release(4)
        await h.wait(lambda: len(cdp.calls("Input.dispatchMouseEvent")) == 4)
        await asyncio.sleep(0.02)
        calls = cdp.calls("Input.dispatchMouseEvent")
        assert [c["type"] for c in calls] == ["mouseWheel", "mouseWheel", "mousePressed", "mouseWheel"]
        assert [c.get("deltaY") for c in calls] == [10.0, 20.0, None, 10.0]
    finally:
        cdp.release(10)
        await h.finish()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


HISTORY = {
    "currentIndex": 1,
    "entries": [
        {"id": 11, "url": "https://a.example/"},
        {"id": 12, "url": "https://b.example/"},
        {"id": 13, "url": "https://c.example/"},
    ],
}


async def _nav_harness(history=HISTORY):
    cdp = FakeCDPSession(replies={"Page.getNavigationHistory": history})
    page = FakePage(cdp)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    await h.wait(lambda: cdp.calls("Page.startScreencast"))
    return h, cdp, ws


@pytest.mark.asyncio
async def test_back_and_forward_walk_chromes_navigation_history():
    h, cdp, ws = await _nav_harness()
    try:
        ws.push({"type": "nav", "action": "back"})
        await h.wait(lambda: cdp.calls("Page.navigateToHistoryEntry"))
        assert cdp.calls("Page.navigateToHistoryEntry")[-1] == {"entryId": 11}
        ws.push({"type": "nav", "action": "forward"})
        await h.wait(lambda: len(cdp.calls("Page.navigateToHistoryEntry")) == 2)
        assert cdp.calls("Page.navigateToHistoryEntry")[-1] == {"entryId": 13}
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_state_reports_history_availability_and_back_at_the_start_is_a_no_op():
    h, cdp, ws = await _nav_harness({"currentIndex": 0, "entries": [{"id": 1, "url": "https://a.example/"}]})
    try:
        await h.wait(lambda: ws.of_type("state"))
        assert ws.of_type("state")[0]["canGoBack"] is False
        assert ws.of_type("state")[0]["canGoForward"] is False
        ws.push({"type": "nav", "action": "back"})
        ws.push({"type": "nav", "action": "reload"})
        await h.wait(lambda: cdp.calls("Page.reload"))
        assert cdp.calls("Page.navigateToHistoryEntry") == []
        assert ws.of_type("error") == []
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_goto_navigates_to_web_urls_only():
    h, cdp, ws = await _nav_harness()
    try:
        ws.push({"type": "nav", "action": "goto", "url": "https://example.org/x"})
        await h.wait(lambda: cdp.calls("Page.navigate"))
        assert cdp.calls("Page.navigate")[-1] == {"url": "https://example.org/x"}
        for bad in ("file:///etc/passwd", "javascript:alert(1)", "example.org", "", 7):
            ws.push({"type": "nav", "action": "goto", "url": bad})
        await h.wait(lambda: len(ws.of_type("error")) == 5)
        assert len(cdp.calls("Page.navigate")) == 1
        ws.push({"type": "nav", "action": "stop"})
        await h.wait(lambda: cdp.calls("Page.stopLoading"))
        ws.push({"type": "nav", "action": "sideways"})
        await h.wait(lambda: len(ws.of_type("error")) == 6)
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_navigation_action_is_counted_in_the_forensic_log():
    cdp = FakeCDPSession(replies={"Page.getNavigationHistory": HISTORY})
    page = FakePage(cdp)
    session = FakeSession()
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp, session_id="sess_1",
              agent_manager=FakeAgentManager(session))
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        ws.push({"type": "nav", "action": "reload"})
        await h.wait(lambda: cdp.calls("Page.reload"))
        await h.wait(lambda: session.meta)
    finally:
        await h.finish()
    counts = {}
    for _e, fields in session.meta:
        for k, v in fields["counts"].items():
            counts[k] = counts.get(k, 0) + v
    assert counts == {"nav.reload": 1}


# ---------------------------------------------------------------------------
# Chrome's Page events push state without waiting for the poll
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_page_events_push_url_loading_and_history_immediately(monkeypatch):
    monkeypatch.setattr(browser_live, "POLL_INTERVAL_S", 5.0)  # the poll must not be what delivers these
    cdp = FakeCDPSession(replies={
        "Page.getFrameTree": {"frameTree": {"frame": {"id": "MAIN"}}},
        "Page.getNavigationHistory": {"currentIndex": 0, "entries": [{"id": 1}]},
    })
    page = FakePage(cdp)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: ws.of_type("state"))
        assert cdp.calls("Page.enable") == [{}]
        cdp.emit("Page.frameStartedLoading", {"frameId": "MAIN"})
        await h.wait(lambda: ws.of_type("state")[-1]["loading"] is True)
        # A sub-frame navigating is not the address bar's business.
        cdp.emit("Page.frameNavigated", {"frame": {"id": "SUB", "parentId": "MAIN", "url": "https://ad.example/"}})
        await asyncio.sleep(0.02)
        assert ws.of_type("state")[-1]["url"] == "https://example.com/"
        cdp._replies["Page.getNavigationHistory"] = {"currentIndex": 1, "entries": [{"id": 1}, {"id": 2}]}
        page.url = "https://example.com/next"
        page._title = "Next"
        cdp.emit("Page.frameNavigated", {"frame": {"id": "MAIN", "url": "https://example.com/next"}})
        await h.wait(lambda: ws.of_type("state")[-1]["url"] == "https://example.com/next")
        assert ws.of_type("state")[-1]["title"] == "Next"
        assert ws.of_type("state")[-1]["canGoBack"] is True
        cdp.emit("Page.frameStoppedLoading", {"frameId": "MAIN"})
        await h.wait(lambda: ws.of_type("state")[-1]["loading"] is False)
        cdp.emit("Page.navigatedWithinDocument", {"frameId": "MAIN", "url": "https://example.com/next#a"})
        page.url = "https://example.com/next#a"
        await h.wait(lambda: ws.of_type("state")[-1]["url"] == "https://example.com/next#a")
    finally:
        await h.finish()


# ---------------------------------------------------------------------------
# Cursor mirroring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_pages_cursor_under_a_resting_pointer_is_sent_when_it_changes(monkeypatch):
    monkeypatch.setattr(browser_live, "CURSOR_PROBE_MIN_INTERVAL_S", 0.0)
    cdp = FakeCDPSession(replies={"Runtime.evaluate": {"result": {"type": "string", "value": "pointer"}}})
    page = FakePage(cdp)
    ws = FakeWebSocket()
    h = start(ws, FakeBrowserManager([page]), cdp)
    try:
        await h.wait(lambda: cdp.calls("Page.startScreencast"))
        ws.push(move(10, 20))
        await h.wait(lambda: ws.of_type("cursor"))
        assert ws.of_type("cursor") == [{"type": "cursor", "cursor": "pointer"}]
        probe = cdp.calls("Runtime.evaluate")[-1]
        assert "elementFromPoint(x, y)" in probe["expression"]
        assert probe["expression"].rstrip().endswith("(10.0, 20.0)")
        assert probe["returnByValue"] is True
        # Same answer again: nothing new to say.
        ws.push(move(11, 21))
        await h.wait(lambda: len(cdp.calls("Runtime.evaluate")) == 2)
        await asyncio.sleep(0.02)
        assert len(ws.of_type("cursor")) == 1
        cdp._replies["Runtime.evaluate"] = {"result": {"type": "string", "value": "text"}}
        ws.push(move(12, 22))
        await h.wait(lambda: len(ws.of_type("cursor")) == 2)
        assert ws.of_type("cursor")[-1]["cursor"] == "text"
    finally:
        await h.finish()


# ---------------------------------------------------------------------------
# Reopen after the idle close (2026-11-18)
# ---------------------------------------------------------------------------


class LoadablePage(FakePage):
    """A fresh page at about:blank that Playwright can load a URL into.
    Records the order of its loads against the CDP traffic, so a test can
    tell "loaded, then attached" from "attached, then navigated"."""

    def __init__(self, cdp, title="Reopened"):
        super().__init__(cdp, title=title)
        self.url = "about:blank"
        self.gotos: list[tuple[str, str, int]] = []  # url, wait_until, cdp calls so far
        self.viewport_at_goto: dict | None = None

    async def goto(self, url, wait_until=None, timeout=None):
        self.gotos.append((url, wait_until, len(self._cdp.sent)))
        self.viewport_at_goto = dict(self.viewport_size)
        self.url = url

    async def wait_for_load_state(self, state, timeout=None):
        return None


class IdleClosedManager(FakeBrowserManager):
    """A manager whose browser the idle close shut: it remembers a URL per
    scope until that scope gets a page again (as the real manager does)."""

    def __init__(self, remembered=None, opens=None, fail_open=False):
        super().__init__([], opens=opens)
        self.remembered: dict[str, str] = dict(remembered or {})
        self.scopes_opened: list[str] = []
        self.lookups: list[tuple[str, str | None]] = []
        self.uses = 0
        self._fail_open = fail_open

    def idle_closed_page(self, project_id, session_id=None):
        self.lookups.append((project_id, session_id))
        from agent_os.daemon_v2.browser_manager import page_scope
        scope = page_scope(project_id, session_id)
        url = self.remembered.get(scope)
        return (scope, url) if url else None

    def note_use(self):
        self.uses += 1

    async def get_page(self, project_id):
        if self._fail_open:
            self.get_page_calls += 1
            raise RuntimeError("Chrome would not start")
        await asyncio.sleep(0.02)  # Chrome takes a moment to start
        page = await super().get_page(project_id)
        self.scopes_opened.append(project_id)
        self.remembered.pop(project_id, None)
        return page

    def idle_close(self, page, url):
        """What the idle close does: the page goes, its URL is remembered."""
        page.close_it()
        self.remembered[SESSION_SCOPE] = url


SESSION_SCOPE = "proj_1::session:sess_1"


@pytest.mark.asyncio
async def test_opening_the_view_after_the_idle_close_reopens_the_last_page():
    cdp = FakeCDPSession()
    page = LoadablePage(cdp)
    bm = IdleClosedManager({SESSION_SCOPE: "https://example.com/docs"}, opens=page)
    ws = FakeWebSocket([{"type": "viewport", "width": 690, "height": 820, "dpr": 2}])
    h = start(ws, bm, cdp, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("frame"))
        # Through the same get-or-create a typed address uses, on the
        # viewer's own session scope.
        assert bm.scopes_opened == [SESSION_SCOPE]
        # Loaded BEFORE the stream attached (no CDP traffic yet), at the
        # viewer's size — so the client never sees about:blank or a page at
        # the default size.
        assert page.gotos == [("https://example.com/docs", "domcontentloaded", 0)]
        assert page.viewport_at_goto == {"width": 690, "height": 820}
        assert cdp.calls("Page.navigate") == []
        # The first thing the client hears is the live page, never a
        # "no browser" in between.
        states = ws.of_type("state")
        assert [s["status"] for s in states][:1] == ["open"]
        assert all(s["status"] == "open" for s in states)
        assert states[0]["url"] == "https://example.com/docs"
        assert ws.of_type("frame")[0]["width"] == 690
        assert ws.of_type("error") == []
        # A use: the reopened browser gets a fresh idle period.
        assert bm.uses >= 1
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_without_a_remembered_url_nothing_is_opened():
    bm = IdleClosedManager({}, opens=LoadablePage(FakeCDPSession()))
    ws = FakeWebSocket()
    h = start(ws, bm, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("state"))
        await asyncio.sleep(POLL_TICKS)
        assert [s["status"] for s in ws.of_type("state")] == ["no_browser"]
        assert bm.get_page_calls == 0
        assert bm.uses == 0
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_another_sessions_remembered_page_is_not_this_views():
    bm = IdleClosedManager({"proj_1::session:other": "https://example.com/"},
                           opens=LoadablePage(FakeCDPSession()))
    ws = FakeWebSocket()
    h = start(ws, bm, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("state"))
        await asyncio.sleep(POLL_TICKS)
        assert ws.of_type("state")[-1]["status"] == "no_browser"
        assert bm.get_page_calls == 0
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_second_idle_close_under_the_same_view_does_not_reopen_again():
    cdp = FakeCDPSession()
    page = LoadablePage(cdp)
    bm = IdleClosedManager({SESSION_SCOPE: "https://example.com/a"}, opens=page)
    ws = FakeWebSocket()
    h = start(ws, bm, cdp, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("frame"))
        assert bm.get_page_calls == 1
        # Idle again, with the same view open and untouched.
        bm._opens = LoadablePage(FakeCDPSession())
        bm.idle_close(page, "https://example.com/a")
        await h.wait(lambda: ws.of_type("state")[-1]["status"] != "open")
        await asyncio.sleep(POLL_TICKS * 4)
        # It stays on the last frame: no relaunch loop. (The watcher turns
        # "closed" into "no_browser" on its next tick, as it always has; the
        # client treats both as "not live".)
        assert bm.get_page_calls == 1
        assert ws.of_type("state")[-1]["status"] in ("closed", "no_browser")
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_click_on_the_stale_picture_reopens_it():
    cdp = FakeCDPSession()
    page = LoadablePage(cdp)
    bm = IdleClosedManager({SESSION_SCOPE: "https://example.com/a"}, opens=page)
    ws = FakeWebSocket()
    h = start(ws, bm, cdp, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("frame"))
        cdp2 = FakeCDPSession(replies={"Page.captureScreenshot": {"data": "QUdBSU4="}})
        page2 = LoadablePage(cdp2)
        bm._opens = page2
        bm.idle_close(page, "https://example.com/b")
        await h.wait(lambda: ws.of_type("state")[-1]["status"] != "open")
        uses_before = bm.uses
        ws.push({"type": "nav", "action": "reopen"})
        await h.wait(lambda: ws.of_type("frame")[-1]["jpeg"] == "QUdBSU4=")
        assert page2.gotos[0][0] == "https://example.com/b"
        assert ws.of_type("state")[-1]["status"] == "open"
        assert bm.get_page_calls == 2
        assert bm.uses > uses_before
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_reopen_click_with_nothing_remembered_does_nothing():
    bm = IdleClosedManager({}, opens=LoadablePage(FakeCDPSession()))
    ws = FakeWebSocket()
    h = start(ws, bm, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("state"))
        ws.push({"type": "nav", "action": "reopen"})
        await asyncio.sleep(POLL_TICKS)
        assert bm.get_page_calls == 0
        assert ws.of_type("error") == []
        assert ws.of_type("state")[-1]["status"] == "no_browser"
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_view_that_found_the_page_open_never_reopens_it_later():
    page, cdp = make_page()
    bm = IdleClosedManager({}, opens=LoadablePage(FakeCDPSession()))
    bm.pages.append(page)
    ws = FakeWebSocket()
    h = start(ws, bm, cdp, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("frame"))
        bm.idle_close(page, "https://example.com/")
        await h.wait(lambda: ws.of_type("state")[-1]["status"] != "open")
        await asyncio.sleep(POLL_TICKS * 4)
        assert bm.get_page_calls == 0
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_a_reopen_that_cannot_start_chrome_falls_back_quietly():
    bm = IdleClosedManager({SESSION_SCOPE: "https://example.com/"}, fail_open=True)
    ws = FakeWebSocket()
    h = start(ws, bm, session_id="sess_1")
    try:
        await h.wait(lambda: ws.of_type("state"))
        await asyncio.sleep(POLL_TICKS)
        assert ws.of_type("state")[-1]["status"] == "no_browser"
        assert ws.of_type("error") == []  # nothing the user asked for failed
        assert bm.get_page_calls == 1  # tried once, not on every tick
    finally:
        await h.finish()


@pytest.mark.asyncio
async def test_the_auto_reopen_is_not_logged_as_user_input():
    cdp = FakeCDPSession()
    page = LoadablePage(cdp)
    bm = IdleClosedManager({SESSION_SCOPE: "https://example.com/"}, opens=page)
    session = FakeSession()
    ws = FakeWebSocket()
    h = start(ws, bm, cdp, session_id="sess_1", agent_manager=FakeAgentManager(session))
    try:
        await h.wait(lambda: ws.of_type("frame"))
        await asyncio.sleep(POLL_TICKS)
    finally:
        await h.finish()
    assert session.meta == []
