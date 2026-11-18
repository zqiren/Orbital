# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live view of the agent's browser (spec 078 §5.6, D9).

One WebSocket route attaches a CDP session to the project's *current* page,
streams ``Page.startScreencast`` JPEG frames to the client, forwards the
client's mouse/keyboard events back into the page via ``Input.*``, and drives
plain browser navigation (back / forward / reload / stop / go to URL).

Protocol (shared contract — mirrored in ``web/src/hooks/useBrowserLive.ts``)::

    WS  /api/v2/agents/{project_id}/browser/live[?session_id=…&token=…]

    server → client
      {"type":"frame","jpeg":"<base64>","width":<css px>,"height":<css px>,"title":"…","url":"…"}
      {"type":"state","status":"no_browser"|"open"|"closed","title":"…","url":"…",
       "loading":bool,"canGoBack":bool,"canGoForward":bool}
      {"type":"cursor","cursor":"pointer"}   — the page's CSS cursor under the pointer
      {"type":"error","message":"…"}

      {"type":"viewport","width":<css px>,"height":<css px>,"owner":bool}
                                           — the page's size, and whether this
                                             viewer is the one that sized it

    client → server
      {"type":"mouse","action":"move"|"down"|"up"|"wheel","x":n,"y":n,
       "button":"left"|"right"|"middle","clickCount":1,"deltaX":n,"deltaY":n,"modifiers":n}
      {"type":"key","action":"down"|"up","key":"Enter","code":"Enter","text":"a","modifiers":n}
      {"type":"text","text":"pasted or IME-composed text"}
      {"type":"nav","action":"back"|"forward"|"reload"|"stop"|"goto"|"reopen","url":"https://…"}
      {"type":"viewport","width":<css px>,"height":<css px>,"dpr":n}
                                           — the size of this viewer's view; sent
                                             on connect and when the view settles
                                             at a new size, never because of a frame

``x``/``y`` are CSS pixels of the page viewport — the same space as a frame's
``width``/``height`` (taken from the screencast metadata's
``deviceWidth``/``deviceHeight``). The JPEG is at the page's device pixels
(``PAGE_DEVICE_SCALE`` in the browser manager), possibly downscaled by
Chrome; the client scales its canvas coordinates into the CSS space.

A ``goto`` is the one client message that may CREATE a page (2026-09-05):
the panel's header is there in every state, so typing an address has to work
before the agent has browsed anything. The page is made through the browser
manager's per-project pool, so it is the same page the agent's browser tool
picks up.

**Reopen after the idle close** (2026-11-18): the agent's browser closes
itself after 15 idle minutes (spec 175), and a viewer opening the view after
that used to find only the agent's last screenshot. Now the view brings the
page back: when a stream's *first* look finds no page but the browser
manager remembers the URL the idle close shut for this scope
(``idle_closed_page``), the stream reopens it through the same path a typed
address takes (``_open_page``: the manager's ``get_page``, so the same
persistent profile and its sign-ins), loads the URL, and only then attaches,
so the client goes from the last screenshot straight to the loaded page at
its own size. No state is pushed while that runs (the client keeps showing
what it shows while connecting). The reopen counts as a use (a fresh 15
minutes); a viewer merely watching still does not, and the automatic reopen
happens once per stream: when the browser idle-closes again under the same
untouched view, the view falls back to the last frame and stays there until
the view is opened again or the user acts — a click on the picture sends
``nav``/``reopen``, the address field a ``goto``. Page state (scroll,
forms, in-page history) is not restored. Apart from these two (a user's
gesture, a viewer opening the view), nothing here ever starts a browser.

**Input never queues up** (2026-09-04, "scrolling feels very laggy"): a
wheel event takes ~20 ms for Chrome to confirm because it scrolls and renders
before answering, while a trackpad emits 120 of them a second — handling them
one at a time turned a two-second swipe into five seconds of scrolling. So
input goes through a queue that merges: consecutive moves keep only the
newest, consecutive wheels sum their deltas, and clicks / keys / text / nav
keep their order relative to everything else. At most one event is ever in
flight, and the backlog can never exceed one merged event of each kind.

**The page takes the viewer's size** (2026-11-18, reversing the 2026-09-05
mirror on purpose: the panel should behave like a browser pane at its real
width). While a viewer watches, the agent's page viewport is that viewer's
view size; when the last viewer leaves it goes back to ``HOME_SIZE``. With
several viewers the most recent one (attached or resized last) owns the size
and the others letterbox. ``_PageSizer`` below is the one owner of a page's
size, and the design keeps the 2026-09-05 size fight from coming back:

* The only thing that writes the page's device metrics is Playwright's own
  session, through ``page.set_viewport_size`` — called by the sizer only
  when the owned size changes. This route's CDP session never sends
  ``Emulation.setDeviceMetricsOverride`` and never captures with a ``clip``
  (Chrome restores the *calling* session's emulation after a clipped
  capture; a session with no emulation "restores" the page to none, which
  is how the September prime armed the fight). Playwright's screenshots
  restore Playwright's own baseline, which is the size we set.
* The client sends its size when its view settles at a new size and on
  connect — never because a frame disagrees. A frame that disagrees is
  repaired here (next point) and not forwarded.
* The sizer compares against the size it applied, not Playwright's cache, so
  a deduped same-size call can never be mistaken for progress, and requests
  coalesce: only the newest wanted size is ever applied.
* Resizing never restarts the screencast. Chrome keeps streaming and the
  frames' metadata carries the new size.

**The capture area is repaired, never the page:** the screencast captures the
page's visible area, which does not follow the page in headless Chrome — it
is the window's content area, 87 px shorter than the window
``set_viewport_size`` asks for, and the side tab the browser tool's
``fetch`` opens resets the window to the context default (probed 2026-09-05
and 2026-11-18). A frame whose size disagrees with the page's size is
dropped and triggers ``Emulation.setVisibleSize`` back to the page's size,
which touches no device metrics and nothing the agent sees.

**No coordination layer** (D9): there is no take-over button and no pause
flag. The user's input and the agent's tool calls simply interleave — the
browser tool snapshots the page before each action, so it sees what the user
did. The only thing guarded here is the page/CDP session going away.
"""

import asyncio
import base64
import collections
import itertools
import logging
import time
from urllib.parse import urlsplit

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

# ---- Dependency holders ----

_browser_manager = None
_agent_manager = None


def configure(browser_manager, agent_manager=None):
    """Called by the app factory to inject dependencies.

    ``agent_manager`` is optional: it is only used for the forensic
    ``browser_user_input`` meta row, which is skipped when it is absent.
    """
    global _browser_manager, _agent_manager
    _browser_manager = browser_manager
    _agent_manager = agent_manager


# ---- Tunables (module level so tests can shrink them) ----

#: How often we look for the project's current page. Also the cadence at
#: which the page title is refreshed and the input log is flushed. Navigation
#: within the attached page is NOT on this clock — Chrome's Page events push
#: url/title/loading/history the moment they change.
POLL_INTERVAL_S = 2.0

#: Screencast parameters for a 1x viewer. ``maxWidth``/``maxHeight`` scale
#: with the viewer's device-pixel ratio (``_frame_cap``): the page renders at
#: 2x, so a Retina viewer gets frames at its own resolution and a 1x viewer
#: keeps the old 1280 ceiling.
SCREENCAST_PARAMS = {
    "format": "jpeg",
    "quality": 60,
    "maxWidth": 1280,
    "maxHeight": 1280,
    "everyNthFrame": 1,
}

#: The page's size when nobody is watching: the browser context's default,
#: the size the agent has always browsed at.
HOME_SIZE = (1280, 720)

#: Bounds for a viewer's size (CSS px). Below the minimum, layouts break in
#: ways no panel needs; above the maximum, frames get needlessly heavy. 280,
#: not 320: the docked panel's own floor is 320 px, and its drawable area
#: (inside the page card's inset) is 292 — that must still fill edge to edge.
MIN_VIEW_WIDTH = 280
MIN_VIEW_HEIGHT = 240
MAX_VIEW_SIDE = 2560

#: Off-size frames are dropped while the capture area is repaired. If this
#: many arrive in a row the repair is not taking, and showing something beats
#: a frozen view.
MAX_DROPPED_FRAMES = 30

#: After a capture-area repair: how long to wait (each step) for the
#: screencast's own fresh frame before capturing one instead.
REPAIR_PRIME_DELAYS_S = (0.05, 0.1, 0.2, 0.4)

#: Reopening an idle-closed page: how long to wait for it to load before
#: attaching anyway (the screencast shows the rest). DOMContentLoaded, as
#: the agent's browser tool waits, then a short best-effort wait for ``load``
#: so the first frame is the page rather than its half-built layout.
REOPEN_DOM_TIMEOUT_MS = 15000
REOPEN_LOAD_TIMEOUT_MS = 3000

#: The page's cursor under the pointer is looked up (one ``Runtime.evaluate``)
#: after a dispatched move when no newer move is waiting, at most this often.
CURSOR_PROBE_MIN_INTERVAL_S = 0.05

# ---- Input mapping ----

_MOUSE_TYPES = {
    "move": "mouseMoved",
    "down": "mousePressed",
    "up": "mouseReleased",
    "wheel": "mouseWheel",
}

_BUTTONS = {"none", "left", "middle", "right", "back", "forward"}

_NAV_ACTIONS = {"back", "forward", "reload", "stop", "goto", "reopen"}

#: Only web URLs may be typed into the address field. ``file:`` would read
#: the daemon host's disk through the agent's browser; ``javascript:`` is
#: script injection; anything else has no business in a live page.
_ALLOWED_NAV_SCHEMES = {"http", "https"}

#: Virtual key codes Chrome needs for the non-printable keys that matter for
#: "log me in here" — everything else falls back to the single-character
#: derivation below, and anything unmapped simply omits the field.
_VIRTUAL_KEY_CODES = {
    "Enter": 13,
    "Backspace": 8,
    "Tab": 9,
    "Escape": 27,
    "Delete": 46,
    "ArrowLeft": 37,
    "ArrowUp": 38,
    "ArrowRight": 39,
    "ArrowDown": 40,
    "Home": 36,
    "End": 35,
    "PageUp": 33,
    "PageDown": 34,
    "Shift": 16,
    "Control": 17,
    "Alt": 18,
    "Meta": 91,
}

#: Evaluated in the page to mirror its cursor onto the client's canvas. A
#: computed ``auto`` is what most elements report, so it is resolved the way
#: a browser would show it: hand over links/buttons, I-beam over editable
#: text, arrow elsewhere.
_CURSOR_JS = """(function (x, y) {
  try {
    var el = document.elementFromPoint(x, y);
    if (!el) return 'default';
    var c = getComputedStyle(el).cursor || 'auto';
    if (c !== 'auto') return c;
    if (el.closest('a[href],button,[role=button],[role=link],select,summary,label,[onclick]')) return 'pointer';
    if (el.closest('input,textarea,[contenteditable=""],[contenteditable=true],[contenteditable=plaintext-only]')) return 'text';
    return 'default';
  } catch (e) { return 'default'; }
})(%r, %r)"""


def _number(value):
    """Coerce a JSON number to float, or None when it is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _virtual_key_code(key: str):
    code = _VIRTUAL_KEY_CODES.get(key)
    if code is not None:
        return code
    if len(key) == 1:
        return ord(key.upper())
    return None


def _clamp_view_size(width, height) -> tuple[int, int] | None:
    """A viewer's size as whole CSS px within the bounds, or None when the
    message does not carry two positive numbers."""
    w, h = _number(width), _number(height)
    if w is None or h is None or w <= 0 or h <= 0:
        return None
    return (
        int(min(max(round(w), MIN_VIEW_WIDTH), MAX_VIEW_SIDE)),
        int(min(max(round(h), MIN_VIEW_HEIGHT), MAX_VIEW_SIDE)),
    )


def _frame_cap(dpr: float) -> int:
    """Screencast ``maxWidth``/``maxHeight`` for a viewer at ``dpr``."""
    scale = min(max(dpr, 1.0), 2.0)
    return int(round(SCREENCAST_PARAMS["maxWidth"] * scale))


_JPEG_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _jpeg_size(b64: str) -> tuple[int, int] | None:
    """Pixel size of a base64 JPEG from its frame header, or None.

    Only the first few KB are decoded: Chrome writes the SOF segment right
    after the quantization tables."""
    if not isinstance(b64, str):
        return None
    head_len = min(len(b64), 4096)
    head_len -= head_len % 4
    try:
        head = base64.b64decode(b64[:head_len])
    except Exception:
        return None
    if head[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 <= len(head):
        if head[i] != 0xFF:
            return None
        marker = head[i + 1]
        if marker in _JPEG_SOF_MARKERS:
            height = int.from_bytes(head[i + 5:i + 7], "big")
            width = int.from_bytes(head[i + 7:i + 9], "big")
            return (width, height) if width and height else None
        i += 2 + int.from_bytes(head[i + 2:i + 4], "big")
    return None


def _same_aspect(pixels: tuple[int, int] | None, size: tuple[int, int]) -> bool:
    """True when an image of ``pixels`` can be a picture of ``size`` CSS px
    (Chrome downscales frames keeping the aspect). Unknown pixels pass."""
    if pixels is None:
        return True
    pw, ph = pixels
    w, h = size
    if pw <= 0 or ph <= 0 or w <= 0 or h <= 0:
        return True
    return abs(pw * h - ph * w) <= 0.015 * max(pw * h, ph * w)


def _size_of(value) -> tuple[int, int] | None:
    """``{"width": w, "height": h}`` (Playwright's ``viewport_size``) as a tuple."""
    if not isinstance(value, dict):
        return None
    w, h = value.get("width"), value.get("height")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return (w, h)
    return None


#: Activity stamps for "the most recent viewer owns the size".
_activity = itertools.count(1)


class _PageSizer:
    """The single owner of one page's viewport size while viewers watch it.

    Every live stream attached to the page registers here with the size of
    its view (or none, for a client that does not send one). The page's size
    is the newest viewer's size; with no sized viewer it is ``HOME_SIZE``.
    ``settle`` applies that through Playwright, one call at a time, always
    the newest target, and only when it differs from what was last applied.
    """

    def __init__(self, page):
        self.page = page
        #: stream -> (wanted size or None, activity stamp)
        self.viewers: dict = {}
        #: What this sizer last applied; None until it applies anything.
        self.applied: tuple[int, int] | None = None
        #: The size being applied right now, while ``set_viewport_size`` runs.
        self.applying: tuple[int, int] | None = None
        self.apply_count = 0
        self._lock = asyncio.Lock()

    def owner(self):
        sized = [(stamp, s) for s, (size, stamp) in self.viewers.items() if size]
        return max(sized, key=lambda item: item[0])[1] if sized else None

    def target(self) -> tuple[int, int] | None:
        """The size the page should have; None when this sizer has nothing
        to do (no sized viewer, and it never changed the page)."""
        owner = self.owner()
        if owner is not None:
            return self.viewers[owner][0]
        if self.applied is None or self.applied == HOME_SIZE:
            return None
        return HOME_SIZE

    def current(self) -> tuple[int, int] | None:
        """The page's size as far as frames are concerned."""
        return self.applying or self.applied or _size_of(
            getattr(self.page, "viewport_size", None))

    async def settle(self) -> None:
        changed = False
        async with self._lock:
            # Re-read after every call: a newer request that landed while
            # Playwright was busy is applied next; older ones are never.
            while True:
                target = self.target()
                if target is None or target == self.applied:
                    break
                if self.applied is None and target == _size_of(
                        getattr(self.page, "viewport_size", None)):
                    # Already that size (a fresh page at the default):
                    # nothing to apply.
                    self.applied = target
                    break
                self.applying = target
                try:
                    await self.page.set_viewport_size(
                        {"width": target[0], "height": target[1]})
                except Exception:
                    logger.debug("browser_live: set_viewport_size failed",
                                 exc_info=True)
                    self.applying = None
                    break
                self.applied = target
                self.applying = None
                self.apply_count += 1
                changed = True
                logger.info("browser_live: page sized to %dx%d", *target)
        for stream in list(self.viewers):
            await stream._on_page_sized(self, changed)


#: id(page) -> its sizer, while any stream is attached to the page.
_SIZERS: dict[int, _PageSizer] = {}


def _sizer_for(page) -> _PageSizer:
    sizer = _SIZERS.get(id(page))
    if sizer is None or sizer.page is not page:
        sizer = _PageSizer(page)
        _SIZERS[id(page)] = sizer
    return sizer


def _nav_url(value) -> str | None:
    """A web URL fit for ``Page.navigate``, or None."""
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in _ALLOWED_NAV_SCHEMES or not parts.netloc:
        return None
    return url


class _LiveStream:
    """One connected client, watching one project's browser.

    Four concurrent pieces: the caller's receive loop (validates input and
    queues it), an input pump (dispatches the queue, merging as it goes), a
    sender task (frames out, coalesced), and a watcher task that
    attaches/detaches as the project's page appears and disappears.
    """

    def __init__(self, websocket, project_id, session_id,
                 browser_manager, agent_manager, dpr=None):
        self._ws = websocket
        self._project_id = project_id
        self._session_id = session_id
        self._bm = browser_manager
        self._am = agent_manager
        #: The viewer's device-pixel ratio (frame resolution it can use).
        self._dpr = _number(dpr) or 1.0
        #: The size of this viewer's view (CSS px, clamped), once it says.
        self._wanted: tuple[int, int] | None = None
        self._sizer: _PageSizer | None = None
        self._last_viewport_msg: dict | None = None
        self._screencast_cap: int | None = None
        self._dropped_frames = 0
        self._good_frames = 0
        #: Capture-area repairs and frames withheld, for tests and the log.
        self.repair_count = 0
        self.dropped_total = 0

        self._page = None
        self._cdp = None
        self._title = ""
        self._url = ""
        self._status = None  # None | "no_browser" | "open" | "closed"
        self._loading = False
        self._can_go_back = False
        self._can_go_forward = False
        self._main_frame_id = None
        self._last_state: dict | None = None
        self._cursor = "default"
        self._last_cursor_probe = 0.0
        #: The page's layout viewport (CSS px) as of the last look — the
        #: fallback for "the page's size" when neither the sizer nor
        #: Playwright knows it — and whether the last screencast frame
        #: matched the page. A frame that does not means the screencast's
        #: capture area is off — not the page.
        self._layout_size: tuple[int, int] | None = None
        self._last_frame_size: tuple[int, int] | None = None
        self._last_frame_ok = True
        self._repairing = False

        # Exactly one frame may be pending. A frame that arrives while the
        # socket is still writing the previous one replaces it, so a slow
        # client sees the newest page state rather than a growing backlog.
        self._pending_frame = None
        self._frame_ready = asyncio.Event()

        # Input queue — see the module docstring. Items are dicts: CDP
        # dispatches carry ``method``/``params``/``merge`` ("move" | "wheel"
        # | None); navigation carries ``nav``/``url``.
        self._input_queue: collections.deque = collections.deque()
        self._input_ready = asyncio.Event()

        #: True until the watcher's first look: only a viewer *opening* the
        #: view brings an idle-closed page back on its own (see the module
        #: docstring), never a view that watched the browser close.
        self._may_auto_reopen = True
        #: A reopen is queued or running: the watcher keeps its hands off
        #: the page it is creating (it is at about:blank until loaded).
        self._reopening = False
        self.reopen_count = 0

        self._closed = False
        self._send_lock = asyncio.Lock()
        # Serializes every attach/detach transition. The watcher polls for the
        # agent's page while a user `goto` can open one of its own; without
        # this the watcher could tear down the page the user just opened.
        self._attach_lock = asyncio.Lock()
        self._bg: set = set()
        self._input_counts: dict[str, int] = {}

    # -- socket writes -------------------------------------------------

    async def _send(self, payload: dict) -> None:
        if self._closed:
            return
        try:
            async with self._send_lock:
                await self._ws.send_json(payload)
        except Exception:
            # The client is gone (or going). Stop producing; the receive loop
            # will unwind and run teardown.
            self._closed = True
            self._frame_ready.set()
            self._input_ready.set()

    async def _send_error(self, message: str) -> None:
        await self._send({"type": "error", "message": message})

    def _state_payload(self) -> dict:
        return {
            "type": "state",
            "status": self._status,
            "title": self._title,
            "url": self._url,
            "loading": self._loading,
            "canGoBack": self._can_go_back,
            "canGoForward": self._can_go_forward,
        }

    async def _send_state(self) -> None:
        """Push the state row when anything in it changed since the last push."""
        if self._status is None:
            return
        payload = self._state_payload()
        if payload == self._last_state:
            return
        self._last_state = payload
        await self._send(payload)

    async def _set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        await self._send_state()

    def _spawn(self, coro) -> None:
        """Run ``coro`` in the background; awaited at teardown."""
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    # -- CDP attach / detach -------------------------------------------

    def _on_screencast_frame(self, params) -> None:
        """CDP event callback. Runs on the event loop, synchronously."""
        if not isinstance(params, dict):
            return
        data = params.get("data")
        if data:
            metadata = params.get("metadata") or {}
            width = int(metadata.get("deviceWidth") or 0)
            height = int(metadata.get("deviceHeight") or 0)
            self._last_frame_size = (width, height)
            expected = self._expected_size()
            ok = expected is None or (
                (width, height) == expected
                and _same_aspect(_jpeg_size(data), expected))
            self._last_frame_ok = ok
            if ok or self._dropped_frames >= MAX_DROPPED_FRAMES:
                self._dropped_frames = 0
                if ok:
                    self._good_frames += 1
                self._pending_frame = {
                    "type": "frame",
                    "jpeg": data,
                    "width": width,
                    "height": height,
                    "title": self._title,
                    "url": self._url,
                }
                self._frame_ready.set()
            else:
                # A picture of the capture area, not of the page: drawing it
                # would flash a wrong-sized frame. Keep the last good one on
                # screen and put the capture area back.
                self._dropped_frames += 1
                self.dropped_total += 1
            if not ok:
                self._spawn(self._repair_visible_size())
        # Ack every frame, including ones we drop — withholding the ack
        # stalls the stream rather than throttling it.
        session_id = params.get("sessionId")
        cdp = self._cdp
        if session_id is not None and cdp is not None:
            self._spawn(self._ack(cdp, session_id))

    async def _ack(self, cdp, session_id) -> None:
        try:
            await cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        except Exception:
            logger.debug("browser_live: screencastFrameAck failed", exc_info=True)

    # Chrome's Page events, so url / title / loading / history reach the
    # client when they change instead of on the 2 s poll. Sub-frames are
    # ignored: the address bar is about the main frame.

    def _is_main_frame(self, frame_id) -> bool:
        return self._main_frame_id is None or frame_id == self._main_frame_id

    def _on_frame_navigated(self, params) -> None:
        frame = (params or {}).get("frame") if isinstance(params, dict) else None
        if not isinstance(frame, dict) or frame.get("parentId"):
            return
        if frame.get("id"):
            self._main_frame_id = frame["id"]
        url = frame.get("url")
        if isinstance(url, str):
            self._url = url
        self._spawn(self._refresh_nav_state())

    def _on_navigated_within_document(self, params) -> None:
        if not isinstance(params, dict) or not self._is_main_frame(params.get("frameId")):
            return
        url = params.get("url")
        if isinstance(url, str):
            self._url = url
        self._spawn(self._refresh_nav_state())

    def _on_frame_started_loading(self, params) -> None:
        if not isinstance(params, dict) or not self._is_main_frame(params.get("frameId")):
            return
        self._loading = True
        self._spawn(self._send_state())

    def _on_frame_stopped_loading(self, params) -> None:
        if not isinstance(params, dict) or not self._is_main_frame(params.get("frameId")):
            return
        self._loading = False
        self._spawn(self._refresh_nav_state())

    async def _load_history(self, cdp) -> None:
        """Back/forward availability from Chrome's navigation history."""
        try:
            history = await cdp.send("Page.getNavigationHistory") or {}
        except Exception:
            logger.debug("browser_live: getNavigationHistory failed", exc_info=True)
            return
        entries = history.get("entries") or []
        index = history.get("currentIndex")
        if not isinstance(index, int):
            return
        self._can_go_back = index > 0
        self._can_go_forward = index < len(entries) - 1

    async def _refresh_nav_state(self) -> None:
        page, cdp = self._page, self._cdp
        if page is None or cdp is None:
            return
        self._title = await self._safe_title(page)
        self._url = self._page_url(page) or self._url
        await self._load_history(cdp)
        await self._send_state()

    async def _read_layout_size(self) -> tuple[int, int] | None:
        """The page's layout viewport in CSS px, or None when unreadable."""
        cdp = self._cdp
        if cdp is None:
            return None
        try:
            metrics = await cdp.send("Page.getLayoutMetrics") or {}
        except Exception:
            return None
        viewport = metrics.get("cssLayoutViewport") or {}
        width = int(viewport.get("clientWidth") or viewport.get("width") or 0)
        height = int(viewport.get("clientHeight") or viewport.get("height") or 0)
        return (width, height) if width and height else None

    def _expected_size(self) -> tuple[int, int] | None:
        """The page's viewport in CSS px — what a frame of it must measure.

        The sizer's applied (or in-flight) size when a viewer sized the page;
        otherwise Playwright's own viewport for the page; otherwise the
        layout size last read from Chrome. Never Playwright's cache alone
        while the sizer is mid-change, so a repair cannot pull the capture
        area back to the size being left."""
        if self._sizer is not None:
            current = self._sizer.current()
            if current is not None:
                return current
        own = _size_of(getattr(self._page, "viewport_size", None))
        return own or self._layout_size

    async def _repair_visible_size(self) -> None:
        """Put the screencast's capture area back to the page's own size,
        and make sure the client gets a frame of it.

        Headless Chrome captures the window's content area, which is 87 px
        shorter than the window ``set_viewport_size`` asks for, and the
        browser tool's ``fetch`` side tab resets the window — while the page
        itself is untouched (see the module docstring).
        ``Emulation.setVisibleSize`` resizes only the captured area, not the
        page's metrics; nothing here sizes the page.

        The window change behind an off-size area can land after our call,
        and a static page sends no frame by itself, so this waits for the
        screencast's own fresh frame and otherwise captures one — putting
        the area back again whenever the capture still shows it off.
        Bounded: ``REPAIR_PRIME_DELAYS_S`` steps, then the 2 s tick retries.
        """
        if self._repairing:
            return
        self._repairing = True
        try:
            cdp = self._cdp
            if cdp is None or self._last_frame_ok:
                return
            if self._sizer is not None and self._sizer.applying is not None:
                return  # the sizer re-checks once its change has landed
            if self._expected_size() is None or self._layout_size is None:
                layout = await self._read_layout_size()
                if layout is not None:
                    self._layout_size = layout
            expected = self._expected_size()
            if expected is None:
                return
            seen = self._good_frames
            if not await self._set_visible_size(cdp, expected):
                return
            # Assume it took; a frame that says otherwise re-arms the tick.
            self._last_frame_ok = True
            for delay in REPAIR_PRIME_DELAYS_S:
                await asyncio.sleep(delay)
                if self._closed or self._cdp is not cdp:
                    return
                if self._good_frames != seen or await self._prime_frame():
                    return
                expected = self._expected_size() or expected
                await self._set_visible_size(cdp, expected)
            self._last_frame_ok = False  # the tick tries again
        finally:
            self._repairing = False

    async def _set_visible_size(self, cdp, size: tuple[int, int]) -> bool:
        try:
            await cdp.send("Emulation.setVisibleSize",
                           {"width": size[0], "height": size[1]})
        except Exception:
            logger.debug("browser_live: setVisibleSize failed", exc_info=True)
            return False
        self.repair_count += 1
        logger.info("browser_live: screencast capture area set to %dx%d", *size)
        return True

    async def _check_capture_area(self) -> None:
        """Tick-time fallback for an off-size frame whose repair was skipped
        (it landed while the sizer was mid-change, or before the page's
        size was known)."""
        if self._expected_size() is None:
            layout = await self._read_layout_size()
            if layout is None:
                return
            self._layout_size = layout
        if not self._last_frame_ok:
            await self._repair_visible_size()

    async def _attach(self, page) -> None:
        cdp = await page.context.new_cdp_session(page)
        self._cdp = cdp
        self._page = page
        self._loading = False
        self._can_go_back = False
        self._can_go_forward = False
        self._main_frame_id = None
        self._cursor = "default"
        self._layout_size = None
        self._last_frame_size = None
        self._last_frame_ok = True
        self._dropped_frames = 0
        cdp.on("Page.screencastFrame", self._on_screencast_frame)
        cdp.on("Page.frameNavigated", self._on_frame_navigated)
        cdp.on("Page.navigatedWithinDocument", self._on_navigated_within_document)
        cdp.on("Page.frameStartedLoading", self._on_frame_started_loading)
        cdp.on("Page.frameStoppedLoading", self._on_frame_stopped_loading)
        # Page events need the domain enabled on *this* session (the
        # screencast does not). Best effort: without it the poll still works.
        try:
            await cdp.send("Page.enable")
            tree = await cdp.send("Page.getFrameTree") or {}
            frame = (tree.get("frameTree") or {}).get("frame") or {}
            if frame.get("id"):
                self._main_frame_id = frame["id"]
        except Exception:
            logger.debug("browser_live: Page.enable/getFrameTree failed", exc_info=True)
        self._layout_size = await self._read_layout_size()
        await self._start_screencast(cdp)
        await self._load_history(cdp)
        # Size the page to this viewer (attaching counts as activity).
        self._sizer = _sizer_for(page)
        stamp = next(_activity) if self._wanted else 0
        self._sizer.viewers[self] = (self._wanted, stamp)
        await self._sizer.settle()

    async def _start_screencast(self, cdp) -> None:
        params = dict(SCREENCAST_PARAMS)
        cap = _frame_cap(self._dpr)
        params["maxWidth"] = params["maxHeight"] = cap
        await cdp.send("Page.startScreencast", params)
        self._screencast_cap = cap
        logger.info("browser_live: screencast started (frames up to %dpx)", cap)

    async def _leave_sizer(self) -> None:
        """Stop counting as a viewer of the current page; the page goes back
        to the next viewer's size, or home when none is left."""
        sizer, self._sizer = self._sizer, None
        self._last_viewport_msg = None
        if sizer is None:
            return
        sizer.viewers.pop(self, None)
        await sizer.settle()
        if not sizer.viewers and _SIZERS.get(id(sizer.page)) is sizer:
            del _SIZERS[id(sizer.page)]

    async def _on_page_sized(self, sizer: _PageSizer, changed: bool) -> None:
        """Called by the sizer after it settled: tell the client the page's
        size and whether it is this viewer's, and — when the size changed —
        put the capture area to it and get the client a frame at it (Chrome
        need not send one for a resize of a static page)."""
        if sizer is not self._sizer:
            return
        size = sizer.current()
        if size is not None:
            payload = {"type": "viewport", "width": size[0], "height": size[1],
                       "owner": sizer.owner() is self}
            if payload != self._last_viewport_msg:
                self._last_viewport_msg = payload
                await self._send(payload)
        if changed:
            self._last_frame_ok = False
        if not self._last_frame_ok:
            self._spawn(self._repair_visible_size())

    async def _handle_viewport(self, msg: dict) -> None:
        """Queue a viewer's new size behind the input it sent before it, so
        a click aimed at the old layout lands before the page changes.
        Consecutive sizes merge (newest wins). Not user input: it neither
        stamps browser use nor enters the forensic log."""
        size = _clamp_view_size(msg.get("width"), msg.get("height"))
        if size is None:
            await self._send_error("viewport needs a numeric width and height")
            return
        dpr = _number(msg.get("dpr"))
        item = {"viewport": size, "dpr": dpr if dpr and dpr > 0 else None,
                "merge": "viewport"}
        queue = self._input_queue
        if queue and queue[-1].get("merge") == "viewport":
            queue[-1] = item
        else:
            queue.append(item)
        self._input_ready.set()

    async def _apply_viewport(self, size: tuple[int, int], dpr) -> None:
        if dpr is not None:
            self._dpr = dpr
            cdp = self._cdp
            if (cdp is not None and self._screencast_cap is not None
                    and self._screencast_cap != _frame_cap(dpr)):
                # The viewer moved to a display of another density — the
                # only thing that restarts the screencast. Resizing never does.
                # A cap of None means this session's screencast has not
                # started yet (``_attach`` is still on its way there and will
                # start it at ``self._dpr``): starting it here would make the
                # attach's own start fail with "Screencast is already active".
                try:
                    await cdp.send("Page.stopScreencast")
                    await self._start_screencast(cdp)
                except Exception:
                    logger.debug("browser_live: screencast restart failed",
                                 exc_info=True)
        self._wanted = size
        sizer = self._sizer
        if sizer is None:
            return  # applied when a page is attached
        sizer.viewers[self] = (size, next(_activity))
        await sizer.settle()

    async def _viewport_size(self, cdp) -> tuple[int, int]:
        """The page viewport in CSS pixels — the frame coordinate space.

        Probed 2026-09-03: ``cssLayoutViewport`` is 1280x720 on the production
        launch config, byte-identical to the screencast metadata's
        ``deviceWidth``/``deviceHeight``, and neither moves when the page is
        scrolled. So a primed frame and a screencast frame describe the same
        space and the client needs no special case.
        """
        try:
            metrics = await cdp.send("Page.getLayoutMetrics") or {}
        except Exception:
            return 0, 0
        viewport = metrics.get("cssVisualViewport") or metrics.get(
            "cssLayoutViewport"
        ) or {}
        return (
            int(viewport.get("clientWidth") or viewport.get("width") or 0),
            int(viewport.get("clientHeight") or viewport.get("height") or 0),
        )

    async def _prime_frame(self) -> bool:
        """Paint the client immediately with one captured screenshot.

        Chrome only emits a screencast frame when something on the page
        changes (probed 2026-09-03: a static page can sit silent after
        ``startScreencast``). Without this the panel would show nothing until
        the agent or the user moved something. Queued through the same
        pending slot as a real frame, so a screencast frame that lands first
        simply wins.
        """
        cdp = self._cdp
        if cdp is None:
            return False
        # Captured the way the screencast captures — whole viewport, device
        # pixels, no ``clip``: a clipped capture makes Chrome rewrite this
        # session's emulation for the shot, which is how the 2026-09-05 size
        # fight was armed.
        params: dict = {"format": "jpeg", "quality": 60}
        try:
            shot = await cdp.send("Page.captureScreenshot", params) or {}
        except Exception:
            logger.debug("browser_live: initial captureScreenshot failed",
                         exc_info=True)
            return False
        data = shot.get("data")
        if not data:
            return False
        expected = self._expected_size()
        if expected is not None:
            if not _same_aspect(_jpeg_size(data), expected):
                # The capture area is still off (a repair not landed yet):
                # the screencast's next frame will do.
                return False
            width, height = expected
        else:
            width, height = await self._viewport_size(cdp)
        self._pending_frame = {
            "type": "frame",
            "jpeg": data,
            "width": width,
            "height": height,
            "title": self._title,
            "url": self._url,
        }
        self._frame_ready.set()
        return True

    async def _detach(self) -> None:
        await self._leave_sizer()
        cdp, self._cdp = self._cdp, None
        self._page = None
        self._screencast_cap = None
        self._pending_frame = None
        self._loading = False
        self._can_go_back = False
        self._can_go_forward = False
        self._main_frame_id = None
        if cdp is None:
            return
        try:
            await cdp.send("Page.stopScreencast")
        except Exception:
            logger.debug("browser_live: stopScreencast failed", exc_info=True)
        try:
            await cdp.detach()
        except Exception:
            logger.debug("browser_live: cdp detach failed", exc_info=True)

    # -- page discovery -------------------------------------------------

    async def _current_page(self):
        """The viewer's session's last open page (spec 155 §5.4: sessions run
        in parallel, each with its own pages); without a session, the
        project's most recent page. None when there is none. Never creates
        one — the panel watches the agent's browser, it does not start one.
        The single exception is an explicit user gesture: typing an address
        (see ``_open_page``)."""
        if self._bm is None:
            return None
        try:
            view_pages = getattr(self._bm, "get_view_pages", None)
            if view_pages is not None:
                pages = await view_pages(self._project_id, self._session_id)
            else:
                pages = await self._bm.get_all_pages(self._project_id)
        except Exception:
            logger.debug("browser_live: get_all_pages failed", exc_info=True)
            return None
        for page in reversed(list(pages or [])):
            try:
                if not page.is_closed():
                    return page
            except Exception:
                continue
        return None

    async def _safe_title(self, page) -> str:
        try:
            return await page.title() or ""
        except Exception:
            return self._title

    def _page_url(self, page) -> str:
        """The page's URL (a sync property in Playwright); '' when unreadable.
        The client uses it to tell a blank page from a real one."""
        try:
            value = page.url
            return value if isinstance(value, str) else ""
        except Exception:
            return self._url

    # -- background tasks -----------------------------------------------

    async def _sender(self) -> None:
        while not self._closed:
            await self._frame_ready.wait()
            self._frame_ready.clear()
            frame, self._pending_frame = self._pending_frame, None
            if frame is None:
                continue
            await self._send(frame)

    async def _attach_to(self, page) -> bool:
        """Attach to ``page``, detaching from any previous one. False on failure."""
        async with self._attach_lock:
            if page is self._page:
                return True
            await self._detach()
            self._title = await self._safe_title(page)
            self._url = self._page_url(page)
            try:
                await self._attach(page)
            except Exception as exc:
                await self._send_error(
                    f"could not attach to the browser page: {exc}"
                )
                return False
            await self._set_status("open")
            # Every attach — the first one and every re-attach after
            # no_browser/closed — primes the client with one frame.
            await self._prime_frame()
            return True

    async def _watcher(self) -> None:
        while not self._closed:
            if self._reopening:
                await asyncio.sleep(POLL_INTERVAL_S)
                continue
            page = await self._current_page()
            first_look, self._may_auto_reopen = self._may_auto_reopen, False
            if page is None and first_look and self._queue_auto_reopen():
                await asyncio.sleep(POLL_INTERVAL_S)
                continue
            if page is not None:
                if page is not self._page:
                    if not await self._attach_to(page):
                        await asyncio.sleep(POLL_INTERVAL_S)
                        continue
                else:
                    self._title = await self._safe_title(page)
                    self._url = self._page_url(page)
                    await self._send_state()
                    await self._check_capture_area()
            else:
                async with self._attach_lock:
                    # Re-read under the lock: a user `goto` may have opened and
                    # attached a page while this tick was looking at a pool
                    # that was still empty, and dropping it here would close
                    # the live view the user just asked for.
                    if await self._current_page() is None:
                        if self._page is not None:
                            await self._detach()
                        # "closed" only once we have actually shown a page;
                        # before that the honest answer is no browser.
                        await self._set_status(
                            "closed" if self._status == "open" else "no_browser"
                        )
            await self._flush_input_log()
            await asyncio.sleep(POLL_INTERVAL_S)

    # -- reopen after the idle close -------------------------------------

    def _idle_closed_page(self):
        """``(scope, url)`` the browser manager remembers for this viewer, or
        None (older managers and fakes have no such call)."""
        lookup = getattr(self._bm, "idle_closed_page", None)
        if not callable(lookup):
            return None
        try:
            found = lookup(self._project_id, self._session_id)
        except Exception:
            logger.debug("browser_live: idle_closed_page failed", exc_info=True)
            return None
        if (isinstance(found, tuple) and len(found) == 2
                and all(isinstance(v, str) and v for v in found)):
            return found
        return None

    def _queue_auto_reopen(self) -> bool:
        """The viewer just opened the view and the browser is not running:
        queue a reopen of the idle-closed page when there is one. Queued
        behind (and serialised with) the user's input, but not counted as
        user input in the forensic log."""
        if self._idle_closed_page() is None:
            return False
        self._reopening = True
        self._input_queue.append({"nav": "reopen", "url": None, "merge": None})
        self._input_ready.set()
        return True

    async def _reopen(self) -> bool:
        """Bring the idle-closed page back and attach to it. False when
        there is nothing to reopen (a page is attached already, the manager
        remembers none) or the browser could not be started."""
        try:
            if self._cdp is not None or self._closed:
                return False
            found = self._idle_closed_page()
            if found is None:
                return False
            self._reopening = True
            scope, url = found
            started = time.monotonic()
            # A use: the reopened browser gets a fresh idle period.
            note_use = getattr(self._bm, "note_use", None)
            if callable(note_use):
                note_use()
            ok = await self._open_page(scope=scope, url=url, quiet=True)
            if ok:
                self.reopen_count += 1
                logger.info("browser_live: reopened the idle-closed page in %.1fs",
                            time.monotonic() - started)
            return ok
        finally:
            self._reopening = False
            if self._cdp is None and not self._closed:
                await self._set_status(
                    "closed" if self._status == "open" else "no_browser")

    def _absorb_queued_viewport(self) -> None:
        """Take the newest size the client has queued (it is applied again,
        as a no-op, when the pump reaches it). The reopen runs ahead of a
        size that arrived while Chrome was starting."""
        for item in reversed(self._input_queue):
            if "viewport" in item:
                self._wanted = item["viewport"]
                if item.get("dpr"):
                    self._dpr = item["dpr"]
                return

    async def _presize(self, page) -> None:
        """Give a page about to be loaded this viewer's size first, through
        its sizer (the one owner of its size), so it loads at the size it
        will be watched at."""
        self._absorb_queued_viewport()
        if self._wanted is None:
            return
        sizer = _sizer_for(page)
        sizer.viewers[self] = (self._wanted, next(_activity))
        await sizer.settle()

    def _unpresize(self, page) -> None:
        sizer = _SIZERS.get(id(page))
        if sizer is None or sizer is self._sizer:
            return
        sizer.viewers.pop(self, None)
        if not sizer.viewers:
            del _SIZERS[id(page)]

    async def _load(self, page, url: str) -> None:
        """Navigate a page that is not attached yet. Best effort: a slow or
        failing page is attached as it is."""
        try:
            await page.goto(url, wait_until="domcontentloaded",
                            timeout=REOPEN_DOM_TIMEOUT_MS)
        except Exception:
            logger.info("browser_live: reopen navigation did not finish",
                        exc_info=True)
            return
        try:
            await page.wait_for_load_state("load", timeout=REOPEN_LOAD_TIMEOUT_MS)
        except Exception:
            pass

    # -- forensic input log ---------------------------------------------

    def _count_input(self, kind: str) -> None:
        self._input_counts[kind] = self._input_counts.get(kind, 0) + 1

    def _append_input_meta(self, counts: dict) -> None:
        session_id, agent_manager = self._session_id, self._am
        if not session_id or agent_manager is None:
            return
        try:
            _resolved, session = agent_manager.peek_chat_session(
                self._project_id, session_id
            )
            if session is None:
                return
            session.append_meta(
                "browser_user_input",
                counts=counts,
                total=sum(counts.values()),
            )
        except Exception:
            # Forensic only — never let it disturb the stream. The session
            # file lock is zero-retry, so a live turn can legitimately win.
            logger.debug("browser_live: input meta append failed", exc_info=True)

    async def _flush_input_log(self) -> None:
        if not self._input_counts:
            return
        counts = dict(self._input_counts)
        self._input_counts.clear()
        await asyncio.to_thread(self._append_input_meta, counts)

    # -- input in --------------------------------------------------------

    async def _cdp_send(self, method: str, params: dict) -> bool:
        ok, _result = await self._cdp_call(method, params)
        return ok

    async def _cdp_call(self, method: str, params: dict):
        cdp = self._cdp
        if cdp is None:
            await self._send_error("no browser page is attached")
            return False, None
        try:
            return True, await cdp.send(method, params)
        except Exception as exc:
            await self._send_error(f"input was not delivered: {exc}")
            return False, None

    def _enqueue(self, item: dict, count_key: str) -> None:
        """Queue one input item, merging into the newest queued item when
        both are moves (newest wins) or both are wheels (deltas add up)."""
        self._count_input(count_key)
        # Spec 175: user input keeps the agent browser from its idle close.
        note_use = getattr(self._bm, "note_use", None)
        if callable(note_use):
            note_use()
        merge = item.get("merge")
        queue = self._input_queue
        if merge and queue and queue[-1].get("merge") == merge:
            last = queue[-1]
            if merge == "wheel":
                item["params"]["deltaX"] += last["params"]["deltaX"]
                item["params"]["deltaY"] += last["params"]["deltaY"]
            queue[-1] = item
        else:
            queue.append(item)
        self._input_ready.set()

    async def _input_pump(self) -> None:
        while not self._closed:
            await self._input_ready.wait()
            self._input_ready.clear()
            while self._input_queue and not self._closed:
                item = self._input_queue.popleft()
                try:
                    await self._run_input(item)
                except Exception:
                    logger.debug("browser_live: input item failed", exc_info=True)

    async def _run_input(self, item: dict) -> None:
        if "viewport" in item:
            await self._apply_viewport(item["viewport"], item.get("dpr"))
            return
        if "nav" in item:
            await self._run_nav(item["nav"], item.get("url"))
            return
        params = item["params"]
        if not await self._cdp_send(item["method"], params):
            return
        if params.get("type") == "mouseMoved":
            await self._maybe_probe_cursor(params["x"], params["y"])

    def _move_pending(self) -> bool:
        return any(i.get("merge") == "move" for i in self._input_queue)

    async def _maybe_probe_cursor(self, x: float, y: float) -> None:
        """Mirror the page's cursor for the point the pointer came to rest at."""
        if self._move_pending():
            return
        now = time.monotonic()
        if now - self._last_cursor_probe < CURSOR_PROBE_MIN_INTERVAL_S:
            return
        self._last_cursor_probe = now
        cdp = self._cdp
        if cdp is None:
            return
        try:
            result = await cdp.send("Runtime.evaluate", {
                "expression": _CURSOR_JS % (float(x), float(y)),
                "returnByValue": True,
            }) or {}
        except Exception:
            logger.debug("browser_live: cursor probe failed", exc_info=True)
            return
        cursor = ((result.get("result") or {}).get("value"))
        if not isinstance(cursor, str) or not cursor:
            return
        if cursor != self._cursor:
            self._cursor = cursor
            await self._send({"type": "cursor", "cursor": cursor})

    async def _run_nav(self, action: str, url) -> None:
        if action in ("back", "forward"):
            ok, history = await self._cdp_call("Page.getNavigationHistory", {})
            if not ok:
                return
            history = history or {}
            entries = history.get("entries") or []
            index = history.get("currentIndex")
            if not isinstance(index, int):
                return
            target = index - 1 if action == "back" else index + 1
            if not (0 <= target < len(entries)):
                return  # nothing there; the client's button was stale
            entry_id = (entries[target] or {}).get("id")
            if entry_id is None:
                return
            await self._cdp_send("Page.navigateToHistoryEntry", {"entryId": entry_id})
        elif action == "reload":
            await self._cdp_send("Page.reload", {})
        elif action == "stop":
            await self._cdp_send("Page.stopLoading", {})
        elif action == "reopen":
            await self._reopen()
            return
        elif action == "goto":
            if self._cdp is None:
                # Nothing is open yet. The address field is live in that state
                # (the panel's header is always there), and typing an address
                # is an explicit request for a page — so make one, through the
                # browser manager's own per-project pool, which is where the
                # agent's browser tool will find it on its next call.
                if not await self._open_page():
                    return
            await self._cdp_send("Page.navigate", {"url": url})
        # Chrome's Page events carry the rest (url, loading, history); this
        # covers a session where they are unavailable.
        await self._refresh_nav_state()

    async def _open_page(self, scope: str | None = None, url: str | None = None,
                         quiet: bool = False) -> bool:
        """Open (and attach to) a page for this project. False on failure.

        Called for a typed address and for a reopen after the idle close
        (``url``: loaded before attaching, at this viewer's size, so the
        client never sees the blank page in between). The watcher still
        never creates a browser on its own. ``quiet`` keeps failures off the
        socket (the client would show an error for something it never asked).
        """
        async def fail(message: str) -> bool:
            if quiet:
                logger.info("browser_live: %s", message)
            else:
                await self._send_error(message)
            return False

        if self._bm is None:
            return await fail("no browser is available")
        try:
            # The viewer's session's scope, so the agent of that session
            # continues on the page the user opened.
            from agent_os.daemon_v2.browser_manager import page_scope
            page = await self._bm.get_page(
                scope or page_scope(self._project_id, self._session_id))
        except Exception as exc:
            return await fail(f"could not open a browser page: {exc}")
        if page is None:
            return await fail("could not open a browser page")
        if url:
            await self._presize(page)
            await self._load(page, url)
        ok = await self._attach_to(page)
        if not ok:
            self._unpresize(page)
        return ok

    async def _handle_mouse(self, msg: dict) -> None:
        action = msg.get("action")
        cdp_type = _MOUSE_TYPES.get(action)
        if cdp_type is None:
            await self._send_error(f"unknown mouse action: {action!r}")
            return
        x, y = _number(msg.get("x")), _number(msg.get("y"))
        if x is None or y is None:
            await self._send_error("mouse event needs numeric x and y")
            return
        button = msg.get("button")
        if button is None:
            button = "left" if action in ("down", "up") else "none"
        if button not in _BUTTONS:
            await self._send_error(f"unknown mouse button: {button!r}")
            return
        params = {
            "type": cdp_type,
            "x": x,
            "y": y,
            "button": button,
            "modifiers": int(_number(msg.get("modifiers")) or 0),
        }
        if action in ("down", "up"):
            params["clickCount"] = int(_number(msg.get("clickCount")) or 1)
        if action == "wheel":
            params["deltaX"] = _number(msg.get("deltaX")) or 0.0
            params["deltaY"] = _number(msg.get("deltaY")) or 0.0
        merge = action if action in ("move", "wheel") else None
        self._enqueue(
            {"method": "Input.dispatchMouseEvent", "params": params, "merge": merge},
            f"mouse.{action}",
        )

    async def _handle_key(self, msg: dict) -> None:
        action = msg.get("action")
        if action not in ("down", "up"):
            await self._send_error(f"unknown key action: {action!r}")
            return
        key = msg.get("key")
        if not isinstance(key, str) or not key:
            await self._send_error("key event needs a non-empty key")
            return
        code = msg.get("code")
        params = {
            "type": "keyDown" if action == "down" else "keyUp",
            "key": key,
            "code": code if isinstance(code, str) else "",
            "modifiers": int(_number(msg.get("modifiers")) or 0),
        }
        text = msg.get("text")
        if action == "down" and isinstance(text, str) and text:
            params["text"] = text
        virtual_key = _virtual_key_code(key)
        if virtual_key is not None:
            params["windowsVirtualKeyCode"] = virtual_key
            params["nativeVirtualKeyCode"] = virtual_key
        self._enqueue(
            {"method": "Input.dispatchKeyEvent", "params": params, "merge": None},
            f"key.{action}",
        )

    async def _handle_text(self, msg: dict) -> None:
        text = msg.get("text")
        if not isinstance(text, str):
            await self._send_error("text event needs a string text")
            return
        if not text:
            return
        self._enqueue(
            {"method": "Input.insertText", "params": {"text": text}, "merge": None},
            "text",
        )

    async def _handle_nav(self, msg: dict) -> None:
        action = msg.get("action")
        if action not in _NAV_ACTIONS:
            await self._send_error(f"unknown nav action: {action!r}")
            return
        url = None
        if action == "reopen" and self._cdp is not None:
            return  # live already: a stale click, not a use
        if action == "goto":
            url = _nav_url(msg.get("url"))
            if url is None:
                await self._send_error("goto needs an http(s) URL")
                return
        self._enqueue({"nav": action, "url": url}, f"nav.{action}")

    async def _handle(self, msg) -> None:
        if not isinstance(msg, dict):
            await self._send_error("expected a JSON object")
            return
        msg_type = msg.get("type")
        if msg_type == "mouse":
            await self._handle_mouse(msg)
        elif msg_type == "key":
            await self._handle_key(msg)
        elif msg_type == "text":
            await self._handle_text(msg)
        elif msg_type == "nav":
            await self._handle_nav(msg)
        elif msg_type == "viewport":
            await self._handle_viewport(msg)
        elif msg_type in ("ping", "pong"):
            pass
        else:
            await self._send_error(f"unknown message type: {msg_type!r}")

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        sender = asyncio.ensure_future(self._sender())
        watcher = asyncio.ensure_future(self._watcher())
        pump = asyncio.ensure_future(self._input_pump())
        try:
            while not self._closed:
                try:
                    msg = await self._ws.receive_json()
                except WebSocketDisconnect:
                    break
                except (ValueError, TypeError):
                    await self._send_error("malformed message: not valid JSON")
                    continue
                await self._handle(msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("browser_live: receive loop ended", exc_info=True)
        finally:
            self._closed = True
            self._frame_ready.set()
            self._input_ready.set()
            for task in (sender, watcher, pump):
                task.cancel()
            await asyncio.gather(sender, watcher, pump, return_exceptions=True)
            if self._bg:
                await asyncio.gather(*list(self._bg), return_exceptions=True)
            await self._detach()
            await self._flush_input_log()


async def serve(websocket, project_id: str, session_id: str | None = None, *,
                browser_manager=None, agent_manager=None, dpr=None) -> None:
    """Run one live-view session on an already-accepted socket.

    Split out from the route so tests can drive it with fakes. ``dpr`` is
    the viewer's device-pixel ratio as of connecting (it sizes the
    screencast's frames before the first ``viewport`` message arrives).
    """
    stream = _LiveStream(
        websocket,
        project_id,
        session_id,
        browser_manager if browser_manager is not None else _browser_manager,
        agent_manager if agent_manager is not None else _agent_manager,
        dpr=dpr,
    )
    await stream.run()


@router.websocket("/agents/{project_id}/browser/live")
async def browser_live(websocket: WebSocket, project_id: str,
                       session_id: str | None = None,
                       token: str | None = None,
                       dpr: float | None = None):
    """Live view of the project's browser page.

    Auth mirrors the daemon's existing ``/ws`` endpoint exactly: the socket is
    accepted unconditionally. The daemon binds loopback and the relay
    terminates device auth upstream, so there is no socket-layer check to
    mirror. ``token`` is accepted so the client can send the relay JWT on the
    query string the way the rest of the surface does, and is deliberately
    not inspected here — introducing a check on this route alone would be a
    new scheme, which §5.6 does not ask for.
    """
    await websocket.accept()
    await serve(websocket, project_id, session_id, dpr=dpr)
