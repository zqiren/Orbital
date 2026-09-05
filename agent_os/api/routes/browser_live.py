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

    client → server
      {"type":"mouse","action":"move"|"down"|"up"|"wheel","x":n,"y":n,
       "button":"left"|"right"|"middle","clickCount":1,"deltaX":n,"deltaY":n,"modifiers":n}
      {"type":"key","action":"down"|"up","key":"Enter","code":"Enter","text":"a","modifiers":n}
      {"type":"text","text":"pasted or IME-composed text"}
      {"type":"nav","action":"back"|"forward"|"reload"|"stop"|"goto","url":"https://…"}

``x``/``y`` are CSS pixels of the page viewport — the same space as a frame's
``width``/``height`` (taken from the screencast metadata's
``deviceWidth``/``deviceHeight``). The JPEG itself may be downscaled by
Chrome; the client scales its canvas coordinates into this space.

**Input never queues up** (2026-09-04, "scrolling feels very laggy"): a
wheel event takes ~20 ms for Chrome to confirm because it scrolls and renders
before answering, while a trackpad emits 120 of them a second — handling them
one at a time turned a two-second swipe into five seconds of scrolling. So
input goes through a queue that merges: consecutive moves keep only the
newest, consecutive wheels sum their deltas, and clicks / keys / text / nav
keep their order relative to everything else. At most one event is ever in
flight, and the backlog can never exceed one merged event of each kind.

**The panel is a mirror** (2026-09-05, "the browser is blinking"): this route
never sizes the page. An earlier amendment had it resize the agent's page to
the panel so the live view would fill it; that made the agent browse a
phone-width layout (and take phone-sized screenshots) whenever someone was
watching, and it was the root of a size fight — Chrome performs a clipped
``Page.captureScreenshot`` by rewriting the calling session's emulation and
restoring that session's baseline, the side tab the browser tool's ``fetch``
opens and closes re-synced the page to it, and Playwright ignores a same-size
re-apply — that strobed the panel between a clipped capture and the page's
real frames. The agent's page keeps its own size (the context default);
frames arrive at that size and the client fits them into the panel. What the
user does in the panel still lands in the page: input is forwarded, only the
page's size is off limits.

**The capture area is repaired, never the page:** the screencast captures the
page's visible area, and the side tab the browser tool's ``fetch`` opens and
closes leaves that area at the window's content size — 87 px shorter than
the page — while the page's layout, its visual viewport and the tool's own
screenshots are untouched (probed 2026-09-05; restarting the screencast does
not help). A frame that disagrees with the page's layout size triggers
``Emulation.setVisibleSize`` back to that size, which touches no device
metrics and nothing the agent sees.

**No coordination layer** (D9): there is no take-over button and no pause
flag. The user's input and the agent's tool calls simply interleave — the
browser tool snapshots the page before each action, so it sees what the user
did. The only thing guarded here is the page/CDP session going away.
"""

import asyncio
import collections
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

#: Screencast parameters. ~10 fps in practice; quality/size chosen so a frame
#: is small enough to survive the relay hop to a phone.
SCREENCAST_PARAMS = {
    "format": "jpeg",
    "quality": 60,
    "maxWidth": 1280,
    "maxHeight": 1280,
    "everyNthFrame": 1,
}

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

_NAV_ACTIONS = {"back", "forward", "reload", "stop", "goto"}

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
                 browser_manager, agent_manager):
        self._ws = websocket
        self._project_id = project_id
        self._session_id = session_id
        self._bm = browser_manager
        self._am = agent_manager

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
        #: The page's layout viewport (CSS px) as of the last look, and the
        #: size the last screencast frame claimed. When they disagree the
        #: screencast's capture area has shrunk — not the page.
        self._layout_size: tuple[int, int] | None = None
        self._last_frame_size: tuple[int, int] | None = None
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

        self._closed = False
        self._send_lock = asyncio.Lock()
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
            self._pending_frame = {
                "type": "frame",
                "jpeg": data,
                "width": width,
                "height": height,
                "title": self._title,
                "url": self._url,
            }
            self._frame_ready.set()
            self._last_frame_size = (width, height)
            if self._layout_size and (width, height) != self._layout_size:
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

    async def _repair_visible_size(self) -> None:
        """Put the screencast's capture area back to the page's own size.

        The browser tool's ``fetch`` side tab leaves it at the window's
        content size (87 px short) while the page is untouched — see the
        module docstring. ``Emulation.setVisibleSize`` resizes only the
        captured frame, not the page's metrics; nothing here sizes the page.
        """
        if self._repairing:
            return
        self._repairing = True
        try:
            cdp = self._cdp
            if cdp is None:
                return
            layout = await self._read_layout_size()
            if layout is None:
                return
            self._layout_size = layout
            if self._last_frame_size == layout:
                return
            try:
                await cdp.send("Emulation.setVisibleSize",
                               {"width": layout[0], "height": layout[1]})
            except Exception:
                logger.debug("browser_live: setVisibleSize failed", exc_info=True)
                return
            logger.info("browser_live: screencast capture area repaired to %dx%d",
                        layout[0], layout[1])
            self._last_frame_size = layout
            await self._prime_frame()
        finally:
            self._repairing = False

    async def _check_capture_area(self) -> None:
        """Tick-time fallback for a cropped frame that landed before the
        layout size was known."""
        layout = await self._read_layout_size()
        if layout is None:
            return
        self._layout_size = layout
        if self._last_frame_size and self._last_frame_size != layout:
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
        await cdp.send("Page.startScreencast", dict(SCREENCAST_PARAMS))
        await self._load_history(cdp)

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

    async def _prime_frame(self) -> None:
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
            return
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
            return
        data = shot.get("data")
        if not data:
            return
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

    async def _detach(self) -> None:
        cdp, self._cdp = self._cdp, None
        self._page = None
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
        """The project's last open page, or None. Never creates one."""
        if self._bm is None:
            return None
        try:
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

    async def _watcher(self) -> None:
        while not self._closed:
            page = await self._current_page()
            if page is not None:
                if page is not self._page:
                    await self._detach()
                    self._title = await self._safe_title(page)
                    self._url = self._page_url(page)
                    try:
                        await self._attach(page)
                    except Exception as exc:
                        await self._send_error(
                            f"could not attach to the browser page: {exc}"
                        )
                        await asyncio.sleep(POLL_INTERVAL_S)
                        continue
                    await self._set_status("open")
                    # Every attach — the first one and every re-attach after
                    # no_browser/closed — primes the client with one frame.
                    await self._prime_frame()
                else:
                    self._title = await self._safe_title(page)
                    self._url = self._page_url(page)
                    await self._send_state()
                    await self._check_capture_area()
            else:
                if self._page is not None:
                    await self._detach()
                # "closed" only once we have actually shown a page; before
                # that the honest answer is that there is no browser.
                await self._set_status(
                    "closed" if self._status == "open" else "no_browser"
                )
            await self._flush_input_log()
            await asyncio.sleep(POLL_INTERVAL_S)

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
        elif action == "goto":
            await self._cdp_send("Page.navigate", {"url": url})
        # Chrome's Page events carry the rest (url, loading, history); this
        # covers a session where they are unavailable.
        await self._refresh_nav_state()

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
                browser_manager=None, agent_manager=None) -> None:
    """Run one live-view session on an already-accepted socket.

    Split out from the route so tests can drive it with fakes.
    """
    stream = _LiveStream(
        websocket,
        project_id,
        session_id,
        browser_manager if browser_manager is not None else _browser_manager,
        agent_manager if agent_manager is not None else _agent_manager,
    )
    await stream.run()


@router.websocket("/agents/{project_id}/browser/live")
async def browser_live(websocket: WebSocket, project_id: str,
                       session_id: str | None = None,
                       token: str | None = None):
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
    await serve(websocket, project_id, session_id)
