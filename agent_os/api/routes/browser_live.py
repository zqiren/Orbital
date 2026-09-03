# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live view of the agent's browser (spec 078 §5.6, D9).

One WebSocket route attaches a CDP session to the project's *current* page,
streams ``Page.startScreencast`` JPEG frames to the client, and forwards the
client's mouse/keyboard events back into the page via ``Input.*``.

Protocol (shared contract — mirrored in ``web/src/hooks/useBrowserLive.ts``)::

    WS  /api/v2/agents/{project_id}/browser/live[?session_id=…&token=…]

    server → client
      {"type":"frame","jpeg":"<base64>","width":<css px>,"height":<css px>,"title":"…"}
      {"type":"state","status":"no_browser"|"open"|"closed","title":"…"}
      {"type":"error","message":"…"}

    client → server
      {"type":"mouse","action":"move"|"down"|"up"|"wheel","x":n,"y":n,
       "button":"left"|"right"|"middle","clickCount":1,"deltaX":n,"deltaY":n,"modifiers":n}
      {"type":"key","action":"down"|"up","key":"Enter","code":"Enter","text":"a","modifiers":n}
      {"type":"text","text":"pasted text"}

``x``/``y`` are CSS pixels of the page viewport — the same space as a frame's
``width``/``height`` (taken from the screencast metadata's
``deviceWidth``/``deviceHeight``). The JPEG itself may be downscaled by
Chrome; the client scales its canvas coordinates into this space.

**No coordination layer** (D9): there is no take-over button and no pause
flag. The user's input and the agent's tool calls simply interleave — the
browser tool snapshots the page before each action, so it sees what the user
did. The only thing guarded here is the page/CDP session going away.
"""

import asyncio
import logging

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
#: which the page title is refreshed and the input log is flushed.
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

# ---- Input mapping ----

_MOUSE_TYPES = {
    "move": "mouseMoved",
    "down": "mousePressed",
    "up": "mouseReleased",
    "wheel": "mouseWheel",
}

_BUTTONS = {"none", "left", "middle", "right", "back", "forward"}

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


class _LiveStream:
    """One connected client, watching one project's browser.

    Three concurrent pieces: the caller's receive loop (input in), a sender
    task (frames out, coalesced), and a watcher task that attaches/detaches
    as the project's page appears and disappears.
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
        self._status = None  # None | "no_browser" | "open" | "closed"

        # Exactly one frame may be pending. A frame that arrives while the
        # socket is still writing the previous one replaces it, so a slow
        # client sees the newest page state rather than a growing backlog.
        self._pending_frame = None
        self._frame_ready = asyncio.Event()

        self._closed = False
        self._send_lock = asyncio.Lock()
        self._acks: set = set()
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

    async def _send_error(self, message: str) -> None:
        await self._send({"type": "error", "message": message})

    async def _set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        await self._send({"type": "state", "status": status, "title": self._title})

    # -- CDP attach / detach -------------------------------------------

    def _on_screencast_frame(self, params) -> None:
        """CDP event callback. Runs on the event loop, synchronously."""
        if not isinstance(params, dict):
            return
        data = params.get("data")
        if data:
            metadata = params.get("metadata") or {}
            self._pending_frame = {
                "type": "frame",
                "jpeg": data,
                "width": int(metadata.get("deviceWidth") or 0),
                "height": int(metadata.get("deviceHeight") or 0),
                "title": self._title,
            }
            self._frame_ready.set()
        # Ack every frame, including ones we drop — withholding the ack
        # stalls the stream rather than throttling it.
        session_id = params.get("sessionId")
        cdp = self._cdp
        if session_id is not None and cdp is not None:
            task = asyncio.ensure_future(self._ack(cdp, session_id))
            self._acks.add(task)
            task.add_done_callback(self._acks.discard)

    async def _ack(self, cdp, session_id) -> None:
        try:
            await cdp.send("Page.screencastFrameAck", {"sessionId": session_id})
        except Exception:
            logger.debug("browser_live: screencastFrameAck failed", exc_info=True)

    async def _attach(self, page) -> None:
        cdp = await page.context.new_cdp_session(page)
        self._cdp = cdp
        self._page = page
        cdp.on("Page.screencastFrame", self._on_screencast_frame)
        await cdp.send("Page.startScreencast", dict(SCREENCAST_PARAMS))

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
        try:
            shot = await cdp.send(
                "Page.captureScreenshot", {"format": "jpeg", "quality": 60}
            ) or {}
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
        }
        self._frame_ready.set()

    async def _detach(self) -> None:
        cdp, self._cdp = self._cdp, None
        self._page = None
        self._pending_frame = None
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
        cdp = self._cdp
        if cdp is None:
            await self._send_error("no browser page is attached")
            return False
        try:
            await cdp.send(method, params)
            return True
        except Exception as exc:
            await self._send_error(f"input was not delivered: {exc}")
            return False

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
        if await self._cdp_send("Input.dispatchMouseEvent", params):
            self._count_input(f"mouse.{action}")

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
        if await self._cdp_send("Input.dispatchKeyEvent", params):
            self._count_input(f"key.{action}")

    async def _handle_text(self, msg: dict) -> None:
        text = msg.get("text")
        if not isinstance(text, str):
            await self._send_error("text event needs a string text")
            return
        if not text:
            return
        if await self._cdp_send("Input.insertText", {"text": text}):
            self._count_input("text")

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
        elif msg_type in ("ping", "pong"):
            pass
        else:
            await self._send_error(f"unknown message type: {msg_type!r}")

    # -- lifecycle --------------------------------------------------------

    async def run(self) -> None:
        sender = asyncio.ensure_future(self._sender())
        watcher = asyncio.ensure_future(self._watcher())
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
            for task in (sender, watcher):
                task.cancel()
            await asyncio.gather(sender, watcher, return_exceptions=True)
            if self._acks:
                await asyncio.gather(*list(self._acks), return_exceptions=True)
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
