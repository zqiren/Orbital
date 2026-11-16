# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Protocol fakes for the bot-source adapter tests: a real local HTTP
server (stdlib, in a thread) and a real local WebSocket server
(``websockets``), so each adapter is exercised over the wire exactly as it
talks to its service. Frame shapes and their provenance are documented in
each service's test file."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from agent_os.sources.context import SourceContext


class FakeHTTP:
    """``route(method, path, fn)``: ``fn(req) -> (status, obj)``; ``path``
    is matched exactly or, ending in ``*``, by prefix. ``req`` has
    ``method, path, query, headers, json, form``. Every request is kept in
    ``requests``. Handlers run in server threads."""

    def __init__(self):
        self.routes: list = []
        self.requests: list = []
        self._srv = None
        self._thread = None
        self.lock = threading.Lock()

    def route(self, method: str, path: str, fn) -> None:
        self.routes.insert(0, (method.upper(), path, fn))

    def calls(self, path: str) -> list:
        with self.lock:
            return [r for r in self.requests if r.path == path or (path.endswith("*") and r.path.startswith(path[:-1]))]

    def start(self) -> "FakeHTTP":
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _handle(self, method):
                u = urlparse(self.path)
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b""
                body_json, form = None, {}
                ctype = self.headers.get("content-type") or ""
                if raw and "json" in ctype:
                    try:
                        body_json = json.loads(raw)
                    except ValueError:
                        body_json = None
                elif raw and "form" in ctype:
                    form = {k: v[0] for k, v in parse_qs(raw.decode()).items()}
                req = SimpleNamespace(method=method, path=u.path,
                                      query={k: v[0] for k, v in parse_qs(u.query).items()},
                                      headers={k.lower(): v for k, v in self.headers.items()},
                                      json=body_json, form=form)
                with fake.lock:
                    fake.requests.append(req)
                fn = None
                for m, p, f in fake.routes:
                    if m == method and (p == u.path or (p.endswith("*") and u.path.startswith(p[:-1]))):
                        fn = f
                        break
                if fn is None:
                    status, obj = 404, {"error": "no route"}
                else:
                    status, obj = fn(req)
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                # One request per connection, so stop() really stops the
                # service (no kept-alive socket keeps answering).
                self.send_header("connection", "close")
                self.close_connection = True
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self._srv.daemon_threads = True
        self._thread = threading.Thread(target=self._srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()


class FakeWS:
    """A local WebSocket server; ``handler(ws)`` is the per-connection
    script (set or replaced by each test). ``connections`` counts them."""

    def __init__(self, handler=None):
        self.handler = handler
        self.connections = 0
        self.paths: list = []
        self._server = None

    async def start(self) -> "FakeWS":
        from websockets.asyncio.server import serve

        async def _h(ws):
            self.connections += 1
            try:
                self.paths.append(ws.request.path)
            except Exception:
                pass
            await self.handler(ws)

        self._server = await serve(_h, "127.0.0.1", 0)
        return self

    @property
    def url(self) -> str:
        port = list(self._server.sockets)[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), 3)
            except asyncio.TimeoutError:
                pass


class Run:
    """A stream adapter running against a fake: a real SourceContext whose
    emits and connects are recorded."""

    def __init__(self, adapter, *, secrets: dict, config: dict | None = None,
                 kinds: dict | None = None, state: dict | None = None, source_id="src_test"):
        self.items: list = []
        self.connects = 0
        self.emits = 0
        self.state = state if state is not None else {}
        self.adapter = adapter
        self.connected_evt = asyncio.Event()

        async def _emit(evs):
            self.emits += 1
            self.items.extend(evs)

        def _connected():
            self.connects += 1
            self.connected_evt.set()

        self.ctx = SourceContext(source_id=source_id, type_="t", config=config or {},
                                 kinds=kinds or {"dm": {}, "at": {}}, account=None,
                                 secrets_fn=dict(secrets).get, state=self.state,
                                 emit_fn=_emit, connected_fn=_connected)
        self.task: asyncio.Task | None = None

    def start(self) -> "Run":
        self.task = asyncio.create_task(self.adapter.stream(self.ctx))
        return self

    async def until(self, cond, timeout: float = 5.0):
        async def _w():
            while not cond():
                if self.task is not None and self.task.done():
                    # surface the stream's own end/exception
                    return
                await asyncio.sleep(0.01)
        await asyncio.wait_for(_w(), timeout)

    async def result(self, timeout: float = 5.0):
        """Wait for the stream to end; return None or its exception."""
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout)
        except asyncio.TimeoutError:
            raise
        except BaseException as e:  # noqa: BLE001
            return e
        return None

    async def cancel(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except BaseException:  # noqa: BLE001
                pass


def verify_ctx(secrets: dict, config: dict | None = None) -> SourceContext:
    return SourceContext(source_id=None, type_="t", config=config or {}, kinds={"dm": {}, "at": {}},
                         account=None, secrets_fn=dict(secrets).get, state={})
