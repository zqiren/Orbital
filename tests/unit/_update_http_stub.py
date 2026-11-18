# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A tiny threaded HTTP server for updater tests: serves byte blobs with
Range / If-Range support, can cut a response short (an interrupted
download), and serves JSON feeds. Loopback only, port chosen by the OS.

Also runnable for the manual macOS rehearsal:

    python -m tests.unit._update_http_stub --file Orbital-macOS.dmg=/path/x.dmg \
        --json latest.json=/path/latest.json --port 8790
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class StubServer:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.json: dict[str, object] = {}
        self.etags: dict[str, str] = {}
        self.cut_after: dict[str, int] = {}   # path -> bytes to send, then drop
        self.ignore_range: set[str] = set()
        self.requests: list[tuple[str, dict]] = []
        self._httpd: ThreadingHTTPServer | None = None

    @property
    def base(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address[:2]
        return f"http://127.0.0.1:{port}"

    def url(self, path: str) -> str:
        return f"{self.base}/{path.lstrip('/')}"

    def start(self, port: int = 0) -> "StubServer":
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # quiet
                pass

            def do_GET(self):  # noqa: N802
                path = self.path.lstrip("/")
                stub.requests.append((path, {k.lower(): v for k, v in self.headers.items()}))
                if path in stub.json:
                    body = json.dumps(stub.json[path]).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path not in stub.files:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = stub.files[path]
                etag = stub.etags.get(path, '"v1"')
                start = 0
                rng = self.headers.get("Range")
                if_range = self.headers.get("If-Range")
                partial = bool(rng) and path not in stub.ignore_range and \
                    (if_range is None or if_range == etag)
                if partial:
                    start = int(rng.split("=", 1)[1].split("-", 1)[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", etag)
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                cut = stub.cut_after.pop(path, None)
                if cut is not None:
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(2)
                    except OSError:
                        pass
                    return
                self.wfile.write(body)

        class Server(ThreadingHTTPServer):
            def handle_error(self, request, client_address):  # cut responses: quiet
                pass

        self._httpd = Server(("127.0.0.1", port), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", default=[], help="name=path")
    ap.add_argument("--json", action="append", default=[], help="name=path")
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args()
    stub = StubServer()
    for spec in args.file:
        name, path = spec.split("=", 1)
        stub.files[name] = Path(path).read_bytes()
    for spec in args.json:
        name, path = spec.split("=", 1)
        stub.json[name] = json.loads(Path(path).read_text())
    stub.start(args.port)
    print(f"serving on {stub.base}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    _main()
