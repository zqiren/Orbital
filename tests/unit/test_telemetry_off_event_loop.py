# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The telemetry sender's blocking work never runs on the event loop.

Spec 171 A1: ``logs/hang.log`` on the owner's machine caught the loop thread
inside ``sender.py _write_pending`` (a 17.8 s stall on 2026-11-09) and inside
``httpx.AsyncClient`` construction (``ssl.create_default_context`` loads the
CA bundle, once per post). While the loop is blocked, chat, the sidebar and
every project stop answering. The snapshot writes and the client build now
run in a worker thread; these tests record which thread they ran on.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agent_os.telemetry import sender as sender_mod
from agent_os.telemetry.identity import InstallIdentity
from agent_os.telemetry.sender import TelemetrySender
from agent_os.telemetry.spool import Spool


def _sender(tmp_path: Path, post=None) -> TelemetrySender:
    events = tmp_path / "telemetry" / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    # A past, unacknowledged day so the rebuild path writes too.
    events.write_text(json.dumps(
        {"event": "app_start", "ts": "2026-01-02T10:00:00+00:00"}) + "\n",
        encoding="utf-8")
    return TelemetrySender(
        tmp_path, InstallIdentity(tmp_path), Spool(tmp_path),
        is_enabled=lambda: True, post=post,
    )


@pytest.mark.asyncio
async def test_snapshot_writes_run_off_the_event_loop(tmp_path, monkeypatch):
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []
    real_write = TelemetrySender._write_pending

    def recording_write(self, ping):
        seen.append((ping["date"], threading.get_ident()))
        real_write(self, ping)

    monkeypatch.setattr(TelemetrySender, "_write_pending", recording_write)

    async def post(url, payload, headers):
        return 200

    s = _sender(tmp_path, post=post)
    await s.run_cycle()

    dates = {d for d, _ in seen}
    assert "2026-01-02" in dates            # past-day rebuild ran
    assert len(dates) == 2                  # ...and today's snapshot
    assert all(tid != loop_thread for _, tid in seen)
    # The pending files really were written (and the past day acked + deleted).
    assert (tmp_path / "telemetry" / "acks.json").exists()


@pytest.mark.asyncio
async def test_default_post_builds_the_http_client_off_the_event_loop(tmp_path, monkeypatch):
    import httpx

    loop_thread = threading.get_ident()
    built_on: list[int] = []

    class _Resp:
        status_code = 204

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            built_on.append(threading.get_ident())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    status = await sender_mod._httpx_post("https://example.invalid/ping", {"a": 1}, {})

    assert status == 204
    assert built_on and all(tid != loop_thread for tid in built_on)


@pytest.mark.asyncio
async def test_client_built_in_a_thread_still_posts_on_the_loop(tmp_path):
    """The real httpx client, built in a worker thread, sends from the loop."""
    import asyncio
    from http.server import BaseHTTPRequestHandler, HTTPServer

    got: list[dict] = []

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            got.append(json.loads(self.rfile.read(n)))
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/ping"
        status = await asyncio.wait_for(
            sender_mod._httpx_post(url, {"date": "2026-01-02"}, {}), timeout=10)
    finally:
        srv.shutdown()
    assert status == 202
    assert got == [{"date": "2026-01-02"}]
