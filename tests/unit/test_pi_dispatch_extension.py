# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 for Pi: the seat-holder Pi mounts the daemon's dispatch endpoint.

Pi (>= 0.99) has its own MCP client, and an extension can register a server
for one session. Orbital ships one tiny extension
(``agent_os/agents/assets/pi/orbital-dispatch.ts``) and passes it with ``-e``
only when the transport carries ``dispatch_mcp``; the URL and token travel in
the spawn env and the extension removes both from Pi's env before any tool
runs. Mounted means Pi said so: the extension's ``orbital-dispatch`` status
record reads ``mounted`` and no ``extension_error`` came from that file, both
written while Pi binds extensions, before it answers ``get_state``.

The fake-child tests pin the transport contract; the real-pi tests (skipped
when ``pi`` is not installed) prove the extension against the released binary
and the real ``/mcp`` endpoint, without a model.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys

import pytest

from agent_os.agent.transports import pi_rpc_transport as pi_mod
from agent_os.agent.transports.pi_rpc_transport import PiRPCTransport
from agent_os.daemon_v2 import dispatch_tools

FIXTURES = os.path.join(os.path.dirname(__file__), os.pardir, "fixtures", "pi_rpc")
FAKE_PI = os.path.join(FIXTURES, "fake_pi.py")
URL = "http://127.0.0.1:8541/mcp"
TOKEN = "tok-pi-dispatch-0123456789"


def _record(tmp_path):
    path = tmp_path / "record.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _spawns(tmp_path):
    return [r for r in _record(tmp_path) if r["event"] == "spawn"]


def _status(text):
    return {"type": "extension_ui_request", "id": "s1", "method": "setStatus",
            "statusKey": "orbital-dispatch", "statusText": text}


async def _started(tmp_path, *, dispatch_mcp=None, startup=None, env=None):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    t = PiRPCTransport(system_prompt="brief", session_dir=str(tmp_path / "pi-sessions"),
                       dispatch_mcp=dispatch_mcp, command_timeout=10.0)
    full_env = {"FAKE_PI_RECORD": str(tmp_path / "record.jsonl"),
                "FAKE_PI_ENV_KEYS": "ORBITAL_DISPATCH_URL,ORBITAL_DISPATCH_TOKEN"}
    if startup is not None:
        path = tmp_path / "startup.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in startup), encoding="utf-8")
        full_env["FAKE_PI_STARTUP_RECORDS"] = str(path)
    full_env.update(env or {})
    await asyncio.wait_for(t.start(sys.executable, [FAKE_PI], str(ws), full_env), 30)
    return t


def test_the_extension_ships_with_orbital_and_names_the_same_env_and_server():
    path = pi_mod.dispatch_extension_path()
    assert path is not None and os.path.isfile(path)
    assert path.endswith(os.path.join("assets", "pi", "orbital-dispatch.ts"))
    source = open(path, encoding="utf-8").read()
    # One source of truth for the names the daemon, transport and extension share.
    assert pi_mod.DISPATCH_TOKEN_ENV == dispatch_tools.TOKEN_ENV
    for name in (pi_mod.DISPATCH_TOKEN_ENV, pi_mod.DISPATCH_URL_ENV,
                 f'"{dispatch_tools.SERVER_NAME}"', f'"{pi_mod.DISPATCH_STATUS_KEY}"'):
        assert name in source
    # The token must leave Pi's env before any tool runs.
    assert f"delete process.env.{pi_mod.DISPATCH_TOKEN_ENV}" in source


@pytest.mark.asyncio
async def test_seat_holder_spawn_passes_the_extension_and_the_env(tmp_path):
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN},
                       startup=[_status("mounted")])
    try:
        spawn = _spawns(tmp_path)[-1]
        argv = spawn["argv"]
        assert argv[argv.index("-e") + 1] == pi_mod.dispatch_extension_path()
        assert spawn["env"] == {"ORBITAL_DISPATCH_URL": URL, "ORBITAL_DISPATCH_TOKEN": TOKEN}
        # Never on argv, where any process listing would show it.
        assert TOKEN not in " ".join(argv)
        assert t.dispatch_mounted is True
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_a_worker_below_the_seat_gets_no_extension_and_no_token(tmp_path):
    t = await _started(tmp_path, dispatch_mcp=None)
    try:
        spawn = _spawns(tmp_path)[-1]
        assert "-e" not in spawn["argv"]
        assert spawn["env"] == {"ORBITAL_DISPATCH_URL": None, "ORBITAL_DISPATCH_TOKEN": None}
        assert t.dispatch_mounted is False
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_no_status_record_means_not_mounted(tmp_path):
    # e.g. a Pi too old for extensions' session_start, or a broken extension
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN}, startup=[])
    try:
        assert t.dispatch_mounted is False
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_a_failed_registration_is_not_mounted(tmp_path):
    # Pi older than 0.99 has no pi.registerMcpServer.
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN},
                       startup=[_status("failed: pi.registerMcpServer is not a function")])
    try:
        assert t.dispatch_mounted is False
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_an_extension_error_from_the_file_cancels_the_mount(tmp_path):
    # Observed on pi 1.5.0 with --no-mcp (or the built-in MCP replaced): the
    # registration succeeds, then nothing connects it.
    error = {"type": "extension_error", "extensionPath": pi_mod.dispatch_extension_path(),
             "event": "register_mcp_server",
             "error": 'MCP server "orbital" is registered, but no loaded extension '
                      "connects MCP servers"}
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN},
                       startup=[_status("mounted"), error])
    try:
        assert t.dispatch_mounted is False
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_another_extensions_error_does_not_cancel_the_mount(tmp_path):
    error = {"type": "extension_error", "extensionPath": "/elsewhere/other.ts",
             "event": "session_start", "error": "boom"}
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN},
                       startup=[_status("mounted"), error])
    try:
        assert t.dispatch_mounted is True
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_a_missing_extension_file_spawns_pi_without_it(tmp_path, monkeypatch):
    # Pi exits at startup on a missing -e path; the worker must still start.
    monkeypatch.setattr(pi_mod, "dispatch_extension_path", lambda: None)
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN},
                       startup=[_status("mounted")])
    try:
        spawn = _spawns(tmp_path)[-1]
        assert "-e" not in spawn["argv"]
        assert spawn["env"]["ORBITAL_DISPATCH_TOKEN"] is None
        assert t.dispatch_mounted is False
    finally:
        await t.stop()


@pytest.mark.asyncio
async def test_the_token_is_redacted_from_pi_output(tmp_path):
    t = await _started(tmp_path, dispatch_mcp={"url": URL, "token": TOKEN},
                       startup=[_status("mounted")])
    try:
        assert TOKEN in t._secrets
    finally:
        await t.stop()


# ---------------------------------------------------------------------------
# The released binary (no model needed): skipped where pi is not installed.
# ---------------------------------------------------------------------------

_PI = shutil.which("pi")
real_pi = pytest.mark.skipif(_PI is None, reason="pi is not installed")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def endpoint(monkeypatch):
    """The real /mcp endpoint on a loopback port, one known token."""
    import uvicorn
    from fastapi import FastAPI
    from unittest.mock import AsyncMock, MagicMock

    port = _free_port()
    monkeypatch.setenv("AGENT_OS_PORT", str(port))
    sam = MagicMock()
    sam.send = AsyncMock(return_value="Message sent to claude-code.")
    contexts = {TOKEN: dispatch_tools.DispatchContext(
        "p1", "s1", "pi", agents=("pi", "claude-code"))}
    seen: list[str] = []

    def lookup(token):
        seen.append(token)
        return contexts.get(token)

    sam.dispatch_context = lookup
    sam._dispatch_url = None
    app = FastAPI()
    dispatch_tools.attach_dispatch_server(app, sam)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.05)
    yield sam._dispatch_url, seen
    server.should_exit = True
    await asyncio.wait_for(task, 10)


async def _real_pi(tmp_path, dispatch_mcp):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    home = tmp_path / "pi-home"   # never the user's ~/.pi/agent
    home.mkdir(exist_ok=True)
    t = PiRPCTransport(session_dir=str(tmp_path / "pi-sessions"),
                       dispatch_mcp=dispatch_mcp, command_timeout=30.0)
    await asyncio.wait_for(t.start(_PI, ["--mode", "rpc", "--no-approve"], str(ws),
                                   {"PI_CODING_AGENT_DIR": str(home)}), 60)
    return t


@real_pi
@pytest.mark.asyncio
async def test_real_pi_mounts_the_endpoint_and_hides_the_token(tmp_path, endpoint):
    url, seen = endpoint
    t = await _real_pi(tmp_path, {"url": url, "token": TOKEN})
    try:
        assert t.dispatch_mounted is True
        # Pi's own MCP client lists the endpoint's tools with the bearer token.
        for _ in range(100):
            if TOKEN in seen:
                break
            await asyncio.sleep(0.05)
        assert TOKEN in seen
        # A shell command Pi runs sees neither the token nor the URL.
        resp = await t._request({"type": "bash", "command":
                                 'echo "T=[$ORBITAL_DISPATCH_TOKEN] U=[$ORBITAL_DISPATCH_URL]"'})
        assert resp.get("success"), resp
        assert resp["data"]["output"].strip() == "T=[] U=[]"
    finally:
        await t.stop()


@real_pi
@pytest.mark.asyncio
async def test_real_pi_below_the_seat_loads_no_extension(tmp_path, endpoint):
    _, seen = endpoint
    t = await _real_pi(tmp_path, None)
    try:
        assert t.dispatch_mounted is False
        await asyncio.sleep(0.5)
        assert seen == []
    finally:
        await t.stop()
