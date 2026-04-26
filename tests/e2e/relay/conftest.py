# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E2E test fixtures for the cloud relay integration.

Provides fixtures that start/stop the daemon (port 8321) and relay (port 3321),
along with a simulated phone client for HTTP and WebSocket interactions.

Skipped wholesale on Windows: the daemon/relay subprocess setup hits
NotADirectoryError [WinError 267] when staging the relay process state — the
fixtures assume POSIX process and path semantics that aren't yet adapted for
Windows. Re-enable by removing the ``collect_ignore_glob`` once a Windows-
compatible relay launcher exists.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Generator

import httpx
import pytest
import pytest_asyncio
import websockets

# Windows compatibility: the relay e2e tests rely on POSIX subprocess /
# filesystem semantics. Skip the directory wholesale on win32.
if sys.platform == "win32":
    collect_ignore_glob = ["test_*.py"]

# ---------------------------------------------------------------------------
# Ports — non-default to avoid conflicts with a running daemon
# ---------------------------------------------------------------------------
DAEMON_PORT = 8321
RELAY_PORT = 3321
DAEMON_BASE = f"http://localhost:{DAEMON_PORT}"
RELAY_BASE = f"http://localhost:{RELAY_PORT}"
RELAY_WS = f"ws://localhost:{RELAY_PORT}"

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # D:/AgentOS

# ---------------------------------------------------------------------------
# Helpers: start / stop processes
# ---------------------------------------------------------------------------


def _wait_http(url: str, timeout: float = 15.0, interval: float = 0.3) -> bool:
    """Block until *url* responds with a non-5xx status or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except (httpx.ConnectError, httpx.ReadTimeout, OSError):
            pass
        time.sleep(interval)
    return False


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process tree (Windows-safe)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# Fixtures: daemon process
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory) -> Path:
    """Isolated data directory so tests don't touch real agent-os data."""
    return tmp_path_factory.mktemp("e2e_data")


@pytest.fixture(scope="session")
def daemon_proc(tmp_data_dir: Path) -> Generator[subprocess.Popen, None, None]:
    """Start the daemon (uvicorn) on DAEMON_PORT with relay URL pointing to our relay."""
    env = {
        **os.environ,
        "AGENT_OS_RELAY_URL": RELAY_BASE,
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        sys.executable, "-m", "uvicorn",
        "agent_os.api.app:create_app", "--factory",
        "--host", "127.0.0.1",
        "--port", str(DAEMON_PORT),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    # Wait for daemon to be ready
    if not _wait_http(f"{DAEMON_BASE}/api/v2/projects", timeout=20):
        out = proc.stdout.read(4096) if proc.stdout else b""
        _kill_tree(proc)
        pytest.fail(f"Daemon did not start on port {DAEMON_PORT}.\nOutput: {out.decode(errors='replace')}")

    yield proc

    _kill_tree(proc)


@pytest.fixture(scope="session")
def relay_proc() -> Generator[subprocess.Popen, None, None]:
    """Start the relay server (Node/tsx) on RELAY_PORT."""
    relay_dir = PROJECT_ROOT / "relay"
    env = {
        **os.environ,
        "PORT": str(RELAY_PORT),
        "RELAY_JWT_SECRET": "test-secret-not-for-production",
    }
    # Use npx tsx to run TypeScript directly
    cmd_tsx = "npx.cmd" if sys.platform == "win32" else "npx"
    cmd = [cmd_tsx, "tsx", "src/index.ts"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(relay_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    # Wait for relay to start
    if not _wait_http(f"{RELAY_BASE}/api/v1/push/vapid-key", timeout=20):
        out = proc.stdout.read(4096) if proc.stdout else b""
        _kill_tree(proc)
        pytest.fail(f"Relay did not start on port {RELAY_PORT}.\nOutput: {out.decode(errors='replace')}")

    yield proc

    _kill_tree(proc)


@pytest.fixture(scope="session")
def services(relay_proc, daemon_proc):
    """Ensure both relay and daemon are running.

    relay_proc is started first because the daemon tries to connect
    to the relay on startup.
    """
    # Give the daemon relay client a moment to connect tunnel
    time.sleep(1)
    return {"daemon": daemon_proc, "relay": relay_proc}


# ---------------------------------------------------------------------------
# Fixtures: phone client (simulated mobile app)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def phone_http() -> Generator[httpx.Client, None, None]:
    """Synchronous httpx client pointed at the relay."""
    with httpx.Client(base_url=RELAY_BASE, timeout=10) as client:
        yield client


@pytest_asyncio.fixture
async def phone_ws_factory():
    """Factory that creates WebSocket connections to the relay.

    Returns an async callable:  ws = await factory(token)
    Caller is responsible for closing.
    """
    connections: list = []

    async def _connect(token: str):
        ws = await websockets.connect(
            f"{RELAY_WS}/ws?token={token}",
            close_timeout=2,
        )
        connections.append(ws)
        return ws

    yield _connect

    for ws in connections:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fixtures: daemon HTTP client (desktop side)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def daemon_http() -> Generator[httpx.Client, None, None]:
    """Synchronous httpx client pointed at the daemon."""
    with httpx.Client(base_url=DAEMON_BASE, timeout=10) as client:
        yield client


@pytest_asyncio.fixture
async def daemon_http_async() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async httpx client pointed at the daemon."""
    async with httpx.AsyncClient(base_url=DAEMON_BASE, timeout=10) as client:
        yield client
