# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The app keeps its port across a relaunch (batch 1114, item 3E).

After an upgrade the relaunched app came up on a random port: the old
instance's closed connections sit in TIME_WAIT on the port for ~30 s on
macOS, and ``find_free_port``'s plain bind probe fails on them, though
uvicorn's own bind (SO_REUSEADDR) would succeed. The probe now retries the
preferred port with SO_REUSEADDR on POSIX, and takes it only when no socket
of another program holds any address on it and nothing accepts a connection.

The occupants run in a child process (another program, as on a real
machine). Ports come from 8790-8799; a port still in TIME_WAIT from an
earlier test is skipped.
"""

import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows keeps today's probe: SO_REUSEADDR there lets a bind take "
    "over a live port, so the TIME_WAIT retry is POSIX-only",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_PORTS = range(8790, 8800)


def _plain_bind_ok(port: int) -> bool:
    for addr in ("0.0.0.0", "127.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((addr, port))
        except OSError:
            return False
        finally:
            s.close()
    return True


def _quiet_port() -> int:
    """A port in 8790-8799 nothing holds, not even a TIME_WAIT."""
    for port in TEST_PORTS:
        if _plain_bind_ok(port):
            return port
    pytest.skip("no quiet port in 8790-8799 on this machine")


def _child(code: str, *args: str) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
        cwd=str(REPO_ROOT),
    )


# A server that answers a few connections, closing each from its side first
# (that side keeps the TIME_WAIT, on the server's port), then exits: the old
# daemon after its last requests.
_SERVE_AND_EXIT = """
import socket, sys
port = int(sys.argv[1])
srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as uvicorn does
srv.bind(("127.0.0.1", port))
srv.listen()
for _ in range(3):
    c = socket.create_connection(("127.0.0.1", port))
    conn, _ = srv.accept()
    conn.sendall(b"ok")
    conn.close()
    c.recv(2)
    c.close()
srv.close()
"""

# Another program holding the port until its stdin closes.
_HOLD = """
import socket, sys
addr, port, reuse, listen = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1"
s = socket.socket()
if reuse:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((addr, port))
if listen:
    s.listen()
print("ready", flush=True)
sys.stdin.read()
"""


def _leave_time_wait(port: int) -> None:
    proc = _child(_SERVE_AND_EXIT, str(port))
    proc.communicate(timeout=20)
    assert proc.returncode == 0
    # The precondition this whole fix is about: the plain probe fails now.
    assert not _plain_bind_ok(port), "expected TIME_WAIT to block a plain bind"


def test_port_in_time_wait_after_its_server_exits_is_taken_again():
    from agent_os.desktop.main import find_free_port

    port = _quiet_port()
    _leave_time_wait(port)
    assert find_free_port(port) == port


@pytest.mark.parametrize(
    "addr,reuse,listen",
    [
        ("127.0.0.1", False, True),   # a plain loopback server
        ("127.0.0.1", True, True),    # a loopback server that sets SO_REUSEADDR
        ("0.0.0.0", True, False),     # a uvicorn-style wildcard socket (bind only: no firewall prompt)
        ("0.0.0.0", False, False),
    ],
)
def test_port_held_by_another_program_after_time_wait_is_not_taken(addr, reuse, listen):
    """Another program holds the port, so the plain probe fails and the
    SO_REUSEADDR retry runs: it must not bind over that program (a wildcard
    SO_REUSEADDR bind on macOS succeeds over a foreign loopback listener).
    A holder that sets SO_REUSEADDR can take the port while the old
    instance's TIME_WAITs remain, so those cases start from TIME_WAIT; one
    without it cannot bind until they expire."""
    from agent_os.desktop.main import find_free_port

    port = _quiet_port()
    if reuse:
        _leave_time_wait(port)
    holder = _child(_HOLD, addr, str(port), "1" if reuse else "0", "1" if listen else "0")
    try:
        assert holder.stdout.readline().strip() == "ready"
        got = find_free_port(port)
        assert got != port
        assert got > 0
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)


def test_port_held_on_a_specific_lan_address_is_not_taken():
    """A foreign socket on one non-loopback address passes both the wildcard
    and the loopback SO_REUSEADDR legs on macOS; the per-address leg must
    catch it."""
    from agent_os.desktop.main import find_free_port
    from tests.unit.test_desktop_main import _non_loopback_ip

    ip = _non_loopback_ip()
    if ip is None:
        pytest.skip("no non-loopback interface on this host")
    port = _quiet_port()
    _leave_time_wait(port)
    holder = _child(_HOLD, ip, str(port), "0", "0")
    try:
        assert holder.stdout.readline().strip() == "ready"
        assert find_free_port(port) != port
    finally:
        holder.stdin.close()
        holder.wait(timeout=10)


def test_time_wait_retry_fails_closed_when_addresses_cannot_be_listed(monkeypatch):
    """Without the list of local addresses the retry cannot rule out a
    foreign socket on one of them, so it falls back as before."""
    import psutil

    from agent_os.desktop.main import find_free_port

    def boom():
        raise OSError("no interface list")

    monkeypatch.setattr(psutil, "net_if_addrs", boom)
    port = _quiet_port()
    _leave_time_wait(port)
    assert find_free_port(port) != port


# Two instances starting together: each picks a port, then binds it the way
# uvicorn does (SO_REUSEADDR on the wildcard; bind only, no firewall prompt).
# A failed bind retries once on a fresh port, as boot_daemon_with_retry does.
_INSTANCE = """
import json, os, socket, sys, time
from agent_os.desktop.main import find_free_port
preferred, go = int(sys.argv[1]), sys.argv[2]
while not os.path.exists(go):
    time.sleep(0.001)

def bind(port):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    return s

port = find_free_port(preferred)
try:
    s = bind(port)
except OSError:
    port = find_free_port(0)
    s = bind(port)
print(json.dumps({"port": s.getsockname()[1]}), flush=True)
sys.stdin.read()
"""


def test_two_instances_starting_together_do_not_share_the_port(tmp_path):
    port = _quiet_port()
    _leave_time_wait(port)
    go = tmp_path / "go"
    procs = [_child(_INSTANCE, str(port), str(go)) for _ in range(2)]
    try:
        time.sleep(0.5)  # both children imported and waiting
        go.write_text("")
        got = [json.loads(p.stdout.readline())["port"] for p in procs]
    finally:
        for p in procs:
            p.stdin.close()
            p.wait(timeout=10)
    assert got[0] != got[1]
    assert port in got  # the one that won kept the preferred port
