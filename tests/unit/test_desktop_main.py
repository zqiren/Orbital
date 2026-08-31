# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import socket
import sys
import time
from unittest.mock import patch, MagicMock

import pytest


def test_find_free_port_preferred():
    from agent_os.desktop.main import find_free_port
    port = find_free_port(preferred=59123)
    assert isinstance(port, int)
    assert port > 0


def test_find_free_port_fallback():
    from agent_os.desktop.main import find_free_port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    occupied_port = sock.getsockname()[1]
    try:
        port = find_free_port(preferred=occupied_port)
        assert port != occupied_port
        assert port > 0
    finally:
        sock.close()


def test_is_already_running_false():
    from agent_os.desktop.main import is_already_running
    assert is_already_running(port=59999) is False


# ---------------------------------------------------------------------------
# Bug #75 — port-8000 conflict on a specific interface (spec 075)
# ---------------------------------------------------------------------------


def _non_loopback_ip():
    """A local non-loopback IPv4 address, or None if the host has none.

    The UDP connect never sends a packet — it only makes the kernel pick the
    outbound interface so getsockname() reveals its address.
    """
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1, unroutable
            ip = probe.getsockname()[0]
        finally:
            probe.close()
    except OSError:
        return None
    return ip if ip and not ip.startswith("127.") else None


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows only conflicts exact-duplicate binds: a specific-interface "
    "occupant coexists with uvicorn's wildcard bind, so the port genuinely IS "
    "usable there (CONFIRMED on CI run 33400107996)",
)
def test_find_free_port_detects_specific_interface_occupant():
    """On macOS/Linux, an occupant bound to a specific non-loopback interface
    conflicts with the wildcard bind uvicorn will attempt — the old
    loopback-only probe missed it. The probe must force the fallback."""
    ip = _non_loopback_ip()
    if ip is None:
        pytest.skip("no non-loopback interface on this host")
    from agent_os.desktop.main import find_free_port
    occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # bind only, no listen(): bind-level conflict suffices here, and a
        # listening non-loopback socket triggers the macOS app-firewall prompt.
        occupant.bind((ip, 0))
        occupied_port = occupant.getsockname()[1]
        port = find_free_port(preferred=occupied_port)
        assert port != occupied_port
        assert port > 0
    finally:
        occupant.close()


def test_find_free_port_detects_wildcard_occupant():
    """The confirmed bug-#75 shape on Windows: the occupant holds 0.0.0.0:P
    itself. Windows lets a specific bind (the old 127.0.0.1 probe) succeed
    OVER a foreign wildcard socket, so the probe reported the port free and
    uvicorn's own wildcard bind then died on the exact duplicate (10048).
    The probe's wildcard leg must catch this on every platform."""
    from agent_os.desktop.main import find_free_port
    occupant = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        occupant.bind(("0.0.0.0", 0))
        occupied_port = occupant.getsockname()[1]
        port = find_free_port(preferred=occupied_port)
        assert port != occupied_port
        assert port > 0
    finally:
        occupant.close()


def test_wait_for_daemon_fails_fast_when_daemon_thread_is_dead():
    """A failed uvicorn bind logs ERROR and returns from server.run() without
    raising — the thread dies with server.started False. wait_for_daemon must
    notice and fail immediately instead of polling HTTP for the full 15s."""
    from agent_os.desktop.main import wait_for_daemon
    server = MagicMock()
    server.started = False
    thread = MagicMock()
    thread.is_alive.return_value = False
    start = time.monotonic()
    ok = wait_for_daemon(59998, timeout=15, server=server, thread=thread)
    assert ok is False
    assert time.monotonic() - start < 5


def test_wait_for_daemon_without_liveness_args_still_times_out():
    """Backward-compatible signature: no server/thread → plain HTTP polling."""
    from agent_os.desktop.main import wait_for_daemon
    assert wait_for_daemon(59997, timeout=1) is False


def test_boot_retries_once_on_a_fresh_port_when_first_start_dies():
    """First start_daemon dies (bind conflict the probe could not see) → one
    retry on a fresh random port, and the app proceeds with that port."""
    from agent_os.desktop import main
    dead_thread = MagicMock()
    dead_thread.is_alive.return_value = False
    live_thread = MagicMock()
    live_thread.is_alive.return_value = True
    server = MagicMock()
    started_ports = []

    def fake_start(port):
        started_ports.append(port)
        return server, (dead_thread if len(started_ports) == 1 else live_thread)

    with patch.object(main, "start_daemon", side_effect=fake_start), \
         patch.object(main, "wait_for_daemon", side_effect=[False, True]), \
         patch.object(main, "find_free_port", side_effect=[8000, 55555]):
        port, srv, thread = main.boot_daemon_with_retry(8000)

    assert started_ports == [8000, 55555]
    assert port == 55555
    assert thread is live_thread


def test_boot_does_not_retry_when_daemon_is_alive_but_slow():
    """A live-but-slow daemon is not a bind failure — starting a second daemon
    beside it would just lose the pid-file race. No retry, report failure."""
    from agent_os.desktop import main
    live_thread = MagicMock()
    live_thread.is_alive.return_value = True
    server = MagicMock()
    started_ports = []

    def fake_start(port):
        started_ports.append(port)
        return server, live_thread

    with patch.object(main, "start_daemon", side_effect=fake_start), \
         patch.object(main, "wait_for_daemon", return_value=False), \
         patch.object(main, "find_free_port", return_value=8000):
        port, srv, thread = main.boot_daemon_with_retry(8000)

    assert started_ports == [8000]
    assert port is None


def test_probe_ignores_system_proxy(monkeypatch):
    """A loopback health probe must never route through a system proxy —
    a machine running Clash/a corp proxy would report a healthy daemon as
    down. The opener must be built with an empty ProxyHandler."""
    import http.server
    import threading

    class _OK(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _OK)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")  # dead proxy
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        from agent_os.desktop.main import is_already_running
        assert is_already_running(port=httpd.server_address[1]) is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_stderr_log_tail_prefers_error_lines(tmp_path):
    """The failure dialog must surface the real error, not just a log path
    the user cannot find (AppData is hidden on Windows)."""
    from agent_os.desktop import main
    log = tmp_path / "orbital-stderr.log"
    log.write_text(
        "INFO:     Started server process [70536]\n"
        "INFO:     Waiting for application startup.\n"
        "ERROR:    [Errno 10048] error while attempting to bind on address\n"
        "INFO:     Application shutdown complete.\n",
        encoding="utf-8",
    )
    with patch.object(main, "_get_log_path", return_value=str(tmp_path)):
        tail = main._stderr_log_tail()
    assert "10048" in tail


def test_stderr_log_tail_missing_file_is_empty():
    from agent_os.desktop import main
    with patch.object(main, "_get_log_path", return_value="/nonexistent/dir"):
        assert main._stderr_log_tail() == ""


def test_resolve_spa_dir_source_mode():
    from agent_os.desktop.main import resolve_spa_dir
    path = resolve_spa_dir()
    assert "web" in path


def test_resolve_window_icon_path_windows_uses_ico():
    """GH #37/#38: webview.start(icon=) on Windows hands the path to .NET
    System.Drawing.Icon, which only accepts .ico. A .png raises
    ArgumentException and crashes the main thread before any window opens
    (taking the daemon thread down with it). Must resolve to a .ico on win32."""
    from agent_os.desktop import main
    with patch("sys.platform", "win32"):
        path = main.resolve_window_icon_path()
    assert path.lower().endswith(".ico")
    assert os.path.exists(path)


def test_resolve_window_icon_path_macos_uses_png():
    """Cocoa/GTK backends accept the .png; only Windows needs the .ico swap."""
    from agent_os.desktop import main
    with patch("sys.platform", "darwin"):
        path = main.resolve_window_icon_path()
    assert path.lower().endswith(".png")


def test_resolve_window_icon_path_windows_falls_back_to_png_when_ico_missing():
    """Defensive: if icon.ico is somehow absent from the bundle, fall back to
    the .png rather than handing webview.start a non-existent path."""
    from agent_os.desktop import main
    with patch("sys.platform", "win32"), patch("os.path.exists", return_value=False):
        path = main.resolve_window_icon_path()
    assert path.lower().endswith(".png")


def test_resolve_icon_path_stays_png_for_hicon_source():
    """Regression guard for the risk in the reporter's blanket fix:
    _png_to_hicon() decodes via PIL and is written for a PNG source, so
    resolve_icon_path() must remain .png. The .ico swap is isolated to
    resolve_window_icon_path() (webview.start only)."""
    from agent_os.desktop.main import resolve_icon_path
    assert resolve_icon_path().lower().endswith(".png")
