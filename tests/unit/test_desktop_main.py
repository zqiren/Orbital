# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import socket
from unittest.mock import patch, MagicMock


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
