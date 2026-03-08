# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

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
