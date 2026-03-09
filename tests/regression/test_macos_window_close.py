# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: macOS must minimize on red X (not close), and Cmd+Q must exit cleanly."""

import inspect

from agent_os.desktop import main as main_mod


def test_macos_close_intercept_exists():
    """main.py must contain macOS-specific window close interception logic."""
    source = inspect.getsource(main_mod)
    assert "miniaturize" in source or "windowShouldClose" in source or "closing" in source, \
        "main.py must intercept window close on macOS (minimize instead of close)"


def test_macos_does_not_sleep_loop():
    """macOS must not enter the while-True sleep loop after webview.start() returns."""
    source = inspect.getsource(main_mod.main)
    loop_idx = source.find("while True")
    if loop_idx == -1:
        return  # Loop removed entirely — fine
    # If loop exists, it must be gated behind a platform check
    preceding = source[max(0, loop_idx - 500):loop_idx]
    assert "darwin" in preceding or "win32" in preceding or "platform" in preceding, \
        "while True sleep loop must be gated behind a platform check — macOS must not enter it"


def test_macos_cmd_q_exits():
    """macOS Cmd+Q path must call os._exit for clean shutdown."""
    source = inspect.getsource(main_mod.main)
    # After open_window, the darwin branch must call os._exit
    open_window_idx = source.find("open_window(port)")
    assert open_window_idx > 0, "main() must call open_window(port)"
    after_open_window = source[open_window_idx:]
    assert 'sys.platform == "darwin"' in after_open_window, \
        "After open_window(), main() must check sys.platform == 'darwin'"
    assert "os._exit" in after_open_window, \
        "macOS quit path must call os._exit(0) for clean shutdown"
