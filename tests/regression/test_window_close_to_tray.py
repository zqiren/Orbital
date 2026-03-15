# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: window close must hide (not destroy) on all platforms.

macOS: miniaturize to Dock. Windows/Linux: hide to system tray.
Both must return False from _on_closing to prevent actual close.
"""

import inspect

from agent_os.desktop import main as main_mod


def test_on_closing_never_returns_true_unconditionally():
    """_on_closing must NOT return True for non-darwin without a condition."""
    source = inspect.getsource(main_mod.open_window)
    # The old bug: `if sys.platform != "darwin": return True`
    assert 'sys.platform != "darwin"' not in source or "return True" not in source.split('sys.platform != "darwin"')[1].split("\n")[0], \
        "_on_closing must not unconditionally return True for non-darwin platforms"


def test_on_closing_hides_window_on_windows():
    """_on_closing must call window.hide() for Windows/Linux platforms."""
    source = inspect.getsource(main_mod.open_window)
    assert "window.hide()" in source or "_window.hide()" in source, \
        "_on_closing must hide the window on Windows/Linux (not close it)"


def test_on_closing_returns_false():
    """_on_closing must return False to prevent the window from being destroyed."""
    source = inspect.getsource(main_mod.open_window)
    assert "return False" in source, \
        "_on_closing must return False to prevent actual window close"


def test_open_window_reuses_hidden_window():
    """open_window() must show existing hidden window instead of creating a new one."""
    source = inspect.getsource(main_mod.open_window)
    assert "_window.show()" in source or "window.show()" in source, \
        "open_window() must call show() on existing window when called again from tray"


def test_module_level_window_reference():
    """Module must store window reference so tray can show it."""
    assert hasattr(main_mod, "_window"), \
        "main module must have _window variable for tray to access"
