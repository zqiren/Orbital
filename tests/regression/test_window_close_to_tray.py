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


def test_is_already_running_path_starts_tray():
    """Whichever shell gets past the guards must own tray AND keep-alive.

    The old bug: is_already_running() returned early with just open_window(),
    skipping tray creation and keep-alive loop. The process would exit
    immediately after the window closed — no tray icon, no persistence.

    Bug #54 narrowed what this branch means without changing that invariant.
    Duplicate *shells* are now stopped earlier, by the single-instance mutex
    guard, before any tray exists (tests/regression/test_single_instance_tray.py).
    is_already_running() only answers "is a daemon already listening" — a repo
    dev daemon, say — and the shell that finds one is still the shell that owns
    the tray and the keep-alive loop.
    """
    source = inspect.getsource(main_mod.main)

    branch_idx = source.find("if is_already_running(")
    assert branch_idx > 0, "main() must still branch on is_already_running()"

    # No early return in that branch — it falls through to the shared path.
    branch_block = "\n".join(source[branch_idx:].split("\n")[:5])
    assert not ("open_window" in branch_block and "\n        return" in branch_block), \
        "is_already_running path must not return early — tray and keep-alive are needed"

    # ...and that shared path starts the tray, then keeps the process alive so
    # the tray outlives the window.
    tail = source[branch_idx:]
    tray_idx = tail.find("start_tray")
    keep_alive_idx = tail.find("while True")
    assert tray_idx > 0, \
        "main() must start the tray after the is_already_running branch"
    assert keep_alive_idx > tray_idx, \
        "the keep-alive loop must follow tray startup so the tray outlives the window"
