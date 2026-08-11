# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import logging
import threading

import pystray
from PIL import Image

logger = logging.getLogger(__name__)

# Seconds between the quit callback returning and the hard exit. Long enough
# for pystray's message loop to unwind (its `finally` is what would otherwise
# have sent NIM_DELETE), short enough that a wedged loop never strands the user.
_EXIT_GRACE_SECONDS = 0.5

# The live pystray.Icon, published so any exit path — not just the tray menu —
# can take the icon out of the notification area. Windows does not reap the
# icon of a dead process until the user happens to sweep the pointer over it.
_icon = None


def _frozen_base_dir() -> str:
    """Return the base directory for bundled resources.

    On macOS .app bundles the executable lives in Contents/MacOS/ but
    manually-copied resources (web/, assets/) live in Contents/Resources/.
    On Windows the resources sit alongside the executable.
    """
    exe_dir = os.path.dirname(sys.executable)
    resources_dir = os.path.join(os.path.dirname(exe_dir), "Resources")
    if os.path.isdir(resources_dir):
        return resources_dir
    return exe_dir


def create_tray_icon() -> Image.Image:
    if getattr(sys, "frozen", False):
        base = _frozen_base_dir()
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    icon_path = os.path.join(base, "assets", "icon.png")
    return Image.open(icon_path)


def hide_tray() -> None:
    """Remove the tray icon from the notification area, if one is showing.

    Safe from any thread and any number of times: pystray's `visible` setter
    short-circuits when the value is unchanged, so a second call (or a call
    before the icon ever ran) does nothing. On Win32 the setter reaches
    Shell_NotifyIcon(NIM_DELETE) synchronously, which is the whole point —
    icon.stop() only *posts* WM_STOP and the deletion in the message loop's
    `finally` never runs when the process exits via os._exit().
    """
    icon = _icon
    if icon is None:
        return
    try:
        icon.visible = False
    except Exception:
        logger.debug("Tray icon hide failed (ignored)", exc_info=True)


def start_tray(port: int, open_window_fn, shutdown_fn):
    global _icon

    def on_open(icon, item):
        open_window_fn()

    def on_quit(icon, item):
        # Delete the icon before stopping the loop: this callback is dispatched
        # on the loop thread, so a posted WM_STOP cannot be processed until it
        # returns, and shutdown_fn() (os._exit) means it never did.
        hide_tray()
        icon.stop()
        # Hand the hard exit to a timer so this callback can return and let the
        # message loop wind down on its own. The timer is the backstop — if the
        # loop hangs, the process still dies.
        threading.Timer(_EXIT_GRACE_SECONDS, shutdown_fn).start()

    try:
        icon = pystray.Icon(
            name="Orbital",
            icon=create_tray_icon(),
            title=f"Orbital \u2014 Running (port {port})",
            menu=pystray.Menu(
                pystray.MenuItem("Open Orbital", on_open, default=True),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit Orbital", on_quit),
            ),
        )
        _icon = icon
        icon.run()
    except Exception:
        if sys.platform == "darwin":
            logger.debug("System tray unavailable (expected on macOS .app bundles)", exc_info=True)
        else:
            logger.error("System tray failed to start", exc_info=True)
