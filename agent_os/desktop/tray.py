# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import logging

import pystray
from PIL import Image

logger = logging.getLogger(__name__)


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


def start_tray(port: int, open_window_fn, shutdown_fn):
    def on_open(icon, item):
        open_window_fn()

    def on_quit(icon, item):
        icon.stop()
        shutdown_fn()

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
        icon.run()
    except Exception:
        if sys.platform == "darwin":
            logger.debug("System tray unavailable (expected on macOS .app bundles)", exc_info=True)
        else:
            logger.error("System tray failed to start", exc_info=True)
