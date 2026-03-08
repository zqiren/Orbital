# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os

import pystray
from PIL import Image


def create_tray_icon() -> Image.Image:
    if getattr(sys, "frozen", False):
        icon_path = os.path.join(os.path.dirname(sys.executable), "assets", "icon.png")
    else:
        icon_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "icon.png")
    icon_path = os.path.abspath(icon_path)
    return Image.open(icon_path)


def start_tray(port: int, open_window_fn, shutdown_fn):
    def on_open(icon, item):
        open_window_fn()

    def on_quit(icon, item):
        icon.stop()
        shutdown_fn()

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
