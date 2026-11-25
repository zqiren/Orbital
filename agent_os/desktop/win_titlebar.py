# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Windows title bar follows the app's light / dark theme.

The window keeps its native title bar on Windows (a frameless one would mean
hand-drawing minimize / maximize / close plus drag, resize and snap). Left
alone that bar is Windows' plain white, a white strip on a near-black window
in the dark theme. What Windows lets a window ask for:

- Windows 10 1809+ : its own dark bar instead of the light one.
- Windows 11       : also an exact caption colour, so the bar continues the
                     tinted chrome under it.

The page owns the theme (``data-theme`` on ``<html>``, see
``web/src/hooks/useTheme.ts``); :data:`THEME_WATCH_JS` reports it to the
shell's ``Api`` now and on every change. Nothing here raises: a title bar is
never worth a failed start.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

DWMWA_USE_IMMERSIVE_DARK_MODE_OLD = 19   # Windows 10 1809 – 1909
DWMWA_USE_IMMERSIVE_DARK_MODE = 20       # Windows 10 2004 and later
DWMWA_CAPTION_COLOR = 35                 # Windows 11

_BUILD_DARK_MODE_OLD = 17763
_BUILD_DARK_MODE = 18985
_BUILD_CAPTION_COLOR = 22000

# The top of look D's chrome gradient (web/src/index.css), where the bar meets it.
CHROME_LIGHT = "#dde4f0"
CHROME_DARK = "#131825"

# The name the page calls on ``window.pywebview.api`` (``Api`` in main.py).
API_METHOD = "set_titlebar_theme"

# Installed on every ``loaded``; the guard makes a re-fire (the origin guard's
# reload) harmless. ``pywebview.api`` is injected asynchronously, so a report
# that finds it missing is sent again on ``pywebviewready``.
THEME_WATCH_JS = """
(function () {
  if (window.__orbitalTitlebarWatch) return;
  window.__orbitalTitlebarWatch = true;
  var root = document.documentElement;
  function report() {
    try {
      var api = window.pywebview && window.pywebview.api;
      if (api && api.%(method)s) api.%(method)s(root.getAttribute('data-theme') || 'light');
    } catch (e) {}
  }
  new MutationObserver(report).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  window.addEventListener('pywebviewready', report);
  report();
})();
""" % {"method": API_METHOD}


def _colorref(hex_rgb: str) -> int:
    """``#rrggbb`` as a Win32 COLORREF (0x00bbggrr)."""
    r, g, b = (int(hex_rgb[i:i + 2], 16) for i in (1, 3, 5))
    return (b << 16) | (g << 8) | r


def windows_build() -> int:
    try:
        return sys.getwindowsversion().build  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — not Windows
        return 0


def titlebar_attributes(theme, build: int) -> list[tuple[int, int]]:
    """The ``(DWM attribute, value)`` pairs to set for ``theme`` on this build.
    Anything but ``"dark"`` is light. Empty when this Windows offers nothing."""
    dark = theme == "dark"
    if build >= _BUILD_DARK_MODE:
        out = [(DWMWA_USE_IMMERSIVE_DARK_MODE, int(dark))]
    elif build >= _BUILD_DARK_MODE_OLD:
        out = [(DWMWA_USE_IMMERSIVE_DARK_MODE_OLD, int(dark))]
    else:
        return []
    if build >= _BUILD_CAPTION_COLOR:
        out.append((DWMWA_CAPTION_COLOR, _colorref(CHROME_DARK if dark else CHROME_LIGHT)))
    return out


def _dwm_set_attribute(hwnd: int, attr: int, value: int) -> int:
    import ctypes
    v = ctypes.c_int(value)
    return ctypes.windll.dwmapi.DwmSetWindowAttribute(
        ctypes.c_void_p(hwnd), attr, ctypes.byref(v), ctypes.sizeof(v))


def _redraw_frame(hwnd: int) -> None:
    """Make Windows repaint the non-client area now. Windows 10 repaints a
    title bar whose dark mode changed only when the window's activation
    changes, so the frame is told it is inactive and then what it really is."""
    import ctypes
    user32 = ctypes.windll.user32
    h = ctypes.c_void_p(hwnd)
    WM_NCACTIVATE = 0x0086
    active = user32.GetForegroundWindow() == hwnd
    user32.SendMessageW(h, WM_NCACTIVATE, int(not active), 0)
    user32.SendMessageW(h, WM_NCACTIVATE, int(active), 0)
    # SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
    user32.SetWindowPos(h, None, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0004 | 0x0010 | 0x0020)


def apply_titlebar_theme(hwnd, theme, *, build: int | None = None,
                         set_attribute=None, redraw=None) -> bool:
    """Set the window's title bar for ``theme``. True when something was asked
    of Windows; never raises."""
    attrs = titlebar_attributes(theme, windows_build() if build is None else build)
    if not hwnd or not attrs:
        return False
    set_attribute = set_attribute or _dwm_set_attribute
    redraw = redraw or _redraw_frame
    for attr, value in attrs:
        try:
            set_attribute(hwnd, attr, value)
        except Exception:  # noqa: BLE001 — display only
            logger.debug("title bar: attribute %s was refused", attr, exc_info=True)
    try:
        redraw(hwnd)
    except Exception:  # noqa: BLE001
        logger.debug("title bar: frame redraw failed", exc_info=True)
    return True


def window_handle(window, title: str = "Orbital") -> int:
    """The HWND of the pywebview window: its WinForms handle, else the
    top-level window with the app's title. 0 when neither is there."""
    try:
        return int(window.native.Handle.ToInt64())
    except Exception:  # noqa: BLE001 — not created yet, or another backend
        pass
    try:
        import ctypes
        return int(ctypes.windll.user32.FindWindowW(None, title) or 0)
    except Exception:  # noqa: BLE001
        return 0


def watch_theme(window) -> None:
    """Start the page-side watcher (``loaded`` handler). Never raises."""
    try:
        window.evaluate_js(THEME_WATCH_JS)
    except Exception:  # noqa: BLE001
        logger.debug("title bar: could not start the theme watcher", exc_info=True)
