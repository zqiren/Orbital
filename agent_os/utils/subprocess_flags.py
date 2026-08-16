# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Windows console-window suppression for child processes (bug #55).

The packaged Windows build is a pure windowed-subsystem executable
(``console=False`` in ``agent_os/desktop/agentos.spec``), so the daemon
process owns no console. When ``CreateProcess`` spawns a *console*-subsystem
child (``icacls.exe``, ``rg.exe``, ``cmd.exe`` …) from a parent with no
console and no ``CREATE_NO_WINDOW``/``DETACHED_PROCESS`` flag, Windows
allocates a **brand-new visible console** for the child — a black
``conhost.exe`` window flashing over the UI mid-turn.

Every subprocess Orbital spawns is a background operation (ACL grants,
ripgrep searches, CLI health checks, sub-agent I/O). None is meant to
present an interactive terminal, so every creation site passes
``creationflags=win_no_window_flags()``.

**This module deliberately lives outside ``agent_os/platform/windows/``.**
That package only imports on Windows, while most of the affected call sites
(``grep_tool``, ``workspace_scan``, ``shell``, ``setup_engine``,
``migration``, ``settings``, and the two transports) are cross-platform
modules that also run on macOS. ``agent_os/utils`` is the existing home for
dependency-free cross-platform helpers and pulls in no package ``__init__``
side effects.

``tests/unit/test_no_console_window.py`` statically asserts that every
``subprocess.run``/``subprocess.Popen`` site under ``agent_os/`` either
carries this flag or sits on an explicit allowlist.
"""

from __future__ import annotations

import sys

#: Win32 ``CREATE_NO_WINDOW`` (``processthreadsapi.h``). Spelled as a literal
#: rather than ``subprocess.CREATE_NO_WINDOW`` because that attribute does not
#: exist off Windows — importing this module must never raise on macOS/Linux.
#: ``agent_os/platform/windows/process.py:27`` carries the same constant for
#: its ctypes ``CreateProcess`` path.
CREATE_NO_WINDOW = 0x08000000


def win_no_window_flags() -> int:
    """Return ``CREATE_NO_WINDOW`` on Windows, ``0`` everywhere else.

    Safe to call and to OR into an existing ``creationflags`` value on any
    platform: ``subprocess`` only rejects a *non-zero* ``creationflags`` on
    POSIX, so the ``0`` returned off Windows is a no-op rather than an error.
    """
    return CREATE_NO_WINDOW if sys.platform == "win32" else 0
