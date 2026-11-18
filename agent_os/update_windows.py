# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows install step of the in-app upgrade (core: ``agent_os.updater``).

The downloaded ``Orbital-Setup.exe`` (sha256-checked; the installer is
unsigned, so the hash is the only gate here) is run silently over the
existing install. Inno Setup is the out-of-process helper:

1. We start Setup detached with :func:`setup_command`. It asks for UAC
   (``PrivilegesRequired=admin``; the non-elevated Setup waits for the
   elevated one, so its exit tells us when the user said No).
2. Elevated Setup's ``InitializeSetup`` (``installer/agentos-setup.iss``,
   only under ``/ORBITALUPDATE=1``) creates the :data:`READY_MUTEX` and
   waits up to 60 s for the app's single-instance mutex to go away. Setup's
   own ``AppMutex`` check runs right after it, and with ``/SUPPRESSMSGBOXES``
   a still-running app makes Setup abort cleanly instead of asking.
3. We see the ready mutex, end any of our child processes that run from the
   install dir (the browser driver's ``node.exe``: Restart Manager cannot
   close a windowless process politely), and exit through the normal path.
   If Setup exits first (UAC declined) we stay up and the pill offers the
   restart again.
4. Setup installs with its progress window (``/SILENT``), closing anything
   still holding a file (``/CLOSEAPPLICATIONS /FORCECLOSEAPPLICATIONS``).
   ``DeinitializeSetup`` relaunches ``{app}\\bin\\Orbital.exe`` as the
   original (non-elevated) user, on success AND on abort, so the user gets
   the new version or, if Setup rolled back, the old one.

There is no Windows machine in this repo's dev loop: the command and the
waits are unit-tested with the platform mocked, and the real run is the
manual runbook ``docs/verification/in-app-upgrade-windows.md``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from agent_os.updater import UpdateError

logger = logging.getLogger(__name__)

#: Must match SINGLE_INSTANCE_MUTEX_NAME in agent_os/desktop/main.py and
#: AppMutex in installer/agentos-setup.iss (a contract test checks all three).
APP_MUTEX = "OrbitalDesktopShell"
#: Created by the elevated Setup when it is past UAC (agentos-setup.iss).
READY_MUTEX = "OrbitalUpdateSetupReady"
UPDATE_PARAM = "ORBITALUPDATE"
RELAUNCH_PARAM = "ORBITALRELAUNCH"

SETUP_FLAGS = ("/SILENT", "/SP-", "/SUPPRESSMSGBOXES", "/NORESTART",
               "/CLOSEAPPLICATIONS", "/FORCECLOSEAPPLICATIONS", "/NORESTARTAPPLICATIONS")

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
ERROR_ACCESS_DENIED = 5
ERROR_ELEVATION_REQUIRED = 740

#: How long we wait for the user to answer UAC before giving up and staying.
UAC_WAIT_S = 30 * 60


def install_layout(executable: str | None = None) -> tuple[Path, Path] | None:
    """``({app}, {app}\\bin\\Orbital.exe)`` for an Inno Setup install, else
    None (a portable copy, a dev run): only an installed Orbital updates."""
    exe = Path(executable or sys.executable)
    if exe.name.lower() != "orbital.exe" or exe.parent.name.lower() != "bin":
        return None
    app_dir = exe.parent.parent
    try:
        if not any(app_dir.glob("unins*.exe")):
            return None
    except OSError:
        return None
    return app_dir, exe


def _quoted(value: str) -> str:
    if '"' in value:
        raise UpdateError("install", "unexpected quote in a path")
    return f'"{value}"'


def setup_command(setup: Path, exe: Path, log: Path) -> str:
    """The exact command line. A string, not a list: Inno Setup parses its
    own ``/NAME="value"`` switches, and ``list2cmdline`` would quote the whole
    switch instead of the value."""
    parts = [_quoted(str(setup)), *SETUP_FLAGS, f"/LOG={_quoted(str(log))}",
             f"/{UPDATE_PARAM}=1", f"/{RELAUNCH_PARAM}={_quoted(str(exe))}"]
    return " ".join(parts)


def mutex_exists(name: str) -> bool:
    """Whether a named mutex exists in this session. ``ERROR_ACCESS_DENIED``
    counts as existing: an elevated Setup's mutex may not be ours to open."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    SYNCHRONIZE = 0x00100000
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, name)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == ERROR_ACCESS_DENIED


def _under(path: str, root: str) -> bool:
    try:
        p = os.path.normcase(os.path.realpath(path))
        r = os.path.normcase(os.path.realpath(root)).rstrip("\\/") + os.sep
    except (OSError, ValueError):
        return False
    return p.startswith(r)


def end_children_in(app_dir: Path) -> int:
    """End our descendants whose executable lives in the install dir (they
    would hold files Setup must replace). Returns how many were ended."""
    import psutil

    try:
        children = psutil.Process().children(recursive=True)
    except Exception:
        return 0
    doomed = []
    for child in children:
        try:
            if _under(child.exe(), str(app_dir)):
                doomed.append(child)
        except Exception:
            continue
    for proc in doomed:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        psutil.wait_procs(doomed, timeout=5)
    except Exception:
        pass
    return len(doomed)


class WindowsInstaller:
    def __init__(self, layout: tuple[Path, Path] | None = None, *,
                 popen: Callable[..., object] = subprocess.Popen,
                 mutex_probe: Callable[[str], bool] | None = None,
                 end_children: Callable[[Path], int] = end_children_in,
                 sleep: Callable[[float], None] = time.sleep,
                 uac_wait_s: float = UAC_WAIT_S) -> None:
        self._layout = layout if layout is not None else install_layout()
        self._popen = popen
        self._mutex_probe = mutex_probe or mutex_exists
        self._end_children = end_children
        self._sleep = sleep
        self._uac_wait_s = uac_wait_s

    def unsupported_reason(self) -> str | None:
        return None if self._layout is not None else "not-installed"

    async def prepare(self, downloaded: Path, version: str) -> Path:
        # Nothing to unpack; the sha256 already matched. Setup carries no
        # signature to check (unsigned installer), so there is no second gate.
        return downloaded

    def _spawn(self, cmd: str):
        flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, close_fds=True)
        try:
            return self._popen(cmd, creationflags=flags | CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError as exc:
            if getattr(exc, "winerror", None) == ERROR_ACCESS_DENIED:
                # Our job object forbids breakaway; a plain detached child
                # still outlives us unless the job kills on close.
                return self._popen(cmd, creationflags=flags, **kwargs)
            raise

    def _wait_ready(self, proc) -> str:
        deadline = time.monotonic() + self._uac_wait_s
        while time.monotonic() < deadline:
            if self._mutex_probe(READY_MUTEX):
                return "go"
            if proc.poll() is not None:
                # One last look: Setup may have created it and died at once.
                return "go" if self._mutex_probe(READY_MUTEX) else "cancelled"
            self._sleep(0.25)
        return "cancelled"

    async def handoff(self, setup: Path, version: str, cache: Path) -> str:
        if self._layout is None:
            raise UpdateError("unsupported", "not-installed")
        app_dir, exe = self._layout
        cmd = setup_command(Path(setup), exe, cache / f"setup-{version}.log")
        try:
            proc = self._spawn(cmd)
        except OSError as exc:
            if getattr(exc, "winerror", None) == ERROR_ELEVATION_REQUIRED:
                raise UpdateError("install", "Setup needs elevation it cannot request") from None
            raise UpdateError("install", f"could not start Setup: {exc}") from None
        outcome = await asyncio.to_thread(self._wait_ready, proc)
        if outcome != "go":
            return "cancelled"
        try:
            ended = self._end_children(app_dir)
            if ended:
                logger.info("ended %d child process(es) running from %s", ended, app_dir)
        except Exception:
            logger.debug("could not end install-dir children", exc_info=True)
        return "go"

    def on_startup(self, cache: Path, pending: dict, current: str) -> None:
        return None

    def cleanup(self, cache: Path) -> None:
        # Keep the last Setup logs for Send feedback; drop old ones.
        try:
            logs = sorted(cache.glob("setup-*.log"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return
        for old in logs[:-3]:
            try:
                old.unlink()
            except OSError:
                pass
