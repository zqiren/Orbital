# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Daemon PID file for singleton enforcement.

Prevents multiple daemon instances from running simultaneously by
writing a PID file at startup and checking for an existing live process.
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PID_PATH = Path.home() / "orbital" / "daemon.pid"
_active_pid_path: Path | None = None


class DaemonAlreadyRunning(RuntimeError):
    """Raised when another daemon instance is already running."""

    def __init__(self, existing_pid: int) -> None:
        self.existing_pid = existing_pid
        super().__init__(
            f"Daemon already running (PID {existing_pid}). "
            f"Stop the existing instance before starting a new one."
        )


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        import psutil
    except ImportError:
        psutil = None

    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            return proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Fallback: use os.kill(pid, 0) if psutil unavailable or errored
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire_pid_file(pid_path: Path | None = None) -> None:
    """Write PID file, refusing to start if another instance is alive.

    Args:
        pid_path: Path for the PID file. Defaults to ~/orbital/daemon.pid.

    Raises:
        DaemonAlreadyRunning: If an existing daemon process is alive.
    """
    global _active_pid_path
    path = pid_path or _DEFAULT_PID_PATH
    path = Path(path)

    # Idempotent: if we already own this PID file, allow re-acquire
    if _active_pid_path is not None and _active_pid_path.resolve() == path.resolve():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    # Claim the file by CREATING it exclusively. Reading, judging the PID dead,
    # and then writing are three steps, and two daemons starting together both
    # passed the judgement before either wrote — that is how two instances came
    # up 377 ms apart on 2026-09-04 and both ran the one-shot credential-card
    # migration, destroying keys. O_CREAT|O_EXCL makes the claim a single
    # atomic step, so exactly one racer can win it.
    for attempt in (1, 2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            break

        # The file exists. Either a live daemon owns it, or it is stale.
        try:
            existing_pid = int(path.read_text().strip())
        except (ValueError, OSError):
            existing_pid = None

        # NOTE: no "is this PID our own?" shortcut here. Re-acquiring our own
        # file is already handled by the _active_pid_path check at the top, and
        # a shortcut on os.getpid() would make a live daemon's own PID look
        # reclaimable — which is exactly the refusal this function exists for.
        if existing_pid is not None and _is_process_alive(existing_pid):
            raise DaemonAlreadyRunning(existing_pid)

        if attempt == 2:
            # Someone re-created it while we were clearing it. Rather than
            # loop, defer to them: a live racer is indistinguishable from a
            # daemon that legitimately owns the file.
            raise DaemonAlreadyRunning(existing_pid or -1)

        logger.warning(
            "Stale PID file found (PID %s is dead), reclaiming", existing_pid,
        )
        try:
            # Unlink rather than overwrite, so the retry goes back through the
            # same exclusive create and only one racer can take it.
            os.unlink(str(path))
        except FileNotFoundError:
            pass

    _active_pid_path = path
    atexit.register(_atexit_cleanup)


def release_pid_file(pid_path: Path | None = None) -> None:
    """Remove the PID file if it contains our PID."""
    global _active_pid_path
    path = pid_path or _active_pid_path or _DEFAULT_PID_PATH
    path = Path(path)

    try:
        if path.exists():
            stored_pid = int(path.read_text().strip())
            if stored_pid == os.getpid():
                path.unlink()
                logger.info("Removed PID file %s", path)
    except (ValueError, OSError):
        pass

    _active_pid_path = None


def _atexit_cleanup() -> None:
    """Atexit handler — removes PID file if it still points to us."""
    release_pid_file()
