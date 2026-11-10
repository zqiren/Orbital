# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Daemon PID file for singleton enforcement.

Prevents multiple daemon instances from running simultaneously by
writing a PID file at startup and checking for an existing live process.

Spec 134 A4 — identity. ``daemon.pid`` stays a bare PID (older builds parse
it), and a sidecar ``daemon.pid.meta`` records ``{pid, create_time, exe,
port}``. When the sidecar describes the recorded PID, that PID counts as
live only if the process now holding it was created at the same moment: the
desktop app ends in ``os._exit`` (no ``atexit``), Windows reuses PIDs fast,
and an unrelated process that inherited a stale PID used to make every
later launch refuse to start. A file without a matching sidecar (an older
build wrote it) keeps the old rule: any live process counts.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PID_PATH = Path.home() / "orbital" / "daemon.pid"
_active_pid_path: Path | None = None


class DaemonAlreadyRunning(RuntimeError):
    """Raised when another daemon instance is already running.

    ``existing_pid`` is the PID read from the PID file, or ``None`` when
    another instance holds the claim lock and no PID could be read (it is
    mid-claim). Callers must not try to end a ``None`` PID.
    """

    def __init__(self, existing_pid: int | None) -> None:
        self.existing_pid = existing_pid
        if existing_pid is None:
            super().__init__(
                "Another daemon instance is starting. "
                "Wait for it, or stop it before starting a new one."
            )
        else:
            super().__init__(
                f"Daemon already running (PID {existing_pid}). "
                f"Stop the existing instance before starting a new one."
            )


#: Seconds two create-time readings may differ by and still be one process
#: (psutil rounds differently across calls on some platforms).
_CREATE_TIME_TOLERANCE_S = 1.0

#: Seconds to wait for another process's claim (or release) to finish. A
#: claim holds the lock for a read, a liveness check and two small writes.
_CLAIM_LOCK_TIMEOUT_S = 10.0
_RELEASE_LOCK_TIMEOUT_S = 2.0


class _ClaimLockTimeout(Exception):
    """The claim lock is held by another live process past the timeout."""


def _claim_lock_path(pid_path: Path) -> Path:
    """The lock beside *pid_path*: isolated daemons that override the PID
    path (``scripts/dev_daemon_isolated.py``) get their own lock."""
    return Path(str(pid_path) + ".lock")


class _claim_lock:
    """Cross-process exclusive lock serializing check-and-claim of a PID file.

    Deliberately NOT ``agent_os.utils.file_lock`` (reserved for session
    JSONL). Held only for one claim or release; the OS frees it when the
    holder's descriptor closes, including when the holder dies, so a crash
    can never wedge startup. The lock file is never deleted (deleting a
    flock'd file lets two processes lock two different inodes; and Windows
    cannot delete an open file); its contents mean nothing.

    POSIX: ``fcntl.flock`` on the whole file. Windows: ``msvcrt.locking`` on
    byte 0 (the file stays empty, locking past EOF is allowed), with an
    explicit seek to 0 before lock and unlock because ``msvcrt.locking``
    locks from the current file position.
    """

    def __init__(self, pid_path: Path, timeout: float) -> None:
        self._path = _claim_lock_path(Path(pid_path))
        self._timeout = timeout
        self._fd: int | None = None

    def _try_lock(self, fd: int) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def __enter__(self) -> "_claim_lock":
        fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self._timeout
        while not self._try_lock(fd):
            if time.monotonic() >= deadline:
                os.close(fd)
                raise _ClaimLockTimeout(str(self._path))
            time.sleep(0.02)
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)


def _read_pid(path: Path) -> int | None:
    """The PID recorded in *path*, or None (missing, empty or garbage)."""
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _write_pid_atomic(path: Path) -> None:
    """Write our PID so no reader ever sees the file empty or partial."""
    pid = os.getpid()
    tmp = Path(f"{path}.{pid}.tmp")
    tmp.write_text(str(pid), encoding="utf-8")
    # Windows refuses to replace a file another process has open (a reader
    # outside the lock, e.g. a release check); that is momentary.
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 19:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(0.05)


def meta_path_for(pid_path: Path) -> Path:
    return Path(str(pid_path) + ".meta")


def _process_create_time(pid: int) -> float | None:
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:
        return None


def read_pid_meta(pid_path: Path | None = None) -> dict | None:
    """The sidecar for *pid_path* (default: the daemon's), or None."""
    path = meta_path_for(Path(pid_path or _active_pid_path or _DEFAULT_PID_PATH))
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _write_meta(pid_path: Path) -> None:
    pid = os.getpid()
    meta = {
        "pid": pid,
        "create_time": _process_create_time(pid),
        "exe": sys.executable,
        "port": os.environ.get("AGENT_OS_PORT") or None,
    }
    target = meta_path_for(pid_path)
    tmp = Path(f"{target}.{pid}.tmp")
    try:
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp, target)
    except OSError:
        logger.warning("Could not write %s", target, exc_info=True)
        try:
            tmp.unlink()
        except OSError:
            pass


def _could_be_daemon(pid: int) -> bool:
    """Whether the process holding *pid* is named like a daemon could be.

    Unreadable names count as a daemon: refusing is the safe side.
    """
    try:
        import psutil

        name = (psutil.Process(pid).name() or "").lower()
    except Exception:
        return True
    return name.startswith("orbital") or name.startswith("python")


def is_recorded_daemon_alive(pid: int, meta: dict | None) -> bool:
    """Whether *pid* is still the daemon that wrote the PID file.

    With a sidecar for this PID, the create time must match. Without one (an
    older build wrote the file, and older builds always left it behind), a
    live process counts only if it could be a daemon at all: the packaged
    app or a Python interpreter. Any other process holding a reused PID is
    reclaimed instead of refusing every launch after an upgrade.
    """
    if not _is_process_alive(pid):
        return False
    if not meta or meta.get("pid") != pid or meta.get("create_time") is None:
        return _could_be_daemon(pid)
    current = _process_create_time(pid)
    if current is None:
        return True  # can't tell; refusing is the safe side
    return abs(current - float(meta["create_time"])) <= _CREATE_TIME_TOLERANCE_S


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

    # The whole check-and-claim runs under one cross-process lock. Reading,
    # judging the recorded PID dead and writing are three steps; without the
    # lock two daemons starting together both judged a stale file dead and
    # both claimed it (two instances came up 377 ms apart on 2026-09-04 and
    # both ran the one-shot credential-card migration, destroying keys), and
    # a reader could catch a claimer's file created but still empty.
    try:
        with _claim_lock(path, timeout=_CLAIM_LOCK_TIMEOUT_S):
            existing_pid = _read_pid(path)
            # NOTE: no "is this PID our own?" shortcut here. Re-acquiring our
            # own file is handled by the _active_pid_path check at the top,
            # and a shortcut on os.getpid() would make a live daemon's own
            # PID look reclaimable — the refusal this function exists for.
            if existing_pid is not None and is_recorded_daemon_alive(
                    existing_pid, read_pid_meta(path)):
                raise DaemonAlreadyRunning(existing_pid)
            if path.exists():
                # Dead, reused, or empty/garbage. Every writer now writes
                # atomically under this lock, so an empty file is a leftover
                # (an older build crashed mid-write), never a claim in flight.
                logger.warning(
                    "Stale PID file found (PID %s is dead or now another "
                    "process), reclaiming", existing_pid,
                )
            _write_pid_atomic(path)
            _write_meta(path)
    except _ClaimLockTimeout:
        # Another process is mid-claim (or hung in one). Report the PID it
        # recorded, if any; never a placeholder a caller might try to end.
        raise DaemonAlreadyRunning(_read_pid(path)) from None

    _active_pid_path = path
    atexit.register(_atexit_cleanup)


def release_pid_file(pid_path: Path | None = None) -> None:
    """Remove the PID file if it contains our PID."""
    global _active_pid_path
    path = pid_path or _active_pid_path or _DEFAULT_PID_PATH
    path = Path(path)

    if not _holds_anything(path):
        # Nothing of ours to remove: touch nothing (no lock file either).
        _active_pid_path = None
        return

    # Check-then-unlink under the claim lock, or a newer daemon's claim
    # landing between our read and our unlink would lose its file.
    try:
        with _claim_lock(path, timeout=_RELEASE_LOCK_TIMEOUT_S):
            _remove_if_ours(path)
    except _ClaimLockTimeout:
        # Someone is mid-claim; leaving our file is safe (the identity
        # check reclaims it once we are gone).
        logger.warning("PID file %s busy; leaving it for the next claim", path)
    except OSError:
        # No lock file can be made (the folder is gone): nothing to remove.
        _remove_if_ours(path)

    _active_pid_path = None


def _holds_anything(path: Path) -> bool:
    meta = read_pid_meta(path)
    return _read_pid(path) == os.getpid() or (
        meta is not None and meta.get("pid") == os.getpid())


def _remove_if_ours(path: Path) -> None:
    if _read_pid(path) == os.getpid():
        try:
            path.unlink()
            logger.info("Removed PID file %s", path)
        except OSError:
            pass
    meta = read_pid_meta(path)
    if meta is not None and meta.get("pid") == os.getpid():
        try:
            meta_path_for(path).unlink()
        except OSError:
            pass


def _atexit_cleanup() -> None:
    """Atexit handler — removes PID file if it still points to us."""
    release_pid_file()
