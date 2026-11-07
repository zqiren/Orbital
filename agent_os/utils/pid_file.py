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

from agent_os.utils.file_lock import DirectFileLock, FileLockError

logger = logging.getLogger(__name__)

# How long a starting daemon waits for another one to finish claiming the file.
_CLAIM_LOCK_TIMEOUT_S = 5.0

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


#: Seconds two create-time readings may differ by and still be one process
#: (psutil rounds differently across calls on some platforms).
_CREATE_TIME_TOLERANCE_S = 1.0


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


def _hold_claim_lock(path: Path) -> DirectFileLock:
    """Take the lock that serialises claiming ``path``.

    Exclusive create only protects the create. Reclaiming a stale file also
    unlinks it, and a racer that judged the file stale could unlink the file
    another racer had just claimed, then create its own: two daemons, each
    holding a PID file. The whole claim therefore runs under this lock, which
    the OS drops if its holder dies.
    """
    lock = DirectFileLock(f"{path}.lock")
    deadline = time.monotonic() + _CLAIM_LOCK_TIMEOUT_S
    while True:
        try:
            lock.acquire()
            return lock
        except FileLockError:
            if time.monotonic() >= deadline:
                raise DaemonAlreadyRunning(-1) from None
            time.sleep(0.02)


def _claim_pid_file(path: Path) -> None:
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
        if existing_pid is not None and is_recorded_daemon_alive(
                existing_pid, read_pid_meta(path)):
            raise DaemonAlreadyRunning(existing_pid)

        if attempt == 2:
            # Someone re-created it while we were clearing it. Rather than
            # loop, defer to them: a live racer is indistinguishable from a
            # daemon that legitimately owns the file.
            raise DaemonAlreadyRunning(existing_pid or -1)

        logger.warning(
            "Stale PID file found (PID %s is dead or now another process), "
            "reclaiming", existing_pid,
        )
        try:
            # Unlink rather than overwrite, so the retry goes back through the
            # same exclusive create and only one racer can take it.
            os.unlink(str(path))
        except FileNotFoundError:
            pass


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

    lock = _hold_claim_lock(path)
    try:
        _claim_pid_file(path)
    finally:
        lock.release()

    _write_meta(path)
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
    meta = read_pid_meta(path)
    if meta is not None and meta.get("pid") == os.getpid():
        try:
            meta_path_for(path).unlink()
        except OSError:
            pass

    _active_pid_path = None


def _atexit_cleanup() -> None:
    """Atexit handler — removes PID file if it still points to us."""
    release_pid_file()
