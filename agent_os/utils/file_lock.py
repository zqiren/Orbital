# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cross-platform advisory file lock using OS-level primitives.

Uses fcntl.flock() on Unix and msvcrt.locking() on Windows.

Two locking strategies are supported:

1. :class:`DirectFileLock` (default) — acquires the OS advisory lock on the
   data file itself via its own file descriptor. No on-disk side artifacts.
2. :class:`SentinelFileLock` (legacy) — acquires the OS advisory lock on a
   sibling ``.lock`` sentinel file. Leaves the sentinel on disk; safe to
   remove later if no process holds it.

Default behaviour ships the direct-lock path. The legacy sentinel path is
retained for one release behind ``AGENT_OS_USE_SENTINEL_LOCK=1`` so an
operator can revert without redeploying if a regression surfaces. The
``cleanup_orphaned_sentinels`` helper removes any ``*.jsonl.lock`` files
left over from the sentinel era (or from prior crashes) at daemon startup.

NOTE: this module's lock strategies are only used for session JSONL files.
Do not reuse them for other file types without explicit review.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature flag — default: direct-on-data-file lock; sentinel kept for revert.
# ---------------------------------------------------------------------------

def _env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


USE_SENTINEL_LOCK: bool = _env_flag("AGENT_OS_USE_SENTINEL_LOCK")
"""When True, sessions use the legacy sentinel-file lock path.

Toggleable via the ``AGENT_OS_USE_SENTINEL_LOCK`` environment variable for
one release. The default (False) acquires the advisory lock directly on the
session JSONL data file via :class:`DirectFileLock`.
"""


class FileLockError(OSError):
    """Raised when a file lock cannot be acquired (contention)."""
    pass


# ---------------------------------------------------------------------------
# OS-level locking primitives — shared by both strategies
# ---------------------------------------------------------------------------

# On Windows, `msvcrt.locking` locks a byte range starting at the current file
# pointer. If that byte sits within the file's data region, normal reads,
# writes, and truncations through other descriptors on the same file are
# denied — which is fatal for DirectFileLock, where the same process needs
# to reopen the JSONL for `open(path, "w")` truncate-write under the lock.
#
# The canonical Windows workaround is to lock a single byte at a position
# far beyond any real file data: the lock then exists only for coordination
# and never conflicts with the data range. 2**62 is well past any plausible
# session-JSONL size and matches the convention used by `portalocker` and
# Boost.Interprocess.
#
# POSIX `fcntl.flock` is per-inode (not byte-ranged), so the offset parameter
# is ignored for Linux/macOS — they always lock the whole file.
_WIN_LOCK_OFFSET = 1 << 62


def _os_acquire_nb(fd: int, path_for_error: str, *, beyond_data: bool = False) -> None:
    """Acquire an exclusive non-blocking advisory lock on ``fd``.

    When ``beyond_data`` is True (DirectFileLock case on Windows), the lock
    byte sits at ``_WIN_LOCK_OFFSET`` so it does not conflict with file I/O
    through other descriptors on the same file. POSIX flock locks the inode,
    not byte ranges, and is unaffected by the parameter.

    Raises ``FileLockError`` on contention. Caller is responsible for closing
    ``fd`` on any raise.
    """
    if sys.platform == "win32":
        import msvcrt
        if beyond_data:
            try:
                os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
            except OSError as exc:
                raise FileLockError(
                    f"Could not seek to lock offset on {path_for_error}: {exc}"
                ) from exc
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except (OSError, IOError) as exc:
            raise FileLockError(
                f"Session file is locked by another process: {path_for_error}"
            ) from exc
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError) as exc:
            raise FileLockError(
                f"Session file is locked by another process: {path_for_error}"
            ) from exc


def _os_release(fd: int, *, beyond_data: bool = False) -> None:
    """Release the OS advisory lock on ``fd``. Best-effort; never raises."""
    if sys.platform == "win32":
        import msvcrt
        if beyond_data:
            try:
                os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
            except OSError:
                pass
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except (OSError, IOError):
            pass
    else:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except (OSError, IOError):
            pass


# ---------------------------------------------------------------------------
# Direct-on-data-file lock (preferred — no sentinel artifact)
# ---------------------------------------------------------------------------

class DirectFileLock:
    """Non-blocking exclusive lock acquired on the data file itself.

    The data file is created if missing (so first-time callers can lock
    before the writer opens its own descriptor). The lock is released and
    the descriptor closed on ``release()`` / context-manager exit; the data
    file is never deleted, so the next caller's ``os.open`` always finds it.

    Usage::

        lock = DirectFileLock("/path/to/session.jsonl")
        with lock:
            # ... protected file I/O ...
    """

    def __init__(self, data_path: str) -> None:
        self._data_path = data_path
        self._fd: int | None = None

    def acquire(self) -> None:
        """Acquire the lock. Raises FileLockError if already held elsewhere."""
        # Ensure the file exists so we have something to lock. O_RDWR is
        # required on Windows for msvcrt.locking() to operate on the byte
        # range. The lock byte sits at _WIN_LOCK_OFFSET on Windows so it
        # doesn't conflict with truncate/write through other descriptors.
        fd = os.open(self._data_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _os_acquire_nb(fd, self._data_path, beyond_data=True)
        except FileLockError:
            os.close(fd)
            raise
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        """Release the lock if held."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            _os_release(fd, beyond_data=True)
        finally:
            os.close(fd)

    def __enter__(self) -> DirectFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        if self._fd is not None:
            self.release()


# ---------------------------------------------------------------------------
# Sentinel-file lock (legacy — kept behind feature flag for one release)
# ---------------------------------------------------------------------------

class SentinelFileLock:
    """Non-blocking exclusive lock on a sibling ``.lock`` sentinel file.

    Legacy strategy retained behind ``AGENT_OS_USE_SENTINEL_LOCK=1`` for
    revert safety during the migration to :class:`DirectFileLock`. Functionally
    identical to the pre-migration ``FileLock`` class.

    Usage::

        lock = SentinelFileLock("/path/to/session.jsonl.lock")
        with lock:
            # ... protected file I/O ...
    """

    def __init__(self, lock_path: str) -> None:
        self._lock_path = lock_path
        self._fd: int | None = None

    def acquire(self) -> None:
        """Acquire the lock. Raises FileLockError if already held."""
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            _os_acquire_nb(fd, self._lock_path)
        except FileLockError:
            os.close(fd)
            raise
        except Exception:
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        """Release the lock if held."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            _os_release(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> SentinelFileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        if self._fd is not None:
            self.release()


# ---------------------------------------------------------------------------
# Backward-compatible alias — `FileLock` is what `tests/regression/...`
# and other call sites import. It must continue to behave like the sentinel
# implementation (i.e. accept a `.lock` path and lock that file) so external
# call sites are not broken by the strategy split.
# ---------------------------------------------------------------------------

FileLock = SentinelFileLock


# ---------------------------------------------------------------------------
# Factory — session.py calls this; selection driven by platform + feature flag.
# ---------------------------------------------------------------------------

def session_lock(jsonl_path: str):
    """Return the appropriate session-file lock for the current platform.

    ``jsonl_path`` is the path to the session JSONL data file (e.g.
    ``.../sessions/<sid>.jsonl``), NOT the ``.lock`` sentinel. The factory
    appends ``.lock`` itself when the sentinel strategy is active.

    Strategy selection:

    * POSIX (Linux, macOS) — :class:`DirectFileLock` via ``fcntl.flock`` on
      the JSONL data file itself. No sentinel artifact. ``fcntl.flock``
      locks the inode, so it composes cleanly with the atomic
      ``open(tmp) + os.replace(tmp, jsonl)`` pattern used by compaction.
    * Windows — :class:`SentinelFileLock` on a sibling ``.jsonl.lock`` file.
      Windows holds an exclusive handle on locked byte ranges that blocks
      ``os.replace`` over the same file, so direct-on-data-file locking is
      fundamentally incompatible with compaction's rename semantics. The
      sentinel sidesteps that — and orphaned sentinels are swept by the
      startup ``cleanup_orphaned_sentinels`` pass.

    The ``AGENT_OS_USE_SENTINEL_LOCK=1`` env var forces the sentinel path
    on ALL platforms for one release, so an operator can revert without
    redeploying if the migration surfaces a regression on POSIX.
    """
    if USE_SENTINEL_LOCK or sys.platform == "win32":
        return SentinelFileLock(jsonl_path + ".lock")
    return DirectFileLock(jsonl_path)


# ---------------------------------------------------------------------------
# Startup cleanup — sweep orphaned `.jsonl.lock` sentinels left over from
# the legacy strategy or from crashes.
# ---------------------------------------------------------------------------

def _try_acquire_then_release(path: str) -> bool:
    """Attempt a non-blocking lock on ``path``; release immediately if ok.

    Returns True if the lock was acquired and released (proof that no other
    process currently holds it). Returns False if the lock could not be
    acquired (some process holds it, or open/lock failed for any other
    reason — in either case we MUST NOT delete the file).
    """
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            _os_acquire_nb(fd, path)
        except FileLockError:
            return False
        except Exception:
            return False
        _os_release(fd)
        return True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def cleanup_orphaned_sentinels(workspaces) -> int:
    """Remove orphaned ``*.jsonl.lock`` sentinels under each workspace.

    For every workspace path in ``workspaces``, scans ``<workspace>/orbital/
    sessions/`` for files matching ``*.jsonl.lock``. For each candidate,
    attempts a non-blocking lock to verify no process holds it; only deletes
    when the lock is freely acquirable. This is safe to run at daemon
    startup because:

    1. The new :class:`DirectFileLock` strategy never creates sentinels, so
       any sentinel found is either legacy or orphaned by a crashed process.
    2. The non-blocking-lock probe means we will NEVER delete a sentinel
       that some other live process is actively holding (multi-daemon or
       concurrent test scenarios).

    Returns the number of sentinels removed (for telemetry / logging).
    """
    from agent_os.agent.project_paths import ProjectPaths  # local import to avoid cycle

    removed = 0
    for workspace in workspaces:
        if not workspace:
            continue
        try:
            sessions_dir = ProjectPaths(workspace).sessions_dir
        except (ValueError, TypeError):
            continue
        if not os.path.isdir(sessions_dir):
            continue
        try:
            entries = os.listdir(sessions_dir)
        except OSError:
            continue
        for name in entries:
            if not name.endswith(".jsonl.lock"):
                continue
            sentinel_path = os.path.join(sessions_dir, name)
            if not os.path.isfile(sentinel_path):
                continue
            if not _try_acquire_then_release(sentinel_path):
                logger.info(
                    "Sentinel lock %s appears in use; skipping cleanup.",
                    sentinel_path,
                )
                continue
            try:
                os.remove(sentinel_path)
                removed += 1
                logger.info("Removed orphaned session lock sentinel: %s", sentinel_path)
            except OSError as exc:
                logger.warning(
                    "Failed to remove orphaned sentinel %s: %s",
                    sentinel_path, exc,
                )
    return removed
