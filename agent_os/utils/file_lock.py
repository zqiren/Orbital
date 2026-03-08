# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cross-platform advisory file lock using OS-level primitives.

Uses fcntl.flock() on Unix and msvcrt.locking() on Windows.
Designed as a context manager for cross-process mutual exclusion
on session JSONL files.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


class FileLockError(OSError):
    """Raised when a file lock cannot be acquired (contention)."""
    pass


class FileLock:
    """Non-blocking exclusive file lock.

    Usage::

        lock = FileLock("/path/to/session.jsonl.lock")
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
            if sys.platform == "win32":
                import msvcrt
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except (OSError, IOError) as exc:
                    os.close(fd)
                    raise FileLockError(
                        f"Session file is locked by another process: {self._lock_path}"
                    ) from exc
            else:
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (OSError, IOError) as exc:
                    os.close(fd)
                    raise FileLockError(
                        f"Session file is locked by another process: {self._lock_path}"
                    ) from exc
            self._fd = fd
        except FileLockError:
            raise
        except Exception:
            os.close(fd)
            raise

    def release(self) -> None:
        """Release the lock if held."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            if sys.platform == "win32":
                import msvcrt
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
        finally:
            os.close(fd)

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __del__(self) -> None:
        if self._fd is not None:
            self.release()
