# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for cross-process session file locking and daemon PID singleton.

Verifies:
- Concurrent appends are blocked (not silently corrupted)
- JSONL append produces only valid JSON lines
- Stale PID files allow daemon startup
- Active PID files block daemon startup
"""

import json
import multiprocessing
import os

import pytest

from agent_os.agent.session import Session
from agent_os.utils.file_lock import FileLock, FileLockError
from agent_os.utils.pid_file import acquire_pid_file, release_pid_file, DaemonAlreadyRunning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _child_append(filepath, result_path):
    """Child process: try to append to a session whose lock is held by parent."""
    try:
        session = Session(filepath)
        session.append({"role": "user", "content": "from child"})
        with open(result_path, "w") as f:
            f.write("OK")
    except FileLockError:
        with open(result_path, "w") as f:
            f.write("LOCKED")
    except Exception as exc:
        with open(result_path, "w") as f:
            f.write(f"ERROR:{exc}")


# ---------------------------------------------------------------------------
# Test: concurrent append is blocked by file lock
# ---------------------------------------------------------------------------

def test_concurrent_append_blocked(tmp_path):
    """A second process cannot append while the first holds the file lock."""
    session = Session.new("test-lock", str(tmp_path))
    lock_path = session._filepath + ".lock"
    result_path = str(tmp_path / "child_result.txt")

    # Parent acquires the file lock
    lock = FileLock(lock_path)
    lock.acquire()
    try:
        # Spawn child that tries to append (which also acquires the same lock)
        ctx = multiprocessing.get_context("fork" if hasattr(os, "fork") else "spawn")
        child = ctx.Process(target=_child_append, args=(session._filepath, result_path))
        child.start()
        child.join(timeout=10)

        assert child.exitcode == 0, f"Child process crashed (exit code {child.exitcode})"
        result = open(result_path).read().strip()
        assert result == "LOCKED", f"Expected LOCKED, got: {result}"
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Test: append produces valid JSONL (atomic writes)
# ---------------------------------------------------------------------------

def test_append_atomic_writes(tmp_path):
    """Every line in the JSONL file is valid JSON after many appends."""
    session = Session.new("test-atomic", str(tmp_path))

    # Append messages of varying sizes
    for i in range(100):
        content = f"Message {i}: " + ("x" * (i * 50))
        session.append({"role": "user", "content": content})

    # Read back raw JSONL and verify every line is valid JSON
    with open(session._filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 100
    for idx, line in enumerate(lines):
        line = line.strip()
        assert line, f"Empty line at index {idx}"
        msg = json.loads(line)  # Raises if invalid JSON
        assert msg["role"] == "user"
        assert f"Message {idx}" in msg["content"]


# ---------------------------------------------------------------------------
# Test: stale PID file allows startup
# ---------------------------------------------------------------------------

def test_stale_pid_file_allows_startup(tmp_path):
    """A PID file pointing to a dead process is treated as stale."""
    pid_path = tmp_path / "daemon.pid"
    # Write a PID that almost certainly doesn't exist
    pid_path.write_text("2000000000")

    # acquire_pid_file should succeed (stale PID)
    acquire_pid_file(pid_path=pid_path)
    try:
        assert pid_path.read_text().strip() == str(os.getpid())
    finally:
        release_pid_file(pid_path=pid_path)


# ---------------------------------------------------------------------------
# Test: active PID file blocks startup
# ---------------------------------------------------------------------------

def test_active_pid_file_blocks_startup(tmp_path):
    """A PID file pointing to a live process blocks a new daemon."""
    pid_path = tmp_path / "daemon.pid"
    # Write our own PID (definitely alive)
    pid_path.write_text(str(os.getpid()))

    with pytest.raises(DaemonAlreadyRunning) as exc_info:
        acquire_pid_file(pid_path=pid_path)

    assert exc_info.value.existing_pid == os.getpid()
    assert "already running" in str(exc_info.value).lower()
