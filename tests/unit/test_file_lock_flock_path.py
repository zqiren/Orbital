# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the direct-on-data-file lock and sentinel cleanup.

Covers the migration in `cleanup/session-jsonl-fcntl-migration`:

1. The new ``DirectFileLock`` strategy does not leave a ``.lock`` sentinel
   on disk after normal session usage. This is the NEW invariant the
   migration introduces.
2. Two concurrent ``Session.append()`` calls from separate processes do
   not corrupt the JSONL data file. (Same correctness guarantee as the
   legacy sentinel strategy — must hold under the new strategy too.)
3. ``cleanup_orphaned_sentinels`` removes legacy sentinels that no process
   holds, and SKIPS sentinels that are currently locked by another
   process.
4. The feature flag ``USE_SENTINEL_LOCK`` swaps the factory back to the
   legacy strategy for one-release revert safety.
"""

import json
import multiprocessing
import os
import sys
import time

import pytest

from agent_os.agent.session import Session
from agent_os.utils.file_lock import (
    DirectFileLock,
    FileLockError,
    SentinelFileLock,
    cleanup_orphaned_sentinels,
    session_lock,
)


# The DirectFileLock-on-data-file strategy is POSIX-only — see the
# `session_lock()` docstring for why Windows is pinned to SentinelFileLock
# even when the legacy flag is off. Tests that assert direct-lock
# behaviour are skipped on Windows.
_is_posix = sys.platform != "win32"
posix_only = pytest.mark.skipif(not _is_posix, reason="DirectFileLock is POSIX-only")


# ---------------------------------------------------------------------------
# Helpers (must be module-level so spawn-mode multiprocessing can pickle them)
# ---------------------------------------------------------------------------

def _child_append_many(filepath: str, n: int, tag: str, ready_path: str) -> None:
    """Append ``n`` messages tagged with ``tag`` to the session at ``filepath``.

    The session lock is non-blocking and raises FileLockError on contention,
    so the child retries each append until it acquires the lock. This models
    the "serialize through the lock, never corrupt" contract.
    """
    # Signal that this child has imported and is about to start hammering.
    with open(ready_path, "w") as f:
        f.write("ready")
    session = Session(filepath)
    for i in range(n):
        backoff = 0.001
        while True:
            try:
                session.append({"role": "user", "content": f"{tag}-{i}"})
                break
            except FileLockError:
                time.sleep(backoff)
                backoff = min(backoff * 2, 0.05)


def _child_hold_sentinel(sentinel_path: str, ready_path: str, hold_seconds: float) -> None:
    """Acquire ``sentinel_path`` via SentinelFileLock and hold it briefly."""
    lock = SentinelFileLock(sentinel_path)
    lock.acquire()
    with open(ready_path, "w") as f:
        f.write("locked")
    try:
        time.sleep(hold_seconds)
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# 1. NEW invariant: DirectFileLock leaves no `.lock` sentinel on disk
# ---------------------------------------------------------------------------

@posix_only
def test_default_session_creates_no_sentinel_file(tmp_path, monkeypatch):
    """After Session.new() + many appends, NO `.jsonl.lock` file exists.

    This is the headline migration invariant on POSIX: the new strategy
    does not pollute the sessions directory with sentinel artifacts. On
    Windows the strategy stays SentinelFileLock and this invariant does
    not apply (see ``session_lock`` docstring).
    """
    # Defensive: ensure env doesn't force sentinel mode for the in-process flag.
    monkeypatch.delenv("AGENT_OS_USE_SENTINEL_LOCK", raising=False)
    # The module-level USE_SENTINEL_LOCK is read at import; the Session
    # uses session_lock() which reads it live, so override it here.
    import agent_os.utils.file_lock as fl
    monkeypatch.setattr(fl, "USE_SENTINEL_LOCK", False)

    session = Session.new("nosentinel", str(tmp_path))
    for i in range(20):
        session.append({"role": "user", "content": f"msg-{i}"})

    sessions_dir = os.path.dirname(session._filepath)
    leftovers = [n for n in os.listdir(sessions_dir) if n.endswith(".jsonl.lock")]
    assert leftovers == [], (
        f"DirectFileLock should not leave sentinel files, found: {leftovers}"
    )


@posix_only
def test_factory_selects_direct_lock_by_default(monkeypatch, tmp_path):
    """``session_lock`` returns a DirectFileLock on POSIX with flag off."""
    import agent_os.utils.file_lock as fl
    monkeypatch.setattr(fl, "USE_SENTINEL_LOCK", False)
    lock = session_lock(str(tmp_path / "s.jsonl"))
    assert isinstance(lock, DirectFileLock)


def test_factory_selects_sentinel_lock_under_flag(monkeypatch, tmp_path):
    """``session_lock`` returns a SentinelFileLock when the flag is on."""
    import agent_os.utils.file_lock as fl
    monkeypatch.setattr(fl, "USE_SENTINEL_LOCK", True)
    lock = session_lock(str(tmp_path / "s.jsonl"))
    assert isinstance(lock, SentinelFileLock)
    # Sentinel mode should append `.lock` to the data path internally.
    assert lock._lock_path.endswith(".jsonl.lock")


def test_factory_selects_sentinel_lock_on_windows(monkeypatch, tmp_path):
    """On Windows, ``session_lock`` returns SentinelFileLock even with flag off.

    Skipped off-Windows because we cannot easily simulate the Windows
    branch without patching ``sys.platform`` (which other imports may
    have already snapshot-captured).
    """
    if sys.platform != "win32":
        pytest.skip("Windows-only branch check")
    import agent_os.utils.file_lock as fl
    monkeypatch.setattr(fl, "USE_SENTINEL_LOCK", False)
    lock = session_lock(str(tmp_path / "s.jsonl"))
    assert isinstance(lock, SentinelFileLock)


# ---------------------------------------------------------------------------
# 2. Correctness: cross-process appends yield valid JSONL (no corruption)
# ---------------------------------------------------------------------------

def test_concurrent_appends_produce_valid_jsonl(tmp_path):
    """Two processes appending in parallel produce a JSONL where every
    line parses as JSON and the expected number of records is present.

    Sanity-checks that whichever session-lock strategy is active
    (DirectFileLock on POSIX, SentinelFileLock on Windows) serializes
    cross-process writes correctly. On main HEAD this also passes — the
    pre-migration sentinel lock already serialized — but the invariant
    must continue to hold under the new strategy.
    """
    session = Session.new("concurrent", str(tmp_path))
    filepath = session._filepath

    n_per_child = 50
    ctx = multiprocessing.get_context("spawn")
    ready_a = str(tmp_path / "ready_a.txt")
    ready_b = str(tmp_path / "ready_b.txt")
    p_a = ctx.Process(target=_child_append_many, args=(filepath, n_per_child, "A", ready_a))
    p_b = ctx.Process(target=_child_append_many, args=(filepath, n_per_child, "B", ready_b))
    p_a.start()
    p_b.start()
    p_a.join(timeout=60)
    p_b.join(timeout=60)

    assert p_a.exitcode == 0, f"Child A crashed (exit {p_a.exitcode})"
    assert p_b.exitcode == 0, f"Child B crashed (exit {p_b.exitcode})"

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = [ln for ln in f.read().splitlines() if ln.strip()]

    # 1 meta header + 2 * n_per_child user messages
    assert len(raw_lines) == 1 + 2 * n_per_child, (
        f"Expected {1 + 2*n_per_child} lines, got {len(raw_lines)}"
    )

    # Every line is valid JSON
    parsed = [json.loads(ln) for ln in raw_lines]
    assert parsed[0]["role"] == "meta"
    user_msgs = parsed[1:]
    assert all(m["role"] == "user" for m in user_msgs)

    a_count = sum(1 for m in user_msgs if m["content"].startswith("A-"))
    b_count = sum(1 for m in user_msgs if m["content"].startswith("B-"))
    assert a_count == n_per_child
    assert b_count == n_per_child


# ---------------------------------------------------------------------------
# 3. cleanup_orphaned_sentinels — removes orphans, preserves live locks
# ---------------------------------------------------------------------------

def _make_workspace_with_sentinels(tmp_path, sentinel_names):
    """Create a fake workspace layout and seed sentinel files.

    Returns ``(workspace_path, sessions_dir, [sentinel_paths])``.
    """
    workspace = tmp_path / "ws"
    sessions_dir = workspace / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    paths = []
    for name in sentinel_names:
        p = sessions_dir / name
        p.write_text("")  # empty sentinel
        paths.append(str(p))
    return str(workspace), str(sessions_dir), paths


def test_cleanup_removes_orphaned_sentinels(tmp_path):
    """Orphaned ``*.jsonl.lock`` files in sessions dirs are removed."""
    workspace, sessions_dir, paths = _make_workspace_with_sentinels(
        tmp_path,
        ["a.jsonl.lock", "b.jsonl.lock", "c.jsonl.lock"],
    )
    # Sanity: all sentinels exist pre-cleanup.
    for p in paths:
        assert os.path.isfile(p)

    removed = cleanup_orphaned_sentinels([workspace])

    assert removed == 3
    # None of the sentinels remain.
    for p in paths:
        assert not os.path.exists(p), f"Sentinel {p} should have been removed"


def test_cleanup_ignores_non_sentinel_files(tmp_path):
    """Files in sessions/ that aren't ``*.jsonl.lock`` are left untouched."""
    workspace = tmp_path / "ws"
    sessions_dir = workspace / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    # Real JSONL session — must NOT be deleted by the cleanup pass.
    jsonl = sessions_dir / "real.jsonl"
    jsonl.write_text('{"role":"meta","event":"session_start"}\n')
    # An unrelated file — also must NOT be deleted.
    other = sessions_dir / "notes.txt"
    other.write_text("scratch notes")

    removed = cleanup_orphaned_sentinels([str(workspace)])

    assert removed == 0
    assert jsonl.exists()
    assert other.exists()


def test_cleanup_skips_held_sentinels(tmp_path):
    """A sentinel held by another process is preserved (NOT deleted)."""
    workspace, sessions_dir, _ = _make_workspace_with_sentinels(
        tmp_path, []
    )
    held_path = os.path.join(sessions_dir, "held.jsonl.lock")
    orphan_path = os.path.join(sessions_dir, "orphan.jsonl.lock")
    # Pre-create the orphan; the helper will create the held one.
    open(orphan_path, "w").close()

    # Spawn a child that holds `held_path` for a few seconds.
    ctx = multiprocessing.get_context("spawn")
    ready_path = str(tmp_path / "child_ready.txt")
    child = ctx.Process(
        target=_child_hold_sentinel,
        args=(held_path, ready_path, 5.0),
    )
    child.start()
    # Wait for child to acquire the lock.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if os.path.exists(ready_path):
            break
        time.sleep(0.05)
    assert os.path.exists(ready_path), "Child failed to acquire lock in time"

    try:
        removed = cleanup_orphaned_sentinels([workspace])
    finally:
        child.join(timeout=15)
        assert child.exitcode == 0

    # Only the orphan should have been removed.
    assert removed == 1
    assert not os.path.exists(orphan_path)
    # The held sentinel still exists (the child released and exited by now,
    # but cleanup was called while the lock was held — so it must have been
    # skipped at that moment).
    assert os.path.exists(held_path), (
        "Held sentinel was deleted by cleanup — this is a correctness bug"
    )


def test_cleanup_handles_empty_or_missing_workspaces(tmp_path):
    """Empty list, missing dirs, and falsy workspace strings are tolerated."""
    # No workspaces — no error, zero removed.
    assert cleanup_orphaned_sentinels([]) == 0
    # Workspace path that doesn't exist on disk.
    assert cleanup_orphaned_sentinels([str(tmp_path / "does-not-exist")]) == 0
    # Falsy workspace entries (empty string, None) are skipped.
    assert cleanup_orphaned_sentinels(["", None]) == 0


# ---------------------------------------------------------------------------
# 4. DirectFileLock contention semantics (same as legacy: second acquire fails)
# ---------------------------------------------------------------------------

def _child_try_direct_lock(data_path: str, result_path: str) -> None:
    """Child: try to acquire DirectFileLock on data_path; write result."""
    lock = DirectFileLock(data_path)
    try:
        lock.acquire()
    except FileLockError:
        with open(result_path, "w") as f:
            f.write("LOCKED")
        return
    try:
        with open(result_path, "w") as f:
            f.write("OK")
    finally:
        lock.release()


def test_direct_lock_blocks_second_process(tmp_path):
    """A second process gets FileLockError while the first holds the lock."""
    data_path = str(tmp_path / "s.jsonl")
    # Create the file so the lock can be acquired.
    open(data_path, "w").close()
    parent_lock = DirectFileLock(data_path)
    parent_lock.acquire()
    try:
        result_path = str(tmp_path / "child_result.txt")
        ctx = multiprocessing.get_context("spawn")
        child = ctx.Process(target=_child_try_direct_lock, args=(data_path, result_path))
        child.start()
        child.join(timeout=15)
        assert child.exitcode == 0
        result = open(result_path).read().strip()
        assert result == "LOCKED", f"Expected LOCKED, got: {result}"
    finally:
        parent_lock.release()
