# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cross-platform process-tree helpers built on :mod:`psutil`.

These helpers deliberately swallow ``psutil.NoSuchProcess`` /
``psutil.AccessDenied`` — by the time a test asserts on tree shape the
process(es) under inspection may already be exiting. Callers that need
strict error propagation can check return values instead.
"""

from __future__ import annotations

import re
import time
from typing import Iterable

import psutil


def get_children(pid: int, recursive: bool = True) -> list[psutil.Process]:
    """Return child processes of ``pid``.

    ``recursive=True`` (the default) walks the full descendant tree.
    Returns an empty list if the parent no longer exists.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    try:
        return list(parent.children(recursive=recursive))
    except psutil.NoSuchProcess:
        return []


def find_children_by_name(pid: int, name_pattern: str) -> list[psutil.Process]:
    """Return descendants of ``pid`` whose ``.name()`` matches ``name_pattern``.

    Matching is case-insensitive regex. Useful for asserting the daemon
    spawned (or reaped) a specific sub-agent binary such as ``claude`` or
    ``python`` on Windows.
    """
    regex = re.compile(name_pattern, re.IGNORECASE)
    matches: list[psutil.Process] = []
    for child in get_children(pid, recursive=True):
        try:
            if regex.search(child.name()):
                matches.append(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def kill_process(proc: psutil.Process, timeout: float = 2.0) -> bool:
    """Terminate ``proc`` then kill if still alive after ``timeout``.

    Returns ``True`` if the process is gone by the time this function
    exits, ``False`` otherwise.
    """
    try:
        proc.terminate()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        pass

    try:
        proc.wait(timeout=timeout)
        return True
    except psutil.TimeoutExpired:
        pass
    except psutil.NoSuchProcess:
        return True

    try:
        proc.kill()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return False

    try:
        proc.wait(timeout=timeout)
        return True
    except (psutil.TimeoutExpired, psutil.NoSuchProcess):
        # NoSuchProcess after kill = success; TimeoutExpired = failure
        try:
            return not proc.is_running()
        except psutil.NoSuchProcess:
            return True


def kill_process_tree(pid: int, timeout: float = 5.0) -> None:
    """Reap ``pid`` and all its descendants.

    Leaf processes are terminated first so the tree collapses cleanly.
    On Windows both ``terminate()`` and ``kill()`` map to
    ``TerminateProcess`` — the distinction matters only on POSIX.
    """
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Capture descendants BEFORE we start terminating; once the parent
    # dies on POSIX the children may be re-parented to init and we lose
    # the relationship.
    try:
        descendants = list(root.children(recursive=True))
    except psutil.NoSuchProcess:
        descendants = []

    # Leaves first: sort by descendant depth (longest ancestor chain = leaf).
    # We approximate by sorting on the len(parents()) list.
    def _depth(p: psutil.Process) -> int:
        try:
            return len(p.parents())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    descendants.sort(key=_depth, reverse=True)

    procs: list[psutil.Process] = list(descendants) + [root]

    # Phase 1: terminate all.
    for proc in procs:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Phase 2: wait briefly for graceful exit.
    deadline = time.monotonic() + timeout
    for proc in procs:
        remaining = max(0.05, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except psutil.TimeoutExpired:
            pass
        except psutil.NoSuchProcess:
            continue

    # Phase 3: hard-kill any stragglers.
    for proc in procs:
        try:
            if proc.is_running():
                proc.kill()
        except psutil.NoSuchProcess:
            continue

    # Final short wait so callers see a clean tree on return.
    for proc in procs:
        try:
            proc.wait(timeout=1.0)
        except (psutil.TimeoutExpired, psutil.NoSuchProcess):
            continue


def count_descendants(pid: int, name_filter: str | None = None) -> int:
    """Count living descendants of ``pid``, optionally filtered by name regex."""
    if name_filter is None:
        return len(get_children(pid, recursive=True))
    return len(find_children_by_name(pid, name_filter))


def wait_for_exit(pid: int, timeout: float = 5.0) -> bool:
    """Block until ``pid`` exits or ``timeout`` elapses.

    Returns ``True`` if the process is gone by the deadline.
    """
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    try:
        proc.wait(timeout=timeout)
        return True
    except psutil.TimeoutExpired:
        return False
    except psutil.NoSuchProcess:
        return True


def iter_pids(procs: Iterable[psutil.Process]) -> list[int]:
    """Return the PID list for a collection of :class:`psutil.Process` objects."""
    out: list[int] = []
    for p in procs:
        try:
            out.append(p.pid)
        except psutil.NoSuchProcess:
            continue
    return out
