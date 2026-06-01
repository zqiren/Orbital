# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Own-the-kill process termination for persistent sub-agent transports.

Background: the claude-agent SDK's graceful ``disconnect()`` raises a cross-task
``RuntimeError`` when called from a different asyncio task than the one that
connected (the daemon's reap/eviction path), so the SDK's own
``process.terminate()`` never runs and the claude subprocess leaks as a live
child of the daemon (FINDINGS-reap-confirmation). Direction B works around this
by terminating the process tree ourselves, independent of the SDK.

Design invariants (TASK-reap-fix-direction-b):
  - **Capture the process object, not a bare PID.** ``psutil.Process`` is
    identity-checked by pid+create_time, so a recycled PID raises
    ``NoSuchProcess`` instead of signalling an innocent process.
  - **Kill the descendant tree, not just the parent.** claude spawns tool
    children; SIGKILL'ing only the parent leaks them.
  - **Tree-walk by descendants, never ``killpg``.** The daemon shares claude's
    process group (Step 0 probe: equal PGID), so ``killpg`` would signal the
    daemon itself. The daemon is claude's *ancestor* and can never appear in
    claude's descendant set, so the tree-walk is safe in all group topologies.
  - **Never block the event loop.** All waits are ``await asyncio.sleep`` polls;
    the single reaping ``waitpid`` (for the parent, our direct child) runs in a
    thread via ``asyncio.to_thread``.
  - **Reap the parent (POSIX zombie), poll the rest.** The daemon is the direct
    parent of claude → reap it. It is *not* the parent of grandchildren →
    SIGKILL + poll only (``waitpid`` on a non-child returns ECHILD).

Cross-platform: ``terminate()``/``kill()`` map to ``TerminateProcess`` on
Windows (no signals/zombies); the reaping step is a harmless no-op there.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

DEFAULT_TERM_GRACE = 2.0
DEFAULT_KILL_GRACE = 1.0
_POLL_INTERVAL = 0.05


@dataclass
class KillOutcome:
    """Result of a ``kill_process_tree`` call."""
    parent_dead: bool
    surviving_pids: list[int] = field(default_factory=list)


def _is_dead(p: psutil.Process) -> bool:
    """True if the process is gone or a zombie (terminated, pending reparent-reap)."""
    try:
        if not p.is_running():
            return True
        return p.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return False


def _safe_status(p: psutil.Process) -> str:
    try:
        return p.status()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "gone"


def _signal(p: psutil.Process, method: str) -> None:
    """Send ``terminate``/``kill`` to ``p``, swallowing benign races."""
    try:
        getattr(p, method)()
    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied:
        logger.warning("process_kill: access denied %s pid=%s", method, _pid(p))


def _pid(p: psutil.Process) -> int | str:
    try:
        return p.pid
    except Exception:
        return "?"


async def _wait_until_dead(procs: list[psutil.Process], grace: float) -> None:
    """Poll (non-blocking) until every proc is dead or ``grace`` elapses."""
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if all(_is_dead(p) for p in procs):
            return
        await asyncio.sleep(_POLL_INTERVAL)


async def _reap_parent(proc: psutil.Process) -> None:
    """Reap the parent zombie off the event loop (bounded).

    ``psutil.Process.wait`` reaps via ``waitpid`` when ``proc`` is our own
    child, and merely polls otherwise — so this is safe whether or not the
    caller is the parent, and never raises ECHILD.
    """
    def _blocking() -> None:
        try:
            proc.wait(timeout=DEFAULT_KILL_GRACE)
        except (psutil.TimeoutExpired, psutil.NoSuchProcess):
            pass
        except Exception:
            logger.debug("process_kill: reap of pid=%s raised", _pid(proc), exc_info=True)

    try:
        await asyncio.to_thread(_blocking)
    except Exception:
        logger.debug("process_kill: to_thread reap failed for pid=%s", _pid(proc), exc_info=True)


async def kill_process_tree(
    proc: psutil.Process,
    *,
    term_grace: float = DEFAULT_TERM_GRACE,
    kill_grace: float = DEFAULT_KILL_GRACE,
    reap_parent: bool = True,
    label: str = "",
) -> KillOutcome:
    """SIGTERM→grace→SIGKILL ``proc`` and its descendant tree; reap the parent.

    Returns a :class:`KillOutcome`. Never raises for process-state races and
    never blocks the event loop.
    """
    tag = f" {label}" if label else ""
    try:
        parent_pid = proc.pid
    except Exception:
        return KillOutcome(parent_dead=True)

    if _is_dead(proc):
        if reap_parent:
            await _reap_parent(proc)
        return KillOutcome(parent_dead=True)

    # 1. Snapshot descendants BEFORE signalling — once the parent dies they
    #    reparent to init and we lose the relationship.
    try:
        descendants = list(proc.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        descendants = []
    targets = [proc] + descendants

    # 2. SIGTERM parent + descendants.
    for p in targets:
        _signal(p, "terminate")

    # 3. Bounded grace for graceful exit.
    await _wait_until_dead(targets, term_grace)

    # 4. SIGKILL survivors.
    survivors = [p for p in targets if not _is_dead(p)]
    for p in survivors:
        _signal(p, "kill")

    # 5. Reap the parent zombie (our direct child) off-loop, so its SIGKILL
    #    actually clears. Grandchildren reparent to init and are reaped there —
    #    we only poll them.
    if reap_parent:
        await _reap_parent(proc)

    # 6. Bounded grace for the rest of the tree to die/reparent-reap.
    await _wait_until_dead(targets, kill_grace)

    surviving_pids = sorted({p.pid for p in targets if not _is_dead(p)})
    parent_dead = _is_dead(proc)

    if not parent_dead:
        logger.error(
            "[reap%s] pid=%s survived SIGKILL (status=%s); not hanging",
            tag, parent_pid, _safe_status(proc),
        )
    elif surviving_pids:
        logger.warning(
            "[reap%s] parent pid=%s dead but descendants survived: %s",
            tag, parent_pid, surviving_pids,
        )
    return KillOutcome(parent_dead=parent_dead, surviving_pids=surviving_pids)
