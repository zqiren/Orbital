# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the sub-agent termination gap (Direction B — own the kill).

The async kill machinery in ``agent_os.agent.transports.process_kill`` must:
  - kill the parent AND its descendant tree (claude spawns tool children),
  - escalate SIGTERM → grace → SIGKILL,
  - reap the parent (our direct child) so no zombie remains,
  - never block the event loop.

These use synthetic processes so they are deterministic and do not depend on
claude's child-spawning behaviour (the real-claude path is covered by
``test_reap_sdk_transport.py``).
"""

from __future__ import annotations

import subprocess
import sys
import time

import psutil
import pytest

from agent_os.agent.transports.process_kill import kill_process_tree


def _dead(pid: int) -> bool:
    """True if pid is gone or a (reparent-pending) zombie."""
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True
    try:
        if not p.is_running():
            return True
        return p.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _spawn_parent_with_child() -> tuple[subprocess.Popen, int]:
    """Spawn a python parent that itself spawns a long-lived child sleeper.

    Returns (parent_popen, child_pid). The child is the parent's child =
    the test process's grandchild — exactly the daemon→claude→tool topology.
    """
    code = (
        "import subprocess,sys,time;"
        "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);"
        "print(c.pid,flush=True);"
        "time.sleep(120)"
    )
    p = subprocess.Popen([sys.executable, "-c", code],
                         stdout=subprocess.PIPE, text=True)
    child_pid = int(p.stdout.readline().strip())
    return p, child_pid


async def test_tree_kill_reaps_descendants():
    p, child_pid = _spawn_parent_with_child()
    try:
        parent = psutil.Process(p.pid)
        # sanity: the child is a descendant before we kill anything
        desc_pids = [d.pid for d in parent.children(recursive=True)]
        assert child_pid in desc_pids, f"child {child_pid} not in {desc_pids}"

        outcome = await kill_process_tree(
            parent, term_grace=1.0, kill_grace=1.0, reap_parent=True,
        )

        assert outcome.parent_dead is True
        # poll briefly for descendant reaping (reparent-to-init is async)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not _dead(child_pid):
            time.sleep(0.05)
        assert _dead(p.pid), "parent still alive"
        assert _dead(child_pid), "descendant survived tree-kill"
    finally:
        for pid in (p.pid, child_pid):
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass


async def test_stop_escalates_sigterm_to_sigkill():
    # Child installs SIG_IGN for SIGTERM, so only SIGKILL can stop it.
    code = (
        "import signal,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "print('ready',flush=True);"
        "time.sleep(120)"
    )
    p = subprocess.Popen([sys.executable, "-c", code],
                         stdout=subprocess.PIPE, text=True)
    try:
        p.stdout.readline()  # wait until the SIGTERM handler is installed
        proc = psutil.Process(p.pid)

        t0 = time.monotonic()
        outcome = await kill_process_tree(
            proc, term_grace=0.5, kill_grace=1.0, reap_parent=True,
        )
        elapsed = time.monotonic() - t0

        assert outcome.parent_dead is True
        assert _dead(p.pid), "SIGTERM-ignoring process survived escalation"
        # Proves we actually waited the SIGTERM grace before SIGKILL.
        assert elapsed >= 0.5, f"escalation did not honour term_grace ({elapsed:.2f}s)"
    finally:
        try:
            psutil.Process(p.pid).kill()
        except psutil.NoSuchProcess:
            pass
