# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The daemon PID claim is one serialized step across processes.

``~/orbital/daemon.pid`` makes the packaged app and a dev daemon mutually
exclusive. Two windows used to let two daemons both claim it:

1. Both read a PID file whose recorded PID is dead, both judged it stale,
   and both claimed it (the liveness check runs between the read and the
   claim, so a slow check widens the window).
2. One process created the file and had not yet written its PID; another
   read it as empty, judged it stale and claimed it.

Real processes, not threads (two threads share a PID). The children widen
each window deterministically by wrapping module functions with a sleep;
the production code carries no test hook.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import agent_os.utils.pid_file as pf
from agent_os.utils.pid_file import DaemonAlreadyRunning, acquire_pid_file

REPO_ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = f"""
import os, sys, time, pathlib
sys.path.insert(0, {str(REPO_ROOT)!r})
import agent_os.utils.pid_file as pf
from agent_os.utils.pid_file import acquire_pid_file, DaemonAlreadyRunning
"""


def _child(body: str, *args: str) -> subprocess.Popen:
    script = _PRELUDE + textwrap.dedent(body)
    return subprocess.Popen([sys.executable, "-c", script, *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)


def _outcomes(procs, timeout=60):
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=timeout)
        assert p.returncode == 0, err
        outs.append(out.strip())
    return outs


@pytest.fixture(autouse=True)
def _not_already_ours():
    pf._active_pid_path = None
    yield
    pf._active_pid_path = None


def test_two_racers_over_a_stale_pid_file_admit_exactly_one(tmp_path):
    """Window 1, widened: each child's liveness check sleeps first, the
    second one longer, so the first finishes its whole claim while the
    second still holds its stale reading of the file."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("999999")  # a PID that cannot be alive
    gate = tmp_path / "go"
    body = f"""
        real_alive = pf._is_process_alive
        delay = float(sys.argv[1])
        def slow_alive(pid):
            time.sleep(delay)
            return real_alive(pid)
        pf._is_process_alive = slow_alive
        gate = pathlib.Path({str(gate)!r})
        while not gate.exists():
            time.sleep(0.002)
        try:
            acquire_pid_file(pathlib.Path({str(pid_path)!r}))
        except DaemonAlreadyRunning as exc:
            print(f"refused {{exc.existing_pid}}", flush=True)
        else:
            print(f"acquired {{os.getpid()}}", flush=True)
            time.sleep(3)   # stay alive while the other racer judges us
    """
    procs = [_child(body, "0.5"), _child(body, "1.0")]
    time.sleep(0.5)
    gate.write_text("go")
    outs = _outcomes(procs)

    acquired = [o for o in outs if o.startswith("acquired")]
    refused = [o for o in outs if o.startswith("refused")]
    assert len(acquired) == 1, f"exactly one racer may claim a stale PID file, got {outs}"
    assert len(refused) == 1, outs
    # The loser is told the winner's real PID.
    assert refused[0].split()[1] == acquired[0].split()[1], outs


def test_a_claim_in_progress_is_never_seen_empty(tmp_path):
    """Window 2, widened: the claimer pauses mid-write for 1 s.

    The second process starts the moment the PID file exists. It must never
    read the file empty, and it must be refused with the claimer's PID.
    """
    pid_path = tmp_path / "daemon.pid"
    claimer = _child(f"""
        target = {str(pid_path)!r}
        real_fdopen, real_replace = os.fdopen, os.replace
        def slow_fdopen(*a, **k):
            fh = real_fdopen(*a, **k)
            time.sleep(1.0)        # file created, PID not written yet
            return fh
        def slow_replace(src, dst, *a, **k):
            real_replace(src, dst, *a, **k)
            if str(dst) == target:
                time.sleep(1.0)    # PID in place, claim not finished yet
        os.fdopen, os.replace = slow_fdopen, slow_replace
        acquire_pid_file(pathlib.Path(target))
        print(f"acquired {{os.getpid()}}", flush=True)
        time.sleep(3)
    """)
    deadline = time.monotonic() + 20
    while not pid_path.exists():
        assert time.monotonic() < deadline, "claimer never created the PID file"
        time.sleep(0.001)
    seen = pid_path.read_text()
    second = _child(f"""
        try:
            acquire_pid_file(pathlib.Path({str(pid_path)!r}))
        except DaemonAlreadyRunning as exc:
            print(f"refused {{exc.existing_pid}}", flush=True)
        else:
            print(f"acquired {{os.getpid()}}", flush=True)
    """)
    outs = _outcomes([claimer, second])

    assert seen.strip(), "the PID file was observable before the PID was written"
    assert outs[0].startswith("acquired"), outs
    assert outs[1] == f"refused {outs[0].split()[1]}", outs


def _hold_claim_lock(pid_path: Path, seconds: float) -> subprocess.Popen:
    holder = _child(f"""
        with pf._claim_lock(pathlib.Path({str(pid_path)!r}), timeout=5):
            print("holding", flush=True)
            time.sleep({seconds})
    """)
    assert holder.stdout.readline().strip() == "holding"
    return holder


def test_lock_timeout_reports_the_real_pid_from_the_file(tmp_path, monkeypatch):
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("4242")
    monkeypatch.setattr(pf, "_CLAIM_LOCK_TIMEOUT_S", 0.3)
    holder = _hold_claim_lock(pid_path, 30)
    try:
        with pytest.raises(DaemonAlreadyRunning) as exc:
            acquire_pid_file(pid_path)
        assert exc.value.existing_pid == 4242
        assert "-1" not in str(exc.value)
    finally:
        holder.kill()
        holder.communicate()
    assert pid_path.read_text() == "4242"  # a refused claim never writes


def test_lock_timeout_with_no_readable_pid_carries_none(tmp_path, monkeypatch):
    pid_path = tmp_path / "daemon.pid"
    monkeypatch.setattr(pf, "_CLAIM_LOCK_TIMEOUT_S", 0.3)
    holder = _hold_claim_lock(pid_path, 30)
    try:
        with pytest.raises(DaemonAlreadyRunning) as exc:
            acquire_pid_file(pid_path)
        assert exc.value.existing_pid is None
        assert "-1" not in str(exc.value) and "None" not in str(exc.value)
    finally:
        holder.kill()
        holder.communicate()


def test_a_holder_that_dies_never_wedges_startup(tmp_path):
    """The OS frees the lock when its holder dies; the lock file may stay."""
    pid_path = tmp_path / "daemon.pid"
    holder = _hold_claim_lock(pid_path, 30)
    os.kill(holder.pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
    holder.communicate()
    acquire_pid_file(pid_path)
    try:
        assert pid_path.read_text().strip() == str(os.getpid())
        assert pf.read_pid_meta(pid_path)["pid"] == os.getpid()
    finally:
        pf.release_pid_file(pid_path)
    assert not pid_path.exists()


def test_an_empty_leftover_pid_file_is_reclaimed(tmp_path):
    """No live writer exposes an empty file any more, so an empty one is a
    leftover (a crash mid-write in an older build) and must not block."""
    pid_path = tmp_path / "daemon.pid"
    pid_path.write_text("")
    acquire_pid_file(pid_path)
    try:
        assert pid_path.read_text().strip() == str(os.getpid())
    finally:
        pf.release_pid_file(pid_path)


def test_lock_path_derives_from_the_pid_path(tmp_path):
    a = pf._claim_lock_path(tmp_path / "daemon.pid")
    b = pf._claim_lock_path(tmp_path / "iso" / "dev-daemon.pid")
    assert a == tmp_path / "daemon.pid.lock"
    assert b == tmp_path / "iso" / "dev-daemon.pid.lock"
