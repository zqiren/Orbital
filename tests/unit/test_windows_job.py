"""Batch 1110 item 14: worker processes are tied to a kill-on-close job on
Windows, so a hard daemon kill takes every worker (and its shells) with it."""

from __future__ import annotations

import subprocess
import sys

import pytest

from agent_os.utils import windows_job


def test_assign_is_a_quiet_no_op_without_a_pid():
    assert windows_job.assign_worker(None) is False
    assert windows_job.assign_worker(0) is False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pass-through")
def test_off_windows_nothing_is_assigned():
    assert windows_job.assign_worker(12345) is False
    assert windows_job.in_daemon_job(12345) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_a_spawned_process_lands_in_the_daemon_job_and_its_child_too():
    # The real interpreter, not the venv launcher (which spawns a child of its
    # own and exits): one pid, then a grandchild spawned from inside the job.
    exe = getattr(sys, "_base_executable", sys.executable)
    proc = subprocess.Popen(
        [exe, "-c",
         "import subprocess, sys, time; "
         "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
         "print(c.pid, flush=True); time.sleep(60)"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert windows_job.assign_worker(proc.pid) is True
        assert windows_job.in_daemon_job(proc.pid) is True
        child_pid = int(proc.stdout.readline().strip())
        # The child was created inside the job, so it inherited membership.
        assert windows_job.in_daemon_job(child_pid) is True
        # A second assign of the same pid is not an error for the caller.
        assert windows_job.assign_worker(proc.pid) in (True, False)
    finally:
        import psutil
        for p in [proc.pid] + [c.pid for c in psutil.Process(proc.pid).children(recursive=True)]:
            try:
                psutil.Process(p).kill()
            except psutil.Error:
                pass
        proc.wait(timeout=10)
