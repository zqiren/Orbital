# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part F: minimal resume backstop — never double-attach.

REPORT-piece3-prerequisites.md Q2 observed that resuming a claude session id
whose process is STILL ALIVE silently double-attaches: same id, two live
processes, shared store file, last-writer-wins history loss on the next
resume. Part E closes the leak by construction; this backstop covers the
OS-won't-let-us-kill / crashed-daemon edge:

- the thread record carries a pid+create_time anchor (PID-reuse-safe);
- _determine_resume kills-then-resumes (confirmed) when the anchor is live;
- if death cannot be confirmed, it starts FRESH (resume_failed) rather than
  attaching to a live session.
"""

from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import psutil
import pytest

from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

PROJ, SID, HANDLE = "proj_f", "sess_f_0001", "claude-code"


def _mgr(record):
    pm = MagicMock()
    mgr = SubAgentManager(process_manager=pm)
    session = MagicMock()
    session.get_sub_agent_thread.return_value = record
    mgr._session_resolver = lambda pid, sid: session
    return mgr


def _live_proc():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    proc = psutil.Process(p.pid)
    return proc


def test_no_anchor_resumes_normally():
    record = {"session_id": "csid-1", "model": "m"}
    mgr = _mgr(record)
    with patch("agent_os.agent.transports.sdk_transport.SDKTransport."
               "resume_source_exists", return_value=True):
        rec, status, reason = mgr._determine_resume("/tmp", PROJ, HANDLE, SID)
    assert status == "resumed" and rec is record


def test_dead_anchor_resumes_normally():
    proc = _live_proc()
    pid, ctime = proc.pid, proc.create_time()
    proc.kill()
    proc.wait(timeout=5)
    record = {"session_id": "csid-2", "model": "m",
              "proc_pid": pid, "proc_create_time": ctime}
    mgr = _mgr(record)
    with patch("agent_os.agent.transports.sdk_transport.SDKTransport."
               "resume_source_exists", return_value=True):
        rec, status, reason = mgr._determine_resume("/tmp", PROJ, HANDLE, SID)
    assert status == "resumed"


def test_pid_reuse_reads_as_dead_not_alive():
    """An unrelated process on the recorded pid (create_time mismatch) must
    NOT be killed — identity is pid+create_time."""
    proc = _live_proc()
    try:
        record = {"session_id": "csid-3", "model": "m",
                  "proc_pid": proc.pid,
                  "proc_create_time": proc.create_time() - 9999.0}
        mgr = _mgr(record)
        with patch("agent_os.agent.transports.sdk_transport.SDKTransport."
                   "resume_source_exists", return_value=True):
            rec, status, reason = mgr._determine_resume("/tmp", PROJ, HANDLE, SID)
        assert status == "resumed"
        assert proc.is_running(), "unrelated process must not be touched"
    finally:
        proc.kill()


def test_live_attachment_killed_confirmed_then_resumed():
    """The core backstop: a live process on the record's session id is killed
    (confirmed) BEFORE the resume proceeds — never a double-attach."""
    proc = _live_proc()
    record = {"session_id": "csid-4", "model": "m",
              "proc_pid": proc.pid, "proc_create_time": proc.create_time()}
    mgr = _mgr(record)
    with patch("agent_os.agent.transports.sdk_transport.SDKTransport."
               "resume_source_exists", return_value=True):
        rec, status, reason = mgr._determine_resume("/tmp", PROJ, HANDLE, SID)
    assert status == "resumed"
    assert (not psutil.pid_exists(proc.pid)
            or psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE), \
        "live attachment must be confirmed dead before resume"


def test_unconfirmable_kill_starts_fresh_never_double_attaches(monkeypatch):
    proc = _live_proc()
    try:
        record = {"session_id": "csid-5", "model": "m",
                  "proc_pid": proc.pid, "proc_create_time": proc.create_time()}
        mgr = _mgr(record)
        monkeypatch.setattr(
            SubAgentManager, "_reap_stragglers",
            staticmethod(lambda victims, grace_s: False),
        )
        rec, status, reason = mgr._determine_resume("/tmp", PROJ, HANDLE, SID)
        assert (rec, status, reason) == (None, "fresh", "resume_failed")
    finally:
        proc.kill()


def test_thread_record_persists_proc_identity():
    """set_sub_agent_thread stores the anchor when provided (fed from the
    consume loop's turn_complete via on_thread_update)."""
    from agent_os.agent.session import Session
    import tempfile
    with tempfile.TemporaryDirectory() as ws:
        s = Session.new("sess_pf", ws)
        s.set_sub_agent_thread(
            HANDLE, session_id="csid-6", model="m",
            proc_pid=4242, proc_create_time=123.45,
        )
        rec = s.get_sub_agent_thread(HANDLE)
        assert rec["proc_pid"] == 4242
        assert rec["proc_create_time"] == 123.45
        # absent when not provided
        s.set_sub_agent_thread(HANDLE, session_id="csid-6", model="m")
        assert "proc_pid" not in s.get_sub_agent_thread(HANDLE)
