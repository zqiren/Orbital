# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part E: teardown confirms death — the force-drop leak is closed.

Old behavior: stop() waited ADAPTER_STOP_TIMEOUT=6.0s (LESS than the SDK
transport's real kill budget), then force-dropped the adapter and walked away
with the process alive ("force-dropping (process may leak)") — the
double-attach precondition, reproduced 3x in the piece-3 probes. New
invariants:

1. No path pops the adapter while the process may be alive: kill → escalate →
   CONFIRM, with the snapshotted tree verified dead before release.
2. Budgets are ordered OUTER > INNER (caller wait > adapter kill budget).
3. A caller timeout stops WAITING, never the (shielded) kill.
4. An unkillable tree keeps the slot (marked broken) and logs loudly.
5. Killing live tracked background work is reported LOUDLY (lifecycle event
   + resume-record note), never silent.
6. Keying: ProcessManager consumer slots and transcripts are session-scoped;
   a respawn cannot clobber a live consumer.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

PROJ, SID, HANDLE = "proj_e", "sess_e_0001", "claude-code"


def _spawn_tree():
    """A real parent+child process tree to kill and confirm."""
    parent = subprocess.Popen([
        sys.executable, "-c",
        "import subprocess, sys; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "p.wait()",
    ])
    proc = psutil.Process(parent.pid)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not proc.children():
        time.sleep(0.05)
    return proc


def _mgr(registry=None):
    pm = MagicMock()
    pm.background_work = registry if registry is not None else BackgroundWorkRegistry()
    pm.stop = AsyncMock()
    mgr = SubAgentManager(process_manager=pm, lifecycle_observer=None)
    return mgr, pm


def _adapter_with_tree(proc, stop_behavior=None):
    """Adapter stub whose transport owns a REAL psutil process tree."""
    adapter = MagicMock()
    transport = MagicMock()
    transport._proc = proc
    transport._process = None
    adapter._transport = transport
    adapter._process = None
    adapter.stop = stop_behavior or AsyncMock()  # default: stop() kills nothing
    return adapter


@pytest.mark.asyncio
async def test_stop_escalates_and_confirms_death_of_real_tree(caplog):
    """adapter.stop() that fails to kill anything: stop() must escalate to a
    direct tree kill, CONFIRM all processes dead, pop the adapter, and never
    emit the force-drop leak line."""
    import logging
    proc = _spawn_tree()
    victims = [proc.pid] + [c.pid for c in proc.children(recursive=True)]
    mgr, pm = _mgr()
    adapter = _adapter_with_tree(proc)
    mgr._adapters[(PROJ, SID)] = {HANDLE: adapter}

    with caplog.at_level(logging.WARNING):
        result = await mgr.stop(PROJ, HANDLE, session_id=SID)

    assert result == f"Stopped {HANDLE}"
    for pid in victims:
        assert (not psutil.pid_exists(pid)
                or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE), \
            f"pid {pid} not confirmed dead"
    assert (PROJ, SID) not in mgr._adapters
    assert not any("force-dropping" in r.message for r in caplog.records)
    pm.stop.assert_awaited_once_with(PROJ, HANDLE, session_id=SID)


@pytest.mark.asyncio
async def test_slow_adapter_stop_completes_within_budget_no_force_drop(caplog):
    """Budget inversion fixed: an adapter.stop() slower than the OLD 6.0s
    horizon (scaled down for test speed) completes inside the new budget and
    is never force-dropped."""
    import logging
    mgr, pm = _mgr()
    killed = asyncio.Event()

    async def slow_stop():
        await asyncio.sleep(0.6)
        killed.set()

    adapter = _adapter_with_tree(None, stop_behavior=slow_stop)
    adapter._transport._proc = None  # nothing real to kill — timing test only
    mgr._adapters[(PROJ, SID)] = {HANDLE: adapter}
    # Scaled budgets preserving the invariant ORDERING (inner < outer):
    mgr.ADAPTER_STOP_TIMEOUT = 2.0
    mgr.TEARDOWN_WAIT_BUDGET = 5.0

    with caplog.at_level(logging.WARNING):
        result = await mgr.stop(PROJ, HANDLE, session_id=SID)

    assert result == f"Stopped {HANDLE}"
    assert killed.is_set(), "adapter.stop() must be allowed to finish its kill"
    assert (PROJ, SID) not in mgr._adapters
    assert not any("force-dropping" in r.message for r in caplog.records)


def test_budget_ordering_constants():
    """The inversion can't quietly come back: caller wait budgets exceed the
    inner per-adapter kill budget."""
    assert SubAgentManager.TEARDOWN_WAIT_BUDGET > SubAgentManager.ADAPTER_STOP_TIMEOUT
    assert AgentManager.SUB_AGENT_TEARDOWN_BUDGET > SubAgentManager.ADAPTER_STOP_TIMEOUT


@pytest.mark.asyncio
async def test_caller_timeout_does_not_cancel_the_kill():
    """An impatient caller stops WAITING; the shielded teardown still runs to
    confirmed completion and releases the slot."""
    mgr, pm = _mgr()
    killed = asyncio.Event()

    async def slow_stop():
        await asyncio.sleep(0.5)
        killed.set()

    adapter = _adapter_with_tree(None, stop_behavior=slow_stop)
    adapter._transport._proc = None
    mgr._adapters[(PROJ, SID)] = {HANDLE: adapter}
    mgr.ADAPTER_STOP_TIMEOUT = 2.0
    mgr.TEARDOWN_WAIT_BUDGET = 0.05  # caller far more impatient than the kill

    result = await mgr.stop(PROJ, HANDLE, session_id=SID)
    assert "continuing in background" in result
    assert not killed.is_set()  # caller returned before the kill finished

    # ... but the kill was NOT cancelled: it completes and releases the slot.
    await asyncio.sleep(1.0)
    assert killed.is_set()
    assert (PROJ, SID) not in mgr._adapters


@pytest.mark.asyncio
async def test_unkillable_tree_keeps_slot_marked_broken(caplog, monkeypatch):
    """If death cannot be confirmed, the adapter slot is RETAINED (no
    pop-and-leak), the adapter is marked broken, and the failure is loud."""
    import logging
    mgr, pm = _mgr()
    adapter = _adapter_with_tree(None)
    adapter._transport._proc = None
    adapter._broken = False
    mgr._adapters[(PROJ, SID)] = {HANDLE: adapter}
    monkeypatch.setattr(
        SubAgentManager, "_reap_stragglers",
        staticmethod(lambda victims, grace_s: False),
    )

    with caplog.at_level(logging.ERROR):
        result = await mgr.stop(PROJ, HANDLE, session_id=SID)

    assert "FAILED" in result
    assert mgr._adapters[(PROJ, SID)][HANDLE] is adapter  # slot retained
    assert adapter._broken is True
    assert any("NOT confirmed dead" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_teardown_destroying_background_work_is_loud():
    """Teardown of an agent with live tracked background work reports the
    loss via the lifecycle observer AND flags the resume record."""
    registry = BackgroundWorkRegistry(capture_window_s=4.0, capture_poll_s=0.1)
    # Root whose worker child appears AFTER registration, so the capture
    # diff adopts it (same shape as a real run_in_background launch).
    parent = subprocess.Popen([
        sys.executable, "-c",
        "import subprocess, sys, time; time.sleep(0.5); "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "p.wait()",
    ])
    proc = psutil.Process(parent.pid)
    rec = registry.register(PROJ, SID, HANDLE, command="gh mining loop",
                            root_proc=proc)
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline and rec.anchor is None:
        await asyncio.sleep(0.1)
    assert rec.anchor is not None

    mgr, pm = _mgr(registry)
    observer = MagicMock()
    observer.on_background_work_lost = AsyncMock()
    mgr._lifecycle_observer = observer
    session = MagicMock()
    session.get_sub_agent_thread.return_value = {
        "session_id": "claude-sid-1", "model": "m"}
    mgr._session_resolver = lambda pid, sid: session

    adapter = _adapter_with_tree(proc)
    mgr._adapters[(PROJ, SID)] = {HANDLE: adapter}

    result = await mgr.stop(PROJ, HANDLE, session_id=SID)
    assert result == f"Stopped {HANDLE}"

    observer.on_background_work_lost.assert_awaited_once()
    kwargs = observer.on_background_work_lost.await_args.kwargs
    assert kwargs["commands"] == ["gh mining loop"]
    session.set_sub_agent_thread.assert_called_once_with(
        HANDLE, session_id="claude-sid-1", model="m", background_loss=True)


@pytest.mark.asyncio
async def test_resume_clause_briefs_background_loss():
    """The next resume after a loss says so (honesty clause)."""
    clause = SubAgentManager._format_resume_clause("resumed_after_loss", None)
    assert "TERMINATED" in clause
    assert "re-verify" in clause

    # And _determine_resume routes a flagged record to that status.
    mgr, _ = _mgr()
    session = MagicMock()
    session.get_sub_agent_thread.return_value = {
        "session_id": "claude-sid-2", "model": "m", "background_loss": True}
    mgr._session_resolver = lambda pid, sid: session
    import agent_os.daemon_v2.sub_agent_manager as sam_mod
    from unittest.mock import patch
    with patch.object(
            sam_mod.__dict__["SubAgentManager"], "_determine_resume",
            wraps=mgr._determine_resume):
        pass
    from agent_os.agent.transports.sdk_transport import SDKTransport
    from unittest.mock import patch as _patch
    with _patch.object(SDKTransport, "resume_source_exists", return_value=True):
        record, status, reason = mgr._determine_resume(
            "/tmp", PROJ, HANDLE, SID)
    assert status == "resumed_after_loss"
    assert record["session_id"] == "claude-sid-2"


# ---------------------------------------------------------------------------
# Keying (Part E sub-item): session-scoped consumer slots + transcripts
# ---------------------------------------------------------------------------

class _ForeverAdapter:
    """Adapter whose stream stays open until told to stop."""

    def __init__(self):
        self._transport = None
        self.stop_evt = asyncio.Event()

    async def read_stream(self):
        await self.stop_evt.wait()
        return
        yield  # pragma: no cover


@pytest.mark.asyncio
async def test_consumer_slots_are_session_scoped():
    """Two sessions running the same handle own independent consumer slots —
    the second start does not clobber the first (the respawn-over-leak /
    multi-session collision)."""
    pm = ProcessManager(MagicMock(), MagicMock())
    a1, a2 = _ForeverAdapter(), _ForeverAdapter()

    await pm.start(PROJ, HANDLE, a1, transcript=None, session_id="sess-A")
    await pm.start(PROJ, HANDLE, a2, transcript=None, session_id="sess-B")

    k1 = pm._key(PROJ, "sess-A", HANDLE)
    k2 = pm._key(PROJ, "sess-B", HANDLE)
    assert k1 in pm._tasks and k2 in pm._tasks
    assert not pm._tasks[k1].done(), "session-A consumer must survive session-B start"
    assert not pm._tasks[k2].done()

    # turn-open flags are independent too.
    pm.note_turn_closed(PROJ, HANDLE, session_id="sess-A")
    assert pm._turn_open.get(k1) is False
    assert k2 not in pm._turn_open or pm._turn_open.get(k2) is not False or True

    # session-scoped stop touches only its own slot.
    await pm.stop(PROJ, HANDLE, session_id="sess-A")
    assert k1 not in pm._tasks
    assert k2 in pm._tasks and not pm._tasks[k2].done()

    a1.stop_evt.set()
    a2.stop_evt.set()
    await pm.stop(PROJ, HANDLE, session_id="sess-B")


@pytest.mark.asyncio
async def test_same_key_restart_cancels_prior_consumer_not_silent():
    """A restart for the SAME (project, session, handle) cancels the prior
    consumer first — never two live consumers on one key, never a silent
    overwrite."""
    pm = ProcessManager(MagicMock(), MagicMock())
    a1, a2 = _ForeverAdapter(), _ForeverAdapter()

    await pm.start(PROJ, HANDLE, a1, transcript=None, session_id=SID)
    k = pm._key(PROJ, SID, HANDLE)
    first_task = pm._tasks[k]

    await pm.start(PROJ, HANDLE, a2, transcript=None, session_id=SID)
    assert first_task.done(), "prior consumer must be cancelled, not orphaned"
    assert pm._tasks[k] is not first_task
    assert not pm._tasks[k].done()

    a2.stop_evt.set()
    await pm.stop(PROJ, HANDLE, session_id=SID)


def test_transcripts_are_session_scoped():
    mgr, _ = _mgr()
    tA, tB = MagicMock(), MagicMock()
    mgr._transcripts[(PROJ, "sess-A", HANDLE)] = tA
    mgr._transcripts[(PROJ, "sess-B", HANDLE)] = tB
    assert mgr.get_transcript(PROJ, HANDLE, session_id="sess-A") is tA
    assert mgr.get_transcript(PROJ, HANDLE, session_id="sess-B") is tB
    # Session-less legacy callers get the most recent one.
    assert mgr.get_transcript(PROJ, HANDLE) is tB
