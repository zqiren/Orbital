# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for daemon shutdown + sleep prevention.

Auto-resume, the daemon-state.json persistence layer, the heartbeat, and the
shutdown_clean machinery were removed (sessions on disk are the only source of
truth; the daemon never auto-resumes). What remains and is tested here:
graceful shutdown stops agents + appends a session marker, and system-sleep
inhibition is reference-counted by running agents.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.platform.null import NullProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path, platform_provider=None, project_store=None,
                  settings_store=None, credential_store=None):
    """Create an AgentManager wired with all-mock dependencies."""
    return AgentManager(
        project_store=project_store or MagicMock(),
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        platform_provider=platform_provider,
        registry=MagicMock(),
        setup_engine=MagicMock(),
        settings_store=settings_store,
        credential_store=credential_store,
    )


def _make_handle(task_done=False, task_is_none=False, config_snapshot=None,
                 started_at="2026-01-01T00:00:00+00:00"):
    """Create a ProjectHandle with a mock task."""
    session = MagicMock()
    session.append = MagicMock()
    loop = MagicMock()

    if task_is_none:
        task = None
    else:
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = task_done

    return ProjectHandle(
        session=session,
        loop=loop,
        provider=MagicMock(),
        registry=MagicMock(),
        context_manager=MagicMock(),
        interceptor=MagicMock(),
        task=task,
        config_snapshot=config_snapshot or {"workspace": "/tmp/test", "model": "gpt-4"},
        started_at=started_at,
    )


# ===========================================================================
# Graceful shutdown
# ===========================================================================

class TestShutdownMethod:

    @pytest.mark.asyncio
    async def test_shutdown_appends_session_markers(self, tmp_path):
        """shutdown() appends a daemon_shutdown marker to each session."""
        mgr = _make_manager(tmp_path)
        handle = _make_handle(task_done=False)
        mgr._handles[("p1", "default")] = handle
        mgr.stop_agent = AsyncMock()

        await mgr.shutdown()

        handle.session.append.assert_called_once()
        call_args = handle.session.append.call_args[0][0]
        assert call_args["role"] == "system"
        assert call_args["type"] == "daemon_shutdown"
        assert "timestamp" in call_args

    @pytest.mark.asyncio
    async def test_shutdown_stops_all_agents(self, tmp_path):
        """shutdown() calls stop_agent for each handle."""
        mgr = _make_manager(tmp_path)
        mgr._handles[("p1", "default")] = _make_handle(task_done=False)
        mgr._handles[("p2", "default")] = _make_handle(task_done=False)
        mgr.stop_agent = AsyncMock()

        await mgr.shutdown()

        assert mgr.stop_agent.call_count == 2
        stopped_ids = {call.args[0] for call in mgr.stop_agent.call_args_list}
        assert stopped_ids == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self, tmp_path):
        """shutdown() handles timeout gracefully when stop_agent is slow."""
        mgr = _make_manager(tmp_path)
        mgr._handles[("p1", "default")] = _make_handle(task_done=False)

        async def slow_stop(pid, *, session_id=None):
            await asyncio.sleep(10)

        mgr.stop_agent = slow_stop
        await asyncio.wait_for(mgr.shutdown(timeout=0.1), timeout=5.0)

    @pytest.mark.asyncio
    async def test_shutdown_no_handles(self, tmp_path):
        """shutdown() with no handles completes without error."""
        mgr = _make_manager(tmp_path)
        await mgr.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_persists_no_state(self, tmp_path):
        """shutdown() does NOT write any daemon-state file (machinery removed)."""
        mgr = _make_manager(tmp_path)
        await mgr.shutdown()
        # No state file is created anywhere; the manager has no persistence hook.
        assert not hasattr(mgr, "_write_state")
        assert list(tmp_path.glob("**/daemon-state.json")) == []


# ===========================================================================
# Sleep prevention (reference-counted by running agents)
# ===========================================================================

class TestSleepPrevention:

    def test_prevent_sleep_on_first_agent(self, tmp_path):
        """_prevent_sleep_if_needed calls prevent_sleep when handle is None."""
        platform = MagicMock()
        platform.prevent_sleep.return_value = "sleep-handle-1"
        mgr = _make_manager(tmp_path, platform_provider=platform)
        assert mgr._sleep_handle is None

        mgr._prevent_sleep_if_needed()

        platform.prevent_sleep.assert_called_once_with("Orbital: agent(s) running")
        assert mgr._sleep_handle == "sleep-handle-1"

    def test_prevent_sleep_idempotent(self, tmp_path):
        """_prevent_sleep_if_needed is a no-op when sleep is already prevented."""
        platform = MagicMock()
        mgr = _make_manager(tmp_path, platform_provider=platform)
        mgr._sleep_handle = "existing-handle"

        mgr._prevent_sleep_if_needed()

        platform.prevent_sleep.assert_not_called()

    def test_allow_sleep_when_last_agent_stops(self, tmp_path):
        """_allow_sleep_if_idle releases sleep inhibit when no agents running."""
        platform = MagicMock()
        mgr = _make_manager(tmp_path, platform_provider=platform)
        mgr._sleep_handle = "handle-1"

        mgr._allow_sleep_if_idle()

        platform.allow_sleep.assert_called_once_with("handle-1")
        assert mgr._sleep_handle is None

    def test_allow_sleep_preserved_when_agents_running(self, tmp_path):
        """_allow_sleep_if_idle keeps sleep inhibit when agents are running."""
        platform = MagicMock()
        mgr = _make_manager(tmp_path, platform_provider=platform)
        mgr._sleep_handle = "handle-1"
        mgr._handles[("p1", "default")] = _make_handle(task_done=False)

        mgr._allow_sleep_if_idle()

        platform.allow_sleep.assert_not_called()
        assert mgr._sleep_handle == "handle-1"

    def test_allow_sleep_noop_when_no_handle(self, tmp_path):
        """_allow_sleep_if_idle does nothing when sleep_handle is None."""
        platform = MagicMock()
        mgr = _make_manager(tmp_path, platform_provider=platform)
        assert mgr._sleep_handle is None

        mgr._allow_sleep_if_idle()

        platform.allow_sleep.assert_not_called()

    def test_null_provider_noop(self):
        """NullProvider prevent_sleep/allow_sleep don't raise."""
        provider = NullProvider()
        handle = provider.prevent_sleep("test reason")
        assert handle is None
        provider.allow_sleep(handle)

    def test_prevent_sleep_no_provider(self, tmp_path):
        """_prevent_sleep_if_needed is a no-op when platform_provider is None."""
        mgr = _make_manager(tmp_path, platform_provider=None)

        mgr._prevent_sleep_if_needed()

        assert mgr._sleep_handle is None

    def test_prevent_sleep_exception_handled(self, tmp_path):
        """_prevent_sleep_if_needed handles exceptions from prevent_sleep."""
        platform = MagicMock()
        platform.prevent_sleep.side_effect = RuntimeError("OS error")
        mgr = _make_manager(tmp_path, platform_provider=platform)

        mgr._prevent_sleep_if_needed()
        assert mgr._sleep_handle is None

    def test_allow_sleep_exception_handled(self, tmp_path):
        """_allow_sleep_if_idle handles exceptions from allow_sleep and clears handle."""
        platform = MagicMock()
        platform.allow_sleep.side_effect = RuntimeError("OS error")
        mgr = _make_manager(tmp_path, platform_provider=platform)
        mgr._sleep_handle = "handle-1"

        mgr._allow_sleep_if_idle()
        assert mgr._sleep_handle is None


# ===========================================================================
# 1109 keyless smoke F4.2: no worker process outlives the daemon
# ===========================================================================

class _FakeWorker:
    """A live sub-agent adapter; ``proc`` (a Popen) stands in for its CLI."""

    def __init__(self, proc=None, hang=None, clears_proc=False):
        self.stopped = False
        self.display_name = "Claude Code"
        self._process = proc
        self._hang = hang
        self._clears_proc = clears_proc

    async def stop(self):
        if self._clears_proc:
            self._process = None  # what a real transport does before its kill
        if self._hang is not None:
            await self._hang.wait()  # a graceful stop that overruns the wait
        self.stopped = True

    def is_alive(self):
        return not self.stopped


def _sam():
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
    sam = SubAgentManager(process_manager=MagicMock(stop=AsyncMock()))
    sam.STOP_CONFIRM_GRACE = 0.5
    return sam


def _sleeper():
    import subprocess
    import sys
    # The real interpreter: a venv's python.exe on Windows is a launcher that
    # spawns the interpreter as a child, which would be a second pid.
    exe = getattr(sys, "_base_executable", None) or sys.executable
    return subprocess.Popen([exe, "-c", "import time; time.sleep(60)"])


def _cleanup(proc, release):
    release.set()
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


class TestSubAgentManagerShutdown:

    @pytest.mark.asyncio
    async def test_every_registered_worker_is_stopped(self):
        """A keyless pinned send / worker onboarding / import scan registers
        its adapter with no manager handle; it is still stopped."""
        from agent_os.daemon_v2.models import make_session_key

        sam = _sam()
        scan, pinned = _FakeWorker(), _FakeWorker()
        sam._adapters[make_session_key("p1", "p1_scan")] = {"claude-code": scan}
        sam._adapters[make_session_key("p2", "p2_pin")] = {"codex": pinned}

        await sam.shutdown(5.0)

        assert scan.stopped and pinned.stopped
        assert sam._adapters == {}

    @pytest.mark.asyncio
    async def test_a_worker_whose_stop_overruns_the_timeout_is_still_killed(self):
        """1109 smoke, F4.2 attempt 1: a mid-turn Claude Code's graceful stop
        outlasted the timeout, its shielded kill died with the event loop and
        the CLI survived with parent 1. Shutdown confirms the process dead."""
        from agent_os.daemon_v2.models import make_session_key

        proc, release = _sleeper(), asyncio.Event()
        try:
            sam = _sam()
            sam._adapters[make_session_key("p1", "p1_scan")] = {
                "claude-code": _FakeWorker(proc, hang=release)}

            await sam.shutdown(0.5)

            assert proc.poll() is not None, "worker process outlived shutdown"
        finally:
            _cleanup(proc, release)
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_an_earlier_snapshot_reaches_a_process_a_stop_already_forgot(self):
        """``procs`` taken before any stop still covers a worker whose stop
        cleared its process reference and left the process running."""
        from agent_os.daemon_v2.models import make_session_key

        proc, release = _sleeper(), asyncio.Event()
        try:
            sam = _sam()
            worker = _FakeWorker(proc, hang=release, clears_proc=True)
            sam._adapters[make_session_key("p1", "s1")] = {"codex": worker}
            procs = sam.snapshot_worker_procs()
            assert [p.pid for p in procs] == [proc.pid]
            worker._process = None

            await sam.shutdown(0.5, procs=procs)

            assert proc.poll() is not None
        finally:
            _cleanup(proc, release)
            await asyncio.sleep(0)


class TestAgentManagerShutdownStopsWorkers:

    @pytest.mark.asyncio
    async def test_sub_agent_shutdown_is_called_with_the_early_snapshot(self, tmp_path):
        mgr = _make_manager(tmp_path)
        sam = mgr._sub_agent_manager
        sam.snapshot_worker_procs = MagicMock(return_value=["snap"])
        sam.shutdown = AsyncMock()

        await mgr.shutdown(timeout=3.0)

        sam.shutdown.assert_awaited_once_with(3.0, procs=["snap"])

    @pytest.mark.asyncio
    async def test_snapshot_is_taken_before_the_handle_loop_stops(self, tmp_path):
        """A handled session's stop clears the transport's process reference
        and leaves the CLI alive (its wait ran out): the snapshot from before
        the handle loop still kills it."""
        from agent_os.daemon_v2.models import make_session_key

        proc, release = _sleeper(), asyncio.Event()
        try:
            sam = _sam()
            mgr = _make_manager(tmp_path)
            mgr._sub_agent_manager = sam
            mgr._handles[make_session_key("p1", "s1")] = _make_handle()
            worker = _FakeWorker(proc, hang=release, clears_proc=True)
            sam._adapters[make_session_key("p1", "s1")] = {"codex": worker}

            async def stop_agent(pid, *, session_id=None, keep_browser=False):
                await sam.stop_all(pid, session_id=session_id)

            mgr.stop_agent = stop_agent

            await mgr.shutdown(timeout=0.5)

            assert proc.poll() is not None, "worker process outlived shutdown"
        finally:
            _cleanup(proc, release)
            await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_a_failing_sub_agent_shutdown_still_tears_down_dispatchers(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr._sub_agent_manager.snapshot_worker_procs = MagicMock(return_value=[])
        mgr._sub_agent_manager.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        dispatcher = MagicMock(shutdown=AsyncMock())
        mgr._dispatchers["p1"] = dispatcher

        await mgr.shutdown(timeout=1.0)

        dispatcher.shutdown.assert_awaited_once()
        assert mgr._dispatchers == {}
