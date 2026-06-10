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
