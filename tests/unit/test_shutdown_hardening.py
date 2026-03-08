# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for daemon shutdown hardening: state file, heartbeat, graceful shutdown, auto-resume, sleep prevention."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.platform.null import NullProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path, platform_provider=None, project_store=None,
                  settings_store=None, credential_store=None):
    """Create an AgentManager wired with all-mock dependencies, state_file in tmp_path."""
    mgr = AgentManager(
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
    # Redirect state file to tmp_path so tests don't touch real ~/.agent-os
    mgr._state_file = tmp_path / "daemon-state.json"
    return mgr


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
# State File
# ===========================================================================

class TestStateFile:

    def test_write_state_creates_file(self, tmp_path):
        """_write_state creates daemon-state.json with correct JSON structure."""
        mgr = _make_manager(tmp_path)
        mgr._handles["proj-1"] = _make_handle(task_done=False)

        mgr._write_state()

        state_file = tmp_path / "daemon-state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert data["version"] == 1
        assert "last_heartbeat" in data
        assert "proj-1" in data["agents"]
        assert data["agents"]["proj-1"]["status"] == "running"
        assert data["agents"]["proj-1"]["shutdown_clean"] is False

    def test_write_state_only_running_agents(self, tmp_path):
        """Only agents with running tasks appear in the state file."""
        mgr = _make_manager(tmp_path)
        mgr._handles["running"] = _make_handle(task_done=False)
        mgr._handles["done"] = _make_handle(task_done=True)
        mgr._handles["no-task"] = _make_handle(task_is_none=True)

        mgr._write_state()

        data = json.loads((tmp_path / "daemon-state.json").read_text(encoding="utf-8"))
        assert "running" in data["agents"]
        assert "done" not in data["agents"]
        assert "no-task" not in data["agents"]

    def test_write_state_atomic(self, tmp_path):
        """_write_state uses a .tmp file and replaces atomically."""
        mgr = _make_manager(tmp_path)
        mgr._handles["p1"] = _make_handle(task_done=False)

        # We verify by checking the final file exists and no .tmp lingers
        mgr._write_state()

        assert (tmp_path / "daemon-state.json").exists()
        assert not (tmp_path / "daemon-state.tmp").exists()

    def test_read_state_returns_dict(self, tmp_path):
        """_read_state returns parsed dict from valid JSON."""
        mgr = _make_manager(tmp_path)
        state = {"version": 1, "agents": {"p1": {"status": "running"}}}
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        result = mgr._read_state()
        assert result == state

    def test_read_state_missing_file(self, tmp_path):
        """_read_state returns None when the file doesn't exist."""
        mgr = _make_manager(tmp_path)
        # No file written
        assert mgr._read_state() is None

    def test_read_state_corrupt_json(self, tmp_path):
        """_read_state returns None for corrupt JSON."""
        mgr = _make_manager(tmp_path)
        (tmp_path / "daemon-state.json").write_text("not valid json {{{{", encoding="utf-8")

        assert mgr._read_state() is None

    def test_write_state_handles_missing_dir(self, tmp_path):
        """_write_state creates parent directories if missing."""
        mgr = _make_manager(tmp_path)
        mgr._state_file = tmp_path / "nested" / "deep" / "daemon-state.json"
        mgr._handles["p1"] = _make_handle(task_done=False)

        mgr._write_state()

        assert mgr._state_file.exists()
        data = json.loads(mgr._state_file.read_text(encoding="utf-8"))
        assert "p1" in data["agents"]

    def test_write_state_preserves_config_snapshot(self, tmp_path):
        """config_snapshot is included in the state file."""
        mgr = _make_manager(tmp_path)
        snapshot = {"workspace": "/home/user/project", "model": "claude-3"}
        mgr._handles["p1"] = _make_handle(task_done=False, config_snapshot=snapshot)

        mgr._write_state()

        data = json.loads((tmp_path / "daemon-state.json").read_text(encoding="utf-8"))
        assert data["agents"]["p1"]["config_snapshot"] == snapshot


# ===========================================================================
# Heartbeat
# ===========================================================================

class TestHeartbeat:

    def test_ensure_heartbeat_starts_task(self, tmp_path):
        """_ensure_heartbeat_running creates an asyncio task."""
        mgr = _make_manager(tmp_path)
        assert mgr._heartbeat_task is None

        with patch("asyncio.create_task") as mock_create:
            mock_task = MagicMock(spec=asyncio.Task)
            mock_task.done.return_value = False
            mock_create.return_value = mock_task

            mgr._ensure_heartbeat_running()

            mock_create.assert_called_once()
            assert mgr._heartbeat_task is mock_task

    def test_ensure_heartbeat_idempotent(self, tmp_path):
        """Calling _ensure_heartbeat_running twice only creates one task."""
        mgr = _make_manager(tmp_path)

        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False

        with patch("asyncio.create_task") as mock_create:
            mock_create.return_value = mock_task
            mgr._ensure_heartbeat_running()
            mgr._ensure_heartbeat_running()

            assert mock_create.call_count == 1

    def test_stop_heartbeat_when_no_agents(self, tmp_path):
        """_stop_heartbeat_if_idle cancels task when no running agents."""
        mgr = _make_manager(tmp_path)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        mgr._heartbeat_task = mock_task
        # No handles = no running agents

        mgr._stop_heartbeat_if_idle()

        mock_task.cancel.assert_called_once()
        assert mgr._heartbeat_task is None

    def test_stop_heartbeat_preserves_when_agents_running(self, tmp_path):
        """_stop_heartbeat_if_idle keeps heartbeat when agents are running."""
        mgr = _make_manager(tmp_path)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        mgr._heartbeat_task = mock_task
        mgr._handles["p1"] = _make_handle(task_done=False)

        mgr._stop_heartbeat_if_idle()

        mock_task.cancel.assert_not_called()
        assert mgr._heartbeat_task is mock_task

    def test_stop_heartbeat_noop_when_none(self, tmp_path):
        """_stop_heartbeat_if_idle does nothing when heartbeat is None."""
        mgr = _make_manager(tmp_path)
        assert mgr._heartbeat_task is None
        # Should not raise
        mgr._stop_heartbeat_if_idle()
        assert mgr._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_heartbeat_writes_state_periodically(self, tmp_path):
        """_start_heartbeat writes state and sleeps in a loop."""
        mgr = _make_manager(tmp_path)
        mgr._handles["p1"] = _make_handle(task_done=False)

        call_count = 0
        original_write = mgr._write_state

        def counting_write():
            nonlocal call_count
            call_count += 1
            original_write()
            if call_count >= 2:
                # Cancel after 2 writes
                raise asyncio.CancelledError()

        mgr._write_state = counting_write

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, None]
            await mgr._start_heartbeat()

        assert call_count >= 2


# ===========================================================================
# Mark Shutdown Clean
# ===========================================================================

class TestMarkShutdownClean:

    def test_mark_sets_shutdown_clean_true(self, tmp_path):
        """mark_shutdown_clean sets shutdown_clean=True for all agents."""
        mgr = _make_manager(tmp_path)
        state = {
            "version": 1,
            "agents": {
                "p1": {"status": "running", "shutdown_clean": False},
                "p2": {"status": "running", "shutdown_clean": False},
            },
        }
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        mgr.mark_shutdown_clean()

        result = json.loads((tmp_path / "daemon-state.json").read_text(encoding="utf-8"))
        assert result["agents"]["p1"]["shutdown_clean"] is True
        assert result["agents"]["p2"]["shutdown_clean"] is True

    def test_mark_no_state_file(self, tmp_path):
        """mark_shutdown_clean with no state file does not raise."""
        mgr = _make_manager(tmp_path)
        # No state file exists
        mgr.mark_shutdown_clean()  # should not raise


# ===========================================================================
# Shutdown Method
# ===========================================================================

class TestShutdownMethod:

    @pytest.mark.asyncio
    async def test_shutdown_marks_clean(self, tmp_path):
        """shutdown() calls mark_shutdown_clean."""
        mgr = _make_manager(tmp_path)
        mgr.mark_shutdown_clean = MagicMock()

        await mgr.shutdown()

        mgr.mark_shutdown_clean.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_appends_session_markers(self, tmp_path):
        """shutdown() appends daemon_shutdown marker to each session."""
        mgr = _make_manager(tmp_path)
        handle = _make_handle(task_done=False)
        mgr._handles["p1"] = handle

        # Mock stop_agent to prevent actual cleanup
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
        mgr._handles["p1"] = _make_handle(task_done=False)
        mgr._handles["p2"] = _make_handle(task_done=False)
        mgr.stop_agent = AsyncMock()

        await mgr.shutdown()

        assert mgr.stop_agent.call_count == 2
        stopped_ids = {call.args[0] for call in mgr.stop_agent.call_args_list}
        assert stopped_ids == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_shutdown_cancels_heartbeat(self, tmp_path):
        """shutdown() cancels the heartbeat task."""
        mgr = _make_manager(tmp_path)
        mock_task = MagicMock(spec=asyncio.Task)
        mock_task.done.return_value = False
        mgr._heartbeat_task = mock_task

        await mgr.shutdown()

        mock_task.cancel.assert_called_once()
        assert mgr._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_shutdown_writes_final_state(self, tmp_path):
        """shutdown() calls _write_state at the end."""
        mgr = _make_manager(tmp_path)
        mgr._write_state = MagicMock()

        await mgr.shutdown()

        mgr._write_state.assert_called()

    @pytest.mark.asyncio
    async def test_shutdown_timeout(self, tmp_path):
        """shutdown() handles timeout gracefully when stop_agent is slow."""
        mgr = _make_manager(tmp_path)
        mgr._handles["p1"] = _make_handle(task_done=False)

        async def slow_stop(pid):
            await asyncio.sleep(10)

        mgr.stop_agent = slow_stop

        # Should complete within timeout, not hang
        await asyncio.wait_for(mgr.shutdown(timeout=0.1), timeout=5.0)

    @pytest.mark.asyncio
    async def test_shutdown_no_handles(self, tmp_path):
        """shutdown() with no handles completes without error."""
        mgr = _make_manager(tmp_path)
        await mgr.shutdown()


# ===========================================================================
# Auto-Resume
# ===========================================================================

class TestAutoResume:

    @pytest.mark.asyncio
    async def test_resume_starts_running_agents(self, tmp_path):
        """auto_resume_agents starts agents from the state file."""
        project_store = MagicMock()
        project_store.get_project.return_value = {
            "workspace": str(tmp_path),
            "name": "TestProject",
            "autonomy": "hands_off",
            "model": "gpt-4",
            "api_key": "test-key",
        }

        settings_store = MagicMock()
        settings_store.get.return_value = MagicMock(
            llm=MagicMock(api_key=None, base_url=None, model=None),
        )
        credential_store = MagicMock()
        credential_store.get_api_key.return_value = None

        mgr = _make_manager(
            tmp_path,
            project_store=project_store,
            settings_store=settings_store,
            credential_store=credential_store,
        )
        mgr.start_agent = AsyncMock()

        state = {
            "version": 1,
            "agents": {
                "proj-1": {
                    "status": "running",
                    "config_snapshot": {"workspace": str(tmp_path)},
                    "shutdown_clean": True,
                },
            },
        }
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        await mgr.auto_resume_agents()

        mgr.start_agent.assert_called_once()
        call_args = mgr.start_agent.call_args
        assert call_args[0][0] == "proj-1"  # project_id
        assert call_args[1]["trigger_source"] == "auto_resume"

    @pytest.mark.asyncio
    async def test_resume_skips_missing_projects(self, tmp_path, caplog):
        """auto_resume_agents warns and skips when project is not in store."""
        project_store = MagicMock()
        project_store.get_project.return_value = None

        mgr = _make_manager(tmp_path, project_store=project_store)

        state = {
            "version": 1,
            "agents": {
                "gone-project": {
                    "status": "running",
                    "config_snapshot": {},
                    "shutdown_clean": True,
                },
            },
        }
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        import logging
        with caplog.at_level(logging.WARNING):
            await mgr.auto_resume_agents()

        assert any("no longer exists" in rec.message for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_resume_no_state_file(self, tmp_path):
        """auto_resume_agents with no state file returns without error."""
        mgr = _make_manager(tmp_path)
        # No state file
        await mgr.auto_resume_agents()

    @pytest.mark.asyncio
    async def test_resume_injects_crash_warning(self, tmp_path):
        """When shutdown_clean=False, a system warning is appended."""
        project_store = MagicMock()
        project_store.get_project.return_value = {
            "workspace": str(tmp_path),
            "name": "CrashedProject",
            "autonomy": "hands_off",
            "model": "gpt-4",
            "api_key": "test-key",
        }

        settings_store = MagicMock()
        settings_store.get.return_value = MagicMock(
            llm=MagicMock(api_key=None, base_url=None, model=None),
        )
        credential_store = MagicMock()
        credential_store.get_api_key.return_value = None

        mgr = _make_manager(
            tmp_path,
            project_store=project_store,
            settings_store=settings_store,
            credential_store=credential_store,
        )

        # Mock start_agent to just populate _handles
        mock_handle = _make_handle(task_done=False)
        async def fake_start(pid, config, initial_message=None, trigger_source=None):
            mgr._handles[pid] = mock_handle

        mgr.start_agent = AsyncMock(side_effect=fake_start)

        state = {
            "version": 1,
            "agents": {
                "proj-crash": {
                    "status": "running",
                    "config_snapshot": {},
                    "shutdown_clean": False,  # NOT clean
                },
            },
        }
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        await mgr.auto_resume_agents()

        # Verify the crash warning was appended
        mock_handle.session.append.assert_called()
        warn_calls = [
            c for c in mock_handle.session.append.call_args_list
            if "interrupted unexpectedly" in str(c)
        ]
        assert len(warn_calls) == 1

    @pytest.mark.asyncio
    async def test_resume_no_warning_on_clean_shutdown(self, tmp_path):
        """When shutdown_clean=True, no extra warning is appended."""
        project_store = MagicMock()
        project_store.get_project.return_value = {
            "workspace": str(tmp_path),
            "name": "CleanProject",
            "autonomy": "hands_off",
            "model": "gpt-4",
            "api_key": "test-key",
        }

        settings_store = MagicMock()
        settings_store.get.return_value = MagicMock(
            llm=MagicMock(api_key=None, base_url=None, model=None),
        )
        credential_store = MagicMock()
        credential_store.get_api_key.return_value = None

        mgr = _make_manager(
            tmp_path,
            project_store=project_store,
            settings_store=settings_store,
            credential_store=credential_store,
        )

        mock_handle = _make_handle(task_done=False)
        async def fake_start(pid, config, initial_message=None, trigger_source=None):
            mgr._handles[pid] = mock_handle

        mgr.start_agent = AsyncMock(side_effect=fake_start)

        state = {
            "version": 1,
            "agents": {
                "proj-clean": {
                    "status": "running",
                    "config_snapshot": {},
                    "shutdown_clean": True,  # Clean shutdown
                },
            },
        }
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        await mgr.auto_resume_agents()

        # No warning should be appended (session.append not called)
        mock_handle.session.append.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_skips_non_running_status(self, tmp_path):
        """auto_resume_agents ignores agents with status != 'running'."""
        project_store = MagicMock()
        mgr = _make_manager(tmp_path, project_store=project_store)
        mgr.start_agent = AsyncMock()

        state = {
            "version": 1,
            "agents": {
                "stopped-agent": {
                    "status": "stopped",
                    "config_snapshot": {},
                    "shutdown_clean": True,
                },
            },
        }
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        await mgr.auto_resume_agents()

        mgr.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_empty_agents(self, tmp_path):
        """auto_resume_agents with empty agents dict returns cleanly."""
        mgr = _make_manager(tmp_path)
        state = {"version": 1, "agents": {}}
        (tmp_path / "daemon-state.json").write_text(json.dumps(state), encoding="utf-8")

        await mgr.auto_resume_agents()


# ===========================================================================
# Sleep Prevention
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
        # No running agents in _handles

        mgr._allow_sleep_if_idle()

        platform.allow_sleep.assert_called_once_with("handle-1")
        assert mgr._sleep_handle is None

    def test_allow_sleep_preserved_when_agents_running(self, tmp_path):
        """_allow_sleep_if_idle keeps sleep inhibit when agents are running."""
        platform = MagicMock()
        mgr = _make_manager(tmp_path, platform_provider=platform)
        mgr._sleep_handle = "handle-1"
        mgr._handles["p1"] = _make_handle(task_done=False)

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
        # allow_sleep with None handle should not raise
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

        # Should not raise
        mgr._prevent_sleep_if_needed()
        assert mgr._sleep_handle is None

    def test_allow_sleep_exception_handled(self, tmp_path):
        """_allow_sleep_if_idle handles exceptions from allow_sleep and clears handle."""
        platform = MagicMock()
        platform.allow_sleep.side_effect = RuntimeError("OS error")
        mgr = _make_manager(tmp_path, platform_provider=platform)
        mgr._sleep_handle = "handle-1"

        # Should not raise
        mgr._allow_sleep_if_idle()
        # Handle should still be cleared even on error
        assert mgr._sleep_handle is None
