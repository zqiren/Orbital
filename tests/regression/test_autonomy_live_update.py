# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: autonomy preset changes must take effect on running agents.

Bug: Changing the autonomy preset in project settings (PUT /api/v2/projects/{id})
only persists to disk. The running agent's AutonomyInterceptor keeps the stale
preset, so changes mid-session have no effect until the agent is restarted.

Fix: AutonomyInterceptor gets an update_preset() method. AgentManager gets an
update_autonomy() method that pushes the new preset to the running interceptor
and injects a system message into the session. The PUT endpoint calls it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


# ---------------------------------------------------------------------------
# Test 1: AutonomyInterceptor.update_preset() changes interception behavior
# ---------------------------------------------------------------------------

class TestInterceptorLiveUpdate:

    def test_update_preset_changes_behavior(self):
        """Changing preset from HANDS_OFF to SUPERVISED should start
        intercepting shell calls that were previously allowed."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.HANDS_OFF, ws, "proj-1")

        shell_call = {"name": "shell", "arguments": {"command": "ls"}}

        # HANDS_OFF: shell is not intercepted
        assert interceptor.should_intercept(shell_call) is False

        # Update to SUPERVISED
        interceptor.update_preset(Autonomy.SUPERVISED)

        # SUPERVISED: shell IS intercepted
        assert interceptor.should_intercept(shell_call) is True

    def test_update_preset_supervised_to_hands_off(self):
        """Changing from SUPERVISED to HANDS_OFF should stop intercepting
        shell and write calls."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.SUPERVISED, ws, "proj-1")

        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        write_call = {"name": "write", "arguments": {"path": "/tmp/x"}}

        assert interceptor.should_intercept(shell_call) is True
        assert interceptor.should_intercept(write_call) is True

        interceptor.update_preset(Autonomy.HANDS_OFF)

        assert interceptor.should_intercept(shell_call) is False
        assert interceptor.should_intercept(write_call) is False

    def test_update_preset_to_check_in(self):
        """CHECK_IN should intercept shell and write but not read."""
        ws = MagicMock()
        interceptor = AutonomyInterceptor(Autonomy.HANDS_OFF, ws, "proj-1")

        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        read_call = {"name": "read", "arguments": {"path": "/tmp/x"}}

        interceptor.update_preset(Autonomy.CHECK_IN)

        assert interceptor.should_intercept(shell_call) is True
        assert interceptor.should_intercept(read_call) is False


# ---------------------------------------------------------------------------
# Test 2: AgentManager.update_autonomy() pushes to running agent
# ---------------------------------------------------------------------------

class TestAgentManagerUpdateAutonomy:

    def _make_manager(self):
        """Create a minimal AgentManager with mocked dependencies."""
        from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
        mgr = AgentManager(
            project_store=MagicMock(),
            ws_manager=MagicMock(),
            sub_agent_manager=MagicMock(),
            activity_translator=MagicMock(),
            process_manager=MagicMock(),
        )
        return mgr, ProjectHandle

    def test_update_autonomy_changes_interceptor(self):
        """update_autonomy() must change the interceptor preset on a running agent."""
        mgr, ProjectHandle = self._make_manager()

        interceptor = AutonomyInterceptor(Autonomy.HANDS_OFF, MagicMock(), "proj-1")
        session = MagicMock()
        handle = ProjectHandle(
            session=session, loop=MagicMock(), provider=MagicMock(),
            registry=MagicMock(), context_manager=MagicMock(),
            interceptor=interceptor, task=MagicMock(),
        )
        mgr._handles["proj-1"] = handle

        result = mgr.update_autonomy("proj-1", Autonomy.SUPERVISED)

        assert result is True
        # Interceptor should now use SUPERVISED
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is True

    def test_update_autonomy_injects_system_message(self):
        """update_autonomy() must inject a system message into the session."""
        mgr, ProjectHandle = self._make_manager()

        interceptor = AutonomyInterceptor(Autonomy.HANDS_OFF, MagicMock(), "proj-1")
        session = MagicMock()
        handle = ProjectHandle(
            session=session, loop=MagicMock(), provider=MagicMock(),
            registry=MagicMock(), context_manager=MagicMock(),
            interceptor=interceptor, task=MagicMock(),
        )
        mgr._handles["proj-1"] = handle

        mgr.update_autonomy("proj-1", Autonomy.SUPERVISED)

        session.append_system.assert_called_once()
        msg = session.append_system.call_args[0][0]
        assert "supervised" in msg.lower()

    def test_update_autonomy_no_running_agent(self):
        """update_autonomy() returns False when no agent is running."""
        mgr, _ = self._make_manager()

        result = mgr.update_autonomy("nonexistent", Autonomy.SUPERVISED)
        assert result is False


# ---------------------------------------------------------------------------
# Test 3: PUT endpoint triggers live update
# ---------------------------------------------------------------------------

class TestPutEndpointTriggersUpdate:

    @pytest.fixture
    def client(self, tmp_path):
        from agent_os.api.app import create_app
        from fastapi.testclient import TestClient
        with patch("agent_os.api.app.acquire_pid_file"):
            app = create_app(data_dir=str(tmp_path / "data"))
        return TestClient(app)

    def test_put_autonomy_calls_update_autonomy(self, client, tmp_path):
        """PUT /api/v2/projects/{id} with autonomy field must call
        agent_manager.update_autonomy()."""
        # Create a project first — workspace directory must exist
        ws = tmp_path / "ws"
        ws.mkdir()
        resp = client.post("/api/v2/projects", json={
            "name": "test-proj",
            "workspace": str(ws),
            "model": "gpt-4",
            "api_key": "sk-test",
            "autonomy": "hands_off",
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        with patch("agent_os.api.routes.agents_v2._agent_manager") as mock_mgr:
            mock_mgr.update_autonomy = MagicMock(return_value=False)
            resp = client.put(f"/api/v2/projects/{pid}", json={
                "autonomy": "supervised",
            })
            assert resp.status_code == 200
            mock_mgr.update_autonomy.assert_called_once_with(pid, Autonomy.SUPERVISED)
