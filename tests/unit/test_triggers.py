# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for the trigger system — tools, data model, and TriggerManager.

Covers:
1. Trigger CRUD tools (create, list, update, delete)
2. Trigger validation (cron expressions, required fields)
3. TriggerManager lifecycle (start, stop, register, unregister)
4. REST endpoint integration (trigger list, toggle)
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from agent_os.agent.tools.base import ToolResult
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.trigger_manager import (
    TriggerManager,
    generate_trigger_id,
    validate_trigger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project_store(triggers=None):
    """Create a real ProjectStore with a temp directory and one project."""
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Test Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    if triggers is not None:
        store.update_project(pid, {"triggers": triggers})
    return store, pid


# ===========================================================================
# Trigger ID Generation
# ===========================================================================

class TestTriggerIdGeneration:

    def test_generate_trigger_id_format(self):
        tid = generate_trigger_id()
        assert tid.startswith("trg_")
        assert len(tid) == 12  # "trg_" + 8 hex chars

    def test_generate_trigger_id_unique(self):
        ids = {generate_trigger_id() for _ in range(100)}
        assert len(ids) == 100


# ===========================================================================
# Trigger Validation
# ===========================================================================

class TestTriggerValidation:

    def test_valid_schedule_trigger(self):
        trigger = {
            "name": "Test",
            "type": "schedule",
            "schedule": {"cron": "0 7 * * *"},
            "task": "Do something",
        }
        assert validate_trigger(trigger) is None

    def test_missing_name(self):
        trigger = {
            "type": "schedule",
            "schedule": {"cron": "0 7 * * *"},
            "task": "Do something",
        }
        assert "name is required" in validate_trigger(trigger)

    def test_invalid_type(self):
        trigger = {
            "name": "Test",
            "type": "invalid",
            "task": "Do something",
        }
        assert "Invalid trigger type" in validate_trigger(trigger)

    def test_missing_cron(self):
        trigger = {
            "name": "Test",
            "type": "schedule",
            "schedule": {},
            "task": "Do something",
        }
        assert "schedule.cron" in validate_trigger(trigger)

    def test_invalid_cron(self):
        trigger = {
            "name": "Test",
            "type": "schedule",
            "schedule": {"cron": "not a cron"},
            "task": "Do something",
        }
        assert "Invalid cron" in validate_trigger(trigger)

    def test_missing_task(self):
        trigger = {
            "name": "Test",
            "type": "schedule",
            "schedule": {"cron": "0 7 * * *"},
        }
        assert "task is required" in validate_trigger(trigger)

    def test_file_watch_requires_watch_path(self):
        trigger = {
            "name": "Test",
            "type": "file_watch",
            "task": "Do something",
        }
        assert "watch_path" in validate_trigger(trigger)

    def test_file_watch_valid(self):
        trigger = {
            "name": "Test",
            "type": "file_watch",
            "watch_path": "incoming",
            "task": "Do something",
        }
        assert validate_trigger(trigger) is None


# ===========================================================================
# CreateTriggerTool
# ===========================================================================

class TestCreateTriggerTool:

    def test_create_schedule_trigger(self):
        from agent_os.agent.tools.triggers import CreateTriggerTool
        store, pid = _make_project_store()
        tool = CreateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(
            name="Morning Report",
            type="schedule",
            task="Generate morning report",
            cron="0 7 * * *",
            human="Every day at 7:00 AM",
        )

        assert isinstance(result, ToolResult)
        data = json.loads(result.content)
        assert data["status"] == "created"
        assert data["trigger"]["name"] == "Morning Report"
        assert data["trigger"]["schedule"]["cron"] == "0 7 * * *"
        assert data["trigger"]["enabled"] is True

        # Verify stored in project
        project = store.get_project(pid)
        assert len(project["triggers"]) == 1
        assert project["triggers"][0]["name"] == "Morning Report"

    def test_create_trigger_with_timezone(self):
        from agent_os.agent.tools.triggers import CreateTriggerTool
        store, pid = _make_project_store()
        tool = CreateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(
            name="Test",
            type="schedule",
            task="Test task",
            cron="0 9 * * *",
            human="Daily at 9am",
            timezone="Asia/Shanghai",
        )
        data = json.loads(result.content)
        assert data["trigger"]["schedule"]["timezone"] == "Asia/Shanghai"

    def test_create_trigger_invalid_cron(self):
        from agent_os.agent.tools.triggers import CreateTriggerTool
        store, pid = _make_project_store()
        tool = CreateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(
            name="Bad Cron",
            type="schedule",
            task="Task",
            cron="not valid",
            human="Invalid",
        )
        assert "Error" in result.content
        assert "Invalid cron" in result.content

    def test_create_trigger_notifies_trigger_manager(self):
        from agent_os.agent.tools.triggers import CreateTriggerTool
        store, pid = _make_project_store()
        mock_tm = MagicMock()
        tool = CreateTriggerTool(project_id=pid, project_store=store, trigger_manager=mock_tm)

        tool.execute(
            name="Test",
            type="schedule",
            task="Task",
            cron="0 7 * * *",
            human="Daily",
        )
        mock_tm.register_trigger.assert_called_once()

    def test_create_trigger_appends_to_existing(self):
        from agent_os.agent.tools.triggers import CreateTriggerTool
        existing = [{"id": "trg_existing", "name": "Existing", "type": "schedule", "enabled": True,
                     "schedule": {"cron": "0 6 * * *"}, "task": "Old task"}]
        store, pid = _make_project_store(triggers=existing)
        tool = CreateTriggerTool(project_id=pid, project_store=store)

        tool.execute(
            name="New Trigger",
            type="schedule",
            task="New task",
            cron="0 8 * * *",
            human="Daily at 8am",
        )

        project = store.get_project(pid)
        assert len(project["triggers"]) == 2


# ===========================================================================
# ListTriggersTool
# ===========================================================================

class TestListTriggersTool:

    def test_list_empty(self):
        from agent_os.agent.tools.triggers import ListTriggersTool
        store, pid = _make_project_store()
        tool = ListTriggersTool(project_id=pid, project_store=store)

        result = tool.execute()
        data = json.loads(result.content)
        assert data["triggers"] == []
        assert "No triggers" in data["message"]

    def test_list_with_triggers(self):
        from agent_os.agent.tools.triggers import ListTriggersTool
        triggers = [
            {"id": "trg_aaa", "name": "A", "type": "schedule", "enabled": True,
             "schedule": {"cron": "0 7 * * *"}, "task": "Task A"},
            {"id": "trg_bbb", "name": "B", "type": "schedule", "enabled": False,
             "schedule": {"cron": "0 12 * * *"}, "task": "Task B"},
        ]
        store, pid = _make_project_store(triggers=triggers)
        tool = ListTriggersTool(project_id=pid, project_store=store)

        result = tool.execute()
        data = json.loads(result.content)
        assert len(data["triggers"]) == 2
        assert "2 trigger(s)" in data["message"]


# ===========================================================================
# UpdateTriggerTool
# ===========================================================================

class TestUpdateTriggerTool:

    def test_update_name(self):
        from agent_os.agent.tools.triggers import UpdateTriggerTool
        triggers = [{"id": "trg_aaa", "name": "Old Name", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        tool = UpdateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_aaa", name="New Name")
        data = json.loads(result.content)
        assert data["status"] == "updated"
        assert data["trigger"]["name"] == "New Name"

        project = store.get_project(pid)
        assert project["triggers"][0]["name"] == "New Name"

    def test_update_cron(self):
        from agent_os.agent.tools.triggers import UpdateTriggerTool
        triggers = [{"id": "trg_aaa", "name": "Test", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        tool = UpdateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_aaa", cron="0 9 * * *", human="Daily at 9am")
        data = json.loads(result.content)
        assert data["trigger"]["schedule"]["cron"] == "0 9 * * *"

    def test_update_invalid_cron(self):
        from agent_os.agent.tools.triggers import UpdateTriggerTool
        triggers = [{"id": "trg_aaa", "name": "Test", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        tool = UpdateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_aaa", cron="not valid")
        assert "Error" in result.content
        assert "invalid cron" in result.content

    def test_update_nonexistent_trigger(self):
        from agent_os.agent.tools.triggers import UpdateTriggerTool
        store, pid = _make_project_store(triggers=[])
        tool = UpdateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_nonexistent", name="X")
        assert "Error" in result.content
        assert "not found" in result.content

    def test_update_enabled_status(self):
        from agent_os.agent.tools.triggers import UpdateTriggerTool
        triggers = [{"id": "trg_aaa", "name": "Test", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        tool = UpdateTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_aaa", enabled=False)
        data = json.loads(result.content)
        assert data["trigger"]["enabled"] is False

    def test_update_notifies_trigger_manager(self):
        from agent_os.agent.tools.triggers import UpdateTriggerTool
        triggers = [{"id": "trg_aaa", "name": "Test", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        mock_tm = MagicMock()
        tool = UpdateTriggerTool(project_id=pid, project_store=store, trigger_manager=mock_tm)

        tool.execute(trigger_id="trg_aaa", name="Updated")
        mock_tm.register_trigger.assert_called_once()


# ===========================================================================
# DeleteTriggerTool
# ===========================================================================

class TestDeleteTriggerTool:

    def test_delete_trigger(self):
        from agent_os.agent.tools.triggers import DeleteTriggerTool
        triggers = [{"id": "trg_aaa", "name": "To Delete", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        tool = DeleteTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_aaa")
        data = json.loads(result.content)
        assert data["status"] == "deleted"

        project = store.get_project(pid)
        assert len(project["triggers"]) == 0

    def test_delete_nonexistent_trigger(self):
        from agent_os.agent.tools.triggers import DeleteTriggerTool
        store, pid = _make_project_store(triggers=[])
        tool = DeleteTriggerTool(project_id=pid, project_store=store)

        result = tool.execute(trigger_id="trg_nonexistent")
        assert "Error" in result.content
        assert "not found" in result.content

    def test_delete_notifies_trigger_manager(self):
        from agent_os.agent.tools.triggers import DeleteTriggerTool
        triggers = [{"id": "trg_aaa", "name": "Test", "type": "schedule",
                     "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}]
        store, pid = _make_project_store(triggers=triggers)
        mock_tm = MagicMock()
        tool = DeleteTriggerTool(project_id=pid, project_store=store, trigger_manager=mock_tm)

        tool.execute(trigger_id="trg_aaa")
        mock_tm.unregister_trigger.assert_called_once_with("trg_aaa")

    def test_delete_preserves_other_triggers(self):
        from agent_os.agent.tools.triggers import DeleteTriggerTool
        triggers = [
            {"id": "trg_aaa", "name": "Keep", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task A"},
            {"id": "trg_bbb", "name": "Delete", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 8 * * *"}, "task": "Task B"},
        ]
        store, pid = _make_project_store(triggers=triggers)
        tool = DeleteTriggerTool(project_id=pid, project_store=store)

        tool.execute(trigger_id="trg_bbb")

        project = store.get_project(pid)
        assert len(project["triggers"]) == 1
        assert project["triggers"][0]["id"] == "trg_aaa"


# ===========================================================================
# TriggerManager
# ===========================================================================

class TestTriggerManager:

    @pytest.mark.asyncio
    async def test_start_registers_enabled_triggers(self):
        triggers = [
            {"id": "trg_aaa", "name": "Active", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task A"},
            {"id": "trg_bbb", "name": "Disabled", "type": "schedule",
             "enabled": False, "schedule": {"cron": "0 8 * * *"}, "task": "Task B"},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()
        # Only the enabled trigger should be registered
        assert "trg_aaa" in tm._timers
        assert "trg_bbb" not in tm._timers
        await tm.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_all_timers(self):
        triggers = [
            {"id": "trg_aaa", "name": "Active", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task A"},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm.start()
        assert len(tm._timers) == 1
        await tm.stop()
        assert len(tm._timers) == 0

    @pytest.mark.asyncio
    async def test_register_trigger(self):
        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)
        await tm.start()

        trigger = {"id": "trg_new", "name": "New", "type": "schedule",
                   "enabled": True, "schedule": {"cron": "0 9 * * *"}, "task": "Task"}
        tm.register_trigger(pid, trigger)
        assert "trg_new" in tm._timers

        # Cleanup
        await tm.stop()

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self):
        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)

        # Should not raise
        tm.unregister_trigger("trg_nonexistent")

    @pytest.mark.asyncio
    async def test_register_replaces_existing(self):
        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)
        await tm.start()

        trigger = {"id": "trg_aaa", "name": "Original", "type": "schedule",
                   "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}
        tm.register_trigger(pid, trigger)
        old_task = tm._timers["trg_aaa"]

        # Re-register with updated cron
        trigger2 = {"id": "trg_aaa", "name": "Updated", "type": "schedule",
                    "enabled": True, "schedule": {"cron": "0 9 * * *"}, "task": "Task"}
        tm.register_trigger(pid, trigger2)
        new_task = tm._timers["trg_aaa"]

        # Old task should have cancel() called (cancelling state),
        # new task should be different from old
        assert old_task.cancelling() > 0 or old_task.cancelled()
        assert new_task is not old_task

        # Cleanup
        await tm.stop()

    @pytest.mark.asyncio
    async def test_fire_trigger_updates_state(self):
        triggers = [
            {"id": "trg_aaa", "name": "Test", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *", "human": "Daily at 7am"},
             "task": "Do the thing", "trigger_count": 0, "last_triggered": None},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm._fire_trigger(pid, "trg_aaa")

        # Trigger state should be updated
        project = store.get_project(pid)
        updated = project["triggers"][0]
        assert updated["trigger_count"] == 1
        assert updated["last_triggered"] is not None

        # Agent should be started with trigger context
        mock_agent_mgr.start_agent.assert_called_once()
        call_kwargs = mock_agent_mgr.start_agent.call_args
        assert call_kwargs.kwargs["trigger_source"] == "schedule"
        assert call_kwargs.kwargs["trigger_name"] == "Test"
        assert "Do the thing" in call_kwargs.kwargs["initial_message"]

    @pytest.mark.asyncio
    async def test_fire_trigger_skips_if_agent_running(self):
        triggers = [
            {"id": "trg_aaa", "name": "Test", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *"},
             "task": "Do the thing", "trigger_count": 0, "last_triggered": None},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=True)
        mock_agent_mgr.start_agent = AsyncMock()
        mock_ws = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr, ws_manager=mock_ws)

        await tm._fire_trigger(pid, "trg_aaa")

        # Agent should NOT be started when already running
        mock_agent_mgr.start_agent.assert_not_called()

        # trigger_count should NOT be incremented
        project = store.get_project(pid)
        assert project["triggers"][0]["trigger_count"] == 0
        assert project["triggers"][0]["last_triggered"] is None

        # Should broadcast a skip event
        mock_ws.broadcast.assert_called_once()
        event = mock_ws.broadcast.call_args[0][1]
        assert event["type"] == "trigger.skipped"
        assert event["reason"] == "agent_busy"

    @pytest.mark.asyncio
    async def test_fire_trigger_disabled(self):
        triggers = [
            {"id": "trg_aaa", "name": "Test", "type": "schedule",
             "enabled": False, "schedule": {"cron": "0 7 * * *"},
             "task": "Do the thing", "trigger_count": 0},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        tm = TriggerManager(store, mock_agent_mgr)

        await tm._fire_trigger(pid, "trg_aaa")

        # Disabled trigger should not fire
        mock_agent_mgr.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_fire_trigger_project_deleted(self):
        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)
        tm._running = True

        # Simulate project deletion by creating a dangling trigger reference
        trigger = {"id": "trg_orphan", "name": "Orphan", "type": "schedule",
                   "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task"}
        tm.register_trigger(pid, trigger)
        store.delete_project(pid)

        # Should not raise, should unregister the trigger
        await tm._fire_trigger(pid, "trg_orphan")
        assert "trg_orphan" not in tm._timers

    @pytest.mark.asyncio
    async def test_fire_trigger_broadcasts_fired_event(self):
        triggers = [
            {"id": "trg_aaa", "name": "Test", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *", "human": "Daily"},
             "task": "Do it", "trigger_count": 0, "last_triggered": None},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        mock_ws = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr, ws_manager=mock_ws)

        await tm._fire_trigger(pid, "trg_aaa")

        # Should broadcast a fired event
        mock_ws.broadcast.assert_called_once()
        event = mock_ws.broadcast.call_args[0][1]
        assert event["type"] == "trigger.fired"
        assert event["trigger_name"] == "Test"


# ===========================================================================
# PromptBuilder Trigger Context
# ===========================================================================

class TestPromptBuilderTriggerContext:

    def test_trigger_context_included_when_trigger_source_set(self):
        from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
        builder = PromptBuilder()
        ctx = PromptContext(
            workspace="/tmp/test",
            model="gpt-4",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[],
            tool_names=["read", "write", "create_trigger"],
            os_type="linux",
            datetime_now="2026-02-27T10:00:00",
            trigger_source="schedule",
            trigger_name="Morning PDF summary",
        )
        _, dynamic = builder.build(ctx)
        assert "Trigger Context" in dynamic
        assert "Morning PDF summary" in dynamic
        assert "schedule" in dynamic

    def test_trigger_context_absent_when_no_trigger(self):
        from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
        builder = PromptBuilder()
        ctx = PromptContext(
            workspace="/tmp/test",
            model="gpt-4",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[],
            tool_names=["read", "write"],
            os_type="linux",
            datetime_now="2026-02-27T10:00:00",
        )
        _, dynamic = builder.build(ctx)
        assert "Trigger Context" not in dynamic


# ===========================================================================
# REST Endpoint Tests
# ===========================================================================

class TestTriggerEndpoints:

    @pytest.fixture
    def client(self):
        """Create a test client with trigger endpoints."""
        from fastapi.testclient import TestClient
        from agent_os.api.routes import agents_v2

        store, pid = _make_project_store(triggers=[
            {"id": "trg_aaa", "name": "Morning", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *", "human": "Daily at 7am"},
             "task": "Do morning task", "trigger_count": 3, "last_triggered": "2026-02-27T07:00:00Z"},
        ])

        # Create minimal mock managers
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_ws = MagicMock()
        mock_tm = MagicMock()

        # Configure routes
        agents_v2.configure(
            store, mock_agent_mgr, mock_ws,
            trigger_manager=mock_tm,
        )

        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(agents_v2.router)
        return TestClient(app), pid, mock_tm

    def test_get_triggers(self, client):
        test_client, pid, _ = client
        resp = test_client.get(f"/api/v2/projects/{pid}/triggers")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Morning"

    def test_get_triggers_not_found(self, client):
        test_client, _, _ = client
        resp = test_client.get("/api/v2/projects/proj_nonexistent/triggers")
        assert resp.status_code == 404

    def test_toggle_trigger_disable(self, client):
        test_client, pid, mock_tm = client
        resp = test_client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_aaa",
            json={"enabled": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        mock_tm.unregister_trigger.assert_called_with("trg_aaa")

    def test_toggle_trigger_enable(self, client):
        test_client, pid, mock_tm = client
        # First disable
        test_client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_aaa",
            json={"enabled": False},
        )
        # Then enable
        resp = test_client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_aaa",
            json={"enabled": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        mock_tm.register_trigger.assert_called()

    def test_toggle_trigger_not_found(self, client):
        test_client, pid, _ = client
        resp = test_client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_nonexistent",
            json={"enabled": False},
        )
        assert resp.status_code == 404

    def test_get_empty_triggers(self):
        from fastapi.testclient import TestClient
        from agent_os.api.routes import agents_v2
        from fastapi import FastAPI

        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        mock_ws = MagicMock()

        agents_v2.configure(store, mock_agent_mgr, mock_ws)
        app = FastAPI()
        app.include_router(agents_v2.router)
        client = TestClient(app)

        resp = client.get(f"/api/v2/projects/{pid}/triggers")
        assert resp.status_code == 200
        assert resp.json() == []
