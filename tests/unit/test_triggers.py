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
from datetime import datetime, timezone
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
        # Only the enabled trigger should be registered in the tick loop's
        # evaluated set (schedule triggers no longer get per-trigger timers —
        # see trigger_manager.py's tick-loop redesign).
        assert "trg_aaa" in tm._schedule_ids
        assert "trg_bbb" not in tm._schedule_ids
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
        assert len(tm._schedule_ids) == 1
        await tm.stop()
        assert len(tm._schedule_ids) == 0

    @pytest.mark.asyncio
    async def test_register_trigger(self):
        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr)
        await tm.start()

        trigger = {"id": "trg_new", "name": "New", "type": "schedule",
                   "enabled": True, "schedule": {"cron": "0 9 * * *"}, "task": "Task"}
        tm.register_trigger(pid, trigger)
        assert "trg_new" in tm._schedule_ids

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
        """Re-registering the same trigger_id is idempotent (no duplicate
        evaluation entries) and the tick loop picks up the updated cron on
        its next evaluation (via a fresh project_store read, not cached
        state)."""
        store, pid = _make_project_store()
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running = MagicMock(return_value=False)
        mock_agent_mgr.start_agent = AsyncMock()
        fixed_now = datetime(2026, 7, 22, 15, 0, 0, tzinfo=timezone.utc)  # Wed 15:00 UTC
        tm = TriggerManager(store, mock_agent_mgr, now_fn=lambda: fixed_now)
        await tm.start()

        trigger = {"id": "trg_aaa", "name": "Original", "type": "schedule",
                   "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Task",
                   "last_triggered": None, "created_at": "2020-01-01T00:00:00+00:00"}
        tm.register_trigger(pid, trigger)
        assert tm._schedule_ids == {"trg_aaa"}

        # Re-register with an updated cron — still exactly one entry.
        trigger2 = {"id": "trg_aaa", "name": "Updated", "type": "schedule",
                    "enabled": True, "schedule": {"cron": "0 9 * * *"}, "task": "Task",
                    "last_triggered": None, "created_at": "2020-01-01T00:00:00+00:00"}
        tm.register_trigger(pid, trigger2)
        assert tm._schedule_ids == {"trg_aaa"}

        # The tick loop reads the trigger fresh from project_store, so the
        # re-registered (updated) cron is what actually gets evaluated —
        # persist trigger2's cron there before evaluating.
        store.update_project(pid, {"triggers": [trigger2]})
        await tm._evaluate_due_triggers()
        mock_agent_mgr.start_agent.assert_called_once()

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
        store.update_project(pid, {
            "sub_agent_deployment_instructions": "Use Codex for trigger work.",
        })
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

        # The config is the canonical builder's, passed through untouched.
        # This used to assert that _fire_trigger copied
        # sub_agent_deployment_instructions off the project itself — one field
        # of a private re-derivation that also resolved model/provider/
        # base_url/api_key independently and paired a project's stale endpoint
        # with the current global key, 401ing every scheduled run. Deriving
        # config is no longer this path's job; carrying it faithfully is.
        # Field-level coverage lives in tests/unit/test_agent_config_parity.py.
        mock_agent_mgr._build_agent_config_from_project.assert_called_once_with(pid)
        assert call_kwargs.args[1] is (
            mock_agent_mgr._build_agent_config_from_project.return_value
        )

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
        assert "trg_orphan" not in tm._schedule_ids

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
        _, semi_stable, _ = builder.build(ctx)
        assert "Trigger Context" in semi_stable
        assert "Morning PDF summary" in semi_stable
        assert "schedule" in semi_stable

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
        _, semi_stable, _ = builder.build(ctx)
        assert "Trigger Context" not in semi_stable


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
        # A disable is an UPDATE, not an unregister-and-forget: the record is
        # still there. apply_trigger_update disarms the scheduler AND announces
        # trigger.updated (see TestTriggerBroadcastSemantics).
        assert mock_tm.apply_trigger_update.call_count == 1
        called_pid, called_trigger = mock_tm.apply_trigger_update.call_args[0]
        assert called_pid == pid
        assert called_trigger["id"] == "trg_aaa"
        assert called_trigger["enabled"] is False
        mock_tm.unregister_trigger.assert_not_called()

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
        assert mock_tm.apply_trigger_update.call_args[0][1]["enabled"] is True

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


# ===========================================================================
# REST create (POST) — the UI's create path. Must produce a record that is
# byte-compatible with CreateTriggerTool's, since both feed the same store,
# scheduler and list surfaces.
# ===========================================================================

def _make_trigger_client(triggers=None, trigger_manager=None):
    """TestClient over the trigger routes. Returns (client, pid, store, tm)."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from agent_os.api.routes import agents_v2

    store, pid = _make_project_store(triggers=triggers if triggers is not None else [])
    mock_agent_mgr = MagicMock()
    mock_agent_mgr.is_running = MagicMock(return_value=False)
    mock_ws = MagicMock()
    tm = trigger_manager if trigger_manager is not None else MagicMock()

    agents_v2.configure(store, mock_agent_mgr, mock_ws, trigger_manager=tm)
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), pid, store, tm


class TestTriggerCreateRoute:

    def test_create_schedule_trigger(self):
        client, pid, store, tm = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={
                "name": "Morning brief",
                "type": "schedule",
                "task": "Summarize the inbox",
                "schedule": {"cron": "0 7 * * *", "human": "Every day at 07:00",
                             "timezone": "Asia/Shanghai"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"].startswith("trg_")
        assert data["enabled"] is True
        assert data["trigger_count"] == 0
        assert data["last_triggered"] is None
        assert data["created_at"]
        assert data["schedule"] == {
            "cron": "0 7 * * *",
            "human": "Every day at 07:00",
            "timezone": "Asia/Shanghai",
        }
        # Persisted into the project record
        assert [t["id"] for t in store.get_project(pid)["triggers"]] == [data["id"]]
        # Armed with the scheduler
        tm.register_trigger.assert_called_once()

    def test_create_file_watch_trigger(self):
        client, pid, store, tm = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={
                "name": "Incoming photos",
                "type": "file_watch",
                "task": "Sort the new photos",
                "watch_path": "incoming",
                "patterns": ["*.jpg", "*.png"],
                "recursive": True,
                "debounce_seconds": 12,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["watch_path"] == "incoming"
        assert data["patterns"] == ["*.jpg", "*.png"]
        assert data["recursive"] is True
        assert data["debounce_seconds"] == 12
        assert "schedule" not in data

    def test_create_defaults_human_to_the_cron_expression(self):
        # A caption is what the strip/list render as the row label — never
        # leave it empty just because the client didn't send one.
        client, pid, _, _ = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={"name": "Hourly", "type": "schedule", "task": "Poll",
                  "schedule": {"cron": "0 * * * *"}},
        )
        assert resp.status_code == 201
        assert resp.json()["schedule"]["human"] == "0 * * * *"
        assert resp.json()["schedule"]["timezone"] == "UTC"

    def test_create_rejects_invalid_cron(self):
        client, pid, store, tm = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={"name": "Bad", "type": "schedule", "task": "x",
                  "schedule": {"cron": "not a cron"}},
        )
        assert resp.status_code == 400
        assert "cron" in resp.json()["detail"].lower()
        assert store.get_project(pid)["triggers"] == []
        tm.register_trigger.assert_not_called()

    def test_create_rejects_unknown_timezone(self):
        client, pid, _, _ = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={"name": "Bad tz", "type": "schedule", "task": "x",
                  "schedule": {"cron": "0 7 * * *", "timezone": "Asia/Shangai"}},
        )
        assert resp.status_code == 400
        assert "timezone" in resp.json()["detail"].lower()

    def test_create_rejects_schedule_without_schedule_block(self):
        client, pid, _, _ = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={"name": "No cron", "type": "schedule", "task": "x"},
        )
        assert resp.status_code == 400

    def test_create_rejects_watch_path_outside_workspace(self):
        # Never trust a client-supplied path — realpath containment, same
        # guard the agent tool goes through.
        client, pid, store, _ = _make_trigger_client()
        resp = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={"name": "Escape", "type": "file_watch", "task": "x",
                  "watch_path": "../../etc"},
        )
        assert resp.status_code == 400
        assert "outside workspace" in resp.json()["detail"]
        assert store.get_project(pid)["triggers"] == []

    def test_create_project_not_found(self):
        client, _, _, _ = _make_trigger_client()
        resp = client.post(
            "/api/v2/projects/proj_nope/triggers",
            json={"name": "x", "type": "schedule", "task": "x",
                  "schedule": {"cron": "0 7 * * *"}},
        )
        assert resp.status_code == 404


# ===========================================================================
# REST update (PATCH) — widened from {enabled} to every editable field,
# including the file_watch ones the agent tool still can't reach (item #61).
# ===========================================================================

SEED_SCHEDULE = {
    "id": "trg_sched", "name": "Morning", "type": "schedule", "enabled": True,
    "schedule": {"cron": "0 7 * * *", "human": "Every day at 07:00", "timezone": "UTC"},
    "task": "Do morning task", "trigger_count": 3,
    "last_triggered": "2026-02-27T07:00:00Z", "created_at": "2026-01-01T00:00:00Z",
}
SEED_WATCH = {
    "id": "trg_watch", "name": "Photos", "type": "file_watch", "enabled": True,
    "watch_path": "incoming", "patterns": ["*.jpg"], "recursive": False,
    "debounce_seconds": 5, "task": "Sort photos", "trigger_count": 0,
    "last_triggered": None, "created_at": "2026-01-01T00:00:00Z",
}


class TestTriggerUpdateRoute:

    def test_patch_name_and_task(self):
        client, pid, store, _ = _make_trigger_client([dict(SEED_SCHEDULE)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched",
            json={"name": "Renamed", "task": "New prompt"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Renamed"
        assert data["task"] == "New prompt"
        # Untouched fields survive a partial update
        assert data["enabled"] is True
        assert data["trigger_count"] == 3
        assert data["schedule"]["cron"] == "0 7 * * *"
        assert store.get_project(pid)["triggers"][0]["name"] == "Renamed"

    def test_patch_schedule_merges_and_keeps_caption(self):
        client, pid, _, _ = _make_trigger_client([dict(SEED_SCHEDULE)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched",
            json={"schedule": {"cron": "30 18 * * 1-5", "human": "Weekdays at 18:30"}},
        )
        assert resp.status_code == 200
        assert resp.json()["schedule"] == {
            "cron": "30 18 * * 1-5",
            "human": "Weekdays at 18:30",
            "timezone": "UTC",  # merged from the stored schedule, not blanked
        }

    def test_patch_file_watch_fields(self):
        # The gap that made edit unshippable: PATCH accepted only {enabled}.
        client, pid, store, _ = _make_trigger_client([dict(SEED_WATCH)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_watch",
            json={
                "watch_path": "uploads",
                "patterns": ["*.png", "*.gif"],
                "recursive": True,
                "debounce_seconds": 30,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["watch_path"] == "uploads"
        assert data["patterns"] == ["*.png", "*.gif"]
        assert data["recursive"] is True
        assert data["debounce_seconds"] == 30
        assert store.get_project(pid)["triggers"][0]["watch_path"] == "uploads"

    def test_patch_can_clear_patterns_to_all_files(self):
        client, pid, _, _ = _make_trigger_client([dict(SEED_WATCH)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_watch",
            json={"patterns": [], "recursive": False},
        )
        assert resp.status_code == 200
        assert resp.json()["patterns"] == []

    def test_patch_rejects_invalid_cron(self):
        client, pid, store, _ = _make_trigger_client([dict(SEED_SCHEDULE)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched",
            json={"schedule": {"cron": "99 99 * * *"}},
        )
        assert resp.status_code == 400
        # Nothing persisted on a rejected edit
        assert store.get_project(pid)["triggers"][0]["schedule"]["cron"] == "0 7 * * *"

    def test_patch_rejects_watch_path_outside_workspace(self):
        client, pid, store, _ = _make_trigger_client([dict(SEED_WATCH)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_watch",
            json={"watch_path": "../../etc"},
        )
        assert resp.status_code == 400
        assert store.get_project(pid)["triggers"][0]["watch_path"] == "incoming"

    def test_patch_rejects_empty_name(self):
        client, pid, _, _ = _make_trigger_client([dict(SEED_SCHEDULE)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched", json={"name": ""},
        )
        assert resp.status_code == 400

    def test_patch_ignores_fields_of_the_other_trigger_kind(self):
        # Type is immutable; a stray watch_path must not land on a schedule
        # record (nor a schedule on a file-watch one).
        client, pid, store, _ = _make_trigger_client([dict(SEED_SCHEDULE)])
        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched",
            json={"name": "Still a schedule", "watch_path": "incoming", "recursive": True},
        )
        assert resp.status_code == 200
        stored = store.get_project(pid)["triggers"][0]
        assert stored["name"] == "Still a schedule"
        assert "watch_path" not in stored
        assert "recursive" not in stored

    def test_patch_keeps_row_position(self):
        # An edit must not reorder the list (the old enable path re-appended).
        client, pid, store, _ = _make_trigger_client(
            [dict(SEED_SCHEDULE), dict(SEED_WATCH)]
        )
        client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched", json={"enabled": False},
        )
        assert [t["id"] for t in store.get_project(pid)["triggers"]] == [
            "trg_sched", "trg_watch",
        ]


# ===========================================================================
# Broadcast semantics — the vanish-on-disable regression.
#
# Toggling a trigger off used to travel over the wire as trigger.deleted
# (the route called unregister_trigger, which broadcast a delete), so every
# live list dropped the row until the next refetch. created/deleted now mean
# a record appeared or went away; everything else is trigger.updated.
# ===========================================================================

def _client_with_real_manager(triggers):
    """TestClient wired to a REAL TriggerManager + a mock ws_manager, so the
    assertions are about the events that actually reach the wire."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from agent_os.api.routes import agents_v2

    store, pid = _make_project_store(triggers=triggers)
    mock_ws = MagicMock()
    tm = TriggerManager(store, MagicMock(), mock_ws)
    agents_v2.configure(store, MagicMock(), mock_ws, trigger_manager=tm)
    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app), pid, mock_ws, tm


class TestTriggerBroadcastSemantics:

    def _events(self, mock_ws):
        return [call.args[1]["type"] for call in mock_ws.broadcast.call_args_list]

    def test_disable_broadcasts_updated_not_deleted(self):
        client, pid, mock_ws, tm = _client_with_real_manager([dict(SEED_SCHEDULE)])
        # Arm it first — a live daemon has every enabled trigger registered
        # (TriggerManager.start), which is exactly the state in which the old
        # disable path emitted a delete.
        tm.register_trigger(pid, dict(SEED_SCHEDULE))
        mock_ws.reset_mock()

        resp = client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched", json={"enabled": False},
        )
        assert resp.status_code == 200

        events = self._events(mock_ws)
        assert "trigger.deleted" not in events, (
            "disabling a trigger must not announce a delete — the row would "
            "vanish from every live list"
        )
        assert events == ["trigger.updated"]
        payload = mock_ws.broadcast.call_args_list[0].args[1]
        assert payload["project_id"] == pid
        assert payload["trigger"]["id"] == "trg_sched"
        assert payload["trigger"]["enabled"] is False
        # …and it really is disarmed, not just relabelled.
        assert "trg_sched" not in tm._schedule_ids

    def test_enable_broadcasts_updated_not_created(self):
        seed = dict(SEED_SCHEDULE)
        seed["enabled"] = False
        client, pid, mock_ws, tm = _client_with_real_manager([seed])

        client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched", json={"enabled": True},
        )
        assert self._events(mock_ws) == ["trigger.updated"]
        assert "trg_sched" in tm._schedule_ids

    def test_edit_broadcasts_updated_with_the_full_record(self):
        client, pid, mock_ws, _ = _client_with_real_manager([dict(SEED_SCHEDULE)])

        client.patch(
            f"/api/v2/projects/{pid}/triggers/trg_sched",
            json={"name": "Evening", "schedule": {"cron": "0 19 * * *"}},
        )
        assert self._events(mock_ws) == ["trigger.updated"]
        trigger = mock_ws.broadcast.call_args_list[0].args[1]["trigger"]
        assert trigger["name"] == "Evening"
        assert trigger["schedule"]["cron"] == "0 19 * * *"
        assert trigger["task"] == "Do morning task"  # full record, not a diff

    def test_create_and_delete_still_broadcast_created_and_deleted(self):
        client, pid, mock_ws, _ = _client_with_real_manager([])

        created = client.post(
            f"/api/v2/projects/{pid}/triggers",
            json={"name": "New", "type": "schedule", "task": "x",
                  "schedule": {"cron": "0 7 * * *"}},
        ).json()
        assert self._events(mock_ws) == ["trigger.created"]

        mock_ws.reset_mock()
        assert client.delete(
            f"/api/v2/projects/{pid}/triggers/{created['id']}"
        ).status_code == 204
        assert self._events(mock_ws) == ["trigger.deleted"]

    def test_re_register_does_not_announce_a_delete(self):
        # register_trigger unregisters first for idempotency; that internal
        # disarm is not a deletion and must stay off the wire.
        store, pid = _make_project_store(triggers=[])
        mock_ws = MagicMock()
        tm = TriggerManager(store, MagicMock(), mock_ws)
        trigger = dict(SEED_SCHEDULE)
        tm.register_trigger(pid, trigger)
        mock_ws.reset_mock()
        tm.register_trigger(pid, trigger)
        assert self._events(mock_ws) == ["trigger.created"]


# ===========================================================================
# Credential-error surfacing: a trigger whose start fails on provider
# construction must broadcast a classified agent.status error (it was
# previously log-only, so users never saw why their trigger did nothing).
# ===========================================================================

class TestTriggerErrorSurfacing:

    @pytest.mark.asyncio
    async def test_fire_broadcasts_classified_error_on_provider_failure(self):
        from agent_os.daemon_v2.provider_errors import ProviderConfigError

        triggers = [
            {"id": "trg_err", "name": "Nightly", "type": "schedule",
             "enabled": True, "schedule": {"cron": "0 7 * * *"}, "task": "Do it"},
        ]
        store, pid = _make_project_store(triggers=triggers)
        mock_agent_mgr = MagicMock()
        mock_agent_mgr.is_running.return_value = False
        mock_agent_mgr._setup_engine = None
        mock_agent_mgr.start_agent = AsyncMock(
            side_effect=ProviderConfigError(
                "missing_api_key", "No LLM API key configured"),
        )
        mock_ws = MagicMock()
        tm = TriggerManager(store, mock_agent_mgr, ws_manager=mock_ws)

        await tm._fire_trigger(pid, "trg_err")

        events = [c.args[1] for c in mock_ws.broadcast.call_args_list]
        err = next(
            (e for e in events
             if e.get("type") == "agent.status" and e.get("status") == "error"),
            None,
        )
        assert err is not None, f"no agent.status error broadcast; got {events}"
        assert err["error_code"] == "missing_api_key"
        assert err["reason"] == "No LLM API key configured"
        assert err["source"] == "trigger"
        assert err["trigger_id"] == "trg_err"
