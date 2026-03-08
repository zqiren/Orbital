# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Trigger CRUD tools — create, list, update, delete triggers.

Management agent calls these when it recognizes scheduling/watching intent.
Tools modify the project's triggers array in project config and notify
the TriggerManager to register/unregister timers.
"""

import json
from datetime import datetime, timezone

from .base import Tool, ToolResult


class CreateTriggerTool(Tool):
    """Create a new trigger for the current project."""

    def __init__(self, project_id: str, project_store, trigger_manager=None):
        self._project_id = project_id
        self._project_store = project_store
        self._trigger_manager = trigger_manager
        self.name = "create_trigger"
        self.description = (
            "Create a trigger that runs a task automatically. "
            "Use type='schedule' for cron-based triggers (e.g., 'every morning at 7am'). "
            "Use type='file_watch' to watch a directory for file changes "
            "(e.g., 'when new .jpg files appear in incoming/')."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the trigger (e.g., 'Morning PDF summary', 'Photo watcher')",
                },
                "type": {
                    "type": "string",
                    "enum": ["schedule", "file_watch"],
                    "description": "Trigger type: 'schedule' for cron-based, 'file_watch' for directory monitoring.",
                },
                "task": {
                    "type": "string",
                    "description": "What the agent should do when triggered (the task description)",
                },
                "cron": {
                    "type": "string",
                    "description": "Cron expression (e.g., '0 7 * * *'). Required for schedule type.",
                },
                "human": {
                    "type": "string",
                    "description": "Human-readable schedule description (e.g., 'Every day at 7:00 AM'). Required for schedule type.",
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone for the schedule (e.g., 'Asia/Shanghai'). Defaults to 'UTC'. Schedule type only.",
                },
                "watch_path": {
                    "type": "string",
                    "description": "Directory to watch, relative to workspace (e.g., 'incoming/', 'uploads/photos'). Required for file_watch type.",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns to filter files (e.g., ['*.jpg', '*.png']). Empty means all files. file_watch type only.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Watch subdirectories too. Default false. file_watch type only.",
                },
                "debounce_seconds": {
                    "type": "integer",
                    "description": "Seconds to wait after last file change before firing. Default 5. file_watch type only.",
                },
                "autonomy": {
                    "type": "string",
                    "enum": ["hands_off", "check_in", "supervised"],
                    "description": "Optional autonomy override for this trigger. Null inherits project default.",
                },
            },
            "required": ["name", "type", "task"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            from agent_os.daemon_v2.trigger_manager import generate_trigger_id, validate_trigger

            name = arguments.get("name", "")
            ttype = arguments.get("type", "schedule")
            task = arguments.get("task", "")
            autonomy = arguments.get("autonomy")

            trigger = {
                "id": generate_trigger_id(),
                "name": name,
                "enabled": True,
                "type": ttype,
                "task": task,
                "autonomy": autonomy,
                "last_triggered": None,
                "trigger_count": 0,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            if ttype == "schedule":
                cron = arguments.get("cron", "")
                human = arguments.get("human", "")
                tz = arguments.get("timezone", "UTC")
                trigger["schedule"] = {
                    "cron": cron,
                    "human": human,
                    "timezone": tz,
                }
            elif ttype == "file_watch":
                trigger["watch_path"] = arguments.get("watch_path", "")
                trigger["patterns"] = arguments.get("patterns", [])
                trigger["recursive"] = arguments.get("recursive", False)
                trigger["debounce_seconds"] = arguments.get("debounce_seconds", 5)

            # Get workspace for path validation
            project = self._project_store.get_project(self._project_id)
            if project is None:
                return ToolResult(content="Error: project not found")
            workspace = project.get("workspace", "")

            error = validate_trigger(trigger, workspace=workspace)
            if error:
                return ToolResult(content=f"Error: {error}")

            triggers = project.get("triggers", [])
            triggers.append(trigger)
            self._project_store.update_project(self._project_id, {"triggers": triggers})

            # Register with TriggerManager if available
            if self._trigger_manager is not None:
                self._trigger_manager.register_trigger(self._project_id, trigger)

            if ttype == "file_watch":
                msg = f"Trigger '{name}' created. Watching '{trigger['watch_path']}' for file changes."
            else:
                msg = f"Trigger '{name}' created. It will run on schedule: {trigger.get('schedule', {}).get('human', '')}"

            return ToolResult(content=json.dumps({
                "status": "created",
                "trigger": trigger,
                "message": msg,
            }))
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")


class ListTriggersTool(Tool):
    """List all triggers for the current project."""

    def __init__(self, project_id: str, project_store):
        self._project_id = project_id
        self._project_store = project_store
        self.name = "list_triggers"
        self.description = (
            "List all triggers for this project. "
            "Shows name, schedule, status, and last fired time."
        )
        self.parameters = {
            "type": "object",
            "properties": {},
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            project = self._project_store.get_project(self._project_id)
            if project is None:
                return ToolResult(content="Error: project not found")

            triggers = project.get("triggers", [])
            if not triggers:
                return ToolResult(content=json.dumps({
                    "triggers": [],
                    "message": "No triggers configured for this project.",
                }))

            return ToolResult(content=json.dumps({
                "triggers": triggers,
                "message": f"{len(triggers)} trigger(s) found.",
            }))
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")


class UpdateTriggerTool(Tool):
    """Update an existing trigger."""

    def __init__(self, project_id: str, project_store, trigger_manager=None):
        self._project_id = project_id
        self._project_store = project_store
        self._trigger_manager = trigger_manager
        self.name = "update_trigger"
        self.description = (
            "Update an existing trigger's settings. Can change name, schedule, "
            "task, autonomy, or enabled status."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "trigger_id": {
                    "type": "string",
                    "description": "The trigger ID to update (e.g., 'trg_abc12345')",
                },
                "name": {
                    "type": "string",
                    "description": "New name for the trigger",
                },
                "task": {
                    "type": "string",
                    "description": "New task description",
                },
                "cron": {
                    "type": "string",
                    "description": "New cron expression",
                },
                "human": {
                    "type": "string",
                    "description": "New human-readable schedule",
                },
                "timezone": {
                    "type": "string",
                    "description": "New timezone",
                },
                "autonomy": {
                    "type": "string",
                    "enum": ["hands_off", "check_in", "supervised"],
                    "description": "New autonomy setting (null to inherit project default)",
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable or disable the trigger",
                },
            },
            "required": ["trigger_id"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            trigger_id = arguments.get("trigger_id", "")
            if not trigger_id:
                return ToolResult(content="Error: trigger_id is required")

            project = self._project_store.get_project(self._project_id)
            if project is None:
                return ToolResult(content="Error: project not found")

            triggers = project.get("triggers", [])
            trigger = next((t for t in triggers if t.get("id") == trigger_id), None)
            if trigger is None:
                return ToolResult(content=f"Error: trigger '{trigger_id}' not found")

            # Apply updates
            if "name" in arguments:
                trigger["name"] = arguments["name"]
            if "task" in arguments:
                trigger["task"] = arguments["task"]
            if "enabled" in arguments:
                trigger["enabled"] = arguments["enabled"]
            if "autonomy" in arguments:
                trigger["autonomy"] = arguments["autonomy"]

            # Update schedule fields
            if "cron" in arguments or "human" in arguments or "timezone" in arguments:
                schedule = trigger.get("schedule", {})
                if "cron" in arguments:
                    from croniter import croniter
                    if not croniter.is_valid(arguments["cron"]):
                        return ToolResult(content=f"Error: invalid cron expression: {arguments['cron']}")
                    schedule["cron"] = arguments["cron"]
                if "human" in arguments:
                    schedule["human"] = arguments["human"]
                if "timezone" in arguments:
                    schedule["timezone"] = arguments["timezone"]
                trigger["schedule"] = schedule

            self._project_store.update_project(self._project_id, {"triggers": triggers})

            # Re-register with TriggerManager
            if self._trigger_manager is not None:
                self._trigger_manager.register_trigger(self._project_id, trigger)

            return ToolResult(content=json.dumps({
                "status": "updated",
                "trigger": trigger,
                "message": f"Trigger '{trigger.get('name', trigger_id)}' updated.",
            }))
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")


class DeleteTriggerTool(Tool):
    """Delete a trigger from the current project."""

    def __init__(self, project_id: str, project_store, trigger_manager=None):
        self._project_id = project_id
        self._project_store = project_store
        self._trigger_manager = trigger_manager
        self.name = "delete_trigger"
        self.description = (
            "Delete a trigger. Permanently removes it from the project."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "trigger_id": {
                    "type": "string",
                    "description": "The trigger ID to delete (e.g., 'trg_abc12345')",
                },
            },
            "required": ["trigger_id"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            trigger_id = arguments.get("trigger_id", "")
            if not trigger_id:
                return ToolResult(content="Error: trigger_id is required")

            project = self._project_store.get_project(self._project_id)
            if project is None:
                return ToolResult(content="Error: project not found")

            triggers = project.get("triggers", [])
            trigger = next((t for t in triggers if t.get("id") == trigger_id), None)
            if trigger is None:
                return ToolResult(content=f"Error: trigger '{trigger_id}' not found")

            trigger_name = trigger.get("name", trigger_id)
            triggers = [t for t in triggers if t.get("id") != trigger_id]
            self._project_store.update_project(self._project_id, {"triggers": triggers})

            # Unregister from TriggerManager
            if self._trigger_manager is not None:
                self._trigger_manager.unregister_trigger(trigger_id)

            return ToolResult(content=json.dumps({
                "status": "deleted",
                "trigger_id": trigger_id,
                "message": f"Trigger '{trigger_name}' deleted.",
            }))
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
