# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""NotifyTool — send a push notification to the user's devices."""

from datetime import datetime, timezone

from .base import Tool, ToolResult


class NotifyTool(Tool):
    """Send a notification to the user via WebSocket (and optionally push)."""

    def __init__(self, ws_manager, project_id: str):
        self._ws_manager = ws_manager
        self._project_id = project_id
        self.name = "notify"
        self.description = (
            "Send a notification to the user. Use this to alert the user about "
            "important events, completed tasks, or anything that needs their attention."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short notification title",
                },
                "body": {
                    "type": "string",
                    "description": "Notification body text",
                },
                "urgency": {
                    "type": "string",
                    "enum": ["high", "normal", "low"],
                    "description": "Notification urgency. 'high' bypasses user prefs, 'low' is silent.",
                },
            },
            "required": ["title", "body"],
        }

    def execute(self, **arguments) -> ToolResult:
        title = arguments.get("title", "")
        body = arguments.get("body", "")
        urgency = arguments.get("urgency", "normal")

        if not title or not body:
            return ToolResult(content="Error: title and body are required")

        self._ws_manager.broadcast(self._project_id, {
            "type": "agent.notify",
            "project_id": self._project_id,
            "title": title,
            "body": body,
            "urgency": urgency,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if urgency == "low":
            return ToolResult(content=f"Notification queued (low urgency)")
        return ToolResult(content=f"Notification sent: {title}")
