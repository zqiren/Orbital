# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""RequestAccessTool — request portal access (always returns pending)."""

import json

from .base import Tool, ToolResult


class RequestAccessTool(Tool):
    """Request access to a path. Always returns pending status."""

    def __init__(self):
        self.name = "request_access"
        self.description = "Request access to a file or directory path."
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to request access to"},
                "reason": {"type": "string", "description": "Reason for access"},
                "access_type": {
                    "type": "string",
                    "description": "Access type: read or read_write",
                    "enum": ["read", "read_write"],
                },
            },
            "required": ["path", "reason", "access_type"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            path = arguments.get("path", "")
            reason = arguments.get("reason", "")
            access_type = arguments.get("access_type", "read")

            return ToolResult(content=json.dumps({
                "path": path,
                "reason": reason,
                "access_type": access_type,
                "status": "pending",
            }))
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
