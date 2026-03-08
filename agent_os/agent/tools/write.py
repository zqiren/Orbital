# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WriteTool — create or overwrite a file within workspace."""

import json
import os

from .base import Tool, ToolResult


class WriteTool(Tool):
    """Create or overwrite a file within the workspace."""

    def __init__(self, workspace: str):
        self._workspace = os.path.realpath(workspace)
        self.name = "write"
        self.description = "Create or overwrite a file within the workspace."
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to workspace"},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["path", "content"],
        }

    def _resolve_safe(self, path: str) -> str | None:
        """Resolve path relative to workspace. Returns None if outside workspace."""
        resolved = os.path.realpath(os.path.join(self._workspace, path))
        if not resolved.startswith(self._workspace):
            return None
        return resolved

    def execute(self, **arguments) -> ToolResult:
        try:
            path = arguments.get("path", "")
            content = arguments.get("content", "")

            resolved = self._resolve_safe(path)
            if resolved is None:
                return ToolResult(content=f"Error: could not write to {path}: path outside workspace")

            # Auto-create parent directories
            parent = os.path.dirname(resolved)
            os.makedirs(parent, exist_ok=True)

            with open(resolved, "w", encoding="utf-8") as f:
                n = f.write(content)

            return ToolResult(content=json.dumps({
                "path": resolved,
                "status": "success",
                "bytes": n,
            }))
        except Exception as e:
            return ToolResult(content=f"Error: could not write to {arguments.get('path', '?')}: {str(e)}")
