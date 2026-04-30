# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""EditTool — surgical find+replace within a file in the workspace."""

import json
import os

from ._path_utils import resolve_safe
from .base import Tool, ToolResult


class EditTool(Tool):
    """Surgical find and replace within a file in the workspace."""

    def __init__(self, workspace: str):
        self._workspace = os.path.realpath(workspace)
        self.name = "edit"
        self.description = "Find and replace text in a file within the workspace."
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path within your workspace. Use a relative path like 'src/main.py' or 'docs/notes.md'. Do NOT start with '/' and do NOT pass an absolute path."},
                "old_text": {"type": "string", "description": "Exact text to find"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            path = arguments.get("path", "")
            old_text = arguments.get("old_text", "")
            new_text = arguments.get("new_text", "")

            resolved = resolve_safe(self._workspace, path)
            if resolved is None:
                return ToolResult(content=f"Error: path outside workspace: {path}")

            if not os.path.isfile(resolved):
                return ToolResult(content=f"Error: file not found: {path}")

            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_text)

            if count == 0:
                preview = old_text[:50]
                suffix = "..." if len(old_text) > 50 else ""
                return ToolResult(content=f"Error: '{preview}{suffix}' not found in {path}")

            if count > 1:
                return ToolResult(content="Error: multiple matches found (expected exactly 1)")

            # Exactly one match — replace it
            new_content = content.replace(old_text, new_text, 1)

            with open(resolved, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult(content=json.dumps({
                "path": path,
                "status": "success",
                "replacements": 1,
            }))
        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
