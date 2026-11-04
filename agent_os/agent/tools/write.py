# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WriteTool — create or overwrite a file within workspace."""

import json
import os

from ._path_utils import resolve_safe
from .base import Tool, ToolResult


class WriteTool(Tool):
    """Create or overwrite a file within the workspace."""

    def __init__(self, workspace: str, *, session_id: str | None = None,
                 worker: str | None = None):
        self._workspace = os.path.realpath(workspace)
        # Spec 065: who this tool writes for, so the file-history version is
        # exact (the session's Orbital mark, or the native fanout worker's).
        self._session_id = session_id
        self._worker = worker
        self.name = "write"
        self.description = "Create or overwrite a file within the workspace."
        self.parameters = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path within your workspace. Use a relative path like 'src/main.py' or 'docs/notes.md'. Do NOT start with '/' and do NOT pass an absolute path."},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["path", "content"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            path = arguments.get("path", "")
            content = arguments.get("content", "")

            resolved = resolve_safe(self._workspace, path)
            if resolved is None:
                return ToolResult(content=f"Error: could not write to {path}: path outside workspace")

            # Layer-1 memory files: stamp system-managed metadata + enforce
            # format (non-destructive). This is the shared persist path so a
            # direct tool write can no longer bypass the memory invariants.
            from agent_os.agent import memory_entries
            content, warns = memory_entries.process_on_write(self._workspace, resolved, content)

            # Auto-create parent directories
            parent = os.path.dirname(resolved)
            os.makedirs(parent, exist_ok=True)

            # Spec 065: the write and its version are one step under the
            # project's history lock (the watcher echo then records nothing).
            from agent_os.history import recording, tool_attrib
            with recording(resolved, tool_attrib(self._session_id, worker=self._worker)):
                with open(resolved, "w", encoding="utf-8") as f:
                    n = f.write(content)

            payload = {"path": resolved, "status": "success", "bytes": n}
            out = json.dumps(payload)
            if warns:
                out += "\n\nNote: " + " ".join(warns)
            return ToolResult(content=out)
        except Exception as e:
            return ToolResult(content=f"Error: could not write to {arguments.get('path', '?')}: {str(e)}")
