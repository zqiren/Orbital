# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GlobTool — find files matching a glob pattern within the workspace.

Read-only. Auto-approved under all autonomy presets. Paths are validated against
the workspace root before expansion; absolute or escape-attempt paths are rejected.
"""

import os
from pathlib import Path

from ._path_utils import resolve_safe
from .base import Tool, ToolResult

_MAX_RESULTS = 1000


class GlobTool(Tool):
    """Find files matching a glob pattern, scoped to the workspace."""

    def __init__(self, workspace: str):
        self._workspace = os.path.realpath(workspace)
        self.name = "glob"
        self.description = (
            "Find files matching a glob pattern (e.g. '**/*.py'). "
            "Returns paths relative to workspace root, sorted alphabetically, "
            "capped at 1000 results."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '*.py', '**/*.md')",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search from, relative to workspace. Defaults to workspace root.",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            pattern = arguments.get("pattern")
            if not pattern or not isinstance(pattern, str):
                return ToolResult(content="Error: 'pattern' argument is required")

            path_arg = arguments.get("path", ".")
            resolved = resolve_safe(self._workspace, path_arg)
            if resolved is None:
                return ToolResult(content=f"Error: path outside workspace: {path_arg}")

            if not os.path.exists(resolved):
                return ToolResult(content=f"Error: path not found: {path_arg}")
            if not os.path.isdir(resolved):
                return ToolResult(content=f"Error: path is not a directory: {path_arg}")

            base = Path(resolved)

            # pathlib.Path.glob supports ** natively for recursive descent.
            try:
                iterator = base.glob(pattern)
            except (ValueError, OSError) as e:
                return ToolResult(content=f"Error: invalid pattern '{pattern}': {str(e)}")

            matches: list[str] = []
            truncated = False
            workspace_root = Path(self._workspace)

            try:
                for entry in iterator:
                    try:
                        entry_real = os.path.realpath(str(entry))
                    except OSError:
                        continue
                    # Defense in depth: drop anything that resolves outside the workspace
                    # (e.g. symlinks pointing out).
                    if not entry_real.startswith(self._workspace):
                        continue
                    try:
                        rel = str(Path(entry_real).relative_to(workspace_root))
                    except ValueError:
                        continue
                    # Normalise to forward slashes for consistent agent-visible output
                    rel = rel.replace(os.sep, "/")
                    matches.append(rel)
                    if len(matches) > _MAX_RESULTS:
                        truncated = True
                        break
            except (ValueError, OSError) as e:
                return ToolResult(content=f"Error: invalid pattern '{pattern}': {str(e)}")

            if not matches:
                return ToolResult(content="(no matches)")

            matches.sort()
            if truncated:
                shown = matches[:_MAX_RESULTS]
                extra = len(matches) - _MAX_RESULTS
                body = "\n".join(shown)
                return ToolResult(
                    content=f"{body}\n[... {extra}+ more paths, refine your pattern]"
                )
            return ToolResult(content="\n".join(matches))

        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")
