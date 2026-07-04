# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GlobTool — find files matching a glob pattern within the workspace.

Read-only. Auto-approved under all autonomy presets. Paths are validated against
the workspace root before expansion; absolute or escape-attempt paths are rejected.
"""

import os
from pathlib import Path
from typing import Callable

from ._path_utils import _excluded_in_secondary, resolve_safe_read
from .base import Tool, ToolResult

_MAX_RESULTS = 1000


class GlobTool(Tool):
    """Find files matching a glob pattern, scoped to the workspace."""

    def __init__(self, workspace: str,
                 read_roots: Callable[[], list[str]] | None = None,
                 root_labels: Callable[[], dict[str, str]] | None = None):
        self._workspace = os.path.realpath(workspace)
        # Callables (evaluated per call) so per-session scope changes apply on
        # the next call. None → single-root, byte-identical to normal projects.
        self._read_roots = read_roots
        self._root_labels = root_labels
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
                    "description": "Directory within your workspace, relative to workspace root (e.g. 'src' or 'docs/notes'). Defaults to workspace root. Do NOT start with '/'.",
                },
            },
            "required": ["pattern"],
        }

    def execute(self, **arguments) -> ToolResult:
        try:
            pattern = arguments.get("pattern")
            if not pattern or not isinstance(pattern, str):
                return ToolResult(content="Error: 'pattern' argument is required")

            roots = self._read_roots() if self._read_roots else [self._workspace]
            if not roots:
                roots = [self._workspace]
            primary_real = os.path.realpath(roots[0])
            secondary_reals = [os.path.realpath(r) for r in roots[1:]]
            labels = self._root_labels() if self._root_labels else {}

            path_arg = arguments.get("path", ".")
            if path_arg not in (".", "", None):
                # Explicit path: scope the search to that single resolved dir
                # (may be a secondary root's subtree via an absolute path).
                resolved = resolve_safe_read(roots, path_arg)
                if resolved is None:
                    return ToolResult(content=f"Error: path outside workspace: {path_arg}")
                if not os.path.exists(resolved):
                    return ToolResult(content=f"Error: path not found: {path_arg}")
                if not os.path.isdir(resolved):
                    return ToolResult(content=f"Error: path is not a directory: {path_arg}")
                search_dirs = [resolved]
            else:
                search_dirs = [primary_real] + secondary_reals

            # Per-owning-root buckets of matches. Primary → relative paths;
            # secondary → absolute paths grouped under a project label.
            primary_matches: list[str] = []
            secondary_matches: dict[str, list[str]] = {r: [] for r in secondary_reals}
            total = 0
            truncated = False

            for search_dir in search_dirs:
                owner = self._owning_root(search_dir, primary_real, secondary_reals)
                try:
                    iterator = Path(search_dir).glob(pattern)
                except (ValueError, OSError) as e:
                    return ToolResult(content=f"Error: invalid pattern '{pattern}': {str(e)}")
                try:
                    for entry in iterator:
                        try:
                            entry_real = os.path.realpath(str(entry))
                        except OSError:
                            continue
                        owner_real = owner if owner is not None else self._owning_root(
                            entry_real, primary_real, secondary_reals
                        )
                        if owner_real is None:
                            continue  # symlink-out / outside every root
                        if owner_real == primary_real:
                            if not (entry_real == primary_real
                                    or entry_real.startswith(primary_real + os.sep)):
                                continue
                            rel = os.path.relpath(entry_real, primary_real).replace(os.sep, "/")
                            primary_matches.append(rel)
                        else:
                            if _excluded_in_secondary(entry_real, owner_real):
                                continue  # other project's orbital/ or .git/
                            secondary_matches[owner_real].append(entry_real)
                        total += 1
                        if total > _MAX_RESULTS:
                            truncated = True
                            break
                except (ValueError, OSError) as e:
                    return ToolResult(content=f"Error: invalid pattern '{pattern}': {str(e)}")
                if truncated:
                    break

            if total == 0:
                return ToolResult(content="(no matches)")

            # Cap displayed rows at _MAX_RESULTS across all sections (primary
            # first, then each secondary project in root order).
            remaining = _MAX_RESULTS
            sections: list[str] = []
            primary_show = sorted(primary_matches)[:remaining]
            remaining -= len(primary_show)
            if primary_show:
                sections.append("\n".join(primary_show))
            for root in secondary_reals:
                if remaining <= 0:
                    break
                hits = sorted(secondary_matches.get(root, []))[:remaining]
                remaining -= len(hits)
                if hits:
                    label = labels.get(root, os.path.basename(root))
                    sections.append(f"[project: {label}]\n" + "\n".join(hits))

            body = "\n".join(sections)
            if truncated:
                body += "\n[... more paths, refine your pattern]"
            return ToolResult(content=body)

        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")

    @staticmethod
    def _owning_root(candidate: str, primary_real: str,
                     secondary_reals: list[str]) -> str | None:
        """Return the root (primary or a secondary) that contains ``candidate``."""
        if candidate == primary_real or candidate.startswith(primary_real + os.sep):
            return primary_real
        for root in secondary_reals:
            if candidate == root or candidate.startswith(root + os.sep):
                return root
        return None
