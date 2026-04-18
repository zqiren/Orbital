# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GrepTool — search for text content within workspace files using ripgrep.

Read-only. Auto-approved under all autonomy presets. Paths are validated against
the workspace root before ripgrep is invoked.

Ripgrep binary resolution:
    1. Bundled binary at agent_os/vendor/rg/{platform}/rg[.exe]
    2. System PATH via shutil.which("rg")
    3. Graceful error if neither is available

MVP limitation: the bundled-binary path is an optional file; the installer
bundling step (packaging/bundle_rg.py) is deferred. In the interim the tool
falls back to the user's system ripgrep. On machines without `rg` on PATH, the
tool returns a clear error rather than crashing — see 'ripgrep not found' test.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .base import Tool, ToolResult

_MAX_MATCHES = 100
_MAX_COLUMNS = 500
_MAX_COUNT_PER_FILE = 50
_TIMEOUT_SECONDS = 10


def _find_ripgrep() -> str | None:
    """Locate ripgrep binary. Prefer bundled over system."""
    # Walk up from this file to find agent_os/vendor/rg/
    here = Path(__file__).resolve()
    vendor = here.parent.parent.parent / "vendor" / "rg"
    if vendor.is_dir():
        # Try platform-specific subdir first
        if sys.platform == "win32":
            candidates = [vendor / "rg.exe", vendor / "windows" / "rg.exe"]
        elif sys.platform == "darwin":
            candidates = [vendor / "rg", vendor / "macos" / "rg"]
        else:
            candidates = [vendor / "rg", vendor / "linux" / "rg"]
        for c in candidates:
            if c.is_file() and os.access(str(c), os.X_OK):
                return str(c)

    return shutil.which("rg")


class GrepTool(Tool):
    """Search for text content across workspace files using ripgrep."""

    def __init__(self, workspace: str):
        self._workspace = os.path.realpath(workspace)
        self.name = "grep"
        self.description = (
            "Search for text within workspace files using ripgrep. "
            "Supports regex by default, or literal matches via fixed_strings. "
            "Returns matches in 'path:line:content' format, capped at 100 matches."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex by default)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search in, relative to workspace. Defaults to workspace root.",
                },
                "glob_filter": {
                    "type": "string",
                    "description": "Optional glob to limit files searched (ripgrep -g flag, e.g. '*.py')",
                },
                "fixed_strings": {
                    "type": "boolean",
                    "description": "Treat pattern as a literal string, not regex (ripgrep -F flag)",
                },
            },
            "required": ["pattern"],
        }

    # NOTE: duplicated from ReadTool._resolve_safe — see comment in glob_tool.py.
    def _resolve_safe(self, path: str) -> str | None:
        """Resolve path relative to workspace. Returns None if outside workspace."""
        path = path.lstrip("/")
        resolved = os.path.realpath(os.path.join(self._workspace, path))
        if not resolved.startswith(self._workspace):
            return None
        return resolved

    def execute(self, **arguments) -> ToolResult:
        try:
            pattern = arguments.get("pattern")
            if not pattern or not isinstance(pattern, str):
                return ToolResult(content="Error: 'pattern' argument is required")

            path_arg = arguments.get("path", ".")
            resolved = self._resolve_safe(path_arg)
            if resolved is None:
                return ToolResult(content=f"Error: path outside workspace: {path_arg}")
            if not os.path.exists(resolved):
                return ToolResult(content=f"Error: path not found: {path_arg}")

            rg = _find_ripgrep()
            if rg is None:
                return ToolResult(
                    content="Error: ripgrep not found. Install ripgrep or ensure the "
                    "bundled binary exists at agent_os/vendor/rg/."
                )

            cmd = [
                rg,
                "--line-number",
                "--no-heading",
                "--color", "never",
                "--max-count", str(_MAX_COUNT_PER_FILE),
                "--max-columns", str(_MAX_COLUMNS),
            ]
            if arguments.get("fixed_strings"):
                cmd.append("-F")
            glob_filter = arguments.get("glob_filter")
            if glob_filter and isinstance(glob_filter, str):
                cmd.extend(["-g", glob_filter])

            cmd.extend(["--", pattern, resolved])

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                    encoding="utf-8",
                    errors="replace",
                )
            except subprocess.TimeoutExpired:
                return ToolResult(content=f"Error: grep timed out after {_TIMEOUT_SECONDS}s")
            except FileNotFoundError:
                return ToolResult(content="Error: ripgrep binary could not be executed")
            except Exception as e:
                return ToolResult(content=f"Error: failed to run grep: {str(e)}")

            if result.returncode == 1:
                return ToolResult(content="No matches found")
            if result.returncode >= 2:
                stderr = (result.stderr or "").strip() or "unknown error"
                # Trim obscenely long stderr
                if len(stderr) > 500:
                    stderr = stderr[:500] + "..."
                return ToolResult(content=f"Error: grep failed (exit {result.returncode}): {stderr}")

            # returncode == 0: matches
            lines = (result.stdout or "").splitlines()
            parsed: list[str] = []
            total = 0
            truncated = False
            workspace_root = self._workspace

            for line in lines:
                # ripgrep format with --line-number --no-heading: path:line:content
                # On Windows, path contains a drive letter colon — split on last 2 colons
                # by splitting only 2 times from the right; we need exactly
                # path:line:content, so split from the left with maxsplit on the
                # first two colons AFTER any drive-letter colon.
                parts = _split_rg_line(line)
                if parts is None:
                    # Keep unparseable lines as-is (defensive)
                    parsed.append(line[:_MAX_COLUMNS])
                    total += 1
                else:
                    abs_path, lineno, content = parts
                    try:
                        rel_path = os.path.relpath(abs_path, workspace_root)
                    except ValueError:
                        rel_path = abs_path
                    rel_path = rel_path.replace(os.sep, "/")
                    parsed.append(f"{rel_path}:{lineno}:{content}")
                    total += 1

                if total >= _MAX_MATCHES:
                    truncated = True
                    # Count remaining lines for the truncation marker
                    remaining = len(lines) - len(parsed)
                    if remaining > 0:
                        body = "\n".join(parsed)
                        return ToolResult(
                            content=f"{body}\n[... truncated, {len(lines)} total matches, refine your pattern]"
                        )
                    break

            if not parsed:
                return ToolResult(content="No matches found")

            body = "\n".join(parsed)
            if truncated:
                body += f"\n[... truncated at {_MAX_MATCHES} matches, refine your pattern]"
            return ToolResult(content=body)

        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")


def _split_rg_line(line: str) -> tuple[str, str, str] | None:
    """Parse a ripgrep --no-heading line into (path, lineno, content).

    On Windows, paths contain 'C:\\...' — the drive colon is not a separator.
    Strategy: find the first ':<digits>:' boundary — that separates path from
    lineno from content.
    """
    if not line:
        return None
    # Find the position where :<digits>: appears
    i = 0
    n = len(line)
    while i < n:
        idx = line.find(":", i)
        if idx == -1:
            return None
        # Check that what follows is <digits>:
        j = idx + 1
        start = j
        while j < n and line[j].isdigit():
            j += 1
        if j > start and j < n and line[j] == ":":
            path = line[:idx]
            lineno = line[start:j]
            content = line[j + 1:]
            return (path, lineno, content)
        i = idx + 1
    return None
