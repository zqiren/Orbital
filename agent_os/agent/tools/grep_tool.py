# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GrepTool — search for text content within workspace files using ripgrep.

Read-only. Auto-approved under all autonomy presets. Paths are validated against
the workspace root before ripgrep is invoked.

Ripgrep binary resolution:
    1. PyInstaller bundle — sys._MEIPASS/agent_os/vendor/rg/{platform}/rg[.exe]
    2. Dev source tree — agent_os/vendor/rg/{platform}/rg[.exe]
    3. Legacy fallback paths (agent_os/vendor/rg/rg, .../macos/rg)
    4. System PATH via shutil.which("rg")
    5. Graceful error if none are available

On macOS the platform subdir is arch-specific: macos-arm64 or macos-x86_64.
Windows installers bundle ripgrep via agent_os/desktop/agentos.spec datas;
macOS installers bundle both arches via agentos-macos.spec.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ._path_utils import resolve_safe
from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

_MAX_MATCHES = 100
_MAX_COLUMNS = 500
_MAX_COUNT_PER_FILE = 50
_TIMEOUT_SECONDS = 10


def _macos_arch_subdir() -> str:
    """Return the macos-{arch} subdirectory name for the current machine."""
    # FOLLOW-UP: x86_64 (Intel) path has not been smoke-tested on Intel hardware
    # yet — ad-hoc signed binary vendored and bundled, runtime selection covers
    # it, but end-to-end grep on an Intel Mac is still pending. Remove this
    # comment once verified.
    machine = platform.machine()
    if machine == "arm64":
        return "macos-arm64"
    if machine == "x86_64":
        return "macos-x86_64"
    return f"macos-{machine}"


def _find_ripgrep() -> str | None:
    """Locate ripgrep binary. Order: PyInstaller bundle, dev vendor dir, system PATH."""
    if sys.platform == "win32":
        # Windows: flat windows/ layout, rg.exe also kept at vendor root for legacy.
        names = ["rg.exe", "windows/rg.exe"]
    elif sys.platform == "darwin":
        arch_dir = _macos_arch_subdir()
        # Prefer the arch-specific dir; keep legacy paths for backward compat.
        names = [f"{arch_dir}/rg", "rg", "macos/rg"]
    else:
        names = ["rg", "linux/rg"]

    search_roots: list[Path] = []

    # 1. PyInstaller frozen bundle — datas=[] entries get extracted here
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        search_roots.append(Path(meipass) / "agent_os" / "vendor" / "rg")

    # 2. Dev source tree — walk up from this file to agent_os/vendor/rg/
    here = Path(__file__).resolve()
    search_roots.append(here.parent.parent.parent / "vendor" / "rg")

    for root in search_roots:
        if not root.is_dir():
            continue
        for name in names:
            candidate = root / name
            if candidate.is_file() and os.access(str(candidate), os.X_OK):
                logger.info(
                    "[ripgrep] resolved to: %s (arch=%s, frozen=%s)",
                    candidate, platform.machine(), bool(meipass),
                )
                return str(candidate)

    # 3. System PATH
    fallback = shutil.which("rg")
    if fallback:
        logger.info("[ripgrep] falling back to system PATH: %s", fallback)
    return fallback


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
                    "description": "Directory or file within your workspace, relative to workspace root (e.g. 'src' or 'docs/notes.md'). Defaults to workspace root. Do NOT start with '/'.",
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
