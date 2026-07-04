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
from typing import Callable

from ._path_utils import _excluded_in_secondary, resolve_safe_read
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


# Public alias so other modules (e.g. workspace_scan) can reuse the vendored
# ripgrep resolution logic without importing a private symbol.
def find_ripgrep() -> str | None:
    """Public wrapper around the vendored-ripgrep resolver."""
    return _find_ripgrep()


class GrepTool(Tool):
    """Search for text content across workspace files using ripgrep."""

    def __init__(self, workspace: str,
                 read_roots: Callable[[], list[str]] | None = None,
                 root_labels: Callable[[], dict[str, str]] | None = None):
        self._workspace = os.path.realpath(workspace)
        # Callables (evaluated per call) so per-session scope changes apply on
        # the next call. None → single-root, byte-identical to normal projects.
        self._read_roots = read_roots
        self._root_labels = root_labels
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

            roots = self._read_roots() if self._read_roots else [self._workspace]
            if not roots:
                roots = [self._workspace]
            primary_real = os.path.realpath(roots[0])
            secondary_reals = [os.path.realpath(r) for r in roots[1:]]
            labels = self._root_labels() if self._root_labels else {}

            path_arg = arguments.get("path", ".")
            if path_arg not in (".", "", None):
                resolved = resolve_safe_read(roots, path_arg)
                if resolved is None:
                    return ToolResult(content=f"Error: path outside workspace: {path_arg}")
                if not os.path.exists(resolved):
                    return ToolResult(content=f"Error: path not found: {path_arg}")
                search_dirs = [resolved]
            else:
                search_dirs = [primary_real] + secondary_reals

            rg = _find_ripgrep()
            if rg is None:
                return ToolResult(
                    content="Error: ripgrep not found. Install ripgrep or ensure the "
                    "bundled binary exists at agent_os/vendor/rg/."
                )

            fixed = bool(arguments.get("fixed_strings"))
            glob_filter = arguments.get("glob_filter")
            glob_filter = glob_filter if isinstance(glob_filter, str) and glob_filter else None

            primary_lines: list[str] = []
            secondary_lines: dict[str, list[str]] = {r: [] for r in secondary_reals}
            total = 0
            truncated = False
            hard_error: tuple[int, str] | None = None

            for search_dir in search_dirs:
                owner = self._owning_root(search_dir, primary_real, secondary_reals)
                is_secondary = owner is not None and owner != primary_real
                rc, stdout, stderr = self._run_rg(
                    rg, search_dir, pattern, fixed=fixed, glob_filter=glob_filter,
                    exclude_internal=is_secondary,
                )
                if rc is None:  # timeout / exec failure
                    return ToolResult(content=stderr)
                if rc == 1:
                    continue  # no matches in this root
                if rc >= 2:
                    hard_error = (rc, stderr)
                    continue

                for line in stdout.splitlines():
                    parts = _split_rg_line(line)
                    if parts is None:
                        (secondary_lines[owner] if is_secondary else primary_lines).append(
                            line[:_MAX_COLUMNS]
                        )
                    else:
                        abs_path, lineno, content = parts
                        if is_secondary:
                            # Defence in depth: rg already excludes these globs,
                            # but never surface another project's internals.
                            if _excluded_in_secondary(abs_path, owner):
                                continue
                            secondary_lines[owner].append(f"{abs_path}:{lineno}:{content}")
                        else:
                            try:
                                rel_path = os.path.relpath(abs_path, primary_real)
                            except ValueError:
                                rel_path = abs_path
                            primary_lines.append(
                                f"{rel_path.replace(os.sep, '/')}:{lineno}:{content}"
                            )
                    total += 1
                    if total >= _MAX_MATCHES:
                        truncated = True
                        break
                if truncated:
                    break

            if total == 0:
                if hard_error is not None:
                    rc, stderr = hard_error
                    stderr = (stderr or "").strip() or "unknown error"
                    if len(stderr) > 500:
                        stderr = stderr[:500] + "..."
                    return ToolResult(content=f"Error: grep failed (exit {rc}): {stderr}")
                return ToolResult(content="No matches found")

            sections: list[str] = []
            if primary_lines:
                sections.append("\n".join(primary_lines))
            for root in secondary_reals:
                hits = secondary_lines.get(root)
                if hits:
                    label = labels.get(root, os.path.basename(root))
                    sections.append(f"[project: {label}]\n" + "\n".join(hits))

            body = "\n".join(sections)
            if truncated:
                body += f"\n[... truncated at {_MAX_MATCHES} matches, refine your pattern]"
            return ToolResult(content=body)

        except Exception as e:
            return ToolResult(content=f"Error: {str(e)}")

    @staticmethod
    def _owning_root(candidate: str, primary_real: str,
                     secondary_reals: list[str]) -> str | None:
        if candidate == primary_real or candidate.startswith(primary_real + os.sep):
            return primary_real
        for root in secondary_reals:
            if candidate == root or candidate.startswith(root + os.sep):
                return root
        return None

    @staticmethod
    def _run_rg(rg: str, search_dir: str, pattern: str, *, fixed: bool,
                glob_filter: str | None, exclude_internal: bool):
        """Run ripgrep in one directory. Returns (returncode, stdout, stderr).

        ``returncode`` is None on timeout / exec failure (stderr carries the
        user-facing error). ``exclude_internal`` adds globs that keep a
        secondary project's ``orbital/`` and ``.git/`` out of the results.
        """
        cmd = [
            rg,
            "--line-number",
            "--no-heading",
            "--color", "never",
            "--max-count", str(_MAX_COUNT_PER_FILE),
            "--max-columns", str(_MAX_COLUMNS),
        ]
        if fixed:
            cmd.append("-F")
        if glob_filter:
            cmd.extend(["-g", glob_filter])
        if exclude_internal:
            cmd.extend(["-g", "!orbital", "-g", "!.git"])
        cmd.extend(["--", pattern, search_dir])
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECONDS,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return None, "", f"Error: grep timed out after {_TIMEOUT_SECONDS}s"
        except FileNotFoundError:
            return None, "", "Error: ripgrep binary could not be executed"
        except Exception as e:
            return None, "", f"Error: failed to run grep: {str(e)}"
        return result.returncode, result.stdout or "", result.stderr or ""


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
