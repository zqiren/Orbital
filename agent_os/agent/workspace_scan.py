# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic, gitignore-respecting, self-bounding workspace skeleton.

Stage 1 of the cold-start scan. No LLM. Produces the one fact the agent
cannot reliably guess — how much is here and where — as a bounded inventory
string suitable for injection as a system message.

Reuses the vendored ripgrep (`rg --files`, which honors .gitignore) for the
file listing, then stats each path for size. Falls back to an os.walk that
skips .git if ripgrep is unavailable.
"""
from __future__ import annotations

import os
import subprocess

from agent_os.agent.tools.grep_tool import find_ripgrep
from agent_os.utils.subprocess_flags import win_no_window_flags

_DEFAULT_MAX_FILES = 1000
_RG_TIMEOUT_SECONDS = 10


def _list_files(workspace: str) -> list[str]:
    """Return workspace-relative file paths, gitignore-respected when possible."""
    rg = find_ripgrep()
    if rg is not None:
        try:
            proc = subprocess.run(
                # --no-require-git: honor .gitignore even when the imported
                # workspace is not itself a git repo (ripgrep's default only
                # applies gitignore rules inside a git repository).
                [rg, "--files", "--hidden", "--no-require-git", "--glob", "!.git"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=_RG_TIMEOUT_SECONDS,
                creationflags=win_no_window_flags(),
            )
            if proc.returncode in (0, 1):  # 1 == no files matched
                return [ln for ln in proc.stdout.splitlines() if ln.strip()]
        except (OSError, subprocess.SubprocessError):
            pass
    # Fallback: os.walk skipping .git (no gitignore semantics).
    out: list[str] = []
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), workspace)
            out.append(rel)
    return out


def scan_workspace(workspace: str, *, max_files: int = _DEFAULT_MAX_FILES) -> str:
    """Build a bounded inventory string of (path, size) for the workspace."""
    rels = _list_files(workspace)
    sized: list[tuple[str, int]] = []
    for rel in rels:
        try:
            size = os.path.getsize(os.path.join(workspace, rel))
        except OSError:
            size = 0
        sized.append((rel, size))

    total = len(sized)
    if total == 0:
        return "[WORKSPACE SKELETON]\nNo files found (empty workspace)."

    truncated = total > max_files
    if truncated:
        # Keep the largest files (high signal) and the shallowest paths
        # (top-level structure). Union, then cap.
        by_size = sorted(sized, key=lambda t: t[1], reverse=True)[:max_files]
        by_depth = sorted(sized, key=lambda t: t[0].count(os.sep))[:max_files]
        seen: set[str] = set()
        kept: list[tuple[str, int]] = []
        for item in by_depth + by_size:
            if item[0] in seen:
                continue
            seen.add(item[0])
            kept.append(item)
            if len(kept) >= max_files:
                break
        sized = kept

    sized.sort(key=lambda t: t[0])
    total_bytes = sum(s for _, s in sized)
    lines = [
        f"[WORKSPACE SKELETON] {total} files"
        + (f" (showing {len(sized)} — largest + top levels)" if truncated else "")
        + f", ~{total_bytes} bytes shown.",
    ]
    for rel, size in sized:
        lines.append(f"{rel}\t{size}")
    if truncated:
        lines.append(
            f"[... {total - len(sized)} more files not shown; "
            "use list/glob/grep to explore specific paths]"
        )
    return "\n".join(lines)
