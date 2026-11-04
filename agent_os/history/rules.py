# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""What file history tracks (spec 065 §5.2).

Tracked = in the ripgrep listing (gitignore semantics, also in non-git
folders) AND not denied AND (text ≤ 1 MB OR an Office document ≤ 20 MB).
The four Layer-1 memory files of the project are tracked by exact path.

Everything here is pure string logic except ``list_tracked`` (one rg run)
and ``classify`` (one stat + an 8 KB read).
"""
from __future__ import annotations

import fnmatch
import os
import subprocess

from agent_os.agent.tools.grep_tool import find_ripgrep
from agent_os.utils.subprocess_flags import win_no_window_flags

# Directory segments that are never tracked anywhere in the tree. ``orbital``
# covers this project's store and every nested project's store.
DENY_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "orbital"})

# Editor / Office temp files, and the restore temp prefix.
RESTORE_TMP_PREFIX = ".orbital-restore-"
DENY_FILE_GLOBS = ("~$*", ".~lock.*", "*.swp", ".#*", "*~", "*.tmp", RESTORE_TMP_PREFIX + "*")

# The four Layer-1 files (D11), workspace-relative, POSIX separators.
LAYER1_FILES: dict[str, str] = {
    "orbital/PROJECT_STATE.md": "state",
    "orbital/DECISIONS.md": "decisions",
    "orbital/LESSONS.md": "lessons",
    "orbital/INDEX.md": "index",
}

TEXT_MAX_BYTES = 1 * 1024 * 1024
OFFICE_MAX_BYTES = 20 * 1024 * 1024
OFFICE_EXTENSIONS = frozenset({".docx", ".xlsx", ".pptx"})
_SNIFF_BYTES = 8192

_RG_TIMEOUT_SECONDS = 60


class ListingUnavailable(RuntimeError):
    """rg is missing or failed: history is ``unavailable`` for the project
    (the os.walk fallback has no .gitignore semantics, §5.2)."""


def to_rel(path: str) -> str:
    """Normalise a workspace-relative path to POSIX separators."""
    return path.replace(os.sep, "/") if os.sep != "/" else path


def is_layer1(rel: str) -> bool:
    return rel in LAYER1_FILES


def is_office(rel: str) -> bool:
    return os.path.splitext(rel)[1].lower() in OFFICE_EXTENSIONS


def is_denied(rel: str, nested: tuple[str, ...] = ()) -> bool:
    """String test, no I/O: is a workspace-relative path never tracked?

    ``nested`` = workspace-relative roots of other registered projects inside
    this one (POSIX, no trailing slash); that project owns those files.
    """
    if not rel or rel == "." or rel.startswith("../") or rel == "..":
        return True
    if rel in LAYER1_FILES:
        return False
    parts = rel.split("/")
    for seg in parts[:-1]:
        if seg in DENY_DIRS:
            return True
    name = parts[-1]
    for pat in DENY_FILE_GLOBS:
        if fnmatch.fnmatchcase(name, pat):
            return True
    for root in nested:
        if rel == root or rel.startswith(root + "/"):
            return True
    return False


def list_tracked(workspace: str, nested: tuple[str, ...] = ()) -> list[str]:
    """The listing that defines membership: rg ``--files`` with .gitignore
    semantics (``--no-require-git``) minus the deny list. Raises
    ``ListingUnavailable`` when rg is missing or fails — never falls back."""
    rg = find_ripgrep()
    if rg is None:
        raise ListingUnavailable("ripgrep not found")
    cmd = [rg, "--files", "--hidden", "--no-require-git", "--glob", "!.git"]
    # Trailing slash = directories only, at any depth (rg override globs).
    for d in sorted(DENY_DIRS):
        cmd += ["--glob", f"!{d}/"]
    for root in nested:
        cmd += ["--glob", f"!/{root}/**"]
    try:
        proc = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=_RG_TIMEOUT_SECONDS,
            creationflags=win_no_window_flags(),
        )
    except Exception as e:  # noqa: BLE001 — any failure means no listing
        raise ListingUnavailable(f"ripgrep failed: {e}") from e
    if proc.returncode not in (0, 1) or proc.stdout is None:
        raise ListingUnavailable(f"ripgrep exited {proc.returncode}")
    out = []
    for ln in proc.stdout.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rel = to_rel(ln[2:] if ln.startswith("./") else ln)
        if not is_denied(rel, nested):
            out.append(rel)
    return out


def classify(abs_path: str, rel: str, size: int | None = None) -> str | None:
    """None when the file is trackable; else the skip reason
    (``binary`` | ``size``). Office documents are trackable whole blobs."""
    if size is None:
        size = os.stat(abs_path).st_size
    if is_office(rel):
        return "size" if size > OFFICE_MAX_BYTES else None
    if size > TEXT_MAX_BYTES:
        return "size"
    with open(abs_path, "rb") as f:
        head = f.read(_SNIFF_BYTES)
    if b"\x00" in head:
        return "binary"
    return None


def classify_bytes(rel: str, data: bytes) -> str | None:
    """``classify`` for content already in memory."""
    if is_office(rel):
        return "size" if len(data) > OFFICE_MAX_BYTES else None
    if len(data) > TEXT_MAX_BYTES:
        return "size"
    if b"\x00" in data[:_SNIFF_BYTES]:
        return "binary"
    return None
