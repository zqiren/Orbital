# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GlobTool — find files matching a glob pattern within the workspace.

Read-only. Auto-approved under all autonomy presets. Paths are validated against
the workspace root before expansion; absolute or escape-attempt paths are rejected.
"""

import fnmatch
import os
import sys
from pathlib import Path, PurePath
from typing import Callable

from ._path_utils import (
    _SECONDARY_EXCLUDED_COMPONENTS,
    OPEN_READS_PATH_SENTENCE,
    OPEN_READS_SENTENCE,
    hidden_in_secondary,
    is_memory_file,
    memory_file_paths,
    open_reads_active,
    real_memory_roots,
    resolve_safe_read,
)
from .base import Tool, ToolResult

_MAX_RESULTS = 1000


# ── Secondary roots: a pruned walk (2026-11-07) ───────────────────────────
#
# ``Path.glob`` cannot skip a subtree, so over another project's root it
# walked that project's whole ``orbital/`` (session logs, history, …) only to
# drop every hit afterwards. Secondary roots use this walk instead: it follows
# the pattern segment by segment like pathlib (``**`` = this dir and every
# subdir, not through symlinked dirs; ``*``/``?``/``[…]`` per name, hidden
# names included) and never enters an ``orbital`` or ``.git`` dir. The
# allowlisted memory files are then matched against the pattern by name.

def _pattern_segments(pattern: str) -> list[str] | None:
    """The pattern's relative segments, or None when only ``Path.glob`` can
    take it (absolute, ``..``, empty) — it then raises or walks as before."""
    try:
        pure = PurePath(pattern)
    except (TypeError, ValueError):
        return None
    if pure.is_absolute() or pure.anchor:
        return None
    segs = [s for s in pure.parts if s not in ("", ".")]
    if not segs or ".." in segs:
        return None
    return segs


# Python 3.13 changed pathlib: a trailing ``**`` matches files as well as
# directories (3.11/3.12: directories only). Follow the running interpreter.
_TRAILING_STARSTAR_FILES = sys.version_info >= (3, 13)


def _pruned(name: str) -> bool:
    return name.casefold() in _SECONDARY_EXCLUDED_COMPONENTS


def _select(dirpath: str, segs: list[str], out: list[str]) -> None:
    if not segs:
        out.append(dirpath)
        return
    seg, rest = segs[0], segs[1:]
    try:
        entries = list(os.scandir(dirpath))
    except OSError:
        return
    if seg == "**":
        _select(dirpath, rest, out)  # zero directories
        for e in entries:
            if _pruned(e.name):
                continue
            try:
                is_dir = e.is_dir() and not e.is_symlink()
            except OSError:
                continue
            if is_dir:
                _select(e.path, segs, out)
            elif not rest and _TRAILING_STARSTAR_FILES:
                out.append(e.path)
        return
    for e in entries:
        if _pruned(e.name) or not fnmatch.fnmatch(e.name, seg):
            continue
        if not rest:
            out.append(e.path)
            continue
        try:
            is_dir = e.is_dir()
        except OSError:
            continue
        if is_dir:
            _select(e.path, rest, out)


def _parts_match(parts: list[str], segs: list[str]) -> bool:
    if not segs:
        return not parts
    if segs[0] == "**":
        return any(_parts_match(parts[i:], segs[1:]) for i in range(len(parts) + 1))
    return (bool(parts) and fnmatch.fnmatch(parts[0], segs[0])
            and _parts_match(parts[1:], segs[1:]))


def _secondary_glob(search_dir: str, owner_real: str, segs: list[str],
                    memory_ok: bool) -> list[str]:
    """Matches under a secondary root's ``search_dir``: the pruned walk, plus
    the owner's memory files when ``memory_ok`` and the pattern names them."""
    found: list[str] = []
    _select(search_dir, segs, found)
    # Before 3.13 a trailing ** matches directories only.
    if memory_ok and (segs[-1] != "**" or _TRAILING_STARSTAR_FILES):
        base = os.path.realpath(search_dir)
        for path in memory_file_paths(owner_real):
            if not path.startswith(base + os.sep) or not os.path.isfile(path):
                continue
            if not is_memory_file(os.path.realpath(path), owner_real):
                continue  # a symlink: the real file is not at that location
            if _parts_match(os.path.relpath(path, base).split(os.sep), segs):
                found.append(path)
    return list(dict.fromkeys(found))


class GlobTool(Tool):
    """Find files matching a glob pattern, scoped to the workspace."""

    def __init__(self, workspace: str,
                 read_roots: Callable[[], list[str]] | None = None,
                 root_labels: Callable[[], dict[str, str]] | None = None,
                 open_reads: Callable[[], bool] | None = None,
                 cross_project: bool | None = None,
                 memory_roots: Callable[[], list[str]] | None = None):
        # ``cross_project`` picks the LLM-facing text: the scratch (Quick
        # Tasks) wording that names other in-scope projects, or the normal
        # single-root wording. Defaults to "a root callable was given" so
        # existing callers are unchanged; spec 117 wires a root callable for
        # EVERY project (granted portals, D7) and passes False for normal ones.
        if cross_project is None:
            cross_project = read_roots is not None
        self._workspace = os.path.realpath(workspace)
        # Callables (evaluated per call) so per-session scope changes apply on
        # the next call. None → single-root, byte-identical to normal projects.
        self._read_roots = read_roots
        self._root_labels = root_labels
        # Spec 117 D3: safe mode OFF → an explicit absolute path may point
        # anywhere on disk; the default root list is unchanged.
        self._open_reads = open_reads
        # In-scope project roots whose memory files are listed (Quick Tasks).
        self._memory_roots = memory_roots
        self.name = "glob"
        self.description = (
            "Find files matching a glob pattern (e.g. '**/*.py'). "
            "Returns paths relative to workspace root, sorted alphabetically, "
            "capped at 1000 results."
        )
        if not cross_project:
            dir_desc = "Directory within your workspace, relative to workspace root (e.g. 'src' or 'docs/notes'). Defaults to workspace root. Do NOT start with '/'."
        else:
            dir_desc = (
                "Directory to search. Relative paths stay in YOUR workspace; by "
                "default ALL in-scope project workspaces are searched and "
                "cross-project matches are prefixed with [project: <name>]."
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
                    "description": dir_desc,
                },
            },
            "required": ["pattern"],
        }

    def schema(self) -> dict:
        """Spec 117 §3.4 — the LLM-facing text follows the project's mode."""
        if not open_reads_active(self._open_reads):
            return super().schema()
        params = {**self.parameters, "properties": dict(self.parameters["properties"])}
        params["properties"]["path"] = {
            **params["properties"]["path"],
            "description": (
                f"Directory to search. {OPEN_READS_PATH_SENTENCE} "
                "Defaults to your workspace."
            ),
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description} {OPEN_READS_SENTENCE}",
                "parameters": params,
            },
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
            memory_list = self._memory_roots() if self._memory_roots else None
            memory_reals = real_memory_roots(memory_list)

            path_arg = arguments.get("path", ".")
            if path_arg not in (".", "", None):
                # Explicit path: scope the search to that single resolved dir
                # (may be a secondary root's subtree via an absolute path).
                resolved = resolve_safe_read(
                    roots, path_arg, open_reads=open_reads_active(self._open_reads),
                    memory_roots=memory_list)
                if resolved is None:
                    return ToolResult(content=f"Error: path outside workspace: {path_arg}")
                if not os.path.exists(resolved):
                    return ToolResult(content=f"Error: path not found: {path_arg}")
                if not os.path.isdir(resolved):
                    return ToolResult(content=f"Error: path is not a directory: {path_arg}")
                search_dirs = [resolved]
                if self._owning_root(resolved, primary_real, secondary_reals) is None:
                    # Spec 117 D3 (open reads): an explicit path outside every
                    # root is searched as an ad-hoc secondary root, labelled
                    # by its own path so the matches stay absolute.
                    secondary_reals = [*secondary_reals, resolved]
                    labels = {**labels, resolved: resolved}
            else:
                search_dirs = [primary_real] + secondary_reals

            # Per-owning-root buckets of matches. Primary → relative paths;
            # secondary → absolute paths grouped under a project label.
            primary_matches: list[str] = []
            secondary_matches: dict[str, list[str]] = {r: [] for r in secondary_reals}
            total = 0
            truncated = False

            segs = _pattern_segments(pattern)
            for search_dir in search_dirs:
                owner = self._owning_root(search_dir, primary_real, secondary_reals)
                try:
                    if owner is not None and owner != primary_real and segs is not None:
                        iterator = _secondary_glob(search_dir, owner, segs,
                                                   memory_ok=owner in memory_reals)
                    else:
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
                            if hidden_in_secondary(entry_real, owner_real,
                                                   memory_ok=owner_real in memory_reals):
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
