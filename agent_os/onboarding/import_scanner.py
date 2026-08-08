# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read-only discovery of importable projects for first-run onboarding
(backlog #34, MVP).

A brand-new Orbital user usually already has project/session data on disk from
other CLI agents (Claude Code, Codex) and note vaults (Obsidian). This module
scans those locations and produces a ranked, deduplicated, path-verified list of
*candidate* projects. Nothing is created here — the caller confirms each
candidate, and confirmed candidates route through the existing
``POST /api/v2/projects`` (link-only: the workspace IS the real folder).

Hard guarantees (see backlog #34):

* **Metadata only.** We read paths, per-session working directory (``cwd``),
  project/vault name, timestamps and counts — never transcript message bodies.
  Concretely: Claude Code sessions are probed for the whitelisted ``cwd`` key
  only; the Codex SQLite index is queried for ``cwd``/``updated_at``/
  ``rollout_path`` columns only (never ``first_user_message``/``preview``/
  ``title``, which are message-derived); Obsidian reads only the vault registry.
* **Strictly read-only.** Nothing here opens a file for writing, deletes, moves
  or mutates anything under ``~/.claude``, ``~/.codex`` or any vault.
* **Dead paths dropped.** Candidates whose target folder no longer exists on
  this machine are filtered out (empirically ~80% of raw candidates are stale).
* **Deduplicated.** Multiple sources / sessions pointing at the same real folder
  collapse to one candidate (session counts summed, latest activity kept).
* **Ranked.** Agent-project candidates rank above Obsidian vaults (a vault is
  often a personal knowledge base, not a project). Within a tier, most-recently-
  active first.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Source identifiers (also the `source` field value on a candidate).
SOURCE_CLAUDE_CODE = "claude-code"
SOURCE_CODEX = "codex"
SOURCE_OBSIDIAN = "obsidian"

# Ranking tier: agent projects sort above Obsidian vaults.
_AGENT_SOURCES = frozenset({SOURCE_CLAUDE_CODE, SOURCE_CODEX})

# Cap how far into a Claude Code session file we look for the `cwd` key. The
# first records are often queue/title bookkeeping without a cwd; the working
# directory appears within the first handful of real turns. Bounding the scan
# keeps us from ever streaming a whole (potentially huge) transcript.
_CWD_PROBE_MAX_LINES = 40


@dataclass
class ImportCandidate:
    """One importable folder discovered on disk.

    ``path`` is the real, on-disk, existence-verified folder that would become
    the linked project workspace. ``last_activity`` is an ISO-8601 UTC string
    (or ``None`` when no timestamp is available, e.g. a vault with no registry
    ``ts`` and an unreadable mtime).
    """

    source: str
    name: str
    path: str
    session_count: int = 0
    last_activity: str | None = None
    # Internal: every source that contributed to this (post-dedup) candidate.
    # Not part of the public {source,name,path,session_count,last_activity}
    # payload, but handy for debugging/telemetry.
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "name": self.name,
            "path": self.path,
            "session_count": self.session_count,
            "last_activity": self.last_activity,
        }


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def scan_importable_projects(
    *,
    home: str | None = None,
    claude_projects_dir: str | None = None,
    codex_dir: str | None = None,
    obsidian_config_path: str | None = None,
) -> list[ImportCandidate]:
    """Scan all v1 sources and return ranked, deduped, path-verified candidates.

    All source locations default to the standard per-user paths derived from
    ``home`` (default ``~``), but every one is injectable so tests can point at
    fixture home directories. Any source that is missing or unreadable is
    skipped silently — discovery never raises.
    """
    home = home or os.path.expanduser("~")

    if claude_projects_dir is None:
        claude_projects_dir = os.path.join(home, ".claude", "projects")
    if codex_dir is None:
        codex_dir = os.path.join(home, ".codex")
    if obsidian_config_path is None:
        obsidian_config_path = _default_obsidian_config_path(home)

    raw: list[ImportCandidate] = []
    raw.extend(_safe(_scan_claude_code, claude_projects_dir))
    raw.extend(_safe(_scan_codex, codex_dir))
    raw.extend(_safe(_scan_obsidian, obsidian_config_path))

    verified = [c for c in raw if _path_is_live(c.path)]
    deduped = _dedupe(verified)
    return _rank(deduped)


# --------------------------------------------------------------------------
# Claude Code — ~/.claude/projects/<encoded-cwd>/<session>.jsonl
# --------------------------------------------------------------------------


def _scan_claude_code(projects_dir: str) -> list[ImportCandidate]:
    """One candidate per Claude Code project directory.

    ``session_count`` is the number of ``*.jsonl`` transcripts; ``path`` is the
    real ``cwd`` read from a session's whitelisted ``cwd`` key (the directory
    name is a lossy encoding of the path, so we prefer the recorded cwd);
    ``last_activity`` is the newest transcript mtime.
    """
    out: list[ImportCandidate] = []
    if not os.path.isdir(projects_dir):
        return out

    for entry in os.scandir(projects_dir):
        if not entry.is_dir():
            continue
        sessions = _jsonl_files(entry.path)
        if not sessions:
            continue
        # Newest first, so the cwd probe uses a recent transcript.
        sessions.sort(key=_safe_mtime, reverse=True)
        cwd = None
        for session in sessions:
            cwd = _claude_session_cwd(session)
            if cwd:
                break
        if not cwd:
            continue
        out.append(
            ImportCandidate(
                source=SOURCE_CLAUDE_CODE,
                name=os.path.basename(cwd.rstrip("/\\")) or cwd,
                path=cwd,
                session_count=len(sessions),
                last_activity=_iso(_safe_mtime(sessions[0])),
                sources=[SOURCE_CLAUDE_CODE],
            )
        )
    return out


def _claude_session_cwd(path: str) -> str | None:
    """Extract the ``cwd`` from a Claude Code transcript — metadata only.

    Reads at most ``_CWD_PROBE_MAX_LINES`` lines and touches ONLY the top-level
    ``cwd`` key of each JSON record. Message/content fields are never accessed.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if i >= _CWD_PROBE_MAX_LINES:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(record, dict):
                    cwd = record.get("cwd")
                    if isinstance(cwd, str) and cwd:
                        return cwd
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------
# Codex — ~/.codex/state_5.sqlite index (fallback: session_meta JSONL)
# --------------------------------------------------------------------------


def _scan_codex(codex_dir: str) -> list[ImportCandidate]:
    """One candidate per distinct Codex working directory.

    Prefers the ``state_5.sqlite`` ``threads`` index (fast, and we query only
    the ``cwd``/``updated_at``/``rollout_path`` metadata columns). Falls back to
    reading each session's first-line ``session_meta`` payload when the index is
    absent.
    """
    index = os.path.join(codex_dir, "state_5.sqlite")
    if os.path.isfile(index):
        rows = _codex_index_rows(index)
        if rows is not None:
            return _codex_candidates_from_rows(rows)
    return _scan_codex_sessions(os.path.join(codex_dir, "sessions"))


def _codex_index_rows(index_path: str) -> list[tuple[str, int | None]] | None:
    """Return ``(cwd, updated_at_epoch)`` rows from the Codex threads index.

    Only metadata columns are selected — never ``first_user_message``,
    ``preview`` or ``title`` (all message-derived). Opened read-only via a URI
    connection so we can never mutate the index. Returns ``None`` if the schema
    is not the expected one (caller falls back to the JSONL scan).
    """
    try:
        con = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        try:
            cur = con.execute(
                "SELECT cwd, updated_at FROM threads WHERE cwd IS NOT NULL"
            )
        except sqlite3.Error:
            return None
        rows: list[tuple[str, int | None]] = []
        for cwd, updated_at in cur.fetchall():
            if isinstance(cwd, str) and cwd:
                ts = updated_at if isinstance(updated_at, (int, float)) else None
                rows.append((cwd, ts))
        return rows
    finally:
        con.close()


def _codex_candidates_from_rows(
    rows: list[tuple[str, int | None]]
) -> list[ImportCandidate]:
    grouped: dict[str, dict] = {}
    for cwd, ts_epoch in rows:
        g = grouped.setdefault(cwd, {"count": 0, "latest": None})
        g["count"] += 1
        iso = _iso_from_epoch(ts_epoch)
        if iso and (g["latest"] is None or iso > g["latest"]):
            g["latest"] = iso
    return [
        ImportCandidate(
            source=SOURCE_CODEX,
            name=os.path.basename(cwd.rstrip("/\\")) or cwd,
            path=cwd,
            session_count=g["count"],
            last_activity=g["latest"],
            sources=[SOURCE_CODEX],
        )
        for cwd, g in grouped.items()
    ]


def _scan_codex_sessions(sessions_dir: str) -> list[ImportCandidate]:
    """Fallback: derive candidates from ``rollout-*.jsonl`` session_meta records."""
    if not os.path.isdir(sessions_dir):
        return []
    grouped: dict[str, dict] = {}
    for root, _dirs, files in os.walk(sessions_dir):
        for fname in files:
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(root, fname)
            cwd = _codex_session_cwd(fpath)
            if not cwd:
                continue
            g = grouped.setdefault(cwd, {"count": 0, "latest": None})
            g["count"] += 1
            iso = _iso(_safe_mtime(fpath))
            if iso and (g["latest"] is None or iso > g["latest"]):
                g["latest"] = iso
    return [
        ImportCandidate(
            source=SOURCE_CODEX,
            name=os.path.basename(cwd.rstrip("/\\")) or cwd,
            path=cwd,
            session_count=g["count"],
            last_activity=g["latest"],
            sources=[SOURCE_CODEX],
        )
        for cwd, g in grouped.items()
    ]


def _codex_session_cwd(path: str) -> str | None:
    """Extract ``cwd`` from a Codex session's first-line ``session_meta`` record.

    Touches only ``payload.cwd`` — never the conversation events that follow.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    if not first:
        return None
    try:
        record = json.loads(first)
    except (ValueError, TypeError):
        return None
    if not isinstance(record, dict):
        return None
    payload = record.get("payload")
    if isinstance(payload, dict):
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return None


# --------------------------------------------------------------------------
# Obsidian — vault registry (obsidian.json) + optional .obsidian marker dirs
# --------------------------------------------------------------------------


def _scan_obsidian(config_path: str) -> list[ImportCandidate]:
    """One candidate per registered Obsidian vault (the vault root IS the
    project; there are no sessions, so ``session_count`` stays 0).

    A vault is confirmed by its ``.obsidian/`` marker directory when present; a
    registry entry without the marker is still surfaced (the folder may exist
    but not yet be open in Obsidian).
    """
    if not os.path.isfile(config_path):
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    vaults = data.get("vaults") if isinstance(data, dict) else None
    if not isinstance(vaults, dict):
        return []

    out: list[ImportCandidate] = []
    for meta in vaults.values():
        if not isinstance(meta, dict):
            continue
        vault_path = meta.get("path")
        if not isinstance(vault_path, str) or not vault_path:
            continue
        ts = meta.get("ts")  # ms since epoch when Obsidian last opened the vault
        last = _iso_from_epoch_ms(ts) if isinstance(ts, (int, float)) else None
        if last is None:
            last = _iso(_safe_mtime(vault_path))
        out.append(
            ImportCandidate(
                source=SOURCE_OBSIDIAN,
                name=os.path.basename(vault_path.rstrip("/\\")) or vault_path,
                path=vault_path,
                session_count=0,
                last_activity=last,
                sources=[SOURCE_OBSIDIAN],
            )
        )
    return out


def _default_obsidian_config_path(home: str) -> str:
    """Platform-specific default location of ``obsidian.json``."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA") or os.path.join(
            home, "AppData", "Roaming"
        )
        return os.path.join(appdata, "obsidian", "obsidian.json")
    if sys.platform == "darwin":
        return os.path.join(
            home, "Library", "Application Support", "obsidian", "obsidian.json"
        )
    # Linux / other: XDG config home.
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(xdg, "obsidian", "obsidian.json")


# --------------------------------------------------------------------------
# Dedup + ranking
# --------------------------------------------------------------------------


def _dedupe(candidates: list[ImportCandidate]) -> list[ImportCandidate]:
    """Collapse candidates that resolve to the same real folder.

    Keyed by ``os.path.realpath`` so a symlink and its target, or two sources
    naming the same directory, merge into one. Session counts sum; the latest
    ``last_activity`` wins; an agent source outranks Obsidian for the merged
    ``source`` label so ranking treats a folder that is *also* an agent project
    as an agent project.
    """
    merged: dict[str, ImportCandidate] = {}
    for cand in candidates:
        # normcase folds case + separators on Windows (identity on POSIX), so
        # C:\Foo and c:/foo collapse to one candidate as they should.
        key = os.path.normcase(os.path.realpath(cand.path))
        existing = merged.get(key)
        if existing is None:
            merged[key] = ImportCandidate(
                source=cand.source,
                name=cand.name,
                path=cand.path,
                session_count=cand.session_count,
                last_activity=cand.last_activity,
                sources=list(cand.sources or [cand.source]),
            )
            continue
        existing.session_count += cand.session_count
        existing.last_activity = _max_iso(existing.last_activity, cand.last_activity)
        for s in cand.sources or [cand.source]:
            if s not in existing.sources:
                existing.sources.append(s)
        # Promote the label to an agent source if either contributor is one, so
        # a vault-that-is-also-a-project ranks in the agent tier.
        if existing.source not in _AGENT_SOURCES and cand.source in _AGENT_SOURCES:
            existing.source = cand.source
    return list(merged.values())


def _rank(candidates: list[ImportCandidate]) -> list[ImportCandidate]:
    """Agent projects above Obsidian vaults; within a tier newest-first."""

    def sort_key(c: ImportCandidate):
        tier = 0 if c.source in _AGENT_SOURCES else 1
        # tier ascending; activity + session_count descending; name as a stable,
        # deterministic final tiebreak.
        return (tier, -_activity_epoch(c.last_activity), -c.session_count, c.name.lower())

    return sorted(candidates, key=sort_key)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _safe(fn, *args) -> list[ImportCandidate]:
    """Run a source scanner, swallowing any error into an empty list."""
    try:
        return fn(*args)
    except Exception:  # pragma: no cover - defensive; discovery never raises
        logger.exception("import scanner source %s failed", getattr(fn, "__name__", fn))
        return []


def _jsonl_files(directory: str) -> list[str]:
    out: list[str] = []
    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.endswith(".jsonl"):
                out.append(entry.path)
    except OSError:
        return []
    return out


def _path_is_live(path: str) -> bool:
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def _safe_mtime(path: str) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _iso_from_epoch(epoch: int | float | None) -> str | None:
    if not isinstance(epoch, (int, float)):
        return None
    return _iso(float(epoch))


def _iso_from_epoch_ms(epoch_ms: int | float | None) -> str | None:
    if not isinstance(epoch_ms, (int, float)):
        return None
    return _iso(float(epoch_ms) / 1000.0)


def _max_iso(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    return a if a >= b else b


def _activity_epoch(iso: str | None) -> float:
    """Parse an ISO-8601 string back to an epoch for sorting (0.0 when absent)."""
    if not iso:
        return 0.0
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0
