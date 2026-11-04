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
* **Folder rules (spec 138 pass 1).** A session folder inside a git repo
  folds onto the repo's main root (worktrees included); ``~``, ``~/Desktop``,
  ``~/Documents``, ``~/Downloads``, ``/`` and temp roots are never offered. A
  folder that is not a git repo, has no ``AGENTS.md`` / ``CLAUDE.md`` and
  holds fewer than 3 sessions is ``minor`` (offered unchecked).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_os.onboarding.history_paths import HistoryEnv, codex_dir as _codex_dir, history_home

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
    # Spec 084 §3.2: True when the folder is already a project's workspace, so
    # the wizard shows it as imported instead of offering an Add that can only
    # 409 against the project it created itself.
    already_imported: bool = False
    # Internal: every source that contributed to this (post-dedup) candidate.
    # Not part of the public {source,name,path,session_count,last_activity}
    # payload, but handy for debugging/telemetry.
    sources: list[str] = field(default_factory=list)
    # Spec 138: sessions per agent, and the pass-1 folder facts.
    by_agent: dict[str, int] = field(default_factory=dict)
    is_git: bool = False
    has_agent_md: bool = False
    minor: bool = False
    # Spec 153: private notes per agent kept about this folder.
    notes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "name": self.name,
            "path": self.path,
            "session_count": self.session_count,
            "last_activity": self.last_activity,
            "already_imported": self.already_imported,
            "kind": "folder",
            "minor": self.minor,
            "is_git": self.is_git,
            "by_agent": dict(self.by_agent),
            "notes": dict(self.notes),
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
    existing_workspaces: Iterable[str] = (),
    env: HistoryEnv | None = None,
) -> list[ImportCandidate]:
    """Scan all v1 sources and return ranked, deduped, path-verified candidates.

    All source locations default to the standard per-user paths derived from
    ``home`` (default ``~``), but every one is injectable so tests can point at
    fixture home directories. Any source that is missing or unreadable is
    skipped silently — discovery never raises.

    ``existing_workspaces`` (spec 084 §3.2) are the workspaces of projects that
    already exist; a candidate resolving to one of them is returned with
    ``already_imported=True`` rather than dropped, so the wizard can show it
    honestly. Matching uses the same realpath/normcase key as ``_dedupe``, so a
    symlinked import still matches its real folder.

    ``home`` defaults to ``$ORBITAL_HISTORY_HOME`` or ``~``; ``env`` (the
    container / temp folders of pass 1) defaults to the one for that home.
    """
    home = str(home) if home else history_home()
    env = env or HistoryEnv.for_home(home)

    if claude_projects_dir is None:
        claude_projects_dir = os.path.join(home, ".claude", "projects")
    if codex_dir is None:
        codex_dir = _codex_dir(home)
    if obsidian_config_path is None:
        obsidian_config_path = _default_obsidian_config_path(home)

    raw: list[ImportCandidate] = []
    raw.extend(_safe(_scan_claude_code, claude_projects_dir))
    raw.extend(_safe(_scan_codex, codex_dir))
    raw.extend(_safe(_scan_obsidian, obsidian_config_path))

    verified = [c for c in raw if _path_is_live(c.path)]
    folded = [c for c in (_fold_to_repo(c, env) for c in verified) if c is not None]
    deduped = _dedupe(folded)
    for cand in deduped:
        _annotate(cand, env)
    imported_keys = {_path_key(w) for w in existing_workspaces if w}
    if imported_keys:
        for cand in deduped:
            cand.already_imported = _path_key(cand.path) in imported_keys
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
                by_agent={SOURCE_CLAUDE_CODE: len(sessions)},
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


def claude_project_dirs_for(projects_dir: str, workspace: str) -> list[str]:
    """Claude Code project dirs whose sessions ran in ``workspace``.

    The cwd-probe fallback for session import (keyless onboarding, 2026-10-21):
    the per-cwd dir name is a lossy encoding of the path, so when the derived
    slug dir is missing, each project dir's newest transcript is probed for its
    recorded ``cwd`` (the same whitelisted-key read the scanner uses) and the
    dirs whose cwd resolves to ``workspace`` are returned. Matching uses
    :func:`path_key`. Never raises; read-only.
    """
    out: list[str] = []
    if not os.path.isdir(projects_dir):
        return out
    want = path_key(workspace)
    try:
        entries = list(os.scandir(projects_dir))
    except OSError:
        return out
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        sessions = _jsonl_files(entry.path)
        if not sessions:
            continue
        sessions.sort(key=lambda p: _safe_mtime(p) or 0.0, reverse=True)
        for session in sessions[:3]:
            cwd = _claude_session_cwd(session)
            if cwd:
                if path_key(cwd) == want:
                    out.append(entry.path)
                break
    return out


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
                rows.append((strip_extended_prefix(cwd), ts))
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
            by_agent={SOURCE_CODEX: g["count"]},
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
            cwd = strip_extended_prefix(_codex_session_cwd(fpath))
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
            by_agent={SOURCE_CODEX: g["count"]},
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
        key = _path_key(cand.path)
        existing = merged.get(key)
        if existing is None:
            merged[key] = ImportCandidate(
                source=cand.source,
                name=cand.name,
                path=cand.path,
                session_count=cand.session_count,
                last_activity=cand.last_activity,
                sources=list(cand.sources or [cand.source]),
                by_agent=dict(cand.by_agent),
            )
            continue
        existing.session_count += cand.session_count
        for agent, n in cand.by_agent.items():
            existing.by_agent[agent] = existing.by_agent.get(agent, 0) + n
        existing.last_activity = _max_iso(existing.last_activity, cand.last_activity)
        for s in cand.sources or [cand.source]:
            if s not in existing.sources:
                existing.sources.append(s)
        # Promote the label to an agent source if either contributor is one, so
        # a vault-that-is-also-a-project ranks in the agent tier.
        if existing.source not in _AGENT_SOURCES and cand.source in _AGENT_SOURCES:
            existing.source = cand.source
    return list(merged.values())


_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"


def strip_extended_prefix(path: str) -> str:
    """Drop the Windows extended-length prefix Codex writes into its index.

    Codex (Rust ``canonicalize``) records thread working directories as
    ``\\\\?\\D:\\proj`` / ``\\\\?\\UNC\\server\\share``. Python's ``realpath``
    and ``isdir`` do not treat that form as the same folder as ``D:\\proj``
    (2026-10-22: the import list showed ``\\\\?\\`` paths, stored them as
    project workspaces verbatim and listed the same folder twice), so every
    Codex-sourced cwd goes through here before it is compared or shown.
    """
    if not isinstance(path, str):
        return path
    if path.startswith(_EXTENDED_UNC_PREFIX):
        return "\\\\" + path[len(_EXTENDED_UNC_PREFIX):]
    if path.startswith(_EXTENDED_PREFIX):
        return path[len(_EXTENDED_PREFIX):]
    return path


def _fold_to_repo(cand: ImportCandidate, env: HistoryEnv) -> ImportCandidate | None:
    """Pass 1: a folder inside a git repo becomes the repo's main root (so a
    worktree folds onto its repo); container and temp folders are dropped."""
    from agent_os.onboarding.history import git_main_root

    path = os.path.normpath(cand.path)
    if cand.source in _AGENT_SOURCES:
        root = git_main_root(path)
        if root and root != path and not env.is_container(root) and not env.is_temp(root) \
                and _path_is_live(root):
            cand.path = root
            cand.name = os.path.basename(root.rstrip("/\\")) or root
            path = root
    if env.is_container(path) or env.is_temp(path):
        return None
    if env.is_chat_folder(path) and git_main_root(path) is None:
        return None  # a Codex app chat folder, not a project
    return cand


def _annotate(cand: ImportCandidate, env: HistoryEnv) -> None:
    from agent_os.onboarding.history import MIN_FOLDER_SESSIONS, git_main_root
    from agent_os.onboarding.history_paths import has_project_instructions

    root = git_main_root(os.path.normpath(cand.path))
    # A repo rooted at a container (a dotfiles repo in ~) says nothing.
    cand.is_git = root is not None and not env.is_container(root)
    cand.has_agent_md = has_project_instructions(cand.path)
    cand.minor = (cand.source in _AGENT_SOURCES and not cand.is_git
                  and not cand.has_agent_md and cand.session_count < MIN_FOLDER_SESSIONS)


def _path_key(path: str) -> str:
    """Identity key for a folder: realpath, then normcase.

    normcase folds case + separators on Windows (identity on POSIX), so
    ``C:\\Foo`` and ``c:/foo`` collapse to one candidate as they should. The
    same key decides dedup and the already-imported match (spec 084) so the
    two can never disagree about which folder is which.
    """
    return os.path.normcase(os.path.realpath(strip_extended_prefix(path)))


def path_key(path: str) -> str:
    """Public alias of :func:`_path_key` for the session importer, so the two
    modules agree on which folder is which."""
    return _path_key(path)


def _rank(candidates: list[ImportCandidate]) -> list[ImportCandidate]:
    """Agent projects above Obsidian vaults; within a tier newest-first."""

    def sort_key(c: ImportCandidate):
        tier = 1 if c.source == SOURCE_OBSIDIAN else 0
        # tier ascending; activity + session_count descending; name as a stable,
        # deterministic final tiebreak.
        return (tier, -_activity_epoch(c.last_activity), -c.session_count, c.name.lower())

    return sorted(candidates, key=sort_key)


def candidates_from_analysis(analysis, *, vaults: Iterable[ImportCandidate] = (),
                             existing_workspaces: Iterable[str] = (),
                             notes: Iterable = ()) -> list[ImportCandidate]:
    """Spec 138 step 2 rows once the history scan is done: one per pass-1
    folder of ``analysis``, plus the Obsidian ``vaults`` (which have no
    history). Unlike the folder scan above, the analysis has dropped the
    sessions Orbital itself ran as a worker, automation runs and probes, and
    counts the sessions it attached to a folder by the files they touched, so
    the rows agree with the groups."""
    out: list[ImportCandidate] = []
    for root, project in analysis.projects.items():
        by_agent = Counter(s.agent for s in project.sessions)
        if not by_agent:
            continue
        ended = [s.ended_at for s in project.sessions if s.ended_at]
        out.append(ImportCandidate(
            source=by_agent.most_common(1)[0][0],
            name=os.path.basename(root.rstrip("/\\")) or root,
            path=root,
            session_count=len(project.sessions),
            last_activity=max(ended) if ended else None,
            sources=sorted(by_agent),
            by_agent=dict(by_agent),
            is_git=project.is_git,
            has_agent_md=project.has_agent_md,
            minor=project.minor,
        ))
    # Folder roots are unique already; a vault that is also a folder row
    # stays the folder row.
    seen = {_path_key(c.path) for c in out}
    out.extend(v for v in vaults
               if v.source == SOURCE_OBSIDIAN and _path_key(v.path) not in seen)
    imported_keys = {_path_key(w) for w in existing_workspaces if w}
    for cand in out:
        cand.already_imported = _path_key(cand.path) in imported_keys
    # Spec 153: the scanner's notes list, matched on the folder they were
    # kept in (a note with no known folder is never attached to a row).
    by_folder: dict[str, Counter] = {}
    for note in notes:
        if getattr(note, "folder", None):
            by_folder.setdefault(_path_key(note.folder), Counter())[note.agent] += 1
    for cand in out:
        counts = by_folder.get(_path_key(cand.path))
        if counts:
            cand.notes = dict(counts)
    return _rank(out)


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
