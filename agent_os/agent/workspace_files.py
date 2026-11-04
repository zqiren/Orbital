# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workspace file management and session-end routine.

Manages the agent-maintained Layer-1 files in {workspace}/orbital/:
  PROJECT_STATE.md (volatile scratchpad), DECISIONS.md, LESSONS.md (durable),
  INDEX.md (navigation map), plus DECISIONS_ARCHIVE.md / LESSONS_ARCHIVE.md
  (demoted durable entries, read-on-demand).

Provides:
  - WorkspaceFileManager: read/write/append workspace files, build cold resume context
  - prepass / postpass_conservation: the deterministic halves of a tidy pass
    (spec 124 §3.7): normalise + stamp + archive contents tables + INDEX
    archive lines before, id conservation after
  - run_session_end_routine: the Memory session's turn (spec 124 §3.7, see
    memory_editor.py) between those two steps; the size floor only logs

The active model (MiniMax-M3) has a 1,000,000-token window; bounds here exist for
attention and a clean, non-contradictory project identity, NOT context pressure.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from agent_os.agent import memory_entries as _mem
from agent_os.utils.text_io import read_text_lenient, read_text_lenient_detail

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# In-memory idempotency guard for run_session_end_routine. Keyed by session_uuid
# (the JSONL filename stem, Format 2 — see Session docstring). Not persisted
# across daemon restarts — each prior session persists as its own log anyway. GIL makes
# set.add/contains effectively atomic for short operations; we do not need an
# asyncio.Lock unless tests show flakiness under concurrent dispatch.
_completed_session_ends: set[str] = set()

# Persisted "last cleanup" marker for the session-end no-delta gate. Maps
# project_id -> {file_key: mtime_ns} captured at the last successful cleanup, so a
# session-end whose memory files are unchanged since then is a no-op (no LLM call).
# Persisted to orbital/.memory_cleanup.json (the in-memory idempotency set resets
# on daemon restart; this gate must survive restarts).
_CLEANUP_MARKER_FILE = ".memory_cleanup.json"


# Last mtime THIS daemon produced for a given Layer-1 path, recorded at every
# WorkspaceFileManager write. Purely diagnostic: it lets an OCC abort say
# whether the conflicting write came from us or from outside.
#
# Without it every abort read "user mid-edit detected", which sent debugging
# after a human who had done nothing — all four aborts observed in
# orbital-marketing (2026-07-28/29) were a second consolidation pass colliding
# with the first. Bounded by Layer-1 file count per project; last-write-wins is
# sufficient because the comparison is always against the file's CURRENT mtime.
_LAST_DAEMON_WRITE: dict[str, int] = {}


def _history_recording(path: str):
    """Spec 065: wrap a daemon-side Layer-1 write so file history records it
    as Orbital's (exact) under the project's store lock. A plain no-op
    context when history is not running or the path is not tracked."""
    from agent_os.history import ORBITAL_ATTRIB, recording
    return recording(path, ORBITAL_ATTRIB)


def _record_daemon_write(path: str) -> None:
    """Remember the mtime we just produced for ``path`` (best effort)."""
    try:
        _LAST_DAEMON_WRITE[path] = os.stat(path).st_mtime_ns
    except OSError:
        pass


def _stat_mtime_ns(path: str) -> int | None:
    """Return st_mtime_ns for path, or None if the file does not exist.

    Used as the OCC baseline / observation for metadata-file writes. Uses
    nanosecond precision so very fast user edits (within the same coarse
    second) are still detected as conflicts.
    """
    try:
        return os.stat(path).st_mtime_ns
    except FileNotFoundError:
        return None
    except OSError:
        # Permission error or transient I/O failure: treat as "unknown",
        # which forces a conflict on comparison and aborts the write.
        # Safer than silently overwriting a file we cannot stat.
        return None


def _occ_write_metadata(
    workspace_files: "WorkspaceFileManager",
    file_key: str,
    new_content: str,
    baseline_mtime: int | None,
    *,
    project_id: str,
) -> bool:
    """Atomic write of a metadata file gated on the baseline mtime.

    Returns True if the write succeeded, False if it was aborted because
    the file's mtime no longer matches the captured baseline (signalling
    that the user — or some other writer — modified the file between
    when the session-end routine read it for the LLM prompt and now).

    On abort, a structured WARNING is logged with project_id, file path,
    both mtimes, and a cache_thrash_telemetry=True marker so the field
    is greppable in production logs for future analysis.

    The "file did not exist at baseline AND still does not exist now"
    case is treated as a non-conflict — both observed states are None,
    so we proceed with the initial write.
    """
    filepath = workspace_files._file_path(file_key)
    observed_mtime = _stat_mtime_ns(filepath)
    if observed_mtime != baseline_mtime:
        # Who actually wrote? If the file's current mtime is one we produced,
        # this is a second consolidation pass colliding with the first — a bug
        # on our side, not a user edit. Anything else is a genuine outside
        # writer, which is exactly what OCC exists to protect.
        self_collision = (
            observed_mtime is not None
            and _LAST_DAEMON_WRITE.get(filepath) == observed_mtime
        )
        cause = (
            "concurrent consolidation pass (daemon-authored write — NOT a user edit)"
            if self_collision else "user mid-edit detected"
        )
        logger.warning(
            "session_end: OCC abort on %s — %s "
            "(project_id=%s baseline_mtime=%s observed_mtime=%s "
            "self_collision=%s cache_thrash_telemetry=True)",
            file_key, cause, project_id, baseline_mtime, observed_mtime,
            self_collision,
            extra={
                "project_id": project_id,
                "file_path": filepath,
                "baseline_mtime": baseline_mtime,
                "observed_mtime": observed_mtime,
                "self_collision": self_collision,
                "cache_thrash_telemetry": True,
            },
        )
        return False
    workspace_files.write(file_key, new_content)
    return True


def _atomic_replace(src: str, dst: str) -> None:
    """os.replace with retry on Windows (target may be briefly open)."""
    for attempt in range(5):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if not _IS_WINDOWS or attempt == 4:
                raise
            time.sleep(0.05)


FILE_NAMES: dict[str, str] = {
    "state": "PROJECT_STATE.md",
    "decisions": "DECISIONS.md",
    "lessons": "LESSONS.md",
    "index": "INDEX.md",
    # Archives: demoted durable entries, read-on-demand, never injected.
    "decisions_archive": "DECISIONS_ARCHIVE.md",
    "lessons_archive": "LESSONS_ARCHIVE.md",
    "state_archive": "PROJECT_STATE_ARCHIVE.md",
}

# Entry-boundary regexes per sanity-checked file. The session-end prompt
# instructs the LLM to emit entries in these formats; the code-side sanity
# checks (_dedupe_exact + _cap_entries) use these patterns to split the
# LLM output into discrete entries before enforcing dedup and cap. If the
# LLM ignores the format contract (e.g. emits a plain paragraph with no
# markers), the splitter yields <2 entries and the helpers return the
# content unchanged — the caller then writes the LLM output verbatim and
# logs a parse-failure warning.
_LESSONS_ENTRY_PATTERN = r"^\d+\.\s+"      # "1. ", "2. ", ...
_DECISIONS_ENTRY_PATTERN = r"^##\s+"        # "## 2026-04-22: ..."
# CONTEXT.md is no longer entry-based (see _cap_context_tokens) — it became a
# section-based project map in the Layer-1 promotion, so it has no entry
# pattern or entry cap.

# Entry caps per file. Code-side backstop matching the prompt's stated caps.
_LESSONS_CAP = 20
_DECISIONS_CAP = 30


def _split_entries(content: str, entry_pattern: str) -> tuple[str, list[str]]:
    """Split content on entry-marker regex, preserving separator text.

    Returns (preamble, entries) where:
      - preamble is any text before the first entry marker (may be empty)
      - entries is a list of strings, each containing the full entry text
        (marker + body, including any trailing whitespace up to the next
        marker or end of file)

    The split uses a capture group so re.split preserves the marker text
    at split positions. Each entry is reconstructed as marker + body so
    the original byte layout is preserved when the caller rejoins them.
    """
    parts = re.split(f"({entry_pattern})", content, flags=re.MULTILINE)
    # parts = [preamble, marker1, body1, marker2, body2, ...]
    if len(parts) < 3:
        return content, []
    preamble = parts[0]
    entries: list[str] = []
    i = 1
    while i + 1 < len(parts):
        entries.append(parts[i] + parts[i + 1])
        i += 2
    return preamble, entries


def _dedupe_exact(content: str, entry_pattern: str) -> tuple[str, int]:
    """Remove byte-identical duplicate entries, keeping the first occurrence.

    Returns (deduped_content, dropped_count). Identity semantics:

      - If parse fails (splitter yields <2 entries from non-empty content),
        returns (content, 0) — the SAME input string object, unchanged.
      - If no duplicates are found, returns (content, 0) — the SAME input
        string object. The caller's byte-identity check is therefore cheap
        and the main-agent prefix cache is preserved.
      - If duplicates ARE found, returns a new string with dropped entries
        removed and `dropped_count > 0`. Order of remaining entries and
        inter-entry whitespace are preserved.

    "Byte-identical" means equal after stripping trailing whitespace on
    each entry (so `"1. a\\n"` and `"1. a\\n\\n"` are duplicates). The
    first occurrence is kept verbatim with its original trailing bytes.
    """
    if not content.strip():
        return content, 0

    preamble, entries = _split_entries(content, entry_pattern)
    if len(entries) < 2:
        # Parse failure OR only one entry — nothing to dedup.
        return content, 0

    seen: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for entry in entries:
        key = entry.rstrip()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(entry)

    if dropped == 0:
        return content, 0

    return preamble + "".join(kept), dropped


def _cap_entries(
    content: str, entry_pattern: str, cap: int, keep: str = "last",
) -> tuple[str, int]:
    """Enforce a maximum entry count, keeping the first or last N entries.

    Returns (capped_content, dropped_count). Identity semantics mirror
    _dedupe_exact:

      - Parse failure (<2 entries) OR entry_count <= cap → returns
        (content, 0) — the SAME input string object, unchanged.
      - entry_count > cap → returns new string with preamble preserved
        and entries trimmed to the first N (keep="first") or last N
        (keep="last"); inter-entry whitespace inside the kept slice is
        preserved verbatim.

    `keep` must be "first" or "last". "first" keeps entries[:cap];
    "last" keeps entries[-cap:].
    """
    if keep not in ("first", "last"):
        raise ValueError(f"keep must be 'first' or 'last', got {keep!r}")

    if not content.strip():
        return content, 0

    preamble, entries = _split_entries(content, entry_pattern)
    if len(entries) < 2:
        return content, 0
    if len(entries) <= cap:
        return content, 0

    dropped = len(entries) - cap
    if keep == "first":
        kept = entries[:cap]
    else:
        kept = entries[-cap:]

    return preamble + "".join(kept), dropped


def _apply_sanity_checks(
    content: str,
    entry_pattern: str,
    cap: int,
    keep: str,
    filename: str,
) -> str:
    """Run dedup → cap on LLM output. Logs per-file INFO on changes or a
    parse-failure WARNING when the content is non-empty but unparseable.

    Dedup runs first so exact duplicates cannot consume cap slots that
    unique entries could fill. Parse-failure is detected by asking the
    splitter itself: if it yields <2 entries from non-trivial content
    (>1 line after stripping), we treat the pattern as mismatched and
    write the LLM output verbatim. Single-entry content is valid output
    and must not trigger the warning.
    """
    stripped = content.strip()
    if not stripped:
        return content

    # Detect parse failure once, up-front, so dedup and cap no-ops that
    # are legitimate (single-entry content, no dups, under cap) don't
    # emit a spurious warning.
    _, entries = _split_entries(content, entry_pattern)
    if len(entries) < 2 and "\n" in stripped:
        logger.warning(
            "session_end: entry parse failed for %s (pattern=%s), "
            "skipping sanity checks",
            filename, entry_pattern,
        )
        return content

    deduped, dup_count = _dedupe_exact(content, entry_pattern)
    if dup_count:
        logger.info("%s: %d exact duplicates removed", filename, dup_count)

    capped, cap_count = _cap_entries(deduped, entry_pattern, cap, keep=keep)
    if cap_count:
        logger.info("%s: cap enforced, %d dropped", filename, cap_count)

    return capped


# CONTEXT.md is no longer an entry-list (it became a section-based project map
# in the Layer-1 promotion). Entry-based dedup/cap does not apply; instead a
# generous token ceiling backstops a runaway LLM. The prompt targets <1000
# tokens; this caps at 1500 so normal output passes through untouched (and
# preserves the input string object so the prefix cache is not thrashed).
_CONTEXT_TOKEN_CAP = 1500


def _cap_context_tokens(
    content: str, cap_tokens: int = _CONTEXT_TOKEN_CAP, filename: str = "context",
) -> str:
    """Truncate CONTEXT.md at the last complete line under ``cap_tokens``.

    Token estimate mirrors the rest of the codebase: ~4 chars/token. If the
    content is within budget it is returned UNCHANGED (same object). If it
    exceeds the budget, lines are kept until the next line would push the
    estimate over the cap; truncation always lands on a line boundary and a
    warning is logged.
    """
    if not content:
        return content
    char_budget = cap_tokens * 4
    if len(content) <= char_budget:
        return content

    kept: list[str] = []
    used = 0
    for line in content.splitlines():
        # +1 for the newline that rejoins this line.
        if used + len(line) + 1 > char_budget:
            break
        kept.append(line)
        used += len(line) + 1

    truncated = "\n".join(kept)
    logger.warning(
        "%s: exceeded token cap (~%d > %d tokens), truncated to %d lines",
        filename, len(content) // 4, cap_tokens, len(kept),
    )
    return truncated


class WorkspaceFileManager:
    """Reads and writes the 5 workspace files under {workspace}/orbital/."""

    def __init__(self, workspace: str):
        from agent_os.agent.project_paths import ProjectPaths
        self._workspace = workspace
        self._paths = ProjectPaths(workspace)
        self._dir = self._paths.orbital_dir

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def dir(self) -> str:
        return self._dir

    def ensure_dir(self) -> None:
        """Create orbital/ directory."""
        os.makedirs(self._dir, exist_ok=True)

    def _file_path(self, file_key: str) -> str:
        """Return the full path for a workspace file key."""
        _key_to_path = {
            "state": self._paths.project_state,
            "decisions": self._paths.decisions,
            "lessons": self._paths.lessons,
            "index": self._paths.index,
            "decisions_archive": self._paths.decisions_archive,
            "lessons_archive": self._paths.lessons_archive,
            "state_archive": self._paths.project_state_archive,
        }
        return _key_to_path[file_key]

    def read(self, file_key: str) -> str | None:
        """Read a workspace file. Returns None if file doesn't exist.

        file_key is one of: state, decisions, lessons, index,
        decisions_archive, lessons_archive
        """
        if file_key not in FILE_NAMES:
            raise ValueError(f"Unknown file_key: {file_key!r}. Must be one of {list(FILE_NAMES)}")
        filepath = self._file_path(file_key)
        res = read_text_lenient_detail(filepath)
        if res is None:
            return None
        text, fallbacks = res
        if fallbacks:
            self._heal_to_utf8(filepath, text)
        return text

    @staticmethod
    def _heal_to_utf8(filepath: str, text: str) -> None:
        """Spec 141 B: rewrite an Orbital-owned Layer-1 file that needed the
        lenient fallback (e.g. a GBK tail from a Windows shell append) as
        UTF-8, so the next read takes the fast path. Same atomic tmp+replace
        as ``write``, but verbatim — no ``prepare`` normalisation, the bytes
        change only in encoding. Only ``orbital/<FILE_NAMES>`` reach here;
        user-owned files are never rewritten. Best effort: a failed heal
        must not fail the read."""
        tmp_path = filepath + ".tmp"
        try:
            with _history_recording(filepath):
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(text)
                _atomic_replace(tmp_path, filepath)
            _record_daemon_write(filepath)
            logger.info("re-encoded %s as UTF-8", filepath)
        except OSError:
            logger.warning("could not re-encode %s as UTF-8", filepath, exc_info=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def prepare(self, file_key: str, content: str, *, today: str | None = None) -> tuple[str, list[str]]:
        """What ``write`` would put on disk for ``file_key``, without writing.

        Entry files (PROJECT_STATE / DECISIONS / LESSONS) get the format
        header, rule N1–N5 normalisation, ids and dates — the same chokepoint
        the agent's write/edit tools go through. Legacy pointer lines whose
        ids are not in the archive are folded INTO the archive here (archive
        first — spec 124 §3.2 N5), so a caller that then writes the returned
        text keeps the move ordered. Archives get their contents table
        regenerated. Returns ``(text, warnings)``.
        """
        today = today or date.today().isoformat()
        if file_key in _mem.ARCHIVE_KEYS:
            return _mem.with_contents_table(content), []
        if file_key in _mem.FORMAT_HEADERS:
            content = _mem.ensure_format_header(content, file_key)
        if file_key not in _mem.ENTRY_KEYS:
            return content, []
        res = _mem.normalise(content, file_key)
        archive_key = _mem.ARCHIVE_OF[file_key]
        if res.pointers:
            ids_in_archive = _mem.archive_ids(self.read(archive_key))
            unresolved = [
                p for p in res.pointers
                if not (p.ids and all(i in ids_in_archive for i in p.ids))
            ]
            if unresolved:
                _append_archive_section(
                    self, archive_key, f"## [pointers folded {today}]",
                    "\n".join(p.line for p in unresolved),
                )
        content = res.text
        warnings = list(res.warnings)
        if file_key == "state":
            from agent_os.agent import flag_chokepoint
            content, reconcile_warns = flag_chokepoint.reconcile_flags(
                self.read("state"), content, today, None,
            )
            warnings.extend(reconcile_warns)
        else:
            content, stamp_warns = _mem.stamp(
                content, self.read(file_key), file_key, today=today,
                archive_ids=_mem.archive_ids(self.read(archive_key)),
            )
            warnings.extend(stamp_warns)
        return content, warnings

    def write(self, file_key: str, content: str) -> None:
        """Write (overwrite) a workspace file atomically. Creates orbital/ if needed.

        Every daemon-side write of an entry file goes through the same
        chokepoint as the agent tool path (``prepare``): format header,
        normalisation into the one entry shape, ids and dates. Archives get
        their contents table regenerated.
        """
        if file_key not in FILE_NAMES:
            raise ValueError(f"Unknown file_key: {file_key!r}. Must be one of {list(FILE_NAMES)}")
        content, warnings = self.prepare(file_key, content)
        for _w in warnings:
            logger.info("memory write (%s): %s", file_key, _w)
        self.ensure_dir()
        filepath = self._file_path(file_key)
        tmp_path = filepath + ".tmp"
        # Spec 065 D11: the daemon's own Layer-1 writes are versioned with
        # exact attribution (Orbital). No-op when file history is not running.
        with _history_recording(filepath):
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            _atomic_replace(tmp_path, filepath)
        _record_daemon_write(filepath)

    def append(self, file_key: str, content: str) -> None:
        """Append to a workspace file atomically. Creates file if needed."""
        if file_key not in FILE_NAMES:
            raise ValueError(f"Unknown file_key: {file_key!r}. Must be one of {list(FILE_NAMES)}")
        self.ensure_dir()
        filepath = self._file_path(file_key)
        existing = read_text_lenient(filepath) or ""
        tmp_path = filepath + ".tmp"
        with _history_recording(filepath):
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(existing + content)
            _atomic_replace(tmp_path, filepath)
        _record_daemon_write(filepath)

    def read_all(self) -> dict[str, str | None]:
        """Read all 5 files. Returns {key: content_or_None}."""
        return {key: self.read(key) for key in FILE_NAMES}

    def exists(self, file_key: str) -> bool:
        """Check if a workspace file exists."""
        if file_key not in FILE_NAMES:
            raise ValueError(f"Unknown file_key: {file_key!r}. Must be one of {list(FILE_NAMES)}")
        filepath = self._file_path(file_key)
        return os.path.isfile(filepath)

async def run_session_end_routine(
    session,
    provider,
    workspace_files: WorkspaceFileManager,
    utility_provider=None,
    *,
    session_uuid: str,
    bypass_idempotency: bool = False,
    project_id: str = "",
    list_sessions=None,
    force: bool = False,
    run_turn=None,
    trigger: str = "over_budget",
    on_pass_done=None,
) -> str:
    """One memory-tidy pass: the Memory session's turn (spec 124 §3.7)
    between the deterministic prepass and postpass, then the size floor.

    Returns an outcome string the loop surfaces to the agent (hygiene flag +
    status line):

    - ``"edited"`` — the Memory turn changed an entry file;
    - ``"index_only"`` — only INDEX changed;
    - ``"restored"`` — an entry left a live file without reaching its
      archive and was put back from the backup;
    - ``"no_change"`` — the turn ran and changed nothing;
    - ``"skipped"`` — the turn did not run (no runner, or the Memory
      session was busy with a user turn); the files stay dirty;
    - ``"backstop_only"`` — the turn failed (model error, timeout); only
      the floor ran;
    - ``"no_delta"`` — no live file changed since the last pass;
    - ``"not_needed"`` — nothing is over its soft budget and the pass was not
      explicitly requested;
    - ``"in_flight"`` — another pass on this project is running;
    - ``"skipped_idempotent"`` — this session already completed its pass.

    ``provider`` / ``utility_provider`` / ``session`` are kept for the
    callers' signature; the turn runs on the project's own provider inside
    the Memory session, which ``run_turn(task_text)`` (supplied by the agent
    manager) starts. ``force`` (an explicit ``checkpoint_state``) runs the
    pass even when no file is over its soft budget. ``on_pass_done(result)``,
    when given, receives the pass's ``EditorResult`` once the pass has run
    (never for the gates above that skip it) — the agent manager uses it to
    leave the one "Memory tidied" row in the triggering session.
    """
    from agent_os.agent import memory_editor

    del session, provider, utility_provider
    if not bypass_idempotency and session_uuid in _completed_session_ends:
        logger.info("session_end skipped: already completed for %s", session_uuid)
        return "skipped_idempotent"

    # No-delta gate: skip entirely (no LLM) when nothing changed since the last
    # successful pass. Persisted, so it survives daemon restarts.
    if not _has_cleanup_delta(workspace_files):
        logger.info("memory pass: no change since the last pass; skipping (no LLM call)")
        return "no_delta"

    over = [
        k for k in _DELTA_KEYS
        if _mem.over_soft_budget(workspace_files.read(k), k)
    ]
    if not over and not force:
        return "not_needed"

    if not memory_editor.claim(workspace_files.workspace):
        logger.info("memory pass: a pass is already running for %s", project_id or workspace_files.workspace)
        return "in_flight"
    try:
        started = datetime.now(timezone.utc)
        logger.info(
            "memory pass: starting (project=%s over_budget=%s forced=%s trigger=%s runner=%s)",
            project_id, over, force, trigger, "memory session" if run_turn else "none",
        )
        if run_turn is None:
            async def run_turn(_task_text):   # noqa: E306 — no Memory session here
                return None
        result = await memory_editor.run_pass(
            workspace_files, run_turn, list_sessions=list_sessions, project_id=project_id,
            over_keys=over, forced=force, trigger=trigger,
        )
        # The size floor never moves an entry (spec 124 §3.5): it only logs
        # what is still past its hard cap. The existing over-budget trigger
        # stays armed: the next change to a memory file is a delta again.
        _apply_hard_caps(workspace_files)
        if on_pass_done is not None:
            try:
                on_pass_done(result)
            except Exception:  # noqa: BLE001 — a reporting hook never fails the pass
                logger.warning("memory pass: on_pass_done failed", exc_info=True)
        if result.ok and result.outcome != "skipped":
            # Only a pass whose turn actually ran counts: stamping the files
            # "clean" after a failure or a declined turn would disarm the
            # retry.
            memory_editor.write_marker(workspace_files.dir, editor_ran_at=started)
    finally:
        memory_editor.release(workspace_files.workspace)

    if not bypass_idempotency:
        _completed_session_ends.add(session_uuid)
    return result.outcome if result.ok else "backstop_only"


# ---------------------------------------------------------------------------
# Session-end helpers: no-delta gate (persisted) + deterministic hard cap.
# ---------------------------------------------------------------------------

# Layer-1 files whose changes count as an un-consolidated delta.
_DELTA_KEYS = ("state", "decisions", "lessons", "index")


def _cleanup_marker_path(workspace_files: "WorkspaceFileManager") -> str:
    return os.path.join(workspace_files.dir, _CLEANUP_MARKER_FILE)


def _current_layer1_mtimes(workspace_files: "WorkspaceFileManager") -> dict[str, int | None]:
    return {k: _stat_mtime_ns(workspace_files._file_path(k)) for k in _DELTA_KEYS}


def _has_cleanup_delta(workspace_files: "WorkspaceFileManager") -> bool:
    """True if any Layer-1 file changed since the last successful pass.

    Missing marker (never cleaned) => delta. Cheap: stat-only.
    """
    from agent_os.agent import memory_editor
    return memory_editor.has_delta(workspace_files.dir)


def _write_cleanup_marker(workspace_files: "WorkspaceFileManager") -> None:
    """Record post-pass mtimes, keeping any editor watermark already there."""
    from agent_os.agent import memory_editor
    memory_editor.write_marker(workspace_files.dir, editor_ran_at=None)


def _occ_unchanged(
    workspace_files: "WorkspaceFileManager",
    file_key: str,
    baseline_mtime: int | None,
    *,
    project_id: str = "",
) -> bool:
    """True if ``file_key`` is untouched since ``baseline_mtime`` was taken.

    The check half of ``_occ_write_metadata``, for callers that must write
    something else (the archive) between checking and writing the file. The
    caller must not await between this and its writes.
    """
    filepath = workspace_files._file_path(file_key)
    observed_mtime = _stat_mtime_ns(filepath)
    if observed_mtime == baseline_mtime:
        return True
    self_collision = (
        observed_mtime is not None
        and _LAST_DAEMON_WRITE.get(filepath) == observed_mtime
    )
    logger.warning(
        "memory editor: OCC abort on %s — %s changed it while the editor ran; "
        "skipping this file (project_id=%s baseline_mtime=%s observed_mtime=%s "
        "self_collision=%s cache_thrash_telemetry=True)",
        file_key,
        "a daemon write" if self_collision else "an agent or user write",
        project_id, baseline_mtime, observed_mtime, self_collision,
        extra={
            "project_id": project_id,
            "file_path": filepath,
            "baseline_mtime": baseline_mtime,
            "observed_mtime": observed_mtime,
            "self_collision": self_collision,
            "cache_thrash_telemetry": True,
        },
    )
    return False


def _append_archive_section(
    workspace_files: "WorkspaceFileManager", archive_key: str, heading: str, text: str,
) -> None:
    """Append ``heading`` + ``text`` to an archive (created if missing), and
    regenerate its contents table (spec 124 §3.6).

    Raises OSError on failure: every caller writes the archive BEFORE it
    shrinks the live file, so a failure here must stop the move.
    """
    archive_filename = FILE_NAMES[archive_key]
    new_archive = _mem.append_archive_section(
        workspace_files.read(archive_key), archive_filename, heading, text,
    )
    workspace_files.write(archive_key, new_archive)


def refresh_index_archive_lines(workspace_files: "WorkspaceFileManager") -> bool:
    """INDEX gets one daemon-maintained line per existing archive (spec 124
    §3.6): count + date range recomputed, the subject kept (or the
    placeholder until a Memory session describes it). Returns True when
    INDEX changed."""
    archives = {
        FILE_NAMES[k]: workspace_files.read(k) or ""
        for k in _mem.ARCHIVE_KEYS if workspace_files.exists(k)
    }
    if not archives:
        return False
    index = workspace_files.read("index") or ""
    refreshed = _mem.refresh_index_archive_lines(index, archives)
    if refreshed != index:
        workspace_files.write("index", refreshed)
        return True
    return False


# Kept under its old name for the callers that run after an archive write.
_ensure_index_archive_pointer = refresh_index_archive_lines


@dataclass
class PrepassResult:
    touched: list[str] = field(default_factory=list)       # file keys rewritten
    pointer_paths: list[str] = field(default_factory=list)  # named by folded pointers
    warnings: dict[str, list[str]] = field(default_factory=dict)


def prepass(workspace_files: "WorkspaceFileManager", *, today: str | None = None) -> PrepassResult:
    """The deterministic step before a tidy pass (spec 124 §3.7): normalise
    (N1–N5, legacy pointers folded into the archive), stamp ids, give every
    existing archive its contents table, refresh the INDEX archive lines.

    Idempotent: a conforming project is left byte-identical. Returns what
    was rewritten and the paths the folded pointers named (the Memory
    session's task message lists them; nothing else consumes them yet).
    """
    today = today or date.today().isoformat()
    result = PrepassResult()
    for key in _mem.ENTRY_KEYS:
        content = workspace_files.read(key)
        if content is None:
            continue
        res = _mem.normalise(content, key)
        for path in res.pointer_paths:
            if path not in result.pointer_paths:
                result.pointer_paths.append(path)
        prepared, warnings = workspace_files.prepare(key, content, today=today)
        if prepared != content:
            workspace_files.write(key, prepared)
            result.touched.append(key)
            if warnings:
                result.warnings[key] = warnings
    for archive_key in _mem.ARCHIVE_KEYS:
        archive = workspace_files.read(archive_key)
        if archive is None:
            continue
        tabled = _mem.with_contents_table(archive)
        if tabled != archive:
            workspace_files.write(archive_key, tabled)
            result.touched.append(archive_key)
    if refresh_index_archive_lines(workspace_files):
        result.touched.append("index")
    if result.touched:
        logger.info(
            "memory prepass: rewrote %s%s", ", ".join(result.touched),
            f"; folded pointers named {len(result.pointer_paths)} path(s)" if result.pointer_paths else "",
        )
    return result


def postpass_conservation(
    workspace_files: "WorkspaceFileManager", snapshot: dict[str, str], keys=None,
) -> list[str]:
    """Id conservation after a pass (spec 124 §3.5): every id ``snapshot``
    had must still be in the live file or in its archive, else the entry
    comes back verbatim. Returns the restored ids (logged as ``restored=``)."""
    restored_all: list[str] = []
    for key in (keys if keys is not None else _mem.ENTRY_KEYS):
        before = snapshot.get(key)
        if not before:
            continue
        current = workspace_files.read(key) or ""
        in_archive = _mem.archive_ids(workspace_files.read(_mem.ARCHIVE_OF[key]))
        text, restored = _mem.restore_missing(before, current, in_archive)
        if restored:
            workspace_files.write(key, text)
            logger.warning(
                "memory pass: %s lost %d entr%s that never reached the archive; "
                "restored=%s", FILE_NAMES[key], len(restored),
                "y" if len(restored) == 1 else "ies", restored,
            )
            restored_all.extend(restored)
    return restored_all


def _apply_hard_caps(workspace_files: "WorkspaceFileManager") -> list[str]:
    """The size floor (spec 124 §3.5): never moves an entry, never an LLM call.

    Archiving is a judgement ("superseded or finished"), never a matter of
    age or size, so the daemon no longer demotes anything. The entry files
    are only measured here: a file past its hard cap is logged and reported;
    the injected view already folds it by section, and the existing
    over-budget trigger runs the next pass once a memory file changes again.
    INDEX keeps its tail trim (no entries, no archive).
    Returns the file keys still past their hard cap.
    """
    over: list[str] = []
    for key in _mem.ENTRY_KEYS:
        content = workspace_files.read(key)
        if not content or not _mem.over_hard_budget(content, key):
            continue
        over.append(key)
        logger.info(
            "size floor: %s is ~%d tok over its hard cap (%d); leaving it — the "
            "injected view folds it by section and the next tidy pass is armed",
            FILE_NAMES[key],
            int(_mem.est_tokens(_mem.strip_format_header(content)) - _mem.FILE_BUDGETS[key]["hard"]),
            _mem.FILE_BUDGETS[key]["hard"],
        )

    index = workspace_files.read("index")
    if index and _mem.over_soft_budget(index, "index"):
        target = _mem.consolidation_target("index")
        trimmed = _mem.trim_volatile(index, target)
        if trimmed != index:
            workspace_files.write("index", trimmed)
            logger.info("size backstop: trimmed INDEX to <= %d tok", target)
    return over
