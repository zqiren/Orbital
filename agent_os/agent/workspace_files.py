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
  - run_session_end_routine: the memory editor (choices by id, see
    memory_editor.py) followed by the deterministic, age-based size floor

The active model (MiniMax-M3) has a 1,000,000-token window; bounds here exist for
attention and a clean, non-contradictory project identity, NOT context pressure.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from datetime import date, datetime, timezone

from agent_os.agent import memory_entries as _mem

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

# Layer-1 files injected every turn (excludes archives). Order for cold resume.
_RESUME_ORDER = ["state", "decisions", "lessons", "index"]

# Section headers for cold resume context
_SECTION_HEADERS: dict[str, str] = {
    "state": "Project State",
    "decisions": "Decisions",
    "lessons": "Lessons Learned",
    "index": "Project Index (navigation map)",
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
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    def write(self, file_key: str, content: str) -> None:
        """Write (overwrite) a workspace file atomically. Creates orbital/ if needed."""
        if file_key not in FILE_NAMES:
            raise ValueError(f"Unknown file_key: {file_key!r}. Must be one of {list(FILE_NAMES)}")
        # Self-heal the in-file format contract on every system write (the
        # agent-tool write path does the same in process_on_write). Archives
        # keep their own header convention and are skipped.
        if file_key in _mem.FORMAT_HEADERS:
            content = _mem.ensure_format_header(content, file_key)
        if file_key == "state":
            # Daemon-side PROJECT_STATE writes (memory editor, size floor)
            # run through the same chokepoint as the agent tool path so every
            # bullet keeps (or gets) its id and dates.
            from agent_os.agent import flag_chokepoint
            prev = self.read("state")
            content, _reconcile_warns = flag_chokepoint.reconcile_flags(
                prev, content, date.today().isoformat(), None
            )
            for _w in _reconcile_warns:
                logger.info("reconcile_flags(state): %s", _w)
        self.ensure_dir()
        filepath = self._file_path(file_key)
        tmp_path = filepath + ".tmp"
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
        existing = ""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = f.read()
        except OSError:
            pass
        tmp_path = filepath + ".tmp"
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

    def build_cold_resume_context(self) -> str:
        """Assemble cold resume context from the Layer-1 files.

        Standard Layer-1 read (PROJECT_STATE, DECISIONS, LESSONS, INDEX) with no
        special-casing. SESSION_LOG was retired; cross-session history is no
        longer kept separately (the Layer-1 files are injected every turn anyway).

        Returns assembled markdown string with section headers.
        """
        sections: list[str] = []
        for key in _RESUME_ORDER:
            content = self.read(key)
            if content is None:
                continue
            header = _SECTION_HEADERS[key]
            sections.append(f"## {header}\n\n{content.strip()}")
        return "\n\n---\n\n".join(sections)


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
) -> str:
    """One memory-tidy pass: the editor (by id), then the deterministic floor.

    Returns an outcome string the loop surfaces to the agent (hygiene flag +
    status line):

    - ``"edited"`` — the editor ran and applied at least one choice;
    - ``"no_change"`` — the editor ran and chose nothing (or only choices that
      could not be applied);
    - ``"backstop_only"`` — the editor could not run (model error, timeout,
      no usable JSON); only the floor ran;
    - ``"no_delta"`` — no live file changed since the last pass;
    - ``"not_needed"`` — nothing is over its soft budget and the pass was not
      explicitly requested;
    - ``"in_flight"`` — another pass on this project is running;
    - ``"skipped_idempotent"`` — this session already completed its pass.

    The whole-file LLM merge this used to run is gone (spec 089 §4.2): it
    regenerated PROJECT_STATE / DECISIONS / LESSONS from a size-capped view and
    so silently dropped whatever it never saw. ``session`` is kept for the
    callers' signature; the editor reads sessions itself, through read-only
    tools, from the time of its last successful run.

    ``force`` (an explicit ``checkpoint_state``) runs the editor even when no
    file is over its soft budget.
    """
    from agent_os.agent import memory_editor

    del session
    if not bypass_idempotency and session_uuid in _completed_session_ends:
        logger.info("session_end skipped: already completed for %s", session_uuid)
        return "skipped_idempotent"

    # No-delta gate: skip entirely (no LLM) when nothing changed since the last
    # successful pass. Persisted, so it survives daemon restarts.
    if not _has_cleanup_delta(workspace_files):
        logger.info("memory editor: no change since the last pass; skipping (no LLM call)")
        return "no_delta"

    over = [
        k for k in _DELTA_KEYS
        if _mem.over_soft_budget(workspace_files.read(k), k)
    ]
    if not over and not force:
        return "not_needed"

    if not memory_editor.claim(workspace_files.workspace):
        logger.info("memory editor: a pass is already running for %s", project_id or workspace_files.workspace)
        return "in_flight"
    try:
        started = datetime.now(timezone.utc)
        llm = utility_provider if utility_provider is not None else provider
        logger.info(
            "memory editor: starting (project=%s over_budget=%s forced=%s model=%s)",
            project_id, over, force, getattr(llm, "model", "?"),
        )
        result = await memory_editor.run_editor(
            workspace_files, llm, list_sessions=list_sessions, project_id=project_id,
        )
        # The deterministic floor ALWAYS runs, editor or not; never an LLM call.
        _apply_hard_caps(workspace_files)
        if result.ok and not result.skipped_files:
            # Only a pass whose editor ran — over every file it chose to edit
            # — counts: stamping the files "clean" after a failure, or after
            # a file was skipped because someone wrote it mid-run, would
            # disarm the retry.
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
    """Append ``heading`` + ``text`` to an archive (created if missing).

    Raises OSError on failure: every caller writes the archive BEFORE it
    shrinks the live file, so a failure here must stop the move.
    """
    archive_filename = FILE_NAMES[archive_key]
    section = f"{heading}\n\n{text.strip(chr(10))}\n"
    existing = workspace_files.read(archive_key) or ""
    if existing.strip():
        new_archive = existing.rstrip() + "\n\n" + section
    else:
        new_archive = f"# {archive_filename} (archived entries, read-on-demand)\n\n" + section
    workspace_files.write(archive_key, new_archive)


def _ensure_index_archive_pointer(workspace_files: "WorkspaceFileManager", archive_filename: str) -> None:
    """Make sure INDEX points to an archive file after a demotion."""
    index = workspace_files.read("index") or ""
    if archive_filename in index:
        return
    pointer = f"\n- {archive_filename} — older entries demoted from the live file (read on demand).\n"
    workspace_files.write("index", (index.rstrip() + "\n" + pointer) if index.strip() else f"# INDEX\n{pointer}")


def _apply_hard_caps(workspace_files: "WorkspaceFileManager") -> None:
    """Deterministic size floor. Moves, never deletes, never an LLM call.

    Fires only on a file over its SOFT budget (the same measure as the
    hygiene flag), cutting toward ``consolidation_target``. Every move leaves
    an ``[archived DATE id:X] … → ARCHIVE`` pointer and writes the archive
    before the live file shrinks.

    - DECISIONS/LESSONS: stamp ids on any unstamped entry (so an unstamped
      NEW entry is not mistaken for the coldest one), then demote the
      coldest-``touched`` entries (oldest 3 and ``pinned`` are protected).
    - PROJECT_STATE: age-based — only lines older than
      ``STATE_PROBATION_DAYS``, oldest first; old pointers simply expire. When
      what is left is all recent, the file stays over budget and that is
      logged: over budget beats losing current work.
    - INDEX: tail trim; archive-pointer lines are pinned.
    """
    from agent_os.agent import memory_editor

    today = date.today().isoformat()
    for key in _mem.DURABLE_KEYS:
        content = workspace_files.read(key)
        if not content or not _mem.over_soft_budget(content, key):
            continue
        memory_editor.normalise_ids(workspace_files, today=today)
        content = workspace_files.read(key) or ""
        archive_key = _mem.ARCHIVE_OF[key]
        archive_filename = FILE_NAMES[archive_key]
        kept, demoted, ids = _mem.demote_with_pointers(
            content, key, _mem.consolidation_target(key), today, archive_filename,
        )
        if not demoted:
            continue
        try:
            _append_archive_section(workspace_files, archive_key, f"## [archived {today}]", demoted)
            workspace_files.write(key, kept)
            _ensure_index_archive_pointer(workspace_files, archive_filename)
        except OSError as e:
            logger.warning(
                "size backstop: could not demote %s to %s (%s); leaving the "
                "live file intact — over budget beats losing entries",
                key, archive_key, e,
            )
            continue
        logger.info(
            "size backstop: demoted %d %s entries to %s by id: %s",
            len(ids), key, archive_key, ", ".join(ids),
        )

    state = workspace_files.read("state")
    if state and _mem.over_soft_budget(state, "state"):
        memory_editor.normalise_ids(workspace_files, today=today)
        state = workspace_files.read("state") or ""
        archive_filename = FILE_NAMES["state_archive"]
        floor = _mem.floor_state(state, _mem.consolidation_target("state"), today, archive_filename)
        if floor.archived or floor.dropped_pointers:
            try:
                if floor.archived:
                    _append_archive_section(
                        workspace_files, "state_archive", f"## [trimmed {today}]", floor.archived)
                workspace_files.write("state", floor.content)
                if floor.archived:
                    _ensure_index_archive_pointer(workspace_files, archive_filename)
                logger.info(
                    "size backstop: PROJECT_STATE moved %d line(s) to %s by id (%s), "
                    "removed %d expired pointer(s)",
                    len(floor.moved_ids), archive_filename, ", ".join(floor.moved_ids),
                    floor.dropped_pointers,
                )
            except OSError as e:
                logger.warning(
                    "size backstop: could not archive PROJECT_STATE lines (%s); "
                    "leaving the live file intact", e,
                )
        if floor.over_by:
            logger.info(
                "size backstop: PROJECT_STATE still over target by %d tok — every "
                "remaining line is younger than %d days or not a bullet; leaving "
                "it (over budget beats losing current work)",
                floor.over_by, _mem.STATE_PROBATION_DAYS,
            )

    index = workspace_files.read("index")
    if index and _mem.over_soft_budget(index, "index"):
        target = _mem.consolidation_target("index")
        trimmed = _mem.trim_volatile(index, target)
        if trimmed != index:
            workspace_files.write("index", trimmed)
            logger.info("size backstop: trimmed INDEX to <= %d tok", target)
