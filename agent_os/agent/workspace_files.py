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
  - run_session_end_routine: deterministic size backstop + best-effort LLM dedup/merge

The active model (MiniMax-M3) has a 1,000,000-token window; bounds here exist for
attention and a clean, non-contradictory project identity, NOT context pressure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import date

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
        logger.warning(
            "session_end: OCC abort on %s — user mid-edit detected "
            "(project_id=%s baseline_mtime=%s observed_mtime=%s "
            "cache_thrash_telemetry=True)",
            file_key, project_id, baseline_mtime, observed_mtime,
            extra={
                "project_id": project_id,
                "file_path": filepath,
                "baseline_mtime": baseline_mtime,
                "observed_mtime": observed_mtime,
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
        self.ensure_dir()
        filepath = self._file_path(file_key)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        _atomic_replace(tmp_path, filepath)

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

    def build_session_end_prompt(self, session_summary: dict) -> str:
        """Build the LLM prompt for the session-end dedup/cleanup pass.

        This pass is the CLEANUP, not the state record — state is recorded
        incrementally during the session. Its job is to merge duplicates,
        supersede stale entries, and keep one clean, non-contradictory identity.

        It is fed the BOUNDED view of each file (<= injection budget) so a
        bloated project cannot make the call time out (the prior failure: ~26k
        tokens of input -> 3x timeout -> no cleanup at all).

        Returns a prompt asking for JSON: project_state, decisions, lessons, index.
        """
        from agent_os.agent import memory_entries as _mem

        def _bounded(file_key: str) -> str:
            raw = self.read(file_key)
            if not raw or not raw.strip():
                return f"(no existing {file_key})"
            view = _mem.inject_view(raw, file_key, _mem.FILE_BUDGETS[file_key]["hard"])
            return view or f"(no existing {file_key})"

        existing_state = _bounded("state")
        existing_decisions = _bounded("decisions")
        existing_lessons = _bounded("lessons")
        existing_index = _bounded("index")

        # Format recent messages
        recent_lines: list[str] = []
        for msg in session_summary.get("recent_messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            recent_lines.append(f"[{role}] {content}")
        recent_formatted = "\n".join(recent_lines) if recent_lines else "(no messages)"

        files_modified = ", ".join(session_summary.get("files_modified", [])) or "(none)"
        message_count = session_summary.get("message_count", 0)
        tool_calls_count = session_summary.get("tool_calls_count", 0)

        # Token targets for the archive instruction (field 5), read from the
        # shared budgets so the numbers never drift from the enforced caps.
        today = date.today().isoformat()
        dec_budget = _mem.FILE_BUDGETS["decisions"]["soft"]
        les_budget = _mem.FILE_BUDGETS["lessons"]["soft"]

        # Ride-along shape tidy: report-only lint on the volatile files. This
        # is the ONLY consumer of shape_report — formatting is fixed inside a
        # pass that runs anyway, never via a new hygiene flag or checkpoint.
        reports = []
        for fk in ("state", "index"):
            r = _mem.shape_report(self.read(fk), fk)
            if r:
                reports.append(f"- {r}\n  Contract: {_mem.FORMAT_HEADERS[fk]}")
        formatting_block = (
            "\nFORMATTING TO FIX (restore each file to its contract in this same pass):\n"
            + "\n".join(reports) + "\n"
        ) if reports else ""

        prompt = f"""You maintain the Layer-1 memory of an AI agent project. Your job THIS PASS is to CLEAN UP: merge duplicates, supersede stale entries, and keep ONE clean, non-contradictory project identity. State is already recorded incrementally during the session — you are the editor, not the primary record.

Produce a JSON object with these fields. Return "" for any field that needs no change (preserve the existing file unchanged).

1. "project_state" (string): The COMPLETE current-state snapshot — what is true NOW: current focus, in-progress work, blockers, next steps. OVERWRITE intent — this is a living scratchpad, NOT a changelog. Replace stale status with current status; do not append a dated history.

2. "decisions" (string): The COMPLETE updated DECISIONS.md. MERGE AND SUPERSEDE — never append a contradiction. When a new decision changes an old one, REPLACE the old entry or mark it superseded; do not leave both. Keep currently-true decisions; drop genuinely obsolete ones. Each entry:
     ## YYYY-MM-DD: Title
     **Chose:** ...
     **Reason:** ...
     **Rejected:** ...
   To retire an entry, append " (superseded)" to its title or drop it.

3. "lessons" (string): The COMPLETE updated LESSONS.md (numbered entries). Merge near-duplicate lessons into one. KEEP detailed technical playbooks intact — do NOT shorten a real lesson to save space. Drop only genuinely obsolete lessons.

4. "index" (string): The COMPLETE updated INDEX.md — NAVIGATION ONLY. A short map: important files/dirs, ONE sentence each ("path — what it is"). NOT a place for decisions, status, or lessons (those live in their own files). If older entries were archived, point to DECISIONS_ARCHIVE.md / LESSONS_ARCHIVE.md.

5. "archive" (OPTIONAL object — omit unless needed): DECISIONS targets ~{dec_budget} tokens and LESSONS ~{les_budget} tokens (~4 chars/token). If either still exceeds its target after you have merged and deduplicated, move its COMPLETED or DORMANT entries — NEVER an entry tagged `pinned` — OUT of the kept file and return them here VERBATIM under the key "decisions" and/or "lessons" (include only the file(s) you trimmed). For each block you move, leave ONE pointer line in the kept file noting what left and when it is worth re-reading, e.g. `[archived {today}] v0.6 release/signing decisions → DECISIONS_ARCHIVE.md — read before installer work`. Archiving is a last resort — prefer merging and superseding.

RULES:
- A non-empty field MUST be the COMPLETE updated file. "" preserves the existing file.
- MERGE AND SUPERSEDE — never append a contradicting entry.
- If you correct a fact in one file, check the other three for stale copies of it and fix them in the same pass.
- Respond with ONLY valid JSON. No markdown fences. No explanation.
{formatting_block}
--- EXISTING FILES (bounded view; full files are on disk) ---
PROJECT_STATE.md:
{existing_state}

DECISIONS.md:
{existing_decisions}

LESSONS.md:
{existing_lessons}

INDEX.md:
{existing_index}

--- THIS SESSION ({message_count} messages, {tool_calls_count} tool calls) ---
Files modified: {files_modified}
Recent conversation:
{recent_formatted}"""

        return prompt


async def run_session_end_routine(
    session,
    provider,
    workspace_files: WorkspaceFileManager,
    utility_provider=None,
    *,
    session_uuid: str,
    bypass_idempotency: bool = False,
    project_id: str = "",
) -> str:
    """Session-end cleanup: deterministic size backstop + best-effort LLM merge.

    Returns an outcome string the loop surfaces to the agent (hygiene flag +
    status line): "llm_merged" (merge applied), "backstop_only" (LLM failed or
    timed out; only the deterministic hard cap ran), "no_delta" (nothing to do),
    "skipped_idempotent" (this session already completed its boundary pass).

    Repositioned as the CLEANUP, not the state record — state is recorded
    incrementally during the session (overwrite STATE; stamped append for
    DECISIONS/LESSONS via the shared persist helper). So even if the LLM pass is
    skipped or times out, the deterministic hard cap (demote-to-archive for
    durable files, trim for volatile) still runs and never deletes durable
    content.

    Two gates keep it cheap. (1) An idempotency guard per ``session_uuid`` for
    the boundary callers. (2) A persisted NO-DELTA gate: if no Layer-1 file
    changed since the last successful cleanup, this is a no-op with NO LLM call
    — this is what stops the ``agent_decided`` trigger from firing redundantly
    (the 78x problem). The trigger set is otherwise unchanged.

    OCC (Optimistic Concurrency Control): the LLM-merge writes to PROJECT_STATE,
    DECISIONS, LESSONS, INDEX are each gated on a baseline mtime captured before
    the LLM runs, so a concurrent user edit aborts that file's write only.
    """
    # Idempotency guard: short-circuit if this session already completed.
    if not bypass_idempotency and session_uuid in _completed_session_ends:
        logger.info("session_end skipped: already completed for %s", session_uuid)
        return "skipped_idempotent"

    # No-delta gate: skip entirely (no LLM) when nothing changed since the last
    # successful cleanup. Persisted, so it survives daemon restarts.
    if not _has_cleanup_delta(workspace_files):
        logger.info(
            "session_end: no un-consolidated delta since last cleanup; "
            "skipping (no LLM call)"
        )
        return "no_delta"

    today = date.today().isoformat()

    # ---- Part A: best-effort LLM dedup/merge (bounded input; never fatal) ----
    summary = _build_session_summary(session)
    prompt = workspace_files.build_session_end_prompt(summary)
    llm = utility_provider if utility_provider is not None else provider
    messages = [
        {"role": "system", "content": "You maintain workspace memory files for an AI agent. Respond with ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]
    _occ_baselines: dict[str, int | None] = {
        key: _stat_mtime_ns(workspace_files._file_path(key))
        for key in ("state", "decisions", "lessons", "index")
    }
    result: dict | None = None
    _per_attempt_timeouts = _dedup_timeouts(llm)
    if len(_per_attempt_timeouts) == 1:
        logger.info(
            "session_end LLM dedup: reasoning locked-on for model %s — "
            "single attempt with %.0fs timeout (no retry ladder)",
            getattr(llm, "model", "?"), _per_attempt_timeouts[0],
        )
    for _attempt, _timeout in enumerate(_per_attempt_timeouts, start=1):
        try:
            response = await asyncio.wait_for(
                llm.complete(messages, disable_reasoning=True), timeout=_timeout
            )
            result = _parse_session_end_response(response.text)
            break
        except asyncio.TimeoutError:
            if _attempt < len(_per_attempt_timeouts):
                logger.info(
                    "session_end LLM dedup: attempt %d timed out, retrying", _attempt
                )
            else:
                # Do NOT propagate — the deterministic backstop below must still
                # run. The old code raised here, which on a bloated project left
                # the caps unapplied (3x timeout -> nothing). Now the floor holds.
                logger.error(
                    "session_end LLM dedup: all %d attempts timed out; running "
                    "deterministic backstop only", len(_per_attempt_timeouts),
                )
        except Exception as e:  # noqa: BLE001 — cleanup must never crash the loop
            logger.warning(
                "session_end LLM dedup failed (%s); deterministic backstop only", e
            )
            break

    if result:
        # PROJECT_STATE: overwrite scratchpad (not entry-structured).
        if result.get("project_state", "").strip():
            _occ_write_metadata(
                workspace_files, "state", result["project_state"],
                _occ_baselines["state"], project_id=project_id,
            )
        # DECISIONS / LESSONS: stamp metadata onto the merged file, then write.
        for key in ("decisions", "lessons"):
            val = result.get(key, "")
            if val and val.strip():
                stamped, warns = _mem.stamp(
                    val, workspace_files.read(key), key, today=today
                )
                for w in warns:
                    logger.info("session_end stamp: %s", w)
                _occ_write_metadata(
                    workspace_files, key, stamped,
                    _occ_baselines[key], project_id=project_id,
                )
        # INDEX: navigation map (overwrite).
        if result.get("index", "").strip():
            _occ_write_metadata(
                workspace_files, "index", result["index"],
                _occ_baselines["index"], project_id=project_id,
            )
        # LLM-driven archive: entries the merge could not condense within budget
        # get MOVED into the read-on-demand archive file (same filename
        # convention as the deterministic hard-cap demotion in _apply_hard_caps),
        # under a date-stamped section, with an INDEX pointer. Never fatal.
        archive = result.get("archive")
        if isinstance(archive, dict):
            for key in ("decisions", "lessons"):
                moved = archive.get(key)
                if not isinstance(moved, str) or not moved.strip():
                    continue
                archive_key = _mem.ARCHIVE_OF[key]
                archive_filename = FILE_NAMES[archive_key]
                section = f"## [archived {today}]\n\n{moved.strip()}\n"
                try:
                    existing_archive = workspace_files.read(archive_key) or ""
                    if existing_archive.strip():
                        new_archive = existing_archive.rstrip() + "\n\n" + section
                    else:
                        new_archive = (
                            f"# {archive_filename} (demoted entries, read-on-demand)\n\n"
                            + section
                        )
                    workspace_files.write(archive_key, new_archive)
                    _ensure_index_archive_pointer(workspace_files, archive_filename)
                except OSError as e:
                    logger.warning(
                        "session_end: could not archive %s entries (%s); skipping",
                        key, e,
                    )
                    continue

    # ---- Part B: deterministic hard cap (ALWAYS runs; NEVER an LLM call) ----
    _apply_hard_caps(workspace_files)

    # Record the cleanup marker (post-write mtimes) so the next no-delta check is
    # accurate, then mark idempotency.
    _write_cleanup_marker(workspace_files)
    if not bypass_idempotency:
        _completed_session_ends.add(session_uuid)

    return "llm_merged" if result else "backstop_only"


# ---------------------------------------------------------------------------
# Session-end helpers: no-delta gate (persisted) + deterministic hard cap.
# ---------------------------------------------------------------------------

# Layer-1 files whose changes count as an un-consolidated delta.
_DELTA_KEYS = ("state", "decisions", "lessons", "index")

# Per-attempt timeout ladders for the LLM dedup/merge call.
_DEDUP_TIMEOUTS_DEFAULT = (30.0, 60.0, 90.0)
_DEDUP_TIMEOUTS_LOCKED_ON = (240.0,)


def _dedup_timeouts(llm) -> list[float]:
    """Timeout ladder for the session-end LLM dedup call, per provider.

    Locked-on reasoning models (e.g. MiniMax-M3, enable='model_only') cannot
    skip their thinking phase — ``disable_reasoning`` is a no-op, and retrying
    the identical prompt on a 30/60/90s ladder can never succeed; it only burns
    tokens (observed: every consolidation for a MiniMax-M3 project timed out
    3/3 for days, 2026-07-03..09). One attempt, one realistic timeout — the
    pass runs in the background, so a long single attempt never blocks the
    loop. The ``is True`` comparison guards against mocks, whose auto-created
    attributes are truthy without being True.
    """
    if getattr(llm, "reasoning_locked_on", False) is True:
        return list(_DEDUP_TIMEOUTS_LOCKED_ON)
    return list(_DEDUP_TIMEOUTS_DEFAULT)


def _cleanup_marker_path(workspace_files: "WorkspaceFileManager") -> str:
    return os.path.join(workspace_files.dir, _CLEANUP_MARKER_FILE)


def _current_layer1_mtimes(workspace_files: "WorkspaceFileManager") -> dict[str, int | None]:
    return {k: _stat_mtime_ns(workspace_files._file_path(k)) for k in _DELTA_KEYS}


def _has_cleanup_delta(workspace_files: "WorkspaceFileManager") -> bool:
    """True if any Layer-1 file changed since the last successful cleanup.

    Missing marker (never cleaned) => delta. Cheap: stat-only.
    """
    try:
        with open(_cleanup_marker_path(workspace_files), "r", encoding="utf-8") as f:
            stored = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True
    current = _current_layer1_mtimes(workspace_files)
    for k in _DELTA_KEYS:
        if str(stored.get(k)) != str(current.get(k)):
            return True
    return False


def _write_cleanup_marker(workspace_files: "WorkspaceFileManager") -> None:
    try:
        workspace_files.ensure_dir()
        with open(_cleanup_marker_path(workspace_files), "w", encoding="utf-8") as f:
            json.dump(_current_layer1_mtimes(workspace_files), f)
    except OSError as e:
        logger.warning("session_end: could not write cleanup marker (%s)", e)


def _ensure_index_archive_pointer(workspace_files: "WorkspaceFileManager", archive_filename: str) -> None:
    """Make sure INDEX points to an archive file after a demotion."""
    index = workspace_files.read("index") or ""
    if archive_filename in index:
        return
    pointer = f"\n- {archive_filename} — older entries demoted from the live file (read on demand).\n"
    workspace_files.write("index", (index.rstrip() + "\n" + pointer) if index.strip() else f"# INDEX\n{pointer}")


def _apply_hard_caps(workspace_files: "WorkspaceFileManager") -> None:
    """Deterministic size backstop. Demotes durable / trims volatile. No LLM.

    - DECISIONS/LESSONS: while over hard budget, demote coldest-``touched``
      entries to the archive (moves, never deletes; protects oldest-3 + pinned).
    - PROJECT_STATE/INDEX: while over hard budget, trim oldest (volatile).
    """
    budgets = _mem.FILE_BUDGETS
    # Durable: demote to archive.
    for key in _mem.DURABLE_KEYS:
        content = workspace_files.read(key)
        if not content or _mem.est_tokens(content) <= budgets[key]["hard"]:
            continue
        kept, demoted = _mem.split_for_demotion(content, key, budgets[key]["hard"])
        if not demoted:
            continue
        archive_key = _mem.ARCHIVE_OF[key]
        archive_filename = FILE_NAMES[archive_key]
        existing_archive = workspace_files.read(archive_key) or ""
        if existing_archive.strip():
            new_archive = existing_archive.rstrip() + "\n\n" + demoted
        else:
            new_archive = f"# {archive_filename} (demoted entries, read-on-demand)\n\n" + demoted
        workspace_files.write(archive_key, new_archive)
        workspace_files.write(key, kept)
        _ensure_index_archive_pointer(workspace_files, archive_filename)
        logger.info(
            "hard cap: demoted %d chars from %s to %s", len(demoted), key, archive_key
        )
    # Volatile: trim oldest.
    for key in _mem.VOLATILE_KEYS:
        content = workspace_files.read(key)
        if not content or _mem.est_tokens(content) <= budgets[key]["hard"]:
            continue
        trimmed = _mem.trim_volatile(content, budgets[key]["hard"])
        if trimmed != content:
            workspace_files.write(key, trimmed)
            logger.info("hard cap: trimmed %s to <= %d tok", key, budgets[key]["hard"])


def _build_session_summary(session) -> dict:
    """Extract summary stats from session for the LLM prompt."""
    messages = session.get_messages()

    tool_calls = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    # Extract modified files from write/edit tool results
    files_modified: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", tc)
                name = func.get("name", "")
                if name in ("write", "edit"):
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    path = args.get("path", args.get("file_path", ""))
                    if path:
                        files_modified.add(path)

    # Recent messages for context (last 30, but cap total chars)
    recent = messages[-30:] if len(messages) > 30 else messages
    recent_trimmed: list[dict] = []
    total_chars = 0
    for msg in reversed(recent):
        content = str(msg.get("content", ""))[:500]
        total_chars += len(content)
        if total_chars > 10000:
            break
        recent_trimmed.insert(0, {
            "role": msg.get("role"),
            "content": content,
            "source": msg.get("source", ""),
        })

    return {
        "session_id": getattr(session, "session_id", "unknown"),
        "message_count": len(messages),
        "tool_calls_count": sum(len(m.get("tool_calls", [])) for m in tool_calls),
        "files_modified": sorted(files_modified),
        "recent_messages": recent_trimmed,
    }


def _parse_session_end_response(text: str | None) -> dict | None:
    """Parse JSON from LLM response, handling markdown fences and whitespace.

    Returns None on failure.
    """
    if not text:
        return None

    # Strip whitespace
    cleaned = text.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        # Remove opening fence (```json or just ```)
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
        cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            # Optional top-level "archive": LLM-moved entries destined for the
            # read-on-demand archive files. Accept it only as a dict; drop any
            # other type so run_session_end_routine can call archive.get(...)
            # without a wrong-typed value leaking a crash into the cleanup loop.
            if "archive" in result and not isinstance(result["archive"], dict):
                result.pop("archive", None)
            return result
        logger.warning("Session-end LLM response was not a JSON object: %s", type(result).__name__)
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Session-end LLM response JSON parse failed: %s", e)
        return None
