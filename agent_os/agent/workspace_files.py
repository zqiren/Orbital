# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workspace file management and session-end routine.

Manages the 5 workspace files in {workspace}/orbital/:
  PROJECT_STATE.md, DECISIONS.md, SESSION_LOG.md, LESSONS.md, CONTEXT.md

Provides:
  - WorkspaceFileManager: read/write/append workspace files, build cold resume context
  - run_session_end_routine: generate and write workspace files at session end via LLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

# In-memory idempotency guard for run_session_end_routine. Keyed by session_id.
# Not persisted across daemon restarts — a restart archives sessions anyway.
# GIL makes set.add/contains effectively atomic for short operations; we do not
# need an asyncio.Lock unless tests show flakiness under concurrent dispatch.
_completed_session_ends: set[str] = set()


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
    "session_log": "SESSION_LOG.md",
    "lessons": "LESSONS.md",
    "context": "CONTEXT.md",
}

# Order for cold resume context assembly
_RESUME_ORDER = ["state", "decisions", "lessons", "context", "session_log"]

# Section headers for cold resume context
_SECTION_HEADERS: dict[str, str] = {
    "state": "Project State",
    "decisions": "Decisions",
    "lessons": "Lessons Learned",
    "context": "External Context",
    "session_log": "Session Log (Recent)",
}

# Max number of sessions to include from SESSION_LOG when building the
# cold resume context or the session-end prompt (READ-time cap).
_SESSION_LOG_MAX_SESSIONS = 3

# Max number of sessions kept on disk in SESSION_LOG.md after an append
# at session-end (WRITE-time cap). Kept distinct from the read cap so
# future readers cannot confuse the two: read-cap trims what the agent
# sees, write-cap prunes what is persisted.
_SESSION_LOG_WRITE_CAP = 10

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
_CONTEXT_ENTRY_PATTERN = r"^-\s+"           # "- **Tencent:** ..."

# Entry caps per file. Code-side backstop matching the prompt's stated caps.
_LESSONS_CAP = 20
_DECISIONS_CAP = 30
_CONTEXT_CAP = 25


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
            "session_log": self._paths.session_log,
            "lessons": self._paths.lessons,
            "context": self._paths.context,
        }
        return _key_to_path[file_key]

    def read(self, file_key: str) -> str | None:
        """Read a workspace file. Returns None if file doesn't exist.

        file_key is one of: state, decisions, session_log, lessons, context
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
        """Assemble cold resume context from available files.

        Read order (skip missing files):
        1. PROJECT_STATE  - "where did I leave off"
        2. DECISIONS      - "what's already been decided"
        3. LESSONS        - "what should I avoid"
        4. CONTEXT        - "who/what am I working with"
        5. SESSION_LOG    - "what's the history" (last 3 sessions only)

        Returns assembled markdown string with section headers.
        """
        sections: list[str] = []

        for key in _RESUME_ORDER:
            content = self.read(key)
            if content is None:
                continue

            # For session_log, truncate to last N sessions
            if key == "session_log":
                content = _truncate_session_log(content, _SESSION_LOG_MAX_SESSIONS)

            header = _SECTION_HEADERS[key]
            sections.append(f"## {header}\n\n{content.strip()}")

        return "\n\n---\n\n".join(sections)

    def build_session_end_prompt(self, session_summary: dict) -> str:
        """Build the LLM prompt for session-end file generation.

        session_summary contains message_count, tool_calls_count, files_modified,
        recent_messages, etc.

        Returns a prompt string that asks the LLM to produce JSON with:
        project_state, decisions, session_log_entry, lessons, context.
        """
        # Read existing files for dedup context
        existing_state = self.read("state") or "(no existing state)"
        existing_decisions = self.read("decisions") or "(no existing decisions)"
        existing_lessons = self.read("lessons") or "(no existing lessons)"
        existing_context = self.read("context") or "(no existing context)"

        # SESSION_LOG tail (in-memory truncation; disk unchanged here).
        # Apply the READ-cap so the LLM sees only the most recent entries
        # for cross-session continuity when writing project_state/lessons.
        _session_log_raw = self.read("session_log")
        if _session_log_raw and _session_log_raw.strip():
            existing_session_log_tail = _truncate_session_log(
                _session_log_raw, _SESSION_LOG_MAX_SESSIONS
            ).strip() or "(no prior session log)"
        else:
            existing_session_log_tail = "(no prior session log)"

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

        prompt = f"""You are maintaining workspace memory files for an AI agent project.

Given the session information below, produce a JSON object with these fields:

1. "project_state" (REQUIRED): A complete snapshot of current project status.
   Include: what was accomplished, what's in progress, blockers, next steps.
   This REPLACES the previous state file entirely.

2. "decisions" (string, empty to preserve existing): The COMPLETE updated
   DECISIONS.md file. This REPLACES the existing file entirely.
   Scope: significant technical, architectural, or strategic decisions
   with rationale and rejected alternatives.
   - Carry forward every still-relevant prior decision
   - Add any new decisions made THIS SESSION
   - Drop decisions that have been superseded or are now obsolete
   - Merge duplicates or closely related decisions into single entries
   - Cap: 30 entries. Prioritize architectural over tactical; recent over old
     when contested.
   - Format each entry: ## YYYY-MM-DD: Title
                        **Chose:** ...
                        **Reason:** ...
                        **Rejected:** ...
   - Return empty string "" ONLY to indicate "no updates needed, preserve
     existing file." Do NOT return empty to mean "drop everything."

3. "session_log_entry" (REQUIRED): A log entry for this session.
   Format: ## Session {{id}} -- {{date}} {{start}}--{{end}}\\n- Completed: ...\\n- Attempted: ...

4. "lessons" (string, empty to preserve existing): The COMPLETE updated
   LESSONS.md file. This REPLACES the existing file entirely.
   Scope: generalizable patterns, pitfalls, and operating rules discovered
   through experience. Not: session facts, one-shot errors, or decisions.
   - Include every still-relevant prior entry verbatim or in equivalent form.
     Do not drop a lesson unless it is genuinely obsolete (the underlying
     issue was resolved, the advice no longer applies, or an equivalent
     lesson already exists).
   - Add any new lessons from THIS SESSION
   - Merge near-duplicates into single entries
   - Cap: 20 entries
   - Return empty string "" ONLY to indicate "no updates needed, preserve
     existing file."

5. "context" (string, empty to preserve existing): The COMPLETE updated
   CONTEXT.md file. This REPLACES the existing file entirely.
   Scope: external entities relevant to this project — people, services,
   platforms, third-party APIs, persistent environmental constraints.
   Exclusions (do NOT include):
     - Workspace files or internal project artifacts
     - One-shot session errors or transient tool failures
     - In-progress work or current task state (belongs in PROJECT_STATE)
     - Decisions or rationale (belongs in DECISIONS)
     - Patterns or advice (belongs in LESSONS)
   - Carry forward every still-relevant entry
   - Add genuinely new external entities discovered THIS SESSION
   - Drop entries not referenced in the last 10 sessions
   - Merge duplicates
   - Cap: 25 entries
   - Return empty string "" ONLY to indicate "no updates needed, preserve
     existing file."

IMPORTANT:
- For decisions, lessons, context: if you return a non-empty string, it
  must be the COMPLETE updated file (not just new entries). Return empty
  string "" to preserve the existing file unchanged.
- For project_state: always produce a complete snapshot.
- Respond with ONLY valid JSON. No markdown fences. No explanation.

--- EXISTING FILES (for context, DO NOT duplicate existing content) ---
PROJECT_STATE.md:
{existing_state}

DECISIONS.md:
{existing_decisions}

LESSONS.md:
{existing_lessons}

CONTEXT.md:
{existing_context}

SESSION_LOG.md (last 3 entries):
{existing_session_log_tail}

--- THIS SESSION ({message_count} messages, {tool_calls_count} tool calls) ---
Files modified: {files_modified}
Recent conversation:
{recent_formatted}"""

        return prompt


def _truncate_session_log(content: str, max_sessions: int) -> str:
    """Extract only the last N sessions from a SESSION_LOG.md content.

    Sessions are delimited by '## Session' headers.
    """
    # Split on session headers, keeping the delimiter
    parts = re.split(r'(?=^## Session )', content, flags=re.MULTILINE)

    # Filter out any preamble (parts before the first session header)
    session_parts = [p for p in parts if p.strip().startswith("## Session")]

    if not session_parts:
        return content

    # Take last N sessions
    recent = session_parts[-max_sessions:]
    return "\n".join(p.strip() for p in recent)


async def run_session_end_routine(
    session,
    provider,
    workspace_files: WorkspaceFileManager,
    utility_provider=None,
    *,
    session_id: str,
) -> None:
    """Generate and write workspace files at session end.

    Uses utility_provider (cheaper model) if available.
    This runs AFTER the agent loop exits but BEFORE session archival.

    session_id is required (keyword-only). Used to short-circuit duplicate
    invocations for the same session — both the loop.py fire-and-forget
    path and agent_manager.new_session() pre-archival path can fire for
    the same boundary, and we must not double-write SESSION_LOG / DECISIONS
    / CONTEXT. The completion set is only updated AFTER all writes succeed,
    so a failed run allows a second caller to retry.
    """
    # Idempotency guard: short-circuit if this session already completed.
    if session_id in _completed_session_ends:
        logger.info("session_end skipped: already completed for %s", session_id)
        return

    # 1. Gather session summary
    summary = _build_session_summary(session)

    # 2. Build prompt
    prompt = workspace_files.build_session_end_prompt(summary)

    # 3. Call LLM (utility model preferred)
    llm = utility_provider if utility_provider is not None else provider
    messages = [
        {"role": "system", "content": "You maintain workspace memory files for an AI agent. Respond with ONLY valid JSON."},
        {"role": "user", "content": prompt},
    ]
    # orbital-marketing 2026-04-22: a single Moonshot timeout produced an
    # amnesiac session. Retry on TimeoutError only; non-timeout errors fail fast.
    _per_attempt_timeouts = [30.0, 60.0, 90.0]
    for _attempt, _timeout in enumerate(_per_attempt_timeouts, start=1):
        try:
            response = await asyncio.wait_for(llm.complete(messages), timeout=_timeout)
            break
        except asyncio.TimeoutError:
            if _attempt < len(_per_attempt_timeouts):
                logger.info(
                    "session_end LLM call: attempt %d timed out, retrying",
                    _attempt,
                )
            else:
                logger.error(
                    "session_end LLM call: all %d attempts timed out, propagating",
                    len(_per_attempt_timeouts),
                )
                raise

    # 4. Parse JSON response
    result = _parse_session_end_response(response.text)
    if result is None:
        logger.warning("Session-end routine: failed to parse LLM response, skipping file updates")
        return

    # 5. Write files
    # PROJECT_STATE: full overwrite (always)
    if result.get("project_state"):
        workspace_files.write("state", result["project_state"])

    # DECISIONS: full overwrite (LLM returns COMPLETE updated file).
    # Empty string => preserve existing file (safer than blanking on a
    # forgetful LLM response).
    if result.get("decisions", "").strip():
        decisions_content = _apply_sanity_checks(
            result["decisions"],
            _DECISIONS_ENTRY_PATTERN,
            _DECISIONS_CAP,
            keep="last",
            filename="decisions",
        )
        workspace_files.write("decisions", decisions_content)
    else:
        logger.info("session_end: no updates for decisions, preserving existing file")

    # SESSION_LOG: always append, then mechanically cap on disk.
    # Read cap lives in build_cold_resume_context / build_session_end_prompt;
    # this block enforces the WRITE cap so the file does not grow without
    # bound. Reuses _truncate_session_log verbatim — no format changes.
    #
    # Guard: only invoke truncation when the marker count genuinely
    # exceeds the write cap. _truncate_session_log's split on '## Session'
    # also discards any pre-header preamble as a side effect; gating on
    # count keeps malformed-but-harmless files intact when no pruning is
    # required.
    if result.get("session_log_entry"):
        workspace_files.append("session_log", "\n" + result["session_log_entry"])

        current = workspace_files.read("session_log") or ""
        before_count = len(re.findall(r'(?m)^## Session ', current))
        if before_count > _SESSION_LOG_WRITE_CAP:
            truncated = _truncate_session_log(current, _SESSION_LOG_WRITE_CAP)
            if truncated != current:
                after_count = len(re.findall(r'(?m)^## Session ', truncated))
                workspace_files.write("session_log", truncated)
                logger.info(
                    "session_log truncated: %d → %d entries",
                    before_count, after_count,
                )

    # LESSONS: full overwrite (LLM returns COMPLETE updated file).
    # Empty string => preserve existing file.
    if result.get("lessons", "").strip():
        lessons_content = _apply_sanity_checks(
            result["lessons"],
            _LESSONS_ENTRY_PATTERN,
            _LESSONS_CAP,
            keep="first",
            filename="lessons",
        )
        workspace_files.write("lessons", lessons_content)
    else:
        logger.info("session_end: no updates for lessons, preserving existing file")

    # CONTEXT: full overwrite (LLM returns COMPLETE updated file).
    # Empty string => preserve existing file.
    if result.get("context", "").strip():
        context_content = _apply_sanity_checks(
            result["context"],
            _CONTEXT_ENTRY_PATTERN,
            _CONTEXT_CAP,
            keep="last",
            filename="context",
        )
        workspace_files.write("context", context_content)
    else:
        logger.info("session_end: no updates for context, preserving existing file")

    # Mark complete only AFTER all writes succeeded. If any write raised,
    # this line is skipped and a retry from the second caller will run.
    _completed_session_ends.add(session_id)


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
            return result
        logger.warning("Session-end LLM response was not a JSON object: %s", type(result).__name__)
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Session-end LLM response JSON parse failed: %s", e)
        return None
