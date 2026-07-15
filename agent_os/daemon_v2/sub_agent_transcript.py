# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent transcript — append-only JSONL log for one sub-agent's output."""

import json
import os
import re
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Matches the "[Using tool: X]" text the SDK transport writes for a
# tool_activity chunk (see process_manager.py). Captures the tool name.
_TOOL_ACTIVITY_RE = re.compile(r"\[Using tool:\s*([^\]]+)\]")


def _parse_ts(value) -> "datetime | None":
    """Best-effort ISO-8601 parse; returns None on anything unparseable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _summarize_turn(entries: list) -> dict:
    """Summarize ONE turn's (boundary-free) chunks into the capsule-shaped dict::

        {
            "tool_rows": [{"name": "Write", "timestamp": "...", "duration_seconds": 1.2}, ...],
            "total_duration_seconds": 7.3,             # first→last chunk in the turn
            "response": "The file already exists ...",  # last response/message in the turn
            "chunk_count": 4,
            "tools_used": ["Write", "Bash", "Read"],   # de-duped, first-seen order
            "duration_seconds": 7.3,                    # alias of total_duration_seconds
        }

    ``response`` is the LAST response/message chunk in THIS turn, or ``""`` if
    the turn produced none (errored / interrupted / tool-only). Callers MUST NOT
    alias a neighboring turn's text into an empty one. Tool-row durations look
    ahead only within the turn (the last tool's gap to the following chunk, or
    0). Only the tool *name* is available (the SDK streams ``[Using tool: X]``).
    """
    chunk_count = len(entries)
    response = ""
    tool_rows: list = []
    tools_used: list = []
    seen_tools: set = set()

    first_ts = None
    last_ts = None
    for e in entries:
        ts = _parse_ts(e.get("timestamp"))
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

    for idx, e in enumerate(entries):
        chunk_type = e.get("chunk_type")
        content = e.get("content") or ""

        if chunk_type == "tool_activity":
            m = _TOOL_ACTIVITY_RE.search(content)
            if not m:
                continue
            name = m.group(1).strip()
            if not name:
                continue
            ts = _parse_ts(e.get("timestamp"))
            next_ts = _parse_ts(entries[idx + 1].get("timestamp")) if idx + 1 < len(entries) else None
            dur = 0.0
            if ts is not None and next_ts is not None:
                dur = max(0.0, (next_ts - ts).total_seconds())
            tool_rows.append({
                "name": name,
                "timestamp": e.get("timestamp") or "",
                "duration_seconds": round(dur, 1),
            })
            if name not in seen_tools:
                seen_tools.add(name)
                tools_used.append(name)
        elif chunk_type in ("response", "message") or chunk_type is None:
            if content:
                response = content

    total_duration_seconds = 0.0
    if first_ts is not None and last_ts is not None:
        total_duration_seconds = round(max(0.0, (last_ts - first_ts).total_seconds()), 1)

    return {
        "tool_rows": tool_rows,
        "total_duration_seconds": total_duration_seconds,
        "response": response,
        "chunk_count": chunk_count,
        # Back-compat aliases.
        "tools_used": tools_used,
        "duration_seconds": total_duration_seconds,
    }


def read_sub_agent_summary(transcript_path: str) -> "list | None":
    """Read a sub-agent JSONL transcript → a LIST of per-turn summary dicts.

    The transcript is split into turns on ``chunk_type="turn_complete"``
    boundary rows (ProcessManager appends one at each terminal turn). Each
    completed turn is the run of chunks BEFORE a boundary; the boundary closes
    it. Returns one ``_summarize_turn`` dict per completed turn, in order.

    Each turn dict also carries ``"dispatch_id"``: the id stamped on the
    boundary row that closed it (TASK-dispatch-id-pairing), or ``None`` when
    the boundary carries none (legacy data written before dispatch_ids
    existed) or there is no boundary at all (flat legacy transcript). The
    chat renderer (``_interleave_sub_agent_summaries``) joins a dispatch
    marker to its turn by this id — NOT by position — since a transcript
    spans many chat sessions and positional pairing only lines up for the
    file's first session.

    Rules:
    - Trailing chunks AFTER the last boundary are an in-flight turn → NOT
      returned (they get a bubble on a later reload, once complete).
    - A transcript with ZERO boundaries is a legacy/flat file → ONE turn (the
      whole file). Retroactivity guard for pre-boundary transcripts.
    - Boundary rows are split out before summarizing, so they never count toward
      a turn's ``chunk_count`` or duration.

    Returns ``None`` when the file does not exist, is empty, or contains no
    parseable JSON lines. Malformed individual lines are skipped, not fatal.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    entries: list = []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
    except OSError:
        return None

    if not entries:
        return None

    # Split on turn_complete boundary rows. Each boundary closes the turn that
    # precedes it; trailing post-boundary chunks (in-flight) are dropped.
    # ``boundary_ids`` tracks each closed turn's dispatch_id in lockstep with
    # ``turns`` so it can be zipped back on below.
    turns: list = []
    boundary_ids: list = []
    current: list = []
    boundary_seen = False
    for e in entries:
        if e.get("chunk_type") == "turn_complete":
            boundary_seen = True
            turns.append(current)
            boundary_ids.append(e.get("dispatch_id"))
            current = []
        else:
            current.append(e)
    if not boundary_seen:
        # Legacy flat transcript (no boundary) → the whole file is one turn,
        # with no boundary to source a dispatch_id from.
        turns = [current]
        boundary_ids = [None]

    result = []
    for turn_entries, dispatch_id in zip(turns, boundary_ids):
        summary = _summarize_turn(turn_entries)
        summary["dispatch_id"] = dispatch_id
        result.append(summary)
    return result


class SubAgentTranscript:
    """Append-only JSONL log for one sub-agent's output within a project."""

    def __init__(self, workspace: str, handle: str, transcript_id: str):
        from agent_os.agent.project_paths import ProjectPaths
        self._dir = ProjectPaths(workspace).sub_agent_dir(handle)
        os.makedirs(self._dir, exist_ok=True)
        self._filename = f"{transcript_id}.jsonl"
        self._filepath = os.path.join(self._dir, self._filename)
        # Open in append mode: a reused id (BACKLOG 005 §4b) keeps prior turns;
        # a fresh id starts empty.
        open(self._filepath, "a", encoding="utf-8").close()
        # Update latest pointer (use .latest text file on Windows)
        self._update_latest()

    @classmethod
    def open_for_handle(cls, workspace: str, handle: str,
                        *, fresh: bool = False) -> "SubAgentTranscript":
        """Resume the per-(workspace, handle) transcript, or mint a new one.

        BACKLOG 005 §4b: a respawn must keep the UI's ``.latest`` transcript
        CONTINUOUS instead of pointing at a fresh empty file every time. When a
        valid ``.latest`` pointer already exists for the handle, reuse that
        filename (append mode) rather than minting a new ``str(uuid4())[:8]``
        id. A new id is minted only when ``.latest`` is missing/corrupt or its
        target file no longer exists.

        ``fresh=True`` (§4d reset) ALWAYS mints a new id, bypassing reuse — the
        prior transcript stays on disk (archived), and ``.latest`` re-points at
        the new file.
        """
        from agent_os.agent.project_paths import ProjectPaths
        from uuid import uuid4

        if not fresh:
            sub_dir = ProjectPaths(workspace).sub_agent_dir(handle)
            existing = cls._read_latest_id(sub_dir)
            if existing is not None:
                return cls(workspace, handle, existing)
        return cls(workspace, handle, str(uuid4())[:8])

    @staticmethod
    def _read_latest_id(sub_dir: str) -> "str | None":
        """Return the transcript id (filename stem) ``.latest`` points at, when
        the pointed-at file still exists; else None (mint a new one).
        """
        latest_path = os.path.join(sub_dir, ".latest")
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                filename = f.read().strip()
        except OSError:
            return None
        if not filename:
            return None
        # The pointer must reference a real file — a dangling pointer (the file
        # was deleted/pruned) means start fresh, not append to a ghost path.
        if not os.path.isfile(os.path.join(sub_dir, filename)):
            return None
        # __init__ rebuilds the filename as f"{transcript_id}.jsonl", so recover
        # the stem here to round-trip cleanly.
        if filename.endswith(".jsonl"):
            return filename[: -len(".jsonl")]
        return filename

    def _update_latest(self) -> None:
        """Point latest to current transcript."""
        latest_path = os.path.join(self._dir, ".latest")
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(self._filename)
        except OSError:
            pass

    def append(self, entry: dict) -> None:
        """Append one JSON line. Entry should have: source, content, timestamp, chunk_type."""
        if "timestamp" not in entry:
            entry["timestamp"] = _now()
        line = json.dumps(entry, ensure_ascii=False)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    @property
    def filepath(self) -> str:
        """Return the absolute path to the transcript file."""
        return self._filepath

    @classmethod
    def read(cls, filepath: str) -> list[dict]:
        """Read all entries from a transcript file."""
        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
