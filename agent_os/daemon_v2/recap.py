# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The "Conversation so far" block a worker receives on dispatch.

The chat is ONE conversation shared by the management agent and every
sub-agent. A worker's own thread (its resumed provider session) holds only
what was addressed to it and what it replied; everything else that was
visible in the chat — the user's messages to other agents, the manager's
replies, other workers' replies — reaches it through this block, prepended
to the message it is dispatched with.

Rules (agent-agnostic — nothing here knows which transport is on the other
end):

* **Scope = the gap since the worker's last dispatch marker.** The marker
  row (``_meta.dispatch_id`` + ``_meta.handle``) is written when a prompt
  actually goes out, so "after the last marker" is exactly "what this worker
  has not been shown". The completion row is deliberately NOT the watermark:
  rows that land while the worker is still running would otherwise sit
  between its marker and its completion and never be delivered.
* **Native rows are skipped** for a resumed thread: user messages addressed
  to this worker and its own completion rows are already in its session.
* **A fresh spawn gets the whole session**, native rows included — its
  thread is empty, so there is nothing to skip.
* **Only user-visible messages** — user rows, manager text rows, and other
  workers' replies (from the completion row's stored ``_meta.reply``).
  Tool rows, status/lifecycle rows, and steering guidance never appear.
* **No per-message truncation.** Only a generous total budget applies,
  evicting oldest-first; the largest realistic message (a pasted form) must
  arrive intact.
* **Worker-anonymous**: every prior reply is labeled ``assistant``.
"""

from __future__ import annotations

import json
import re
from typing import Callable

# Total budget for one block. Measured on a full day's session: ~19k chars
# of visible messages, so this only ever bites on a pathological paste.
RECAP_CAP_CHARS = 60_000

# Cap for the full reply stored on a completion row (``_meta.reply``). The
# old 500-char summary stays as the timeline text.
REPLY_CAP_CHARS = 16_000

# Legacy completion rows (written before ``_meta.reply`` existed) carry the
# summary only inside their display text. Greedy body + the literal tail
# anchors on the LAST ". Transcript: ", so summaries containing periods
# survive.
_LEGACY_COMPLETED_RE = re.compile(
    r"\[Sub-agent\] \S+ completed\. Summary: ([\s\S]*)\. Transcript: "
)

_HEADER = (
    "Conversation so far (earlier messages in this chat, provided as "
    "context; prior replies are labeled \"assistant\"):\n\n"
)
_FOOTER = "\n\n--- end of conversation so far ---\n\n"


def _is_dispatch_marker(row: dict, handle: str) -> bool:
    meta = row.get("_meta")
    return (
        row.get("role") == "system"
        and isinstance(meta, dict)
        and bool(meta.get("dispatch_id"))
        and meta.get("handle") == handle
    )


def _completion_of(row: dict) -> tuple[str, str] | None:
    """``(handle, reply)`` for a completed-terminal row, else None."""
    meta = row.get("_meta")
    if (
        row.get("role") != "system"
        or not isinstance(meta, dict)
        or meta.get("event") != "sub_agent_terminal"
        or meta.get("kind") != "completed"
    ):
        return None
    content = row.get("content")
    display = meta.get("display_content") or (
        content if isinstance(content, str) else "")
    handle = ""
    if display.startswith("[Sub-agent] "):
        handle = display[len("[Sub-agent] "):].split(" ", 1)[0]
    reply = meta.get("reply")
    if not isinstance(reply, str):
        match = _LEGACY_COMPLETED_RE.match(display)
        reply = match.group(1) if match else ""
    return handle, reply.strip()


def recap_scope(messages: list[dict], handle: str, *, fresh: bool) -> list[dict]:
    """Rows this worker has not been shown: after its last dispatch marker,
    or the whole session for a fresh spawn / a worker never dispatched."""
    if fresh:
        return list(messages)
    last = -1
    for i, row in enumerate(messages):
        if _is_dispatch_marker(row, handle):
            last = i
    return list(messages[last + 1:])


def _visible_line(row: dict, handle: str, *, fresh: bool) -> str | None:
    # Spec 112 A6: the memory record turn (the daemon's [MEMORY] ask, the
    # model's edits and its "nothing to record", a pinned worker's reminder
    # and its reply) is bookkeeping, not conversation. The reply before it
    # is what the chat said.
    if row.get("_kind") in ("memory_record", "memory_tidy", "ask_card"):
        return None
    role = row.get("role")
    content = row.get("content")
    if role == "user":
        if not fresh and row.get("target") == handle:
            return None  # native: addressed to this worker
        if isinstance(content, str) and content.strip():
            return f"user: {content.strip()}"
        return None
    if role == "assistant":
        if isinstance(content, str) and content.strip():
            return f"assistant: {content.strip()}"
        return None
    if role == "system":
        completed = _completion_of(row)
        if completed is None:
            return None
        owner, reply = completed
        if not fresh and owner == handle:
            return None  # native: its own reply
        if reply and reply != "(no output)":
            return f"assistant: {reply}"
    return None


def build_recap(messages: list[dict], handle: str, *, fresh: bool) -> str:
    """Render the block for ``handle``; "" when it missed nothing."""
    lines: list[str] = []
    total = 0
    for row in reversed(recap_scope(messages, handle, fresh=fresh)):
        line = _visible_line(row, handle, fresh=fresh)
        if line is None:
            continue
        if total + len(line) > RECAP_CAP_CHARS:
            break
        lines.append(line)
        total += len(line)
    if not lines:
        return ""
    lines.reverse()
    return _HEADER + "\n\n".join(lines) + _FOOTER


# ---------------------------------------------------------------------------
# Spec 128: where the earlier details live (paths only, read on demand)
# ---------------------------------------------------------------------------
#
# The recap above is conversational text; tool calls, results and lifecycle
# rows are filtered out of it on purpose. The agent taking over after a pin
# switch therefore gets, alongside the recap, the file paths of the prior
# agents' ORIGINAL session logs — Claude Code's own
# ``~/.claude/projects/<slug>/<id>.jsonl``, Codex's rollout JSONL — plus the
# canonical Orbital chat log. Only when a native log cannot be resolved is
# the Orbital worker transcript offered instead, labelled as the partial
# capture it is. Nothing is read or injected here: the recipient opens a
# file when it needs earlier details. The block is kept apart from
# ``_visible_line`` and the recap budget so neither filtering nor eviction
# can drop it.

SOURCE_LOGS_HEADER = (
    "Previous session logs — read only when you need earlier details:\n")

_UNAVAILABLE = "unavailable"


def _dispatch_marker_paths(messages: list[dict]) -> dict[str, list[str]]:
    """``handle -> [exact transcript paths]`` from this chat's own dispatch
    markers, first-seen order, deduplicated. The memory record / tidy turns
    are bookkeeping (spec 112 A6) and never a source."""
    paths: dict[str, list[str]] = {}
    for row in messages:
        if row.get("_kind") in ("memory_record", "memory_tidy", "ask_card"):
            continue
        meta = row.get("_meta")
        if (
            row.get("role") != "system"
            or not isinstance(meta, dict)
            or not meta.get("dispatch_id")
            or not meta.get("handle")
        ):
            continue
        path = meta.get("transcript_path")
        if not isinstance(path, str) or not path or path == "unknown":
            continue
        known = paths.setdefault(meta["handle"], [])
        if path not in known:
            known.append(path)
    return paths


def source_log_entries(
    messages: list[dict],
    handle: str,
    *,
    fresh: bool,
    thread_history: dict[str, list[dict]],
    chat_log_path: str | None,
    native_path_for: Callable[[str, dict], str | None],
) -> list[tuple[str, str | None]]:
    """``(label, path)`` entries for the dispatch to ``handle``; ``path`` is
    None for a known source that could not be resolved (rendered as
    "unavailable" — never a fabricated path).

    ``thread_history`` is every persisted resume record per handle, in
    order, for THIS chat (``Session.sub_agent_thread_history``), so a reset
    does not lose the earlier threads' logs. ``native_path_for`` resolves
    one record to its native log file (metadata only — it must not read
    transcript bodies).

    A resumed worker's latest record IS the thread it is resuming, so that
    one — and its own transcript — are left out; its older records are
    prior logs like any other agent's. A fresh spawn gets everything,
    including its own lost threads.
    """
    entries: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    def add(label: str, path: str | None) -> None:
        if path is not None:
            if path in seen:
                return
            seen.add(path)
        entries.append((label, path))

    marker_paths = _dispatch_marker_paths(messages)
    handles: list[str] = []
    for h in list(thread_history) + list(marker_paths):
        if h not in handles:
            handles.append(h)

    for other in handles:
        records = list(thread_history.get(other) or [])
        own_live = other == handle and not fresh
        if own_live:
            # The latest record is the thread being resumed — every record
            # carrying that session id (a completion rewrites it each turn)
            # is skipped. The earlier (lost) threads stay; the worker is
            # never offered its own current log or transcript.
            live = records[-1].get("session_id") if records else None
            records = [r for r in records
                       if isinstance(r, dict) and r.get("session_id") != live]
            if not records:
                continue
        resolved = False
        for record in records:
            if not isinstance(record, dict):
                continue
            path = native_path_for(other, record)
            if isinstance(path, str) and path:
                add(f"Original agent session log ({other})", path)
                resolved = True
        if resolved or own_live:
            continue
        fallback = marker_paths.get(other) or []
        if fallback:
            for path in fallback:
                add(f"Orbital worker transcript ({other}, partial capture)", path)
        elif records:
            add(f"Original agent session log ({other})", None)

    if chat_log_path:
        add("Orbital chat log", chat_log_path)
    return entries


def format_source_logs(entries: list[tuple[str, str | None]]) -> str:
    """Render the path block, or "" when there is nothing to point at.

    Paths are JSON-encoded so a quote or newline in a filename cannot break
    the block; they are never truncated. Windows paths are rendered with
    forward slashes first: JSON would otherwise double every backslash
    (``"D:\\\\proj\\\\orbital\\\\sessions\\\\x.jsonl"``), which is not a path
    an agent can open verbatim, while ``D:/proj/orbital/sessions/x.jsonl``
    is accepted by every Windows file API and reads cleanly."""
    if not entries:
        return ""
    lines = [SOURCE_LOGS_HEADER]
    for label, path in entries:
        value = (json.dumps(_portable_path(path), ensure_ascii=False)
                 if path else _UNAVAILABLE)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines) + "\n\n"


def _portable_path(path: str) -> str:
    """``C:\\a\\b`` → ``C:/a/b``; POSIX paths are returned unchanged. Only a
    drive-letter or UNC path is rewritten, so a POSIX filename that happens
    to contain a backslash keeps it."""
    if len(path) >= 3 and path[1] == ":" and path[2] in "\\/":
        return path.replace("\\", "/")
    if path.startswith("\\\\"):
        return "//" + path[2:].replace("\\", "/")
    return path
