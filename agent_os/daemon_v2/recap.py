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
* **No per-message truncation.** Only a total budget applies, in UTF-8
  bytes for the whole rendered block (spec 181), evicting oldest-first; the
  conversation's first user message is kept, and when anything is left out
  a note says how many messages and which file holds them all.
* **Worker-anonymous**: every prior reply is labeled ``assistant``.
"""

from __future__ import annotations

import json
import re
from typing import Callable

# Spec 181: the most the WHOLE rendered block (header, note, lines, joins,
# footer) may weigh, in UTF-8 bytes. Bytes, not characters: a CJK character
# costs 3 bytes and an ASCII one 1, which is roughly how tokens behave, so
# this is ~15-20K tokens of prose in any script (it was 60,000 characters:
# up to ~60K tokens of Chinese).
RECAP_CAP_BYTES = 60_000

# The budget's room term (T3 Code's handoffBudget shape): what the receiving
# thread has left, charged one token per byte, minus a reserve for the
# agent's own work. The window and usage come only from what the agent itself
# reported for this thread; unknown → this window and nothing used, which
# leaves the fixed cap in charge.
DEFAULT_CONTEXT_WINDOW = 256_000
_MIN_RESERVE_TOKENS = 16_000

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

# Rows that are bookkeeping, not conversation: the memory record / tidy turns
# (spec 112 A6), the ask card (spec 146), and the daemon's worker-result turn
# (spec 161: the instruction to the worker that dispatched; the child's reply
# is already in the block as its own completion row).
BOOKKEEPING_KINDS = frozenset(
    {"memory_record", "memory_tidy", "ask_card", "worker_result"})

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


# Batch 1118: a worker's own command (`/compact`, `/review x`) runs only when
# it is the very start of the prompt, so a user-addressed message that starts
# with one is dispatched ALONE and everything held for the worker (first-turn
# note, skills note, this block) waits for the next ordinary message. The rule
# is the composer picker's (`slashQuery` in web/src/lib/skill-catalog.ts): a
# `/`, then a name with no whitespace and no second `/`, then whitespace or the
# end — so `/Users/x/a.txt look` and `/tmp/x` are text, not commands.
_COMMAND_RE = re.compile(r"/[^\s/]+(?:\s|$)")

# The marker's display prose (stable: old UIs and the legacy migration read
# it): ``[Sub-agent] Message sent to <handle>: "<first 100 chars>". …``.
_MARKER_PREVIEW_RE = re.compile(r'\[Sub-agent\] Message sent to \S+: "([\s\S]*)')


def is_worker_command(text: str | None) -> bool:
    """Does ``text`` start with a worker command (the picker's rule)?"""
    return isinstance(text, str) and _COMMAND_RE.match(text) is not None


def _marker_held_context(row: dict) -> bool:
    """Was this dispatch a command sent alone? A user-addressed dispatch
    (the only kind that holds context) is wake-suppressed; its marker's
    preview is the message's start, which decides it the same way the
    dispatch did. Read from the row, so a respawned worker (idle stop,
    daemon restart) still knows the gap before the command is unsent."""
    meta = row.get("_meta")
    if not isinstance(meta, dict) or meta.get("suppress_wake") is not True:
        return False
    display = meta.get("display_content") or row.get("content")
    match = _MARKER_PREVIEW_RE.match(display) if isinstance(display, str) else None
    return match is not None and is_worker_command(match.group(1))


def recap_scope(messages: list[dict], handle: str, *, fresh: bool) -> list[dict]:
    """Rows this worker has not been shown: after its last dispatch marker
    (a command sent alone does not count: nothing was shown with it), or the
    whole session for a fresh spawn / a worker never dispatched."""
    if fresh:
        return list(messages)
    last = -1
    for i, row in enumerate(messages):
        if _is_dispatch_marker(row, handle) and not _marker_held_context(row):
            last = i
    return list(messages[last + 1:])


def _native_from(messages: list[dict], handle: str, *,
                 message: str | None,
                 held_dispatch_ids: frozenset[str]) -> int | None:
    """For a FRESH thread: the index of the first row it holds natively, or
    None. Rows from there on are filtered like a resumed thread's.

    That is the user row of the first command this process was sent alone
    (``held_dispatch_ids``: its thread has every message addressed to it
    since), else the message being sent — persisted just before the
    dispatch, so it is in the session and would otherwise be repeated inside
    the block. The message is the first not-yet-dispatched user row addressed
    to the worker whose text it starts with (the route may append a note)."""
    if held_dispatch_ids:
        for i, row in enumerate(messages):
            meta = row.get("_meta")
            if (_is_dispatch_marker(row, handle) and isinstance(meta, dict)
                    and meta.get("dispatch_id") in held_dispatch_ids):
                for j in range(i - 1, -1, -1):
                    prior = messages[j]
                    if prior.get("role") == "user" and prior.get("target") == handle:
                        return j
                return i
    if not message:
        return None
    start = 0
    for i, row in enumerate(messages):
        if _is_dispatch_marker(row, handle):
            start = i + 1
    for i in range(start, len(messages)):
        row = messages[i]
        content = row.get("content")
        if (row.get("role") == "user" and row.get("target") == handle
                and isinstance(content, str) and content
                and message.startswith(content)):
            return i
    return None


def _visible_line(row: dict, handle: str, *, fresh: bool) -> str | None:
    # Spec 112 A6: the memory record turn (the daemon's [MEMORY] ask, the
    # model's edits and its "nothing to record", a pinned worker's reminder
    # and its reply) is bookkeeping, not conversation. The reply before it
    # is what the chat said.
    if row.get("_kind") in BOOKKEEPING_KINDS:
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
        # Spec 180: the move note is part of the conversation for every
        # worker (each is a fresh spawn in the new project).
        meta = row.get("_meta")
        if isinstance(meta, dict) and meta.get("event") == "session_moved":
            if isinstance(content, str) and content.strip():
                return f"system: {content.strip()}"
            return None
        completed = _completion_of(row)
        if completed is None:
            return None
        owner, reply = completed
        if not fresh and owner == handle:
            return None  # native: its own reply
        if reply and reply != "(no output)":
            return f"assistant: {reply}"
    return None


def handoff_budget_bytes(*, message_bytes: int = 0,
                         reported: tuple | None = None) -> int:
    """The byte budget for one block: the fixed cap, shrunk (never grown) by
    the room the receiving thread has left.

    ``reported`` is ``(window, used)`` in tokens as the agent itself reported
    them for this thread, or None. The user's own message is charged against
    the room (it is never truncated)."""
    window, used = DEFAULT_CONTEXT_WINDOW, 0
    if (isinstance(reported, tuple) and len(reported) == 2
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                    for v in reported)
            and reported[0] > 0):
        window, used = reported
    reserve = max(_MIN_RESERVE_TOKENS, window // 4)
    room = window - used - max(0, int(message_bytes)) - reserve
    return max(0, min(RECAP_CAP_BYTES, room))


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _left_out_note(n: int, chat_log_path: str | None) -> str:
    if n == 1:
        head = ("(The 1 earlier message of this conversation is not shown "
                "here.")
    else:
        head = (f"(The {n} earlier messages of this conversation are not "
                f"shown here.")
    if chat_log_path:
        path = json.dumps(_portable_path(chat_log_path), ensure_ascii=False)
        return (f"{head} The whole conversation is in the Orbital chat log: "
                f"{path})")
    return head + ")"


def build_recap(messages: list[dict], handle: str, *, fresh: bool,
                chat_log_path: str | None = None,
                budget_bytes: int | None = None,
                message: str | None = None,
                held_dispatch_ids=None) -> str:
    """Render the block for ``handle``; "" when it missed nothing.

    ``message`` is the text being dispatched with the block, and
    ``held_dispatch_ids`` the commands this (fresh) process was already sent
    alone; with them a fresh thread's block leaves out what that thread
    holds natively (see ``_native_from``).

    The whole rendered block is at most ``budget_bytes`` UTF-8 bytes
    (default ``RECAP_CAP_BYTES``). When every visible line fits, the block is
    exactly the old shape. Otherwise: the first user message in scope (unless
    it alone is over a quarter of the budget), then a note with the number of
    lines left out and the chat log, then the newest lines that fit. When
    nothing fits, the block is header + note + footer — never ""."""
    budget = RECAP_CAP_BYTES if budget_bytes is None else max(0, budget_bytes)
    native_from = (_native_from(messages, handle, message=message,
                                held_dispatch_ids=frozenset(held_dispatch_ids or ()))
                   if fresh else None)
    lines: list[tuple[str, bool]] = []  # (line, is the user's own)
    for i, row in enumerate(recap_scope(messages, handle, fresh=fresh)):
        row_fresh = fresh and (native_from is None or i < native_from)
        line = _visible_line(row, handle, fresh=row_fresh)
        if line is not None:
            lines.append((line, row.get("role") == "user"))
    if not lines:
        return ""
    frame = _utf8_len(_HEADER) + _utf8_len(_FOOTER)
    sizes = [_utf8_len(line) for line, _ in lines]
    if frame + sum(sizes) + 2 * (len(lines) - 1) <= budget:
        return _HEADER + "\n\n".join(line for line, _ in lines) + _FOOTER

    # Eviction. The note is reserved at its largest (every line left out),
    # whether or not it ends up that long, so the fill stays one pass; each
    # kept line then costs its bytes plus the "\n\n" before it.
    left = budget - frame - _utf8_len(_left_out_note(len(lines), chat_log_path))
    first = next((i for i, (_, own) in enumerate(lines) if own), None)
    if first is not None and (sizes[first] + 2 > budget / 4
                              or sizes[first] + 2 > left):
        first = None
    if first is not None:
        left -= sizes[first] + 2
    kept: list[int] = []
    for i in range(len(lines) - 1, -1, -1):
        if i == first:
            continue
        if sizes[i] + 2 > left:
            break
        kept.append(i)
        left -= sizes[i] + 2
    kept.reverse()
    if first is not None and kept and kept[0] < first:
        # The fill reached past the first user message (lines before it are
        # the agent's, e.g. an onboarding greeting): it sits in its own place.
        kept = sorted(kept + [first])
        first = None
    n = len(lines) - len(kept) - (1 if first is not None else 0)
    parts = ([lines[first][0]] if first is not None else [])
    parts.append(_left_out_note(n, chat_log_path))
    parts.extend(lines[i][0] for i in kept)
    return _HEADER + "\n\n".join(parts) + _FOOTER


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
        if row.get("_kind") in BOOKKEEPING_KINDS:
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
    fork_cut: str | None = None,
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

    ``fork_cut`` (spec 156 §5.5): in a session forked from an edited message,
    the edited row's timestamp. The Orbital worker transcripts its copied
    dispatch markers name are per handle and kept growing in the parent past
    that point, so their labels say which entries belong here. The fork's
    own thread records are its own native sessions and need no label.
    """
    cut_suffix = (f" (only entries before {fork_cut} belong to this conversation)"
                  if fork_cut else "")
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
                add(f"Orbital worker transcript ({other}, partial capture)"
                    f"{cut_suffix}", path)
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
