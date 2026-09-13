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

import re

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
