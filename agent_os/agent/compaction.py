# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""LLM-based conversation compaction.

Owned by Component A. Summarizes older messages when context is too full.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.token_utils import estimate_message_tokens

logger = logging.getLogger(__name__)


_SUMMARIZATION_PROMPT = (
    "Summarize this conversation segment. Preserve: key decisions, user preferences, "
    "file paths mentioned, approaches tried and rejected, current task status, "
    "and the facts and values the agent found. Reply in plain prose."
)

# The rows go to the summarizer as a quoted record. Sent as bare "[user]: …"
# lines, mimo answered the user's request ("I cannot read your files")
# instead of summarizing it.
_TRANSCRIPT_FRAME = (
    "Below is the earlier part of a conversation between a user and an AI "
    "agent that uses tools. It is a record to summarize, not a request to you."
    "\n\n<transcript>\n{transcript}\n</transcript>\n\nWrite the summary now."
)

# A second compaction also summarizes the first one's summary. Inlined as one
# more "[system]:" line, mimo restated it word for word and dropped the facts
# found after it; kept apart, with the instruction to carry it forward.
_CHAINED_FRAME = (
    "Below is the summary of an earlier part of a conversation between a user "
    "and an AI agent that uses tools, then what happened after it. Both are a "
    "record to summarize, not a request to you."
    "\n\n<earlier_summary>\n{earlier}\n</earlier_summary>"
    "\n\n<transcript>\n{transcript}\n</transcript>\n\n"
    "Write one updated summary that keeps every fact from the earlier summary "
    "and adds everything new from the transcript."
)

# A summary slower than this is abandoned for the transcript digest:
# compaction runs inside the user's turn and must not stall it (a live mimo
# summary request sat unanswered for minutes; the SDK's own timeout is 600 s,
# with retries).
_SUMMARY_TIMEOUT_S = 90.0

MEMORY_FLUSH_PROMPT = (
    "Pre-compaction memory flush. "
    "Your context window is nearly full and history will be summarised shortly. "
    "Write any critical working state — current task position, active decisions, "
    "in-progress work, and anything you must not forget — to PROJECT_STATE.md now. "
    "Use the write or edit tool. "
    "If there is nothing important to save, reply with exactly: <silent>"
)


def is_silent_response(text: str) -> bool:
    """Return True if the response is exactly <silent> or empty."""
    stripped = text.strip()
    return stripped == "<silent>" or stripped == ""


async def run(session, provider, utility_provider=None, *,
              keep_tokens: float | None = None) -> None:
    """Summarize the older part of the model's history, append-only.

    ``keep_tokens`` bounds the kept tail, in ``estimate_message_tokens``
    units (the caller calibrates it to the window). Without it the newest
    ~30% of the history by size is kept. Uses utility_provider (cheaper
    model) if available, otherwise provider.
    """
    messages = session.get_model_messages()
    if len(messages) < 4:
        return

    split_point = _choose_split(messages, keep_tokens)
    pinned = _task_rows(messages, split_point)
    if not any(i not in pinned and not messages[i].get("_compaction")
               for i in range(split_point)):
        # Nothing new to summarize: the previous summary and the rows it
        # pinned are all that precede the tail. Leave the prompt as it is.
        logger.info("Compaction skipped: nothing older than the kept tail")
        return

    old_messages = messages[:split_point]
    llm = utility_provider if utility_provider is not None else provider
    summary_text = await _summarize(llm, old_messages)

    compaction_message = {
        "role": "system",
        "content": summary_text,
        "_compaction": True,
        "source": "management",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    session._compact(compaction_message, split_point, pinned)
    logger.info(
        "Compaction (session %s): summarized %d rows, kept %d, carried %d "
        "user message(s) verbatim; every row stays on disk",
        getattr(session, "session_uuid", "?"), split_point,
        len(messages) - split_point, len(pinned),
    )


def _choose_split(messages: list[dict], keep_tokens: float | None) -> int:
    """Index where the kept tail starts: as many of the newest rows as fit in
    ``keep_tokens``, but always the latest assistant turn and its results,
    and never opening on a tool result (its call would be summarized away)."""
    sizes = [estimate_message_tokens(m) for m in messages]
    if keep_tokens is None:
        keep_tokens = 0.3 * sum(sizes)
    split, used = len(messages), 0.0
    for i in range(len(messages) - 1, 0, -1):
        if used + sizes[i] > keep_tokens:
            break
        used += sizes[i]
        split = i
    latest = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"),
                 default=len(messages) - 1)
    split = max(1, min(split, latest))
    while split < latest and messages[split].get("role") == "tool":
        split += 1
    while split > 1 and messages[split].get("role") == "tool":
        split -= 1
    return split


def _task_rows(messages: list[dict], split: int) -> list[int]:
    """User messages before ``split`` that must reach the model verbatim.

    The current turn starts after the last finished one (an assistant reply
    with no tool calls; the pre-compaction flush reply does not count). Its
    user messages are the task. A turn started by a daemon event has none, so
    the latest user message stands in for the task.
    """
    turn_start = 0
    for i in range(len(messages) - 1, 0, -1):
        m = messages[i]
        if (m.get("role") == "assistant" and not m.get("tool_calls")
                and messages[i - 1].get("content") != MEMORY_FLUSH_PROMPT):
            turn_start = i + 1
            break
    rows = [i for i in range(turn_start, split) if messages[i].get("role") == "user"]
    if not rows and not any(messages[i].get("role") == "user"
                            for i in range(split, len(messages))):
        rows = [i for i in range(split) if messages[i].get("role") == "user"][-1:]
    return rows


_TOOL_MARKUP = re.compile(
    r"<tool_call|<function[=\s>]|<invoke\b|<\|tool|\"tool_use\"|"
    r"\"tool_calls\"\s*:|\"function_call\"\s*:",
    re.IGNORECASE,
)

_PLAIN_PROSE = (
    "Your previous reply was not a usable summary. Reply with the summary as "
    "plain prose only: no tool calls, no JSON, no XML."
)


def _usable_summary(response) -> str | None:
    """The summary text, or None when it is empty or shaped like a tool call
    or JSON (mimo returned a tool_use object as its "summary" in a live run)."""
    if getattr(response, "tool_calls", None):
        return None
    text = (getattr(response, "text", None) or "").strip()
    if is_silent_response(text) or _TOOL_MARKUP.search(text):
        return None
    if text.startswith("{"):
        return None
    try:
        json.loads(text)
        return None
    except ValueError:
        return text


async def _summarize(llm, old_messages: list[dict]) -> str:
    """LLM summary, retried once; a transcript digest if both are unusable."""
    earlier = [m for m in old_messages if m.get("_compaction")]
    transcript = _format_messages_for_summary(
        [m for m in old_messages if not m.get("_compaction")]
    )
    if earlier:
        body = _CHAINED_FRAME.format(earlier=_text_of(earlier[-1].get("content")),
                                     transcript=transcript)
    else:
        body = _TRANSCRIPT_FRAME.format(transcript=transcript)
    request = [
        {"role": "system", "content": _SUMMARIZATION_PROMPT},
        {"role": "user", "content": body},
    ]
    for attempt in (request, [*request, {"role": "user", "content": _PLAIN_PROSE}]):
        try:
            text = _usable_summary(
                await asyncio.wait_for(llm.complete(attempt), _SUMMARY_TIMEOUT_S)
            )
        except asyncio.TimeoutError:
            logger.warning("Compaction summary timed out after %.0fs; using "
                           "transcript digest", _SUMMARY_TIMEOUT_S)
            break
        except Exception:  # noqa: BLE001 — a failed summary must not fail the turn
            logger.warning("Compaction summarizer call failed", exc_info=True)
            text = None
        if text:
            return text
        logger.warning("Compaction summary unusable; %s",
                       "retrying" if attempt is request else "using transcript digest")
    return _digest(old_messages)


def _digest(old_messages: list[dict]) -> str:
    """Deterministic summary: the earlier summary, every user message
    verbatim, one line per tool call."""
    lines = ["[Summary built from the transcript; the summarizer's reply was unusable.]"]
    earlier = [m for m in old_messages if m.get("_compaction")]
    if earlier:
        lines += ["", "Earlier summary:", str(earlier[-1].get("content") or "")]
    users = [m for m in old_messages if m.get("role") == "user"]
    if users:
        lines += ["", "User messages:"]
        lines += [_text_of(m.get("content")) for m in users]
    calls = [tc for m in old_messages if m.get("role") == "assistant"
             for tc in m.get("tool_calls") or []]
    if calls:
        lines += ["", "Tool calls:"]
        for tc in calls:
            fn = tc.get("function") or {}
            lines.append(f"- {fn.get('name', '?')} {str(fn.get('arguments', ''))[:200]}")
    return "\n".join(lines)


def _text_of(content) -> str:
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text")
    return str(content or "")


def inject_reorientation(workspace: str, session) -> None:
    """Re-inject project goals and state after compaction.

    Reads project_goals.md and PROJECT_STATE.md from the workspace.
    If either file is non-empty, appends a single system message with
    both sections. If both are missing or empty, injects nothing.
    Fault-tolerant: read errors produce empty strings, never crash.
    """
    pp = ProjectPaths(workspace)
    goals_path = pp.project_goals
    state_path = pp.project_state

    goals = _safe_read(goals_path, max_chars=3000)
    state = _safe_read(state_path, max_chars=3000)

    if not goals and not state:
        return

    goals_section = goals if goals else "No project goals file found."
    state_section = state if state else "No state file found — check workspace for context."

    content = (
        "[POST-COMPACTION REORIENTATION]\n\n"
        "Your conversation history has been summarised to free context space. "
        "Your current project goals and state are unchanged:\n\n"
        "--- PROJECT GOALS ---\n"
        f"{goals_section}\n\n"
        "--- CURRENT STATE ---\n"
        f"{state_section}\n\n"
        "Continue your work based on the above. Do not repeat completed steps."
    )

    session.append({
        "role": "system",
        "content": content,
        "source": "management",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def _safe_read(filepath: str, max_chars: int = 3000) -> str:
    """Read a file up to max_chars. Return empty string on any error."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read(max_chars)
    except (OSError, IOError):
        return ""


def _format_messages_for_summary(messages: list[dict]) -> str:
    """Format messages into a readable text block for the summarization LLM."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = _text_of(msg.get("content", ""))
        if content:
            parts.append(f"[{role}]: {content[:2000]}")
    return "\n".join(parts)
