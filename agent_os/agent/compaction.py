# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""LLM-based conversation compaction.

Owned by Component A. Summarizes older messages when context is too full.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from agent_os.agent.project_paths import ProjectPaths

logger = logging.getLogger(__name__)


_SUMMARIZATION_PROMPT = (
    "Summarize this conversation segment. Preserve: key decisions, user preferences, "
    "file paths mentioned, approaches tried and rejected, current task status."
)

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


async def run(session, provider, utility_provider=None) -> None:
    """Summarize older messages and compact the session.

    Uses utility_provider (cheaper model) if available, otherwise provider.
    """
    messages = session.get_messages()
    if len(messages) < 4:
        return

    # Keep last ~30% intact, compact the rest
    keep_count = max(1, int(len(messages) * 0.3))
    split_point = len(messages) - keep_count

    if split_point < 1:
        return

    old_messages = messages[:split_point]

    # Build summarization request
    summary_messages = [
        {"role": "system", "content": _SUMMARIZATION_PROMPT},
        {"role": "user", "content": _format_messages_for_summary(old_messages)},
    ]

    # Use utility provider if available
    llm = utility_provider if utility_provider is not None else provider
    response = await llm.complete(summary_messages)

    summary_text = response.text or "Summary of prior conversation."

    compaction_message = {
        "role": "system",
        "content": summary_text,
        "_compaction": True,
        "source": "management",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    session._compact(compaction_message, split_point)


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
        content = msg.get("content", "")
        if content:
            parts.append(f"[{role}]: {content[:2000]}")
    return "\n".join(parts)
