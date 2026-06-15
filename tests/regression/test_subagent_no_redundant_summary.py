# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: the management agent must not re-summarize a sub-agent's output.

Contract (small-bug: "management agent re-produces the sub-agent's summary"):
    After the subagent-last-message-display feature, a sub-agent's full final
    message is already shown to the user in chat as its own bubble. The
    management agent therefore must NOT restate / re-summarize it. The guidance
    lives on two channels:

    1. The completion marker injected into the management session
       (``LifecycleObserver.on_completed``) — the freshest context when the
       management loop resumes — tells it the summary is already user-visible
       and to add value instead of echoing. When the sub-agent produced NO
       final message (nothing shown to the user), the marker honestly says so
       and asks the agent to report the outcome itself.

    2. The system prompt (``PromptBuilder`` sub-agent sections) reinforces the
       same rule and reframes "verify" as an internal check, not a cue to write
       a user-facing summary.
"""

from __future__ import annotations

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver


# ---------------------------------------------------------------------------
# Channel 1: the completion marker injected into the management session
# ---------------------------------------------------------------------------


def _observer():
    mock_agent_mgr = MagicMock()
    mock_agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
    ws = MagicMock()
    observer = LifecycleObserver(agent_manager=mock_agent_mgr, ws_manager=ws)
    return observer, mock_agent_mgr


@pytest.mark.asyncio
async def test_completion_marker_says_summary_is_user_visible_and_not_to_repeat():
    """A real summary → marker tells the agent it's already shown to the user
    and not to repeat it, while preserving the existing handle/summary/path."""
    observer, mgr = _observer()

    await observer.on_completed(
        "proj", "claude-code", "Found 4 products that listen live.",
        "/tmp/t.jsonl", session_id="sess",
    )

    content = mgr.inject_system_message.await_args.args[1]
    # Existing format preserved (other tests/consumers rely on it).
    assert "claude-code completed" in content
    assert "Found 4 products that listen live." in content
    assert "/tmp/t.jsonl" in content
    # New guidance: already user-visible, do not repeat / re-summarize.
    low = content.lower()
    assert "already" in low
    assert "re-summarize" in low
    assert "do not repeat" in low
    assert "user" in low


@pytest.mark.asyncio
async def test_completion_marker_no_output_asks_agent_to_report_itself():
    """No final message → nothing was shown to the user, so the marker must NOT
    claim it's already visible; it asks the agent to report the outcome."""
    observer, mgr = _observer()

    await observer.on_completed(
        "proj", "claude-code", "", "/tmp/t.jsonl", session_id="sess",
    )

    content = mgr.inject_system_message.await_args.args[1]
    low = content.lower()
    assert "(no output)" in content
    assert "no final message" in low
    assert "tell the user" in low
    # Honesty: must not falsely claim the (absent) summary is already shown.
    assert "already" not in low


# ---------------------------------------------------------------------------
# Channel 2: the system prompt reinforces the rule
# ---------------------------------------------------------------------------


def _ctx(**overrides) -> PromptContext:
    defaults = dict(
        workspace=tempfile.gettempdir(),
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="windows",
        datetime_now="2026-06-15T10:00:00Z",
        project_name="test-project",
        project_id="proj1",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


def test_coordination_section_forbids_re_summarizing():
    """The Sub-Agent Coordination section tells the agent not to re-summarize a
    sub-agent's output because the user already sees it."""
    ctx = _ctx(active_sub_agents=[{"handle": "claude-code", "status": "running"}])
    _, semi_stable, _ = PromptBuilder().build(ctx)

    low = semi_stable.lower()
    assert "re-summarize" in low
    assert "already" in low
    # Existing non-blocking-model contract must still hold.
    assert "AUTOMATICALLY RESUMED" in semi_stable


def test_verification_subsection_is_framed_as_internal_not_a_summary():
    """The 'Verifying Sub-Agent Output' subsection reframes verification as an
    internal check, decoupled from writing a user-facing summary."""
    ctx = _ctx(enabled_agents=[
        {"handle": "claudecode", "display_name": "Claude Code", "type": "cli"},
    ])
    _, semi_stable, _ = PromptBuilder().build(ctx)

    tail = semi_stable.split("Verifying Sub-Agent Output")[1]
    low = tail.lower()
    assert "internal" in low
    assert "already see" in low  # matches "already sees" / "already see"
    # Existing verification guidance (read/shell) must survive.
    assert "read" in low
    assert "shell" in low
