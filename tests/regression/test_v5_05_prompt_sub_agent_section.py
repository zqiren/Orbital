# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-V5-05: PromptBuilder sub-agent awareness section."""

import tempfile

import pytest

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext


def _make_context(**overrides) -> PromptContext:
    """Create a minimal PromptContext for testing."""
    defaults = dict(
        workspace=tempfile.gettempdir(),
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="windows",
        datetime_now="2026-03-10T10:00:00Z",
        project_name="test-project",
        project_id="proj1",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


ENABLED = [{"handle": "claude-code", "display_name": "Claude Code", "type": "cli"}]


class TestSubAgentAwarenessSection:
    """Spec 139 B: the coordination rules are gated on the project's ENABLED
    agents (fixed for the session) and never carry live state; the live state
    is one line in the per-call tail (``truly_dynamic``)."""

    def test_section_included_when_sub_agents_enabled(self):
        """Section is in the system region as soon as an agent is enabled —
        before anything has been dispatched."""
        ctx = _make_context(enabled_agents=ENABLED, active_sub_agents=[])
        _, semi_stable, _ = PromptBuilder().build(ctx)

        assert "### Dispatching" in semi_stable

    def test_section_omitted_when_no_sub_agents_enabled(self):
        """No enabled agents → no section, whatever happens to be running."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
        ])
        _, semi_stable, _ = PromptBuilder().build(ctx)

        assert "### Dispatching" not in semi_stable

    def test_section_bytes_do_not_follow_live_state(self):
        """Dispatch and report-back must not rewrite the system region."""
        builder = PromptBuilder()
        semis = []
        for active in ([], [{"handle": "claude-code", "status": "running"}],
                       [{"handle": "claude-code", "status": "idle"}]):
            cached, semi_stable, _ = builder.build(
                _make_context(enabled_agents=ENABLED, active_sub_agents=active))
            semis.append((cached, semi_stable))
        assert semis[0] == semis[1] == semis[2]
        assert "Current Sub-Agent States" not in semis[0][1]

    def test_section_describes_nonblocking_model(self):
        """Section describes the non-blocking dispatch model."""
        ctx = _make_context(enabled_agents=ENABLED)
        _, semi_stable, _ = PromptBuilder().build(ctx)

        # Wording updated by 16f9f42 (yield_turn dispatch + push-based
        # sub-agent resume): "does NOT wait" became "ENDS YOUR TURN" /
        # "AUTOMATICALLY RESUMED" — the non-blocking model is now described as
        # turn-ending dispatch with push-based resume, not polling-free waiting.
        # 1109 R5: one sentence; agent_message send sets yield_turn and the
        # loop ends the turn, so the prompt no longer shouts it.
        assert "It ends your turn" in semi_stable
        assert "resumed automatically" in semi_stable

    def test_state_line_in_the_tail_shows_multiple_agents(self):
        """Live states ride the tail as one line: handle + status."""
        ctx = _make_context(enabled_agents=ENABLED, active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
            {"handle": "aider", "status": "idle"},
        ])
        _, semi_stable, dynamic = PromptBuilder().build(ctx)

        assert "Sub-agents: claude-code running, aider idle" in dynamic.split("\n")
        assert "aider" not in semi_stable

    def test_state_line_omitted_when_nothing_is_active(self):
        ctx = _make_context(enabled_agents=ENABLED, active_sub_agents=[])
        _, _, dynamic = PromptBuilder().build(ctx)

        assert "Sub-agents:" not in dynamic

    def test_state_line_carries_no_activity_or_timestamps(self):
        """Only handle + status — free text and times would churn the tail."""
        ctx = _make_context(enabled_agents=ENABLED, active_sub_agents=[
            {"handle": "claude-code", "status": "running",
             "last_activity": "refactoring auth", "started_at": "2026-10-27T10:00:00"},
        ])
        cached, semi_stable, dynamic = PromptBuilder().build(ctx)

        assert "Sub-agents: claude-code running" in dynamic.split("\n")
        for part in (cached, semi_stable, dynamic):
            assert "refactoring auth" not in part
            assert "2026-10-27T10:00:00" not in part

    def test_prompt_context_defaults_empty_list(self):
        """PromptContext.active_sub_agents defaults to empty list."""
        ctx = _make_context()
        assert ctx.active_sub_agents == []

    def test_section_mentions_transcript_reading(self):
        """Section tells the agent how to read transcripts."""
        ctx = _make_context(enabled_agents=ENABLED)
        _, semi_stable, _ = PromptBuilder().build(ctx)

        assert "transcript" in semi_stable.lower()

    def test_section_mentions_system_messages(self):
        """Section mentions [Sub-agent] message notifications."""
        ctx = _make_context(enabled_agents=ENABLED)
        _, semi_stable, _ = PromptBuilder().build(ctx)

        assert "[Sub-agent]" in semi_stable
