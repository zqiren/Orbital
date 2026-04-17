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
        project_dir_name="test-project_proj1",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


class TestSubAgentAwarenessSection:
    def test_section_included_when_sub_agents_active(self):
        """Section appears in dynamic suffix when sub-agents are active."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
        ])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "Sub-Agent Coordination" in semi_stable
        assert "claude-code" in semi_stable
        assert "running" in semi_stable

    def test_section_omitted_when_no_sub_agents(self):
        """Section is absent when no sub-agents are active."""
        ctx = _make_context(active_sub_agents=[])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "Sub-Agent Coordination" not in semi_stable

    def test_section_describes_nonblocking_model(self):
        """Section describes the non-blocking dispatch model."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
        ])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "returns IMMEDIATELY" in semi_stable
        assert "does NOT wait" in semi_stable

    def test_section_shows_multiple_agents(self):
        """Multiple sub-agents listed with correct states."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
            {"handle": "aider", "status": "idle"},
        ])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "claude-code" in semi_stable
        assert "running" in semi_stable
        assert "aider" in semi_stable
        assert "idle" in semi_stable

    def test_section_includes_last_activity(self):
        """Last activity info is shown when present."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running", "last_activity": "refactoring auth"},
        ])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "refactoring auth" in semi_stable

    def test_prompt_context_defaults_empty_list(self):
        """PromptContext.active_sub_agents defaults to empty list."""
        ctx = _make_context()
        assert ctx.active_sub_agents == []

    def test_section_mentions_transcript_reading(self):
        """Section tells the agent how to read transcripts."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
        ])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "transcript" in semi_stable.lower()

    def test_section_mentions_system_messages(self):
        """Section mentions [Sub-agent] system message notifications."""
        ctx = _make_context(active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
        ])
        builder = PromptBuilder()
        _, semi_stable, _ = builder.build(ctx)

        assert "[Sub-agent]" in semi_stable
