# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for PromptBuilder settings-related features: identity, global preferences,
standing rules, scratch memory, artifact instructions, memory management."""

import os
import pytest

from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy


def _make_context(tmp_path, **overrides) -> PromptContext:
    """Create a minimal PromptContext for testing."""
    defaults = dict(
        workspace=str(tmp_path),
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-02-24T00:00:00",
        project_name="TestProject",
        project_instructions="",
    )
    defaults.update(overrides)
    return PromptContext(**defaults)


class TestIdentityUsesAgentName:
    def test_identity_uses_agent_name(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, agent_name="Archie", project_name="MyProject")
        cached, _ = builder.build(ctx)
        assert "You are Archie" in cached
        assert "MyProject project" in cached

    def test_identity_falls_back_to_project_name(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, agent_name="", project_name="FallbackProject")
        cached, _ = builder.build(ctx)
        assert "You are FallbackProject" in cached

    def test_identity_falls_back_to_agent(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, agent_name="", project_name="")
        cached, _ = builder.build(ctx)
        assert "You are Agent" in cached


class TestGlobalPreferences:
    def test_global_preferences_included_when_file_exists(self, tmp_path):
        prefs_path = tmp_path / "user_preferences.md"
        prefs_path.write_text("Always use type hints\nPrefer pytest over unittest")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, global_preferences_path=str(prefs_path))
        _, dynamic = builder.build(ctx)
        assert "Global User Preferences" in dynamic
        assert "Always use type hints" in dynamic

    def test_global_preferences_absent_when_no_path(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, global_preferences_path="")
        _, dynamic = builder.build(ctx)
        assert "Global User Preferences" not in dynamic

    def test_global_preferences_absent_when_file_missing(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, global_preferences_path=str(tmp_path / "nonexistent.md"))
        _, dynamic = builder.build(ctx)
        assert "Global User Preferences" not in dynamic


class TestStandingRules:
    def test_standing_rules_included_when_file_exists(self, tmp_path):
        rules_dir = tmp_path / ".agent-os" / "instructions"
        rules_dir.mkdir(parents=True)
        (rules_dir / "user_directives.md").write_text("Never commit to main directly")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        _, dynamic = builder.build(ctx)
        assert "Project Instructions" in dynamic
        assert "Never commit to main directly" in dynamic

    def test_standing_rules_absent_when_no_file(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        _, dynamic = builder.build(ctx)
        assert "Project Instructions" not in dynamic


class TestScratchMemoryVariant:
    def test_scratch_mode_gives_lightweight_memory(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        _, dynamic = builder.build(ctx)
        assert "quick questions" in dynamic
        assert "PROJECT_STATE.md" not in dynamic

    def test_non_scratch_mode_gives_full_memory(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, dynamic = builder.build(ctx)
        assert "PROJECT_STATE.md" in dynamic
        assert "DECISIONS.md" in dynamic


class TestArtifactInstruction:
    def test_agent_output_folder_mentioned_in_non_scratch(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, dynamic = builder.build(ctx)
        assert "agent_output/" in dynamic

    def test_agent_output_folder_not_in_scratch(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        _, dynamic = builder.build(ctx)
        assert "agent_output/" not in dynamic


class TestMemoryManagementInstructions:
    def test_remember_instruction_present(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, dynamic = builder.build(ctx)
        assert 'remember X' in dynamic
        assert "user_directives.md" in dynamic

    def test_forget_instruction_present(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, dynamic = builder.build(ctx)
        assert 'forget X' in dynamic

    def test_global_preferences_path_in_memory(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        prefs_path = str(tmp_path / "prefs.md")
        ctx = _make_context(tmp_path, is_scratch=False, global_preferences_path=prefs_path)
        _, dynamic = builder.build(ctx)
        assert prefs_path in dynamic

    def test_default_preferences_path_when_not_set(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False, global_preferences_path="")
        _, dynamic = builder.build(ctx)
        assert "~/.agent-os/user_preferences.md" in dynamic


class TestAutonomyDirective:
    """Test A: Autonomy directive appears in cached prefix for each level."""

    def test_hands_off_directive_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, autonomy=Autonomy.HANDS_OFF)
        cached, _ = builder.build(ctx)
        lower = cached.lower()
        assert "act immediately" in lower or "autonomous" in lower

    def test_check_in_directive_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, autonomy=Autonomy.CHECK_IN)
        cached, _ = builder.build(ctx)
        lower = cached.lower()
        assert "briefly state" in lower or "check-in" in lower

    def test_supervised_directive_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, autonomy=Autonomy.SUPERVISED)
        cached, _ = builder.build(ctx)
        lower = cached.lower()
        assert "wait for" in lower or "supervised" in lower or "confirmation" in lower


class TestAntiOverConfirmation:
    """Test B: Scratch project gets anti-over-confirmation directive."""

    def test_scratch_has_anti_over_confirmation(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        _, dynamic = builder.build(ctx)
        lower = dynamic.lower()
        assert "never present numbered" in lower or "option" in lower

    def test_non_scratch_no_anti_over_confirmation(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, dynamic = builder.build(ctx)
        lower = dynamic.lower()
        assert "never present numbered" not in lower


class TestAutonomyLevelsDiffer:
    """Test C: Different autonomy levels produce different directive text."""

    def test_all_three_levels_produce_different_text(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        texts = {}
        for level in Autonomy:
            ctx = _make_context(tmp_path, autonomy=level)
            cached, _ = builder.build(ctx)
            texts[level] = cached
        assert texts[Autonomy.HANDS_OFF] != texts[Autonomy.CHECK_IN]
        assert texts[Autonomy.CHECK_IN] != texts[Autonomy.SUPERVISED]
        assert texts[Autonomy.HANDS_OFF] != texts[Autonomy.SUPERVISED]


class TestScratchIdentityTweak:
    """Scratch agents get action-biased identity framing."""

    def test_scratch_identity_mentions_quick_action(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        cached, _ = builder.build(ctx)
        lower = cached.lower()
        assert "quick-action" in lower or "concise" in lower or "act immediately" in lower

    def test_non_scratch_identity_is_methodical(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        cached, _ = builder.build(ctx)
        assert "methodical" in cached.lower()
