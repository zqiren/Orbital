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
        cached, _, _ = builder.build(ctx)
        assert "You are Archie" in cached
        assert "MyProject project" in cached

    def test_identity_falls_back_to_project_name(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, agent_name="", project_name="FallbackProject")
        cached, _, _ = builder.build(ctx)
        assert "You are FallbackProject" in cached

    def test_identity_falls_back_to_agent(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, agent_name="", project_name="")
        cached, _, _ = builder.build(ctx)
        assert "You are Agent" in cached


class TestGlobalPreferences:
    def test_global_preferences_included_when_file_exists(self, tmp_path):
        prefs_path = tmp_path / "user_preferences.md"
        prefs_path.write_text("Always use type hints\nPrefer pytest over unittest")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, global_preferences_path=str(prefs_path))
        _, semi_stable, _ = builder.build(ctx)
        assert "Global User Preferences" in semi_stable
        assert "Always use type hints" in semi_stable

    def test_global_preferences_absent_when_no_path(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, global_preferences_path="")
        _, semi_stable, _ = builder.build(ctx)
        assert "Global User Preferences" not in semi_stable

    def test_global_preferences_absent_when_file_missing(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, global_preferences_path=str(tmp_path / "nonexistent.md"))
        _, semi_stable, _ = builder.build(ctx)
        assert "Global User Preferences" not in semi_stable


class TestStandingRules:
    def test_standing_rules_included_when_file_exists(self, tmp_path):
        rules_dir = tmp_path / "orbital" / "instructions"
        rules_dir.mkdir(parents=True)
        (rules_dir / "user_directives.md").write_text("Never commit to main directly")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        _, semi_stable, _ = builder.build(ctx)
        assert "Project Instructions" in semi_stable
        assert "Never commit to main directly" in semi_stable

    def test_standing_rules_absent_when_no_file(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        _, semi_stable, _ = builder.build(ctx)
        assert "Project Instructions" not in semi_stable


class TestScratchMemoryVariant:
    def test_scratch_mode_gives_lightweight_memory(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        _, semi_stable, _ = builder.build(ctx)
        assert "quick questions" in semi_stable
        assert "PROJECT_STATE.md" not in semi_stable

    def test_non_scratch_mode_gives_full_memory(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert "PROJECT_STATE.md" in semi_stable
        assert "DECISIONS.md" in semi_stable


class TestArtifactInstruction:
    def test_deliverables_go_outside_orbital_in_non_scratch(self, tmp_path):
        # Contract: user-facing deliverables go anywhere in the workspace OUTSIDE
        # orbital/ so they survive a project reset. orbital/ is system state and
        # is wiped on delete (TASK-05). Tool outputs (system-managed) go under
        # orbital/output/ — but the agent does not write deliverables there.
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        # The prompt must instruct the agent NOT to put user-facing deliverables under orbital/.
        assert "DO NOT place user-facing deliverables under orbital/" in semi_stable
        # The prompt must reference the tool-output directory so the agent knows it exists.
        assert "orbital/output/" in semi_stable

    def test_deliverable_instruction_absent_in_scratch(self, tmp_path):
        # Scratch mode uses a lightweight memory section that does not include
        # the deliverable-placement guidance.
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        _, semi_stable, _ = builder.build(ctx)
        assert "DO NOT place user-facing deliverables under orbital/" not in semi_stable
        assert "orbital/output/" not in semi_stable


class TestMemoryManagementInstructions:
    def test_remember_instruction_present(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert 'remember X' in semi_stable
        assert "user_directives.md" in semi_stable

    def test_forget_instruction_present(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert 'forget X' in semi_stable

    def test_global_preferences_path_in_memory(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        prefs_path = str(tmp_path / "prefs.md")
        ctx = _make_context(tmp_path, is_scratch=False, global_preferences_path=prefs_path)
        _, semi_stable, _ = builder.build(ctx)
        assert prefs_path in semi_stable

    def test_default_preferences_path_when_not_set(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False, global_preferences_path="")
        _, semi_stable, _ = builder.build(ctx)
        assert "~/orbital/user_preferences.md" in semi_stable


class TestAutonomyDirective:
    """Test A: Autonomy directive appears in cached prefix for each level."""

    def test_hands_off_directive_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, autonomy=Autonomy.HANDS_OFF)
        cached, _, _ = builder.build(ctx)
        lower = cached.lower()
        assert "act immediately" in lower or "autonomous" in lower

    def test_check_in_directive_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, autonomy=Autonomy.CHECK_IN)
        cached, _, _ = builder.build(ctx)
        lower = cached.lower()
        assert "briefly state" in lower or "check-in" in lower

    def test_supervised_directive_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, autonomy=Autonomy.SUPERVISED)
        cached, _, _ = builder.build(ctx)
        lower = cached.lower()
        assert "wait for" in lower or "supervised" in lower or "confirmation" in lower


class TestAntiOverConfirmation:
    """Test B: Scratch project gets anti-over-confirmation directive."""

    def test_scratch_has_anti_over_confirmation(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        _, semi_stable, _ = builder.build(ctx)
        lower = semi_stable.lower()
        assert "never present numbered" in lower or "option" in lower

    def test_non_scratch_no_anti_over_confirmation(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        lower = semi_stable.lower()
        assert "never present numbered" not in lower


class TestAutonomyLevelsDiffer:
    """Test C: Different autonomy levels produce different directive text."""

    def test_all_three_levels_produce_different_text(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        texts = {}
        for level in Autonomy:
            ctx = _make_context(tmp_path, autonomy=level)
            cached, _, _ = builder.build(ctx)
            texts[level] = cached
        assert texts[Autonomy.HANDS_OFF] != texts[Autonomy.CHECK_IN]
        assert texts[Autonomy.CHECK_IN] != texts[Autonomy.SUPERVISED]
        assert texts[Autonomy.HANDS_OFF] != texts[Autonomy.SUPERVISED]


class TestScratchIdentityTweak:
    """Scratch agents get action-biased identity framing."""

    def test_scratch_identity_mentions_quick_action(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=True)
        cached, _, _ = builder.build(ctx)
        lower = cached.lower()
        assert "quick-action" in lower or "concise" in lower or "act immediately" in lower

    def test_non_scratch_identity_is_methodical(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        cached, _, _ = builder.build(ctx)
        assert "methodical" in cached.lower()
