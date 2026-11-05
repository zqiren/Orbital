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


class TestUserMemorySection:
    """Spec 073 §5.2/§10 — the "## About the User" section: semi_stable only,
    immediately after Global User Preferences, absent whenever there is
    nothing to inject."""

    def _write_memory(self, tmp_path) -> str:
        path = tmp_path / "user_memory.md"
        path.write_text(
            "- Works as a PM at Tencent <!--from:proj-a 2026-08-24-->\n",
            encoding="utf-8")
        return str(path)

    def test_renders_when_file_exists(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, user_memory_path=self._write_memory(tmp_path))
        _, semi_stable, _ = builder.build(ctx)
        assert "## About the User" in semi_stable
        assert "Works as a PM at Tencent" in semi_stable

    def test_absent_when_path_empty(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, user_memory_path="")
        _, semi_stable, _ = builder.build(ctx)
        assert "## About the User" not in semi_stable

    def test_absent_when_file_missing(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(
            tmp_path, user_memory_path=str(tmp_path / "nonexistent.md"))
        _, semi_stable, _ = builder.build(ctx)
        assert "## About the User" not in semi_stable

    def test_absent_when_toggle_off(self, tmp_path):
        # The toggle reaches the prompt layer as an empty path — the config
        # builder (_build_agent_config_from_project) is the single gate and
        # leaves user_memory_path "" when user_memory_enabled is False, even
        # though the file exists on disk.
        memory_path = self._write_memory(tmp_path)
        assert os.path.exists(memory_path)
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, user_memory_path="")
        _, semi_stable, _ = builder.build(ctx)
        assert "## About the User" not in semi_stable

    def test_lands_in_semi_stable_never_cached_or_dynamic(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, user_memory_path=self._write_memory(tmp_path))
        cached, semi_stable, truly_dynamic = builder.build(ctx)
        assert "## About the User" in semi_stable
        assert "## About the User" not in cached
        assert "## About the User" not in truly_dynamic

    def test_ordered_immediately_after_global_preferences(self, tmp_path):
        prefs_path = tmp_path / "prefs.md"
        prefs_path.write_text("Prefer concise replies", encoding="utf-8")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(
            tmp_path,
            global_preferences_path=str(prefs_path),
            user_memory_path=self._write_memory(tmp_path))
        _, semi_stable, _ = builder.build(ctx)
        prefs_at = semi_stable.index("## Global User Preferences")
        memory_at = semi_stable.index("## About the User")
        assert prefs_at < memory_at


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

    # Spec 073: the old "append to ~/orbital/user_preferences.md" instruction
    # was impossible (write/edit and the sandbox are workspace-scoped), so the
    # global fork of the routing now points at the daemon-side tool — or at
    # Settings when the user-memory toggle is off (empty path = no tool).
    def test_global_preference_routes_to_remember_tool(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False,
                            user_memory_path=str(tmp_path / "user_memory.md"))
        _, semi_stable, _ = builder.build(ctx)
        assert "call remember_about_user" in semi_stable
        assert "append to ~/orbital/user_preferences.md" not in semi_stable

    def test_agent_routing_preference_is_a_global_preference(self, tmp_path):
        """Brief 1105-route: "以后代码调查用 claude code，轻任务用 codex" is a
        user-level preference (the Workbench's 自动 reads user memory), not a
        project directive."""
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False,
                            user_memory_path=str(tmp_path / "user_memory.md"))
        _, semi_stable, _ = builder.build(ctx)
        line = next(l for l in semi_stable.splitlines()
                    if "call remember_about_user" in l)
        assert "which agent" in line and "Claude Code" in line

    def test_global_preference_routes_to_settings_when_disabled(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False, user_memory_path="")
        _, semi_stable, _ = builder.build(ctx)
        assert "call remember_about_user" not in semi_stable
        assert "Global Settings" in semi_stable
        assert "append to ~/orbital/user_preferences.md" not in semi_stable

    def test_project_directive_fork_kept_verbatim(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert f"{tmp_path}/orbital/instructions/user_directives.md" in semi_stable
        assert ('ask: "Should this apply to just this project or all your '
                'projects?"') in semi_stable


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


class TestChatReplyPathLinks:
    """Spec 002: chat replies must reference workspace files as full
    workspace-relative markdown links so the UI can render cards/chips."""

    def test_chat_reply_link_convention_in_cached_prefix(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        cached, _, _ = builder.build(ctx)
        assert "PATHS IN CHAT REPLIES" in cached
        assert "[the file's title](path/from/workspace/root.md)" in cached
        # The two observed failure modes must be called out explicitly.
        assert "never abbreviate" in cached
        assert "bare filename" in cached


class TestMemoryFormatHeaderPointer:
    """Layer-1 format contracts: the prompt points at the in-file headers
    (which carry the detail) and adds the INDEX tripwire."""

    def test_memory_section_references_format_headers(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        _, semi_stable, _ = builder.build(ctx)
        assert "<!--format" in semi_stable
        assert "restores it if removed" in semi_stable

    def test_memory_section_has_index_tripwire(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path)
        _, semi_stable, _ = builder.build(ctx)
        assert "date, status, or decision into INDEX.md" in semi_stable


_CODEX = [{"handle": "codex", "display_name": "Codex", "type": "cli"}]


class TestWhoDoesWhat:
    """Owner 2026-11-05: the manager must decide WHO does substantive work
    before doing it itself, from the user's own who-does-what sources."""

    def test_rule_present_when_agents_enabled(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, enabled_agents=_CODEX)
        _, semi_stable, _ = builder.build(ctx)
        assert "### Who does what" in semi_stable
        assert "Preferences in project_goals.md" in semi_stable
        assert '"About the User"' in semi_stable
        flat = " ".join(semi_stable.split())
        assert 'Don\'t start the work yourself "to get it going" first.' in flat

    def test_rule_absent_when_no_agents(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, enabled_agents=[])
        full = "".join(builder.build(ctx))
        assert "Who does what" not in full

    def test_rule_is_static_never_cached_prefix_or_dynamic(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, enabled_agents=_CODEX)
        cached, semi_stable, dynamic = builder.build(ctx)
        assert "Who does what" not in cached
        assert "Who does what" not in dynamic
        assert semi_stable.count("### Who does what") == 1

    def test_rule_sits_after_deployment_instructions_before_dispatch(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, enabled_agents=_CODEX,
                            sub_agent_deployment_instructions="Codex writes code.")
        _, semi_stable, _ = builder.build(ctx)
        deploy = semi_stable.index("</sub_agent_deployment_instructions>")
        rule = semi_stable.index("### Who does what")
        dispatch = semi_stable.index("To interact with sub-agents")
        assert deploy < rule < dispatch
        # The verify-output text is kept as is.
        assert "### Verifying Sub-Agent Output" in semi_stable

    def test_rule_bytes_do_not_move_between_turns(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        a = builder.build(_make_context(tmp_path, enabled_agents=_CODEX,
                                        datetime_now="2026-11-05T10:00:00"))
        b = builder.build(_make_context(tmp_path, enabled_agents=_CODEX,
                                        datetime_now="2026-11-05T11:30:00"))
        assert a[1] == b[1]

    def test_recording_routes_to_settings_when_user_memory_off(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, enabled_agents=_CODEX, user_memory_path="")
        _, semi_stable, _ = builder.build(ctx)
        rule = semi_stable.split("### Who does what", 1)[1].split("To interact", 1)[0]
        assert "remember_about_user" not in rule
        assert "About You" in rule

    def test_recording_uses_remember_tool_when_user_memory_on(self, tmp_path):
        mem = tmp_path / "user_memory.md"
        mem.write_text("likes tea")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = _make_context(tmp_path, enabled_agents=_CODEX,
                            user_memory_path=str(mem))
        _, semi_stable, _ = builder.build(ctx)
        rule = semi_stable.split("### Who does what", 1)[1].split("To interact", 1)[0]
        assert "remember_about_user only when they state it as a personal habit" in rule


class TestAskQuestionCap:
    def test_asks_caps_the_question_at_about_20_words(self, tmp_path):
        builder = PromptBuilder(workspace=str(tmp_path))
        _, semi_stable, _ = builder.build(_make_context(tmp_path))
        flat = " ".join(semi_stable.split())
        assert "one sentence of at most about 20 words" in flat
        assert ("dates, prices, names, numbers and ①②③ points go on lines under it "
                "indented by two spaces") in flat
        assert "never in the question" in flat
        assert "`options: A | B` (2-4 short options)" in flat
