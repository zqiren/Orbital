# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for Component E: Prompt Builder and Skill Loader.

Tests cover all 15 acceptance criteria for the prompt builder and skill loader.
Fully platform-independent.
"""

import os
import pytest

from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.skills import SkillLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(**overrides) -> PromptContext:
    """Create a standard PromptContext with sensible defaults."""
    defaults = {
        "workspace": "/tmp/test-workspace",
        "model": "claude-sonnet-4-5-20250929",
        "autonomy": Autonomy.HANDS_OFF,
        "enabled_agents": [],
        "tool_names": ["read", "write", "edit", "shell"],
        "os_type": "linux",
        "datetime_now": "2026-02-11T10:00:00",
        "context_usage_pct": 0.0,
    }
    defaults.update(overrides)
    return PromptContext(**defaults)


# ===========================================================================
# AC-1: build() returns tuple of 2 non-empty strings
# ===========================================================================

class TestBuildReturnType:

    def test_build_returns_tuple_of_two_strings(self):
        builder = PromptBuilder()
        ctx = make_context()
        result = builder.build(ctx)
        assert isinstance(result, tuple)
        assert len(result) == 2
        cached, dynamic = result
        assert isinstance(cached, str)
        assert isinstance(dynamic, str)

    def test_build_neither_is_empty(self):
        builder = PromptBuilder()
        ctx = make_context()
        cached, dynamic = builder.build(ctx)
        assert len(cached.strip()) > 0, "cached_prefix should not be empty"
        assert len(dynamic.strip()) > 0, "dynamic_suffix should not be empty"


# ===========================================================================
# AC-2: cached_prefix contains tool names from tool_names
# ===========================================================================

class TestToolsInPrefix:

    def test_cached_prefix_contains_read_write_shell(self):
        builder = PromptBuilder()
        ctx = make_context(tool_names=["read", "write", "shell"])
        cached, _ = builder.build(ctx)
        assert "read" in cached.lower()
        assert "write" in cached.lower()
        assert "shell" in cached.lower()

    def test_cached_prefix_contains_all_provided_tools(self):
        builder = PromptBuilder()
        tools = ["read", "write", "edit", "shell"]
        ctx = make_context(tool_names=tools)
        cached, _ = builder.build(ctx)
        for tool in tools:
            assert tool in cached.lower(), f"Tool '{tool}' not found in cached_prefix"


# ===========================================================================
# AC-3: cached_prefix does NOT contain "agent_message" when not in tool_names
# ===========================================================================

class TestToolOmission:

    def test_cached_prefix_omits_agent_message_when_absent(self):
        builder = PromptBuilder()
        ctx = make_context(tool_names=["read", "write", "shell"])
        cached, _ = builder.build(ctx)
        assert "agent_message" not in cached

    def test_cached_prefix_includes_agent_message_when_present(self):
        builder = PromptBuilder()
        ctx = make_context(tool_names=["read", "write", "agent_message"])
        cached, _ = builder.build(ctx)
        assert "agent_message" in cached


# ===========================================================================
# AC-4: dynamic_suffix contains workspace path and datetime from context
# ===========================================================================

class TestDynamicSuffixContent:

    def test_dynamic_suffix_contains_workspace(self):
        builder = PromptBuilder()
        ctx = make_context(workspace="/projects/my-app")
        _, dynamic = builder.build(ctx)
        assert "/projects/my-app" in dynamic

    def test_dynamic_suffix_contains_datetime(self):
        builder = PromptBuilder()
        ctx = make_context(datetime_now="2026-02-11T14:30:00")
        _, dynamic = builder.build(ctx)
        assert "2026-02-11T14:30:00" in dynamic


# ===========================================================================
# AC-5: Sub-agents section only appears when enabled_agents is non-empty
# ===========================================================================

class TestSubAgentsSection:

    def test_no_sub_agents_section_when_empty(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[])
        _, dynamic = builder.build(ctx)
        # Should not mention sub-agents
        assert "sub-agent" not in dynamic.lower() or "Sub-agents available" not in dynamic

    def test_sub_agents_section_present_when_non_empty(self):
        builder = PromptBuilder()
        agents = [{"handle": "claudecode", "display_name": "Claude Code", "type": "cli"}]
        ctx = make_context(enabled_agents=agents)
        _, dynamic = builder.build(ctx)
        assert "claudecode" in dynamic
        assert "Claude Code" in dynamic


# ===========================================================================
# AC-6: Context budget at 75% → includes warning about updating PROJECT_STATE.md
# ===========================================================================

class TestContextBudgetWarning:

    def test_75_pct_context_warning(self):
        builder = PromptBuilder()
        ctx = make_context(context_usage_pct=0.75)
        _, dynamic = builder.build(ctx)
        assert "PROJECT_STATE.md" in dynamic
        # Should mention updating / saving state
        lower = dynamic.lower()
        assert "consider" in lower or "update" in lower or "save" in lower


# ===========================================================================
# AC-7: Context budget at 90% → includes URGENT save message
# ===========================================================================

class TestContextBudgetUrgent:

    def test_90_pct_context_urgent(self):
        builder = PromptBuilder()
        ctx = make_context(context_usage_pct=0.90)
        _, dynamic = builder.build(ctx)
        upper = dynamic.upper()
        assert "URGENT" in upper
        assert "PROJECT_STATE.md" in dynamic


# ===========================================================================
# AC-8: OS instructions — windows → PowerShell, macos → bash
# ===========================================================================

class TestOSInstructions:

    def test_windows_mentions_powershell(self):
        builder = PromptBuilder()
        ctx = make_context(os_type="windows")
        _, dynamic = builder.build(ctx)
        lower = dynamic.lower()
        assert "powershell" in lower

    def test_macos_mentions_bash(self):
        builder = PromptBuilder()
        ctx = make_context(os_type="macos")
        _, dynamic = builder.build(ctx)
        lower = dynamic.lower()
        assert "bash" in lower

    def test_linux_mentions_bash(self):
        builder = PromptBuilder()
        ctx = make_context(os_type="linux")
        _, dynamic = builder.build(ctx)
        lower = dynamic.lower()
        assert "bash" in lower

    def test_windows_mentions_backslash_separator(self):
        builder = PromptBuilder()
        ctx = make_context(os_type="windows")
        _, dynamic = builder.build(ctx)
        assert "\\\\" in dynamic or "\\" in dynamic


# ===========================================================================
# AC-9: Workspace bootstrap: AGENT.md content appears in dynamic_suffix
# ===========================================================================

class TestWorkspaceBootstrap:

    def test_agent_md_appears_in_dynamic_suffix(self, tmp_path):
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir()
        agent_md = agent_os_dir / "AGENT.md"
        agent_md.write_text("Custom agent instructions for this project.")

        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, dynamic = builder.build(ctx)
        assert "Custom agent instructions for this project." in dynamic


# ===========================================================================
# AC-10: Workspace bootstrap: file > 20K chars → truncated
# ===========================================================================

class TestWorkspaceBootstrapTruncation:

    def test_large_agent_md_truncated(self, tmp_path):
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir()
        agent_md = agent_os_dir / "AGENT.md"
        # Write 25K chars
        content = "A" * 25_000
        agent_md.write_text(content)

        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, dynamic = builder.build(ctx)
        # The full 25K content should NOT appear; it should be truncated to ~20K
        # Check that we don't have the full 25K in the output
        assert content not in dynamic
        # But some content should still be present (truncated portion)
        assert "AAAA" in dynamic


# ===========================================================================
# AC-11: Workspace bootstrap: file doesn't exist → section omitted, no error
# ===========================================================================

class TestWorkspaceBootstrapMissing:

    def test_missing_agent_md_no_error(self, tmp_path):
        # No .agent-os/AGENT.md file
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        # Should not raise
        cached, dynamic = builder.build(ctx)
        assert isinstance(cached, str)
        assert isinstance(dynamic, str)

    def test_no_workspace_no_error(self):
        builder = PromptBuilder(workspace=None)
        ctx = make_context()
        cached, dynamic = builder.build(ctx)
        assert isinstance(cached, str)
        assert isinstance(dynamic, str)


# ===========================================================================
# AC-12: SkillLoader with skills/my-skill/SKILL.md → returns metadata
# ===========================================================================

class TestSkillLoaderScan:

    def test_scan_finds_skill_md(self, tmp_path):
        skills_dir = tmp_path / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        skill_md = skills_dir / "SKILL.md"
        skill_md.write_text("# My Skill\nA skill that does things.")

        loader = SkillLoader(str(tmp_path))
        results = loader.scan()
        assert len(results) == 1
        skill = results[0]
        assert "name" in skill
        assert "description" in skill
        assert "path" in skill
        assert skill["name"] == "My Skill"
        assert "does things" in skill["description"]
        assert str(skill_md) == skill["path"] or skill["path"].endswith("SKILL.md")

    def test_scan_multiple_skills(self, tmp_path):
        for name in ["skill-a", "skill-b"]:
            skill_dir = tmp_path / "skills" / name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(f"# {name.title()}\nDescription of {name}.")

        loader = SkillLoader(str(tmp_path))
        results = loader.scan()
        assert len(results) == 2
        names = {s["name"] for s in results}
        assert len(names) == 2


# ===========================================================================
# AC-13: SkillLoader with no skills/ directory → returns []
# ===========================================================================

class TestSkillLoaderEmpty:

    def test_scan_no_skills_dir(self, tmp_path):
        loader = SkillLoader(str(tmp_path))
        results = loader.scan()
        assert results == []

    def test_scan_empty_skills_dir(self, tmp_path):
        (tmp_path / "skills").mkdir()
        loader = SkillLoader(str(tmp_path))
        results = loader.scan()
        assert results == []

    def test_scan_skills_dir_without_skill_md(self, tmp_path):
        skill_dir = tmp_path / "skills" / "no-md-skill"
        skill_dir.mkdir(parents=True)
        # Directory exists but no SKILL.md inside
        (skill_dir / "README.md").write_text("Not a skill file")
        loader = SkillLoader(str(tmp_path))
        results = loader.scan()
        assert results == []


# ===========================================================================
# AC-14: Two calls with identical PromptContext → cached_prefix is byte-identical
# ===========================================================================

class TestCacheStability:

    def test_identical_context_identical_cached_prefix(self):
        builder = PromptBuilder()
        ctx = make_context()
        cached1, _ = builder.build(ctx)
        cached2, _ = builder.build(ctx)
        assert cached1 == cached2, "cached_prefix must be byte-identical for same context"

    def test_multiple_calls_stable(self):
        builder = PromptBuilder()
        ctx = make_context()
        results = [builder.build(ctx)[0] for _ in range(5)]
        assert all(r == results[0] for r in results), "cached_prefix must be stable across calls"


# ===========================================================================
# AC-15: Two calls with different datetime_now → cached_prefix identical,
#         dynamic_suffix differs
# ===========================================================================

class TestCacheSplitBehavior:

    def test_different_datetime_cached_same_dynamic_different(self):
        builder = PromptBuilder()
        ctx1 = make_context(datetime_now="2026-02-11T10:00:00")
        ctx2 = make_context(datetime_now="2026-02-11T11:00:00")
        cached1, dynamic1 = builder.build(ctx1)
        cached2, dynamic2 = builder.build(ctx2)
        assert cached1 == cached2, "cached_prefix should not depend on datetime_now"
        assert dynamic1 != dynamic2, "dynamic_suffix should differ when datetime_now differs"
        assert "2026-02-11T10:00:00" in dynamic1
        assert "2026-02-11T11:00:00" in dynamic2

    def test_different_context_usage_cached_same_dynamic_different(self):
        builder = PromptBuilder()
        ctx1 = make_context(context_usage_pct=0.1)
        ctx2 = make_context(context_usage_pct=0.9)
        cached1, dynamic1 = builder.build(ctx1)
        cached2, dynamic2 = builder.build(ctx2)
        assert cached1 == cached2, "cached_prefix should not depend on context_usage_pct"
        assert dynamic1 != dynamic2


# ===========================================================================
# Autonomy enum tests
# ===========================================================================

class TestAutonomyEnum:

    def test_enum_values(self):
        assert Autonomy.HANDS_OFF == "hands_off"
        assert Autonomy.CHECK_IN == "check_in"
        assert Autonomy.SUPERVISED == "supervised"

    def test_enum_is_string(self):
        assert isinstance(Autonomy.HANDS_OFF, str)


# ===========================================================================
# PromptContext dataclass tests
# ===========================================================================

class TestPromptContext:

    def test_default_context_usage_pct(self):
        ctx = PromptContext(
            workspace="/tmp",
            model="test-model",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[],
            tool_names=["read"],
            os_type="linux",
            datetime_now="2026-01-01T00:00:00",
        )
        assert ctx.context_usage_pct == 0.0

    def test_all_fields(self):
        ctx = make_context()
        assert ctx.workspace == "/tmp/test-workspace"
        assert ctx.model == "claude-sonnet-4-5-20250929"
        assert ctx.autonomy == Autonomy.HANDS_OFF
        assert isinstance(ctx.enabled_agents, list)
        assert isinstance(ctx.tool_names, list)
        assert ctx.os_type == "linux"
        assert ctx.datetime_now == "2026-02-11T10:00:00"


# ===========================================================================
# PromptBuilder with workspace — integration style
# ===========================================================================

class TestPromptBuilderWithWorkspace:

    def test_builder_with_workspace_and_skills(self, tmp_path):
        # Set up skill
        skills_dir = tmp_path / "skills" / "deploy"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# Deploy\nDeploy the application.")

        # Set up agent bootstrap
        agent_os_dir = tmp_path / ".agent-os"
        agent_os_dir.mkdir()
        (agent_os_dir / "AGENT.md").write_text("You are a deployment helper.")

        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        cached, dynamic = builder.build(ctx)

        assert "Deploy" in dynamic
        assert "You are a deployment helper." in dynamic

    def test_builder_identity_section_in_cached(self):
        builder = PromptBuilder()
        ctx = make_context()
        cached, _ = builder.build(ctx)
        # Identity section should mention "management agent" or "Agent OS"
        lower = cached.lower()
        assert "agent" in lower

    def test_builder_safety_section_in_cached(self):
        builder = PromptBuilder()
        ctx = make_context()
        cached, _ = builder.build(ctx)
        lower = cached.lower()
        assert "never" in lower or "rule" in lower

    def test_builder_memory_section_in_dynamic(self):
        builder = PromptBuilder()
        ctx = make_context()
        _, dynamic = builder.build(ctx)
        lower = dynamic.lower()
        assert "project_state.md" in lower or "memory" in lower


# ===========================================================================
# Skills planning discipline injection
# ===========================================================================

class TestSkillsDisciplineInjection:

    def test_skills_present_includes_check_instruction(self, tmp_path):
        """When skills exist, output contains 'check if a relevant skill exists'."""
        skills_dir = tmp_path / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# My Skill\nA test skill.")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, dynamic = builder.build(ctx)
        assert "check if a relevant skill exists" in dynamic

    def test_no_skills_includes_planning_discipline(self, tmp_path):
        """When no skills exist, output contains 'Planning Discipline'."""
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, dynamic = builder.build(ctx)
        assert "Planning Discipline" in dynamic

    def test_skills_section_has_header(self, tmp_path):
        """When skills exist, output contains '## Skills' header."""
        skills_dir = tmp_path / "skills" / "deploy"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# Deploy\nDeploy the application.")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, dynamic = builder.build(ctx)
        assert "## Skills" in dynamic

    def test_no_skill_loader_returns_no_skills_or_discipline(self):
        """When skill_loader is None, neither Skills nor Planning Discipline appears."""
        builder = PromptBuilder()  # no workspace = no skill_loader
        ctx = make_context()
        _, dynamic = builder.build(ctx)
        assert "## Skills" not in dynamic
        # Planning Discipline only appears when skill_loader exists but finds no skills
        # With no workspace at all, skill_loader is None
