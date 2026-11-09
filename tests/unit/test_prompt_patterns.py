# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the prompt-patterns feature.

Covers:
  - LESSONS.md Layer 1 injection in ContextManager (context.py)
  - Session-end consolidation prompt (workspace_files.py)
  - Sub-agent verification guidance (prompt_builder.py)
"""

import os

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_base_prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str, str]:
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


class CapturingPromptBuilder:
    def __init__(self):
        self.deployment_values = []

    def build(self, context: PromptContext) -> tuple[str, str, str]:
        self.deployment_values.append(context.sub_agent_deployment_instructions)
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


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
# Change 1: LESSONS.md Layer 1 Injection (context.py)
# ===========================================================================


class TestLessonsLayer1Injection:

    def test_lessons_md_injected_as_layer1(self, tmp_path):
        """LESSONS.md read every iteration alongside PROJECT_STATE and DECISIONS."""
        agent_os_dir = tmp_path / "orbital"
        agent_os_dir.mkdir()
        (agent_os_dir / "LESSONS.md").write_text(
            "## Don't use rm -rf\n**Problem:** Deleted everything"
        )

        session = Session.new("lessons-test", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx)

        messages = cm.prepare()
        layer1_content = " ".join(
            m["content"] for m in messages
            if m["role"] == "system" and "LESSONS" in m.get("content", "")
        )
        assert "Don't use rm -rf" in layer1_content

    def test_lessons_md_missing_no_error(self, tmp_path):
        """No LESSONS.md file -> no injection, no error."""
        session = Session.new("lessons-miss", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx)

        messages = cm.prepare()  # must not raise
        layer1_content = " ".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        # No "[LESSONS.md]" header should appear
        assert "[LESSONS.md]" not in layer1_content

    def test_lessons_md_updates_visible_next_run(self, tmp_path):
        """Write LESSONS.md between two prepare() calls of the SAME run -> the
        injected row stays the run-start snapshot (spec 112 A1; the tail names
        the drift instead); the next run (``begin_run()``) sees the new
        content."""
        agent_os_dir = tmp_path / "orbital"
        agent_os_dir.mkdir()

        session = Session.new("lessons-live", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx)

        cm.begin_run()
        msgs1 = cm.prepare()
        content1 = " ".join(m["content"] for m in msgs1 if m["role"] == "system")
        assert "Avoid eval" not in content1

        (agent_os_dir / "LESSONS.md").write_text(
            "## Avoid eval\n**Problem:** Security risk\n**Fix:** Use ast.literal_eval"
        )

        msgs2 = cm.prepare()
        content2 = " ".join(m["content"] for m in msgs2 if m["role"] == "system")
        assert "Avoid eval" not in content2

        cm.begin_run()
        msgs3 = cm.prepare()
        content3 = " ".join(m["content"] for m in msgs3 if m["role"] == "system")
        assert "Avoid eval" in content3


class TestLiveSubAgentDeploymentInstructions:
    def test_provider_value_is_refreshed_on_each_prepare(self, tmp_path):
        current = {"value": "Use Codex."}
        session = Session.new("deployment-live", str(tmp_path))
        builder = CapturingPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(
            session,
            builder,
            ctx,
            sub_agent_deployment_instructions_provider=lambda: current["value"],
        )

        cm.prepare()
        current["value"] = "Use Claude Code."
        cm.prepare()

        assert builder.deployment_values == ["Use Codex.", "Use Claude Code."]

    def test_without_provider_keeps_creation_time_value(self, tmp_path):
        session = Session.new("deployment-static", str(tmp_path))
        builder = CapturingPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        ctx.sub_agent_deployment_instructions = "Static routing."
        cm = ContextManager(session, builder, ctx)

        cm.prepare()

        assert builder.deployment_values == ["Static routing."]


# ===========================================================================
# Change 1 (cont): Session-End Consolidation (workspace_files.py)
# ===========================================================================


class TestSessionEndConsolidation:
    """Spec 124 §3.7: the session-end pass is the Memory session's turn — it
    moves whole entries into archives and never regenerates a file, so
    existing memory cannot be silently dropped by the pass (the intent these
    tests always guarded)."""

    def _prompt(self):
        from agent_os.agent import memory_editor
        return memory_editor.build_task_message(
            sizes={k: (1, 10, 5) for k in memory_editor.LIVE_KEYS}, folded=[], over=["state"],
            forced=False, pointer_paths=[], last_pass=None, sessions=[], today="2026-09-19", tool_cap=12,
        )

    def test_task_asks_for_moves_not_whole_files(self):
        prompt = self._prompt()
        assert "Move entries intact" in prompt
        assert '"project_state"' not in prompt
        assert "COMPLETE updated" not in prompt

    def test_task_never_deletes_or_invents(self):
        prompt = self._prompt()
        assert "Archive only what is no longer true or no longer live" in prompt
        assert "Never add facts." in prompt


# ===========================================================================
# Change 2: Sub-Agent Verification Guidance (prompt_builder.py)
# ===========================================================================


class TestSubAgentVerification:

    def test_verification_with_sub_agents(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[
            {"handle": "claudecode", "display_name": "Claude Code", "type": "cli"}
        ])
        _, semi_stable, _ = builder.build(ctx)
        # 1109 R5: folded into the Dispatching block.
        tail = semi_stable.split("### Dispatching")[1]
        assert "verify the work silently" in tail
        assert "files exist, code builds" in tail

    def test_verification_multiple_sub_agents(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[
            {"handle": "claudecode", "display_name": "Claude Code", "type": "cli"},
            {"handle": "aider", "display_name": "Aider", "type": "cli"},
        ])
        _, semi_stable, _ = builder.build(ctx)
        assert semi_stable.count("verify the work silently") == 1

    def test_no_verification_without_sub_agents(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[])
        _, semi_stable, _ = builder.build(ctx)
        assert "verify the work silently" not in semi_stable


# ===========================================================================
# Skill Injection Strength — Change 1: Skill creation trigger in _memory
# ===========================================================================


class TestSkillCreationTrigger:

    def test_skill_creation_in_normal_project(self):
        """Non-scratch prompt contains skill creation instructions."""
        builder = PromptBuilder()
        ctx = make_context(is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert "Skill creation:" in semi_stable

    def test_skill_creation_not_in_scratch(self):
        """Scratch prompt does NOT contain skill creation instructions."""
        builder = PromptBuilder()
        ctx = make_context(is_scratch=True)
        _, semi_stable, _ = builder.build(ctx)
        assert "Skill creation:" not in semi_stable


# ===========================================================================
# Skill Injection Strength — Change 2: MUST language in skills section
# ===========================================================================


class TestSkillConsultationStrength:

    def test_read_rule_when_skills_exist(self, tmp_path):
        """When skills exist, a matching skill is read before starting (1109
        R14: plain wording instead of 'MUST read')."""
        # Skills live at {workspace}/orbital/skills/ in the new layout (TASK-02).
        skills_dir = tmp_path / "orbital" / "skills" / "deploy"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# Deploy\nDeploy the app.")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, semi_stable, _ = builder.build(ctx)
        assert "matches your task, read its SKILL.md before you start" in semi_stable

    def test_planning_discipline_fallback_when_no_skills(self, tmp_path):
        """When no skills exist, fallback to Planning Discipline (unchanged)."""
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, semi_stable, _ = builder.build(ctx)
        assert "Planning Discipline" in semi_stable

    def test_skill_names_still_listed(self, tmp_path):
        """Skill index still lists skill names after preamble change."""
        # Skills live at {workspace}/orbital/skills/ in the new layout (TASK-02).
        skills_dir = tmp_path / "orbital" / "skills" / "my-deploy"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# My Deploy\nDeploys things.")
        builder = PromptBuilder(workspace=str(tmp_path))
        ctx = make_context(workspace=str(tmp_path))
        _, semi_stable, _ = builder.build(ctx)
        assert "My Deploy" in semi_stable
        assert "Deploys things" in semi_stable


# ===========================================================================
# Skill Injection Strength — Change 3: Context budget reflection
# ===========================================================================


class TestContextBudgetReflection:

    def test_70pct_includes_skill_reflection(self):
        """At 75% context, reflection prompt about skills is present."""
        builder = PromptBuilder()
        ctx = make_context(context_usage_pct=0.75)
        _, _, dynamic = builder.build(ctx)
        assert "workflow worth saving as a skill" in dynamic

    def test_50pct_no_skill_reflection(self):
        """At 50% context, no reflection prompt about skills."""
        builder = PromptBuilder()
        ctx = make_context(context_usage_pct=0.50)
        _, _, dynamic = builder.build(ctx)
        assert "workflow worth saving as a skill" not in dynamic

    def test_85pct_still_has_urgent(self):
        """At 90% context, URGENT warning still present (not broken)."""
        builder = PromptBuilder()
        ctx = make_context(context_usage_pct=0.90)
        _, _, dynamic = builder.build(ctx)
        assert "URGENT" in dynamic
        assert "workflow worth saving as a skill" in dynamic


# ===========================================================================
# Skill Injection Strength — Change 4: LESSONS.md instruction updated
# ===========================================================================


class TestLessonsInstructionUpdate:

    def test_write_at_run_end_instruction(self):
        """Normal project prompt says the daemon asks at the end of every run.

        Spec 112 A6b / 124 §3.8 replaced "update them proactively as you
        work" with a single, deterministic write point: the daemon asks what
        to record at run end (and before compaction) and hands over the
        current file text; the trigger to add a lesson on error-recovery
        stays.
        """
        builder = PromptBuilder()
        ctx = make_context(is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert "Do not update the memory files during a run." in semi_stable
        assert "Update them proactively" not in semi_stable
        assert "LESSONS.md" in semi_stable
        # 1109 R4: what a lesson is stays in the prompt; the trigger to add
        # one rides the [MEMORY] row, where it is written.
        assert "rules learned from errors" in semi_stable
        from agent_os.agent.loop import build_memory_record_text
        assert "an error recovered from" in build_memory_record_text("/nonexistent", [], [])

    def test_old_do_not_write_removed(self):
        """Old contradictory 'do NOT need to read or write' instruction is gone."""
        builder = PromptBuilder()
        ctx = make_context(is_scratch=False)
        _, semi_stable, _ = builder.build(ctx)
        assert "do NOT need to read or write" not in semi_stable
