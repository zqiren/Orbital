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
    def build(self, context: PromptContext) -> tuple[str, str]:
        return ("cached-system-prefix", "dynamic-suffix")


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

    def test_lessons_md_updates_visible_next_iteration(self, tmp_path):
        """Write LESSONS.md between two prepare() calls -> second sees new content."""
        agent_os_dir = tmp_path / "orbital"
        agent_os_dir.mkdir()

        session = Session.new("lessons-live", str(tmp_path))
        builder = MockPromptBuilder()
        ctx = _make_base_prompt_context(str(tmp_path))
        cm = ContextManager(session, builder, ctx)

        msgs1 = cm.prepare()
        content1 = " ".join(m["content"] for m in msgs1 if m["role"] == "system")
        assert "Avoid eval" not in content1

        (agent_os_dir / "LESSONS.md").write_text(
            "## Avoid eval\n**Problem:** Security risk\n**Fix:** Use ast.literal_eval"
        )

        msgs2 = cm.prepare()
        content2 = " ".join(m["content"] for m in msgs2 if m["role"] == "system")
        assert "Avoid eval" in content2


# ===========================================================================
# Change 1 (cont): Session-End Consolidation (workspace_files.py)
# ===========================================================================


class TestSessionEndConsolidation:

    def test_session_end_prompt_has_replace_instruction(self):
        """Session-end prompt tells LLM to REPLACE lessons, not append."""
        wfm = WorkspaceFileManager("/tmp/test")
        prompt = wfm.build_session_end_prompt({
            "message_count": 5,
            "tool_calls_count": 2,
            "files_modified": [],
            "recent_messages": [],
        })
        assert "REPLACES" in prompt or "REPLACE" in prompt
        assert "Cap at 20" in prompt or "cap at 20" in prompt.lower()
        # The lessons field should NOT contain the old append-only wording
        # Extract just the lessons instruction (field 4) from the prompt
        lessons_section = prompt.split('"lessons"')[1].split('"context"')[0]
        assert "Only include genuinely new" not in lessons_section

    def test_session_end_prompt_carries_forward(self):
        """Session-end prompt instructs carrying forward existing lessons."""
        wfm = WorkspaceFileManager("/tmp/test")
        prompt = wfm.build_session_end_prompt({
            "message_count": 1,
            "tool_calls_count": 0,
            "files_modified": [],
            "recent_messages": [],
        })
        assert "Carry forward" in prompt or "carry forward" in prompt.lower()


# ===========================================================================
# Change 2: Sub-Agent Verification Guidance (prompt_builder.py)
# ===========================================================================


class TestSubAgentVerification:

    def test_verification_with_sub_agents(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[
            {"handle": "claudecode", "display_name": "Claude Code", "type": "cli"}
        ])
        _, dynamic = builder.build(ctx)
        assert "Verifying Sub-Agent Output" in dynamic
        assert "read" in dynamic.split("Verifying")[1]
        assert "shell" in dynamic.split("Verifying")[1]

    def test_verification_multiple_sub_agents(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[
            {"handle": "claudecode", "display_name": "Claude Code", "type": "cli"},
            {"handle": "aider", "display_name": "Aider", "type": "cli"},
        ])
        _, dynamic = builder.build(ctx)
        assert dynamic.count("Verifying Sub-Agent Output") == 1

    def test_no_verification_without_sub_agents(self):
        builder = PromptBuilder()
        ctx = make_context(enabled_agents=[])
        _, dynamic = builder.build(ctx)
        assert "Verifying Sub-Agent Output" not in dynamic
