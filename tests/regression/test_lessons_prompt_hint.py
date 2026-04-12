# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test for LESSONS.md prompt hint in _memory() section.

Ensures the agent system prompt mentions LESSONS.md (auto-injected each turn)
so the agent knows to consult it rather than treating it as inert reference.
See TASK/small-fix/TASK-lessons-prompt-hint.md.
"""

from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext


def _make_context(*, is_scratch: bool) -> PromptContext:
    return PromptContext(
        workspace="/tmp/ws",
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-04-09T00:00:00",
        context_usage_pct=0.0,
        project_name="test-proj",
        project_dir_name="test-proj",
        is_scratch=is_scratch,
    )


def test_memory_non_scratch_mentions_lessons_md():
    """Non-scratch _memory() output must mention LESSONS.md by name."""
    builder = PromptBuilder(workspace="/tmp/ws")
    ctx = _make_context(is_scratch=False)
    output = builder._memory(ctx)
    assert "LESSONS.md" in output, (
        "Expected _memory() to mention LESSONS.md so the agent knows to "
        "consult auto-injected lessons. Got:\n" + output
    )


def test_memory_non_scratch_describes_consulting_lessons():
    """Non-scratch _memory() must describe LESSONS.md purpose."""
    builder = PromptBuilder(workspace="/tmp/ws")
    ctx = _make_context(is_scratch=False)
    output = builder._memory(ctx).lower()
    # Must contain language about lessons or memory persistence.
    keywords = ("lessons", "memory across sessions", "force-injected", "session end")
    assert any(kw in output for kw in keywords), (
        "Expected _memory() to contain language about lessons or memory persistence. "
        f"Looked for {keywords}. Got:\n{output}"
    )


def test_memory_scratch_does_not_mention_lessons_md():
    """Scratch mode has no project dir, so the prompt must not mention LESSONS.md."""
    builder = PromptBuilder(workspace="/tmp/ws")
    ctx = _make_context(is_scratch=True)
    output = builder._memory(ctx)
    assert "LESSONS.md" not in output, (
        "Scratch mode _memory() should NOT mention LESSONS.md. Got:\n" + output
    )
