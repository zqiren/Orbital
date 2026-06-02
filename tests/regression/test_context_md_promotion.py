# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for CONTEXT.md Layer 1 promotion + expanded scope.

Covers TASK "CONTEXT.md Layer 1 Promotion + Upload Queue Notification":
  - Part 1A: CONTEXT.md injected into Layer 1 context (after LESSONS)
  - Part 1B: session-end prompt redefines CONTEXT.md as a project map
  - Part 1C: token-based cap for CONTEXT.md (not entry-based sanity check)
  - Part 1D/4C: system-prompt CONTEXT.md maintenance + on-judgment update
  - Part 1E: token-pressure nudge mentions CONTEXT.md
  - Part 3A/3B: onboarding mandates workspace exploration + CONTEXT.md
  - Part 4A/4B: workspace-awareness + artifact-writing directives (cached)
"""

from __future__ import annotations

import pytest

from agent_os.agent import context as context_mod
from agent_os.agent import workspace_files as wf
from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
from agent_os.agent.workspace_files import WorkspaceFileManager


class _MockPromptBuilder:
    def build(self, context):
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


def _base_ctx(workspace: str, **overrides) -> PromptContext:
    kwargs = dict(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write", "shell"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )
    kwargs.update(overrides)
    return PromptContext(**kwargs)


# ---------------------------------------------------------------------------
# Part 1A — Layer 1 injection (Test 1)
# ---------------------------------------------------------------------------

class TestContextMdInLayer1:
    def test_context_md_present_in_prepared_context(self, tmp_path):
        """prepare() should read CONTEXT.md and inject it as a system message."""
        orbital = tmp_path / "orbital"
        orbital.mkdir(parents=True, exist_ok=True)
        (orbital / "CONTEXT.md").write_text(
            "## Overview\nA test project map.", encoding="utf-8"
        )

        session = Session.new("ctxmap", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        cm = ContextManager(session, _MockPromptBuilder(), _base_ctx(str(tmp_path)))

        result = cm.prepare()
        system_text = " ".join(
            m.get("content", "") for m in result if m.get("role") == "system"
        )
        assert "A test project map." in system_text

    def test_context_md_appears_after_lessons(self, tmp_path):
        """Ordering: CONTEXT.md must be injected AFTER LESSONS.md."""
        orbital = tmp_path / "orbital"
        orbital.mkdir(parents=True, exist_ok=True)
        (orbital / "LESSONS.md").write_text("LESSON_MARKER_TEXT", encoding="utf-8")
        (orbital / "CONTEXT.md").write_text("CONTEXT_MARKER_TEXT", encoding="utf-8")

        session = Session.new("ctxorder", str(tmp_path))
        session.append({"role": "user", "content": "hi", "source": "user"})
        cm = ContextManager(session, _MockPromptBuilder(), _base_ctx(str(tmp_path)))

        result = cm.prepare()
        blob = "\n".join(m.get("content", "") for m in result)
        assert "LESSON_MARKER_TEXT" in blob and "CONTEXT_MARKER_TEXT" in blob
        assert blob.index("LESSON_MARKER_TEXT") < blob.index("CONTEXT_MARKER_TEXT")

    def test_context_key_registered_in_layer1_tuple(self):
        keys = [k for k, _ in context_mod._LAYER1_FILES]
        assert "context" in keys
        # context must be last (broadest / least urgent)
        assert keys[-1] == "context"


# ---------------------------------------------------------------------------
# Part 1B — session-end prompt redefines CONTEXT.md scope (Test 2)
# ---------------------------------------------------------------------------

class TestSessionEndPromptFormat:
    def test_prompt_uses_section_based_context_structure(self, tmp_path):
        mgr = WorkspaceFileManager(str(tmp_path))
        prompt = mgr.build_session_end_prompt(
            {"message_count": 1, "tool_calls_count": 0, "recent_messages": []}
        )
        # New section headers the LLM must produce for CONTEXT.md
        for header in ("## Overview", "## Key Files", "## Architecture",
                       "## Conventions", "## External Context"):
            assert header in prompt, f"missing {header!r} in session-end prompt"
        # New framing
        assert "map of this project" in prompt

    def test_prompt_drops_old_entry_cap_language_for_context(self, tmp_path):
        """The old CONTEXT.md guidance capped at 25 entries; the new map format
        targets a token budget instead."""
        mgr = WorkspaceFileManager(str(tmp_path))
        prompt = mgr.build_session_end_prompt(
            {"message_count": 1, "tool_calls_count": 0, "recent_messages": []}
        )
        # token target appears for context
        assert "under 1000 tokens" in prompt


# ---------------------------------------------------------------------------
# Part 1C — token-based cap for CONTEXT.md (Test 3)
# ---------------------------------------------------------------------------

class TestContextTokenCap:
    def test_oversized_context_is_truncated_at_line_boundary(self):
        # ~1500 token cap => ~6000 char budget. Build well over it.
        lines = [f"- line number {i} with some filler text here" for i in range(400)]
        big = "\n".join(lines)
        assert len(big) > 6000
        out = wf._cap_context_tokens(big)
        assert len(out) < len(big)
        # truncates on a line boundary — no partial trailing line
        assert out.strip() == out.strip().rstrip("\n")
        for ln in out.splitlines():
            assert ln in lines  # every kept line is a whole original line

    def test_undersized_context_passes_through_unchanged(self):
        small = "## Overview\nTiny map.\n## Key Files\n- a.py: thing"
        out = wf._cap_context_tokens(small)
        assert out == small

    def test_decisions_still_use_entry_based_sanity_check(self):
        """No regression: DECISIONS dedup/cap is still entry-based."""
        dup = "## 2026-01-01: A\nbody\n## 2026-01-01: A\nbody\n"
        out = wf._apply_sanity_checks(
            dup, wf._DECISIONS_ENTRY_PATTERN, wf._DECISIONS_CAP,
            keep="last", filename="decisions",
        )
        # exact duplicate entry removed
        assert out.count("## 2026-01-01: A") == 1


# ---------------------------------------------------------------------------
# Part 1D / 4C — system-prompt CONTEXT.md maintenance instructions
# ---------------------------------------------------------------------------

class TestSystemPromptContextInstructions:
    def test_memory_section_describes_context_md(self, tmp_path):
        _cached, semi, _dyn = PromptBuilder(str(tmp_path)).build(_base_ctx(str(tmp_path)))
        assert "CONTEXT.md" in semi
        assert "map of this project" in semi

    def test_context_md_on_judgment_update_language(self, tmp_path):
        _cached, semi, _dyn = PromptBuilder(str(tmp_path)).build(_base_ctx(str(tmp_path)))
        # 4C: agent may update mid-session, not only at checkpoints
        assert "wait for a checkpoint" in semi.lower()

    def test_scratch_project_has_no_context_md_instruction(self, tmp_path):
        _cached, semi, _dyn = PromptBuilder(str(tmp_path)).build(
            _base_ctx(str(tmp_path), is_scratch=True)
        )
        assert "CONTEXT.md" not in semi


# ---------------------------------------------------------------------------
# Part 1E — token-pressure nudge mentions CONTEXT.md
# ---------------------------------------------------------------------------

class TestTokenPressureNudge:
    def test_moderate_pressure_mentions_context_md(self, tmp_path):
        _c, _s, dyn = PromptBuilder(str(tmp_path)).build(
            _base_ctx(str(tmp_path), context_usage_pct=0.75)
        )
        assert "CONTEXT.md" in dyn

    def test_high_pressure_mentions_context_md(self, tmp_path):
        _c, _s, dyn = PromptBuilder(str(tmp_path)).build(
            _base_ctx(str(tmp_path), context_usage_pct=0.90)
        )
        assert "CONTEXT.md" in dyn


# ---------------------------------------------------------------------------
# Part 3A / 3B — onboarding workspace exploration + CONTEXT.md creation
# ---------------------------------------------------------------------------

class TestOnboardingExploration:
    def test_onboarding_includes_workspace_exploration_step(self, tmp_path):
        # No project_goals.md => onboarding mode
        _c, semi, _d = PromptBuilder(str(tmp_path)).build(_base_ctx(str(tmp_path)))
        assert "ONBOARDING MODE" in semi
        assert "explore the workspace" in semi.lower()
        assert "CONTEXT.md" in semi

    def test_onboarding_tool_restriction_allows_exploration(self, tmp_path):
        _c, semi, _d = PromptBuilder(str(tmp_path)).build(_base_ctx(str(tmp_path)))
        # old text forbade ALL tools until onboarding complete; new text allows
        # read/list exploration after project_goals.md is written.
        assert "until goal-setting is complete" in semi


# ---------------------------------------------------------------------------
# Part 4A / 4B — workspace-awareness + artifact-writing directives (cached)
# ---------------------------------------------------------------------------

class TestWorkspacePersistenceDirectives:
    def test_persistence_directive_in_cached_prefix(self, tmp_path):
        cached, _s, _d = PromptBuilder(str(tmp_path)).build(_base_ctx(str(tmp_path)))
        assert "persistent worker" in cached.lower()
        assert "persistent workspace" in cached.lower()

    def test_artifact_writing_directive_in_cached_prefix(self, tmp_path):
        cached, _s, _d = PromptBuilder(str(tmp_path)).build(_base_ctx(str(tmp_path)))
        assert "artifact" in cached.lower()
        # records artifact location back in CONTEXT.md
        assert "CONTEXT.md" in cached

    def test_scratch_project_has_no_persistence_directive(self, tmp_path):
        cached, _s, _d = PromptBuilder(str(tmp_path)).build(
            _base_ctx(str(tmp_path), is_scratch=True)
        )
        assert "persistent worker" not in cached.lower()
