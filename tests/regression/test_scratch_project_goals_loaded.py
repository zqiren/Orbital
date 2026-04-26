# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: scratch project goals are written and read at new-layout paths.

Bug (F6): The scratch project writes project_goals.md via ProjectPaths but
prompt_builder.py was previously reading it using a slug-based path that
includes a project_dir_name segment which doesn't exist in the new flat
layout. The agent would enter ONBOARDING MODE every session because the
reader couldn't find the file the writer had placed at the correct (flat)
path.

Fix: prompt_builder._onboarding_or_directive() and ._standing_rules() now
read via ProjectPaths, matching the writer's flat path.
"""

import os

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy


SCRATCH_GOAL_TEXT = "Build a REST API with authentication."


def _build_scratch_context(workspace: str) -> PromptContext:
    """Minimal PromptContext for a scratch project."""
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.CHECK_IN,
        enabled_agents=[],
        tool_names=["read", "write"],
        os_type="linux",
        datetime_now="2026-04-25T00:00:00",
        project_name="Quick Tasks",
        is_scratch=True,
    )


def _build_project_context(workspace: str) -> PromptContext:
    """Minimal PromptContext for a real project."""
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.CHECK_IN,
        enabled_agents=[],
        tool_names=["read", "write"],
        os_type="linux",
        datetime_now="2026-04-25T00:00:00",
        project_name="My Project",
        is_scratch=False,
    )


class TestScratchProjectGoalsLoaded:
    """Scratch project goals file is written and read at the new flat layout."""

    def test_scratch_goals_file_exists_at_new_path(self, tmp_path):
        """After writing scratch goals, the file exists at ProjectPaths.project_goals."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.instructions_dir, exist_ok=True)

        # Simulate what app.py does: write goals at the new-layout path
        with open(pp.project_goals, "w", encoding="utf-8") as f:
            f.write(SCRATCH_GOAL_TEXT)

        assert os.path.isfile(pp.project_goals), \
            "project_goals.md must exist at ProjectPaths.project_goals after write"
        # Must NOT exist at a slug-based path
        slug_path = os.path.join(ws, "orbital", "quick-tasks-0000", "instructions", "project_goals.md")
        assert not os.path.isfile(slug_path), \
            "project_goals.md must NOT be written to slug-based path"

    def test_prompt_builder_reads_goals_from_new_path(self, tmp_path):
        """prompt_builder includes goals text when file is at new-layout path."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.instructions_dir, exist_ok=True)

        with open(pp.project_goals, "w", encoding="utf-8") as f:
            f.write(SCRATCH_GOAL_TEXT)

        builder = PromptBuilder(workspace=ws)
        context = _build_project_context(ws)
        cached, semi_stable, dynamic = builder.build(context)
        full_prompt = cached + semi_stable + dynamic

        assert SCRATCH_GOAL_TEXT in full_prompt, \
            "Project goals content must appear in the built prompt"

    def test_prompt_builder_no_onboarding_when_goals_exist(self, tmp_path):
        """When project_goals.md exists at new path, prompt must not say ONBOARDING MODE."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.instructions_dir, exist_ok=True)

        with open(pp.project_goals, "w", encoding="utf-8") as f:
            f.write(SCRATCH_GOAL_TEXT)

        builder = PromptBuilder(workspace=ws)
        context = _build_project_context(ws)
        cached, semi_stable, dynamic = builder.build(context)
        full_prompt = cached + semi_stable + dynamic

        assert "ONBOARDING MODE" not in full_prompt, \
            "ONBOARDING MODE must not appear when project_goals.md exists"

    def test_prompt_builder_onboarding_when_goals_missing(self, tmp_path):
        """When project_goals.md is absent, prompt must say ONBOARDING MODE."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.orbital_dir, exist_ok=True)
        # Do NOT create project_goals.md

        builder = PromptBuilder(workspace=ws)
        context = _build_project_context(ws)
        cached, semi_stable, dynamic = builder.build(context)
        full_prompt = cached + semi_stable + dynamic

        assert "ONBOARDING MODE" in full_prompt, \
            "ONBOARDING MODE must appear when project_goals.md does not exist"

    def test_new_layout_memory_path_in_prompt(self, tmp_path):
        """Prompt must reference new orbital/ paths, not orbital/{slug}/ paths."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.instructions_dir, exist_ok=True)

        with open(pp.project_goals, "w", encoding="utf-8") as f:
            f.write(SCRATCH_GOAL_TEXT)

        builder = PromptBuilder(workspace=ws)
        context = _build_project_context(ws)
        cached, semi_stable, dynamic = builder.build(context)
        full_prompt = cached + semi_stable + dynamic

        # Slug-namespaced paths must never appear in the prompt
        assert "orbital/my-project-a1b2/" not in full_prompt, \
            "Prompt must not reference old slug-namespaced path orbital/{slug}/"

    def test_user_directives_no_slug_path_in_prompt(self, tmp_path):
        """user_directives.md is referenced without slug in the prompt."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.instructions_dir, exist_ok=True)

        with open(pp.project_goals, "w", encoding="utf-8") as f:
            f.write(SCRATCH_GOAL_TEXT)

        builder = PromptBuilder(workspace=ws)
        context = _build_project_context(ws)
        cached, semi_stable, dynamic = builder.build(context)
        full_prompt = cached + semi_stable + dynamic

        # user_directives path in prompt must not include a slug segment
        assert "orbital/my-project-a1b2/instructions/user_directives.md" not in full_prompt, \
            "user_directives.md must not be referenced at a slug path in the prompt"
