# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: compaction reorientation reads files at new-layout paths.

Bug (F4): After TASK-02, PROJECT_STATE.md is written to {workspace}/orbital/
PROJECT_STATE.md (flat). The compaction reader constructed its own path via
os.path.join(workspace, "orbital") which happened to match the flat layout —
BUT if it still carried a slug segment the file would not be found.

This test verifies that inject_reorientation reads PROJECT_STATE.md from
the correct new-layout location (ProjectPaths.project_state) and injects
its content into the session after compaction.
"""

import os

import pytest

from agent_os.agent.compaction import inject_reorientation
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.session import Session


STATE_MARKER = "STATE-MARKER-ABC123"


@pytest.fixture
def session(tmp_path):
    return Session.new("test-reorient-newlayout", str(tmp_path))


@pytest.fixture
def workspace_with_state(tmp_path):
    """Seed workspace with PROJECT_STATE.md at the new flat path."""
    ws = str(tmp_path)
    pp = ProjectPaths(ws)
    os.makedirs(pp.orbital_dir, exist_ok=True)
    os.makedirs(pp.instructions_dir, exist_ok=True)

    with open(pp.project_state, "w", encoding="utf-8") as f:
        f.write(f"## Status\n{STATE_MARKER}\n\nWorking on the feature.")

    with open(pp.project_goals, "w", encoding="utf-8") as f:
        f.write("# Goals\nBuild the API.")

    return ws


class TestCompactionReorientationNewLayout:
    """inject_reorientation reads from the new flat layout."""

    def test_reorientation_contains_state_marker(self, session, workspace_with_state):
        """Reorientation message contains content from PROJECT_STATE.md at new path."""
        inject_reorientation(workspace_with_state, session)

        messages = session.get_messages()
        reorient_msgs = [
            m for m in messages
            if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")
        ]
        assert len(reorient_msgs) == 1, "Expected exactly one reorientation message"
        assert STATE_MARKER in reorient_msgs[0]["content"], \
            f"Reorientation message must contain '{STATE_MARKER}' from PROJECT_STATE.md"

    def test_reorientation_reads_project_goals(self, session, workspace_with_state):
        """Reorientation message also contains project goals from new-layout path."""
        inject_reorientation(workspace_with_state, session)

        messages = session.get_messages()
        reorient_msgs = [
            m for m in messages
            if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")
        ]
        assert len(reorient_msgs) == 1
        assert "Build the API" in reorient_msgs[0]["content"], \
            "Reorientation message must contain project goals from new-layout path"

    def test_missing_state_file_no_exception(self, session, workspace_with_state):
        """If PROJECT_STATE.md is deleted, reorientation falls back cleanly."""
        pp = ProjectPaths(workspace_with_state)
        # Delete the state file
        os.remove(pp.project_state)

        # Must not raise
        inject_reorientation(workspace_with_state, session)

        messages = session.get_messages()
        reorient_msgs = [
            m for m in messages
            if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")
        ]
        # Goals still exist so a reorientation message should be injected
        # but without the state marker
        assert len(reorient_msgs) == 1
        assert STATE_MARKER not in reorient_msgs[0]["content"], \
            "Deleted state file must not contribute content to reorientation"

    def test_both_files_missing_injects_nothing(self, session, tmp_path):
        """If both files are missing, no reorientation message is injected."""
        ws = str(tmp_path)
        pp = ProjectPaths(ws)
        os.makedirs(pp.orbital_dir, exist_ok=True)

        msg_count_before = len(session.get_messages())
        inject_reorientation(ws, session)
        assert len(session.get_messages()) == msg_count_before, \
            "No message should be injected when both files are missing"

    def test_no_slug_path_used(self, workspace_with_state):
        """Files must NOT be written to a slug subdirectory."""
        pp = ProjectPaths(workspace_with_state)
        # The project_state path must be directly under orbital/, not under orbital/{slug}/
        expected_path = pp.project_state
        assert "orbital" in expected_path
        # There should be no extra segment between orbital and PROJECT_STATE.md
        orbital_rel = os.path.relpath(expected_path, pp.orbital_dir)
        assert orbital_rel == "PROJECT_STATE.md", \
            f"project_state must be directly under orbital/, got: {orbital_rel}"
