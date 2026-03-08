# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: post-compaction reorientation prevents agent disorientation.

Bug: After compaction, the agent's awareness of its instructions and project
goals may be lost if those messages were in the summarised portion. The agent
becomes disoriented, forgets its task, or ignores its behavioural guidelines.

Fix: After compaction completes, re-inject project goals and PROJECT_STATE.md
as a system message so the agent retains operating context after history summary.
Fault-tolerant: missing files inject nothing, never crash compaction.
"""

import os
import json

import pytest

from agent_os.agent.session import Session
from agent_os.agent.compaction import inject_reorientation


@pytest.fixture
def session(tmp_path):
    return Session.new("test-reorient", str(tmp_path))


@pytest.fixture
def workspace(tmp_path):
    """Workspace with .agent-os directory structure."""
    agent_os_dir = os.path.join(str(tmp_path), ".agent-os")
    instructions_dir = os.path.join(agent_os_dir, "instructions")
    os.makedirs(instructions_dir, exist_ok=True)
    return str(tmp_path)


class TestReorientationInjection:
    """Post-compaction reorientation injects goals + state as system message."""

    def test_injects_goals_and_state(self, session, workspace):
        """When both files exist, reorientation message contains both."""
        agent_os_dir = os.path.join(workspace, ".agent-os")

        # Create project goals
        goals_path = os.path.join(agent_os_dir, "instructions", "project_goals.md")
        with open(goals_path, "w") as f:
            f.write("Build a REST API with authentication.")

        # Create PROJECT_STATE.md
        state_path = os.path.join(agent_os_dir, "PROJECT_STATE.md")
        with open(state_path, "w") as f:
            f.write("Currently implementing JWT middleware.")

        # Seed session with a compaction summary (simulating post-compaction state)
        session.append({
            "role": "system",
            "content": "Summary of prior conversation.",
            "_compaction": True,
            "source": "management",
        })

        msg_count_before = len(session.get_messages())

        inject_reorientation(workspace, session)

        messages = session.get_messages()
        assert len(messages) == msg_count_before + 1

        reorient_msg = messages[-1]
        assert reorient_msg["role"] == "system"
        assert "[POST-COMPACTION REORIENTATION]" in reorient_msg["content"]
        assert "Build a REST API" in reorient_msg["content"]
        assert "JWT middleware" in reorient_msg["content"]

    def test_goals_only(self, session, workspace):
        """When only goals file exists, state section shows fallback text."""
        agent_os_dir = os.path.join(workspace, ".agent-os")
        goals_path = os.path.join(agent_os_dir, "instructions", "project_goals.md")
        with open(goals_path, "w") as f:
            f.write("Implement user authentication.")

        inject_reorientation(workspace, session)

        messages = session.get_messages()
        reorient_msgs = [m for m in messages if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")]
        assert len(reorient_msgs) == 1
        assert "user authentication" in reorient_msgs[0]["content"]
        assert "No state file found" in reorient_msgs[0]["content"]

    def test_state_only(self, session, workspace):
        """When only state file exists, goals section shows fallback text."""
        agent_os_dir = os.path.join(workspace, ".agent-os")
        state_path = os.path.join(agent_os_dir, "PROJECT_STATE.md")
        with open(state_path, "w") as f:
            f.write("Working on database schema.")

        inject_reorientation(workspace, session)

        messages = session.get_messages()
        reorient_msgs = [m for m in messages if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")]
        assert len(reorient_msgs) == 1
        assert "database schema" in reorient_msgs[0]["content"]
        assert "No project goals file found" in reorient_msgs[0]["content"]

    def test_no_files_injects_nothing(self, session, workspace):
        """When both files are missing, no message is injected."""
        msg_count_before = len(session.get_messages())
        inject_reorientation(workspace, session)
        assert len(session.get_messages()) == msg_count_before

    def test_empty_files_inject_nothing(self, session, workspace):
        """When both files exist but are empty, no message is injected."""
        agent_os_dir = os.path.join(workspace, ".agent-os")
        goals_path = os.path.join(agent_os_dir, "instructions", "project_goals.md")
        state_path = os.path.join(agent_os_dir, "PROJECT_STATE.md")

        with open(goals_path, "w") as f:
            f.write("")
        with open(state_path, "w") as f:
            f.write("")

        msg_count_before = len(session.get_messages())
        inject_reorientation(workspace, session)
        assert len(session.get_messages()) == msg_count_before

    def test_large_files_truncated_at_3000(self, session, workspace):
        """Files larger than 3000 chars are truncated."""
        agent_os_dir = os.path.join(workspace, ".agent-os")
        goals_path = os.path.join(agent_os_dir, "instructions", "project_goals.md")
        with open(goals_path, "w") as f:
            f.write("X" * 5000)

        inject_reorientation(workspace, session)

        messages = session.get_messages()
        reorient_msgs = [m for m in messages if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")]
        assert len(reorient_msgs) == 1
        # The goals portion should be capped at 3000 chars
        content = reorient_msgs[0]["content"]
        # Count X chars — should be at most 3000
        x_count = content.count("X")
        assert x_count <= 3000

    def test_reorientation_after_compaction_message(self, session, workspace):
        """Reorientation messages must appear AFTER the compaction summary."""
        agent_os_dir = os.path.join(workspace, ".agent-os")
        goals_path = os.path.join(agent_os_dir, "instructions", "project_goals.md")
        with open(goals_path, "w") as f:
            f.write("Build the API.")

        # Simulate compaction result
        session.append({
            "role": "system",
            "content": "Compaction summary of conversation.",
            "_compaction": True,
            "source": "management",
        })

        inject_reorientation(workspace, session)

        messages = session.get_messages()
        compaction_idx = None
        reorient_idx = None
        for i, m in enumerate(messages):
            if m.get("_compaction"):
                compaction_idx = i
            if "[POST-COMPACTION REORIENTATION]" in m.get("content", ""):
                reorient_idx = i

        assert compaction_idx is not None
        assert reorient_idx is not None
        assert reorient_idx > compaction_idx, \
            "Reorientation must appear AFTER compaction summary"

    def test_fault_tolerant_read_errors(self, session, workspace):
        """File read errors must not crash — inject nothing."""
        # Point to a non-existent workspace
        inject_reorientation("/nonexistent/path/workspace", session)
        # Should not raise, should inject nothing
        assert len(session.get_messages()) == 0

    def test_persisted_to_jsonl(self, session, workspace):
        """Reorientation message must be persisted to JSONL."""
        agent_os_dir = os.path.join(workspace, ".agent-os")
        state_path = os.path.join(agent_os_dir, "PROJECT_STATE.md")
        with open(state_path, "w") as f:
            f.write("Current task: testing persistence.")

        inject_reorientation(workspace, session)

        # Reload from disk
        reloaded = Session.load(session._filepath)
        messages = reloaded.get_messages()
        reorient_msgs = [m for m in messages if "[POST-COMPACTION REORIENTATION]" in m.get("content", "")]
        assert len(reorient_msgs) == 1
        assert "testing persistence" in reorient_msgs[0]["content"]
