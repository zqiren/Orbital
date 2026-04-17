# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: context.prepare() must read Layer 1 state files
from the namespaced path (workspace/orbital/{project_dir_name}/PROJECT_STATE.md)
when a WorkspaceFileManager is provided.

Bug: context.py previously read from the flat path workspace/orbital/PROJECT_STATE.md
always, ignoring the project_dir_name. WorkspaceFileManager writes state files
to the namespaced path, so after the fix context.py must read from the same
path -- otherwise the agent has amnesia across sessions for namespaced projects.
"""
from __future__ import annotations

import os

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager


class _StubPromptBuilder:
    def build(self, context):
        return ("cached-prefix", "semi-stable-suffix", "dynamic-runtime")


def _make_base_ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class TestLayer1NamespacedPath:
    """context.prepare() must read namespaced Layer 1 files via WorkspaceFileManager."""

    def test_reads_namespaced_state_file(self, tmp_path):
        """prepare() should include PROJECT_STATE.md content from the namespaced
        directory when a WorkspaceFileManager is provided."""
        workspace = str(tmp_path)
        project_dir_name = "my-project-abc1"

        # Create the namespaced state file
        ns_dir = tmp_path / "orbital" / project_dir_name
        ns_dir.mkdir(parents=True)
        (ns_dir / "PROJECT_STATE.md").write_text(
            "namespaced state content", encoding="utf-8"
        )

        # Also create a flat state file with different content to make sure
        # we do NOT accidentally read from it.
        flat_dir = tmp_path / "orbital"
        (flat_dir / "PROJECT_STATE.md").write_text(
            "flat state content", encoding="utf-8"
        )

        session = Session.new("layer1ns", workspace)
        session.append({"role": "user", "content": "hi", "source": "user"})
        wf = WorkspaceFileManager(
            workspace=workspace, project_dir_name=project_dir_name
        )
        ctx_mgr = ContextManager(
            session,
            _StubPromptBuilder(),
            _make_base_ctx(workspace),
            workspace_files=wf,
        )

        result = ctx_mgr.prepare()
        system_blob = " ".join(
            m.get("content", "") for m in result if m.get("role") == "system"
        )

        assert "namespaced state content" in system_blob, (
            "Expected Layer 1 reader to pick up the namespaced PROJECT_STATE.md"
        )
        assert "flat state content" not in system_blob, (
            "Layer 1 reader must NOT read the flat fallback when a namespaced"
            " WorkspaceFileManager is provided"
        )

    def test_reads_namespaced_decisions_and_lessons_as_layer1(self, tmp_path):
        """All three Layer 1 files (state, decisions, lessons) should be
        read from the namespaced directory AND appear as Layer 1 messages
        (with the [FILENAME.md] header), not just as cold-resume context."""
        workspace = str(tmp_path)
        project_dir_name = "testproj-beef"
        ns_dir = tmp_path / "orbital" / project_dir_name
        ns_dir.mkdir(parents=True)
        (ns_dir / "PROJECT_STATE.md").write_text("ns-state", encoding="utf-8")
        (ns_dir / "DECISIONS.md").write_text("ns-decisions", encoding="utf-8")
        (ns_dir / "LESSONS.md").write_text("ns-lessons", encoding="utf-8")

        session = Session.new("layer1ns2", workspace)
        session.append({"role": "user", "content": "hi", "source": "user"})
        wf = WorkspaceFileManager(
            workspace=workspace, project_dir_name=project_dir_name
        )
        ctx_mgr = ContextManager(
            session,
            _StubPromptBuilder(),
            _make_base_ctx(workspace),
            workspace_files=wf,
        )

        result = ctx_mgr.prepare()

        # Collect messages that are specifically Layer 1 (have the bracketed
        # filename header). Cold-resume uses a different header
        # ("[WORKSPACE MEMORY — Resume Context]") so we can distinguish them.
        layer1_blob = " ".join(
            m.get("content", "")
            for m in result
            if m.get("role") == "system"
            and any(
                f"[{name}]" in m.get("content", "")
                for name in ("PROJECT_STATE.md", "DECISIONS.md", "LESSONS.md")
            )
        )

        assert "ns-state" in layer1_blob, (
            "PROJECT_STATE.md content missing from Layer 1 messages; "
            "the Layer 1 reader is not using the namespaced path"
        )
        assert "ns-decisions" in layer1_blob, (
            "DECISIONS.md content missing from Layer 1 messages"
        )
        assert "ns-lessons" in layer1_blob, (
            "LESSONS.md content missing from Layer 1 messages"
        )

    def test_flat_fallback_when_no_workspace_files(self, tmp_path):
        """Backward compatibility: if no WorkspaceFileManager is provided
        (scratch mode / legacy tests), fall back to the flat path."""
        workspace = str(tmp_path)
        flat_dir = tmp_path / "orbital"
        flat_dir.mkdir()
        (flat_dir / "PROJECT_STATE.md").write_text(
            "flat state content", encoding="utf-8"
        )

        session = Session.new("layer1flat", workspace)
        session.append({"role": "user", "content": "hi", "source": "user"})
        ctx_mgr = ContextManager(
            session,
            _StubPromptBuilder(),
            _make_base_ctx(workspace),
            workspace_files=None,
        )

        result = ctx_mgr.prepare()
        system_blob = " ".join(
            m.get("content", "") for m in result if m.get("role") == "system"
        )

        assert "flat state content" in system_blob
