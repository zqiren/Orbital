# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: context.prepare() must read Layer 1 state files
from the canonical flat orbital/ path via ProjectPaths.

History: Originally this test enforced reads from a slug-namespaced path
(workspace/orbital/{project_dir_name}/PROJECT_STATE.md). After TASK-02 collapsed
the on-disk layout to a single flat orbital/ tree, the namespaced layout no
longer exists. After TASK-03, all readers route through ProjectPaths so the
flat tree is the only valid location. Tests below now seed the flat layout
and assert prepare() picks it up — both with an explicit WorkspaceFileManager
and with the default (which ContextManager auto-constructs).
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
    """context.prepare() must read Layer 1 files from the flat orbital/ tree."""

    def test_reads_state_file_with_workspace_file_manager(self, tmp_path):
        """prepare() should include PROJECT_STATE.md content from
        workspace/orbital/PROJECT_STATE.md when an explicit WFM is provided."""
        workspace = str(tmp_path)

        # Create the flat state file (only valid post-TASK-02 location).
        flat_dir = tmp_path / "orbital"
        flat_dir.mkdir()
        (flat_dir / "PROJECT_STATE.md").write_text(
            "flat state content", encoding="utf-8"
        )

        session = Session.new("layer1ns", workspace)
        session.append({"role": "user", "content": "hi", "source": "user"})
        wf = WorkspaceFileManager(workspace=workspace)
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

        assert "flat state content" in system_blob, (
            "Expected Layer 1 reader to pick up the flat PROJECT_STATE.md"
        )

    def test_reads_decisions_and_lessons_as_layer1(self, tmp_path):
        """All three Layer 1 files (state, decisions, lessons) should be
        read from workspace/orbital/ AND appear as Layer 1 messages
        (with the [FILENAME.md] header), not just as cold-resume context."""
        workspace = str(tmp_path)
        flat_dir = tmp_path / "orbital"
        flat_dir.mkdir()
        (flat_dir / "PROJECT_STATE.md").write_text("ns-state", encoding="utf-8")
        (flat_dir / "DECISIONS.md").write_text("ns-decisions", encoding="utf-8")
        (flat_dir / "LESSONS.md").write_text("ns-lessons", encoding="utf-8")

        session = Session.new("layer1ns2", workspace)
        session.append({"role": "user", "content": "hi", "source": "user"})
        wf = WorkspaceFileManager(workspace=workspace)
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
            "PROJECT_STATE.md content missing from Layer 1 messages"
        )
        assert "ns-decisions" in layer1_blob, (
            "DECISIONS.md content missing from Layer 1 messages"
        )
        assert "ns-lessons" in layer1_blob, (
            "LESSONS.md content missing from Layer 1 messages"
        )

    def test_default_workspace_file_manager_used_when_none_passed(self, tmp_path):
        """When workspace_files=None is passed, ContextManager builds its own
        WorkspaceFileManager from the prompt context's workspace, and Layer 1
        files at the flat orbital/ path are still read."""
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
