# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a resume after a chat-mode write to PROJECT_STATE.md must
cause the next prepare() to re-read the file. The mechanism is the
_cold_resume_injected flag being cleared by AgentManager.switch_session
(used by QueueDispatcher.resume).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager


class _StubPromptBuilder:
    def build(self, context):
        return ("CACHED", "SEMI_STABLE", "DYNAMIC")


@pytest.mark.asyncio
async def test_cold_resume_flag_reset_causes_memory_reread(tmp_path):
    workspace = str(tmp_path)
    paths = ProjectPaths(workspace)
    os.makedirs(paths.orbital_dir, exist_ok=True)

    # Initial PROJECT_STATE.md content
    with open(paths.project_state, "w", encoding="utf-8") as f:
        f.write("# Initial state")

    sess = Session.new("sess_mem_test", workspace)
    wf = WorkspaceFileManager(workspace)

    ctx = PromptContext(
        workspace=workspace,
        model="test",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00Z",
        project_name="test",
        agent_name="test",
    )
    cm = ContextManager(
        sess, _StubPromptBuilder(), ctx,
        model_context_limit=200_000,
        workspace_files=wf,
        sub_agent_provider=lambda: [],
    )

    # First prepare() reads memory
    cm.prepare()
    assert cm._cold_resume_injected is True

    # Modify the file. Without flag reset, the cached injection wins.
    with open(paths.project_state, "w", encoding="utf-8") as f:
        f.write("# Mutated state — after chat")

    # Simulate what AgentManager.switch_session does on resume
    cm._cold_resume_injected = False

    # Spy on the WorkspaceFileManager.build_cold_resume_context output
    second = cm.prepare()
    assert cm._cold_resume_injected is True

    # The newly built result must contain the post-chat content somewhere
    # in the system messages it produced.
    all_text = " ".join(
        m.get("content", "") if isinstance(m.get("content"), str) else ""
        for m in second
        if isinstance(m, dict)
    )
    assert "Mutated state" in all_text
