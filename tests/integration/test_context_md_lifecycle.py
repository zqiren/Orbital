# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test (Test 5) — CONTEXT.md session-end lifecycle.

Uses a mock LLM (no live API key) to drive run_session_end_routine and verify:
  - an existing CONTEXT.md (now INDEX.md — the one file the memory editor may
    rewrite whole, spec 089) is UPDATED (overwritten), not destroyed
  - old flat "- entry" content is replaced by the new section format the LLM
    is now asked to produce
  - the updated CONTEXT.md is then injected into the next prepare() context
"""

from __future__ import annotations

import pytest

from agent_os.agent import memory_editor
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine
from agent_os.agent.session import Session
from agent_os.agent.context import ContextManager
from agent_os.agent.prompt_builder import PromptContext, Autonomy


class _MockPromptBuilder:
    def build(self, context):
        return ("cached", "semi", "dynamic")


def _base_ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read"], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )


def _memory_turn(mgr, index_text):
    """A Memory-session turn that rewrites INDEX.md to ``index_text``."""
    async def run_turn(task_text):
        mgr.write("index", index_text)
        return memory_editor.TurnResult(mode="disk", llm_calls=1, ended="text_complete")
    return run_turn


NEW_CONTEXT = (
    "## Overview\nOrbital agent OS test project.\n\n"
    "## Key Files\n- agent_os/agent/loop.py: core loop\n\n"
    "## Architecture\nFastAPI backend + React frontend.\n\n"
    "## Conventions\npytest, ruff.\n\n"
    "## External Context\n- Moonshot API (Kimi)"
)


@pytest.mark.asyncio
async def test_session_end_restructures_old_context_md(tmp_path):
    mgr = WorkspaceFileManager(str(tmp_path))
    mgr.ensure_dir()
    # Old flat format CONTEXT.md (external-entities-only era)
    mgr.write("index", "- **Tencent:** vendor\n- **Moonshot:** LLM API\n")

    session = Session.new("ctxlife", str(tmp_path))
    session.append({"role": "user", "content": "did stuff", "source": "user"})

    run_turn = _memory_turn(mgr, NEW_CONTEXT)

    await run_session_end_routine(
        session, None, mgr, session_uuid=session.session_uuid, force=True,
        run_turn=run_turn,
    )

    updated = mgr.read("index")
    assert updated is not None
    assert "## Overview" in updated and "## Key Files" in updated
    # old flat-only content replaced (no longer just dash entries); the write
    # path self-heals the <!--format--> header on top.
    assert updated.endswith(NEW_CONTEXT)
    assert "- **Tencent:** vendor" not in updated


@pytest.mark.asyncio
async def test_updated_context_md_appears_in_next_prepare(tmp_path):
    mgr = WorkspaceFileManager(str(tmp_path))
    mgr.ensure_dir()
    mgr.write("index", "- old entry\n")

    session = Session.new("ctxlife2", str(tmp_path))
    session.append({"role": "user", "content": "x", "source": "user"})

    run_turn = _memory_turn(mgr, NEW_CONTEXT)
    await run_session_end_routine(
        session, None, mgr, session_uuid=session.session_uuid, force=True,
        run_turn=run_turn,
    )

    # Fresh session reads the updated file from disk
    session2 = Session.new("ctxlife2b", str(tmp_path))
    session2.append({"role": "user", "content": "y", "source": "user"})
    cm = ContextManager(session2, _MockPromptBuilder(), _base_ctx(str(tmp_path)))
    result = cm.prepare()
    blob = "\n".join(m.get("content", "") for m in result)
    assert "Orbital agent OS test project." in blob
