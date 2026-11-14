# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Batch 1114 part C: the Workbench router routes a project with no goals and
no state on the head of its CLAUDE.md / user-written AGENTS.md / README
(daemon-read, capped like the goal excerpt, no model call). Separate from the
rest of batch 1114 so the coordinator can merge or drop it alone: the code is
``workbench._doc_head_part`` + ``workspace_pointers.doc_head``."""

from __future__ import annotations


def test_router_block_carries_the_doc_head_only_without_goals_or_state(tmp_path):
    from agent_os.api.routes.workbench import DOC_HEAD_CAP, project_block

    ws = tmp_path / "shop"
    ws.mkdir()
    project = {"project_id": "p1", "name": "shop", "workspace": str(ws)}
    assert "(head)" not in project_block(project, [])
    (ws / "README.md").write_text("# Etsy shop helpers\nPrice updates and stock check.\n")
    block = project_block(project, ["Take a look"])
    assert "README.md (head):\n# Etsy shop helpers\nPrice updates and stock check." in block
    assert block.index("README.md (head)") < block.index("Latest sessions")
    (ws / "CLAUDE.md").write_text("# Shop\n" + "line of rules\n" * 400)
    block = project_block(project, [])
    head = block.split("CLAUDE.md (head):\n", 1)[1]
    assert len(head) <= DOC_HEAD_CAP and "README" not in block
    goals = ws / "orbital" / "instructions" / "project_goals.md"
    goals.parent.mkdir(parents=True)
    goals.write_text("# Goals\nSell candles.\n")
    block = project_block(project, [])
    assert "Goals:\n# Goals\nSell candles." in block and "(head)" not in block


def test_router_block_skips_the_seeded_agents_md(tmp_path):
    from agent_os.api.routes.workbench import project_block
    from agent_os.daemon_v2.agent_md_seeder import AGENT_MD_TEMPLATE

    ws = tmp_path / "x"
    ws.mkdir()
    (ws / "AGENTS.md").write_text(AGENT_MD_TEMPLATE)
    assert "(head)" not in project_block({"project_id": "p", "workspace": str(ws)}, [])
