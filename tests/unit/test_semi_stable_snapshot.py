# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""1109 R6: the semi-stable system block is frozen per run, like Layer 1.

The semi-stable block (user memory, global preferences, project_goals.md,
user_directives.md, the skills list, deployment instructions …) sits ahead of
the whole history. It used to be rebuilt from disk on every call, so a
mid-run ``remember_about_user``, a "remember X" append to
user_directives.md, a new skill, or the user editing goals in the UI
rewrote the block, and the provider re-prefilled the entire conversation
from that point. The block is now taken once per run (``begin_run()``: run
start and after a compaction) and reused byte-for-byte; a change takes
effect the next run, and the tail names what changed meanwhile.
"""

from __future__ import annotations

import json
import os

from agent_os.agent.context import ContextManager
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager


def _rig(tmp_path, *, deploy=None):
    ws = str(tmp_path / "ws")
    os.makedirs(os.path.join(ws, "orbital", "instructions"))
    paths = ProjectPaths(ws)
    with open(paths.project_goals, "w", encoding="utf-8") as f:
        f.write("# Mission\nShip a newsletter weekly.\n")
    umem = str(tmp_path / "user_memory.md")
    with open(umem, "w", encoding="utf-8") as f:
        f.write("- Works at a startup\n")
    ctx = PromptContext(
        workspace=ws, model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[{"handle": "codex", "display_name": "Codex", "type": "cli"}],
        tool_names=["read", "write", "edit", "remember_about_user"],
        os_type="macos", datetime_now="2026-11-09T10:00", project_name="News",
        user_memory_path=umem,
    )
    session = Session.new("semi", ws)
    files = WorkspaceFileManager(ws)
    files.ensure_dir()
    mgr = ContextManager(
        session, PromptBuilder(ws), ctx, model_context_limit=200_000,
        workspace_files=files,
        sub_agent_deployment_instructions_provider=deploy,
    )
    session.append({"role": "user", "content": "hello", "source": "user"})
    return mgr, session, paths, umem


def _semi(messages):
    """The semi-stable system row (the second system message)."""
    return [m for m in messages if m.get("role") == "system"][1]["content"]


def _tail(messages):
    content = messages[-1].get("content")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _drift_line(messages):
    return next((l for l in _tail(messages).splitlines()
                 if l.startswith("[PROMPT SNAPSHOT]")), None)


def test_user_memory_written_mid_run_leaves_the_semi_block_byte_identical(tmp_path):
    mgr, session, _paths, umem = _rig(tmp_path)
    mgr.begin_run()
    first = mgr.prepare()
    assert "Works at a startup" in _semi(first)
    with open(umem, "a", encoding="utf-8") as f:
        f.write("- Prefers short answers\n")
    second = mgr.prepare()
    assert _semi(second) == _semi(first)
    assert "Prefers short answers" not in _semi(second)
    line = _drift_line(second)
    assert line is not None and "About the User" in line


def test_a_directive_appended_mid_run_takes_effect_next_run(tmp_path):
    mgr, session, paths, _umem = _rig(tmp_path)
    mgr.begin_run()
    first = mgr.prepare()
    with open(paths.user_directives, "w", encoding="utf-8") as f:
        f.write("- Always use British spelling\n")
    second = mgr.prepare()
    assert _semi(second) == _semi(first)
    assert "user_directives.md" in (_drift_line(second) or "")

    mgr.begin_run()
    third = mgr.prepare()
    assert "Always use British spelling" in _semi(third)
    assert _drift_line(third) is None


def test_goals_skills_and_deployment_instructions_are_frozen_too(tmp_path):
    deploy = {"text": "Use codex for code."}
    mgr, session, paths, _umem = _rig(tmp_path, deploy=lambda: deploy["text"])
    mgr.begin_run()
    first = mgr.prepare()
    with open(paths.project_goals, "a", encoding="utf-8") as f:
        f.write("- GOAL-EDITED-IN-UI\n")
    skill = os.path.join(paths.skills_dir, "weekly-digest")
    os.makedirs(skill)
    with open(os.path.join(skill, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: weekly-digest\ndescription: Build the digest\n---\nsteps\n")
    deploy["text"] = "Use claude-code for everything."
    second = mgr.prepare()
    assert _semi(second) == _semi(first)
    line = _drift_line(second)
    for name in ("project_goals.md", "skills", "deployment instructions"):
        assert name in line, (name, line)

    mgr.begin_run()
    semi = _semi(mgr.prepare())
    assert "GOAL-EDITED-IN-UI" in semi and "weekly-digest" in semi
    assert "Use claude-code for everything." in semi


def test_extra_instruction_files_are_frozen_per_run(tmp_path):
    mgr, session, paths, _umem = _rig(tmp_path)
    with open(os.path.join(paths.instructions_dir, "style.md"), "w", encoding="utf-8") as f:
        f.write("Write in a calm voice.\n")
    mgr.begin_run()
    first = mgr.prepare()
    row = [m["content"] for m in first if str(m.get("content", "")).startswith("[Project Instructions]")]
    assert row and "calm voice" in row[0]
    with open(os.path.join(paths.instructions_dir, "style.md"), "w", encoding="utf-8") as f:
        f.write("Write in a LOUD voice.\n")
    second = mgr.prepare()
    row2 = [m["content"] for m in second if str(m.get("content", "")).startswith("[Project Instructions]")]
    assert row2 == row
    assert "instructions/" in (_drift_line(second) or "")


def test_no_drift_line_when_nothing_changed(tmp_path):
    mgr, session, _paths, _umem = _rig(tmp_path)
    mgr.begin_run()
    mgr.prepare()
    assert _drift_line(mgr.prepare()) is None


def test_the_prefix_only_changes_at_the_tail_across_a_mid_run_remember(tmp_path):
    """The whole point: every message before the newest turn is byte-equal."""
    mgr, session, _paths, umem = _rig(tmp_path)
    mgr.begin_run()
    first = mgr.prepare()
    with open(umem, "a", encoding="utf-8") as f:
        f.write("- Prefers short answers\n")
    session.append({"role": "assistant", "content": "Noted.", "source": "management"})
    session.append({"role": "user", "content": "next", "source": "user"})
    second = mgr.prepare()
    diff_at = next(
        (i for i, (a, b) in enumerate(zip(first, second))
         if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)),
        min(len(first), len(second)),
    )
    assert diff_at >= len(first) - 1, f"prefix rewritten at {diff_at} of {len(first)}"
