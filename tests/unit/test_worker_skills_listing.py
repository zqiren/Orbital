# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workers learn the skills: the listing a worker is sent.

The same enabled set the manager lists (the project's own skills, Orbital's
global ones, the other agents' home folders), minus what that worker loads
natively. Established by asking the live CLIs (2026-11-17, scratch HOME with
one probe skill per folder):

- Claude Code 2.1.326's ``init.skills`` listed only ``~/.claude/skills`` and
  ``<ws>/.claude/skills`` (not ``~/.agents``, ``~/.codex``, ``<ws>/.agents``).
- Codex 0.157.1's ``skills/list`` listed ``~/.agents/skills``,
  ``~/.codex/skills``, ``<ws>/.agents/skills`` and ``<ws>/.codex/skills``
  (not ``~/.claude``, ``<ws>/.claude``).

Any other agent gets the full list.
"""

from __future__ import annotations

import os

import pytest

from agent_os.agent.skills import worker_skills, worker_skills_note


def _skill(folder, name, description="Does a thing."):
    os.makedirs(os.path.join(folder, name), exist_ok=True)
    with open(os.path.join(folder, name, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\nBody.\n")
    return os.path.join(folder, name, "SKILL.md")


@pytest.fixture
def layout(tmp_path):
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    glob = tmp_path / "data" / "skills"
    paths = {
        "project": _skill(str(ws / "orbital" / "skills"), "proj-skill"),
        "global": _skill(str(glob), "global-skill"),
        "agents": _skill(str(home / ".agents" / "skills"), "agents-skill"),
        "claude": _skill(str(home / ".claude" / "skills"), "claude-skill"),
        "codex": _skill(str(home / ".codex" / "skills"), "codex-skill"),
    }
    return {"home": str(home), "ws": str(ws), "global": str(glob), "paths": paths}


def _names(rows):
    return [r["name"] for r in rows]


def _ws(layout, agent, **kw):
    return worker_skills(layout["ws"], agent, home=layout["home"],
                         global_dir=layout["global"], disabled=kw.pop("disabled", set()), **kw)


def test_unknown_agent_gets_the_managers_full_set(layout):
    rows = _ws(layout, "aider")
    assert _names(rows) == ["proj-skill", "global-skill", "agents-skill",
                            "claude-skill", "codex-skill"]
    assert all(os.path.isabs(r["path"]) for r in rows)


def test_claude_code_is_not_told_what_it_loads_itself(layout):
    assert _names(_ws(layout, "claude-code")) == [
        "proj-skill", "global-skill", "agents-skill", "codex-skill"]


def test_codex_is_not_told_what_it_loads_itself(layout):
    assert _names(_ws(layout, "codex")) == ["proj-skill", "global-skill", "claude-skill"]


def test_a_symlink_from_a_native_folder_counts_as_native(layout, tmp_path):
    # The usual Mac layout: ~/.claude/skills/x -> ~/.agents/skills/x. The
    # catalog lists it once under "agents"; Claude Code loads it natively.
    target = os.path.dirname(_skill(os.path.join(layout["home"], ".agents", "skills"), "linked"))
    link = os.path.join(layout["home"], ".claude", "skills", "linked")
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    assert "linked" not in _names(_ws(layout, "claude-code"))
    assert "linked" in _names(_ws(layout, "aider"))


def test_a_workspace_native_skill_with_the_same_name_wins(layout):
    _skill(os.path.join(layout["ws"], ".claude", "skills"), "global-skill", "ws copy")
    assert "global-skill" not in _names(_ws(layout, "claude-code"))
    assert "global-skill" in _names(_ws(layout, "codex"))


def test_switched_off_skills_are_left_out(layout):
    rows = _ws(layout, "aider", disabled={"global:global-skill", "claude:claude-skill"})
    assert _names(rows) == ["proj-skill", "agents-skill", "codex-skill"]


def test_nothing_to_tell_gives_an_empty_note(tmp_path):
    rows = worker_skills(str(tmp_path / "ws"), "claude-code", home=str(tmp_path / "h"),
                         global_dir=None, disabled=set())
    assert rows == []
    assert worker_skills_note(rows) == ""


def test_the_note_is_one_instruction_line_and_one_line_per_skill(layout):
    rows = _ws(layout, "codex")
    note = worker_skills_note(rows)
    lines = note.split("\n")
    assert len(lines) == 1 + len(rows)
    assert "SKILL.md" in lines[0]
    assert lines[1] == f"- proj-skill: Does a thing. ({layout['paths']['project']})"
    assert lines[3] == f"- claude-skill: Does a thing. ({layout['paths']['claude']})"


def test_descriptions_are_one_line_and_capped(layout):
    long = "word " * 200
    _skill(layout["global"], "long-skill", long)
    rows = _ws(layout, "codex")
    line = next(l for l in worker_skills_note(rows).split("\n") if l.startswith("- long-skill"))
    desc = line[len("- long-skill: "):line.rindex(" (")]
    assert len(desc) <= 200 and desc.endswith("…")


def test_the_note_is_deterministic(layout):
    assert worker_skills_note(_ws(layout, "codex")) == worker_skills_note(_ws(layout, "codex"))


def test_no_workspace_still_lists_shared_skills(layout):
    rows = worker_skills("", "aider", home=layout["home"], global_dir=layout["global"],
                         disabled=set())
    assert _names(rows) == ["global-skill", "agents-skill", "claude-skill", "codex-skill"]
