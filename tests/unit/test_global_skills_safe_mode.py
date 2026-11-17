# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Safe mode: the manager can read the shared skills its prompt lists (the
global folder and the user's ~/.agents, ~/.claude, ~/.codex skill folders),
and nothing beside them. Each skills folder itself is a read-only root of
the read tool only; grep/glob roots are unchanged."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from agent_os.agent import skills as skills_mod
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.tools.write import WriteTool
from agent_os.daemon_v2.agent_manager import AgentManager


def _manager(projects):
    store = MagicMock()
    by_id = {p["project_id"]: p for p in projects}
    store.get_project.side_effect = lambda pid: by_id.get(pid)
    store.list_projects.side_effect = lambda: list(projects)
    return AgentManager(project_store=store, ws_manager=MagicMock(),
                        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
                        process_manager=MagicMock(), platform_provider=None)


def _write(path, text):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    data = tmp_path / "data"
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    gskill = _write(data / "skills" / "tidy" / "SKILL.md", "# Tidy\nTidy things.\n")
    connectors = _write(data / "connectors.json", '{"secret": "x"}')
    hskill = _write(home / ".claude" / "skills" / "front" / "SKILL.md", "# Front\nF.\n")
    helper = _write(home / ".claude" / "skills" / "front" / "scripts" / "run.py", "print('HELPER')\n")
    settings = _write(home / ".claude" / "settings.json", "{}")
    monkeypatch.setattr(skills_mod, "_global_skills_dir", None)
    skills_mod.configure_global_skills_dir(str(data / "skills"))
    monkeypatch.setenv("ORBITAL_SKILLS_HOME", str(home))
    projects = [
        {"project_id": "safe", "name": "safe", "workspace": str(ws), "safe_mode": True},
        {"project_id": "open", "name": "open", "workspace": str(ws), "safe_mode": False},
    ]
    return dict(mgr=_manager(projects), ws=str(ws), data=data, home=home,
                gskill=gskill, connectors=connectors, hskill=hskill, settings=settings,
                helper=helper)


def _read_tool(rig, pid):
    mgr = rig["mgr"]
    roots_cb = lambda: mgr._read_roots_with_attached(pid, None, [rig["ws"]])
    return ReadTool(workspace=rig["ws"], read_roots=roots_cb, open_reads=lambda: False,
                    cross_project=False)


def test_safe_mode_read_roots_are_the_skill_folders_themselves(rig):
    roots = rig["mgr"]._read_roots_with_attached("safe", None, [rig["ws"]])
    assert roots == [rig["ws"],
                     os.path.realpath(str(rig["data"] / "skills")),
                     os.path.realpath(str(rig["home"] / ".claude" / "skills"))]


def test_safe_mode_manager_reads_listed_skills_and_nothing_beside_them(rig):
    tool = _read_tool(rig, "safe")
    assert "Tidy things." in tool.execute(path=rig["gskill"]).content
    assert "F." in tool.execute(path=rig["hskill"]).content
    assert "HELPER" in tool.execute(path=rig["helper"]).content
    for path in (rig["connectors"], rig["settings"]):
        result = tool.execute(path=path).content
        assert result.startswith("Error"), path
        assert "secret" not in result


def test_skill_folders_are_never_writable(rig):
    target = os.path.join(str(rig["data"] / "skills"), "tidy", "evil.md")
    result = WriteTool(workspace=rig["ws"]).execute(path=target, content="x")
    assert result.content.startswith("Error")
    assert not os.path.exists(target)


def test_open_project_and_missing_folders_change_nothing(rig, tmp_path, monkeypatch):
    mgr = rig["mgr"]
    assert mgr._read_roots_with_attached("open", None, [rig["ws"]]) == [rig["ws"]]
    monkeypatch.setenv("ORBITAL_SKILLS_HOME", str(tmp_path / "nobody"))
    skills_mod.configure_global_skills_dir(str(tmp_path / "absent" / "skills"))
    assert mgr._read_roots_with_attached("safe", None, [rig["ws"]]) == [rig["ws"]]


def test_grep_and_glob_roots_do_not_walk_skill_folders(rig):
    roots, _labels = rig["mgr"]._compute_scope_roots("safe", None, include_portals=True)
    assert roots == [rig["ws"]]


def test_home_folder_is_a_root_only_while_one_of_its_skills_is_offered(rig):
    skills_mod.set_skill_enabled("claude:front", False)
    roots = rig["mgr"]._read_roots_with_attached("safe", None, [rig["ws"]])
    assert roots == [rig["ws"], os.path.realpath(str(rig["data"] / "skills"))]
    assert _read_tool(rig, "safe").execute(path=rig["hskill"]).content.startswith("Error")
    skills_mod.set_skill_enabled("claude:front", True)
    assert os.path.realpath(str(rig["home"] / ".claude" / "skills")) in \
        rig["mgr"]._read_roots_with_attached("safe", None, [rig["ws"]])


def test_default_grep_does_not_search_skill_folders(rig):
    from agent_os.agent.tools.grep_tool import GrepTool, _find_ripgrep
    if _find_ripgrep() is None:
        pytest.skip("ripgrep not available")
    _write(os.path.join(rig["ws"], "notes.txt"), "needle in the workspace\n")
    _write(rig["data"] / "skills" / "tidy" / "more.md", "needle in a skill\n")
    mgr = rig["mgr"]
    tool = GrepTool(workspace=rig["ws"],
                    read_roots=lambda: mgr._compute_scope_roots("safe", None, include_portals=True)[0],
                    open_reads=lambda: False, cross_project=False)
    out = tool.execute(pattern="needle").content
    assert "needle in the workspace" in out
    assert "needle in a skill" not in out
