# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""1109 Task B: new projects no longer get the four bundled skills.

The bundled skills (efficient-execution, learning-capture, process-capture,
task-planning) were opened in ~2% of 1,510 real sessions while users' own
skills are used heavily, so they stopped being installed. Existing projects
keep whatever copies they have: the project is the user's asset, so the
installer never deletes or rewrites a skill. It still heals skills stranded
at the legacy ``{workspace}/skills/`` path (test_installer_migrates_legacy_
skills.py) and still marks the project reconciled, so a downgrade to v0.21.0
does not push the bundled skills into a project created without them.
"""

import os
import pathlib

from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore

BUNDLED = ("efficient-execution", "learning-capture", "process-capture", "task-planning")
REPO = pathlib.Path(__file__).resolve().parents[2]


def _store_and_project(tmp_path, **fields):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({"name": "P", "workspace": str(workspace), **fields})
    return store, pid, workspace


def test_new_project_gets_no_bundled_skills(tmp_path):
    store, pid, workspace = _store_and_project(tmp_path)

    result = install_default_skills(store, pid)

    assert result["status"] == "ok"
    assert result["installed"] == []
    skills_dir = workspace / "orbital" / "skills"
    present = {p.name for p in skills_dir.iterdir()} if skills_dir.exists() else set()
    assert not present & set(BUNDLED)
    # Reconciled, so v0.21.0's installer (which copies when the flag is
    # False) leaves the project alone after a downgrade.
    assert store.get_project(pid)["default_skills_reconciled"] is True


def test_existing_copies_and_user_skills_are_left_byte_identical(tmp_path):
    store, pid, workspace = _store_and_project(tmp_path)
    skills_dir = workspace / "orbital" / "skills"
    before = {}
    for name in ("learning-capture", "task-planning", "my-own-skill"):
        d = skills_dir / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f"# {name}\nhand-tuned.\n", encoding="utf-8")
        before[name] = (d / "SKILL.md").read_bytes()

    result = install_default_skills(store, pid)

    assert result["status"] == "ok"
    assert {p.name for p in skills_dir.iterdir()} == set(before)
    for name, data in before.items():
        assert (skills_dir / name / "SKILL.md").read_bytes() == data


def test_existing_unreconciled_project_is_not_seeded(tmp_path):
    # A pre-fix project whose first install failed (flag still False) used to
    # be healed with the bundled skills on the next start. It no longer is.
    store, pid, workspace = _store_and_project(tmp_path)
    assert store.get_project(pid)["default_skills_reconciled"] is False

    install_default_skills(store, pid)

    skills_dir = workspace / "orbital" / "skills"
    assert not (skills_dir.exists() and any(skills_dir.iterdir()))


def test_bundled_skills_directory_is_gone_and_specs_do_not_bundle_it():
    assert not (REPO / "agent_os" / "default_skills").exists()
    for spec in ("agentos.spec", "agentos-macos.spec"):
        text = (REPO / "agent_os" / "desktop" / spec).read_text(encoding="utf-8")
        assert "default_skills" not in text, spec


def test_user_skill_still_reaches_the_prompt(tmp_path):
    """The skills index and the read-when-it-matches rule still work for a
    skill the user (or the agent) created."""
    from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext

    ws = tmp_path / "ws2"
    d = ws / "orbital" / "skills" / "weekly-report"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: weekly-report\ndescription: Build the Friday report.\n---\n",
        encoding="utf-8")
    _, semi, _ = PromptBuilder(str(ws)).build(PromptContext(
        workspace=str(ws), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write"], os_type="macos",
        datetime_now="2026-11-09T10:00"))
    assert "## Skills" in semi
    assert "read its SKILL.md" in semi
    assert ("- weekly-report: Build the Friday report. "
            "(at orbital/skills/weekly-report/SKILL.md)") in semi
    assert os.sep + "ws2" + os.sep not in semi.split("## Skills", 1)[1]
