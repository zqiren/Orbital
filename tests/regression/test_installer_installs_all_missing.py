# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fresh non-scratch project: installer creates all 4 default skills with a
valid SKILL.md, and sets ``default_skills_reconciled=True``."""

import os

from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


EXPECTED_SKILLS = {
    "efficient-execution",
    "learning-capture",
    "process-capture",
    "task-planning",
}


def test_installer_installs_all_four_skills(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "Fresh",
        "workspace": str(workspace),
    })

    result = install_default_skills(store, pid)
    assert result["status"] == "ok"
    assert set(result["installed"]) == EXPECTED_SKILLS
    assert result["skipped_existing"] == []

    skills_dir = workspace / "skills"
    assert skills_dir.is_dir()
    subdirs = {d.name for d in skills_dir.iterdir() if d.is_dir()}
    assert subdirs == EXPECTED_SKILLS
    for name in EXPECTED_SKILLS:
        skill_md = skills_dir / name / "SKILL.md"
        assert skill_md.is_file(), f"missing SKILL.md for {name}"
        assert skill_md.read_text(encoding="utf-8").strip(), (
            f"SKILL.md for {name} is empty"
        )

    project = store.get_project(pid)
    assert project["default_skills_reconciled"] is True
