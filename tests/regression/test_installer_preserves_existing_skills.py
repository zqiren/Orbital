# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-existing custom skill survives byte-for-byte. Other 3 bundled skills
install alongside it. Reconciled flag is set."""

import os

from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


def test_installer_does_not_overwrite_existing_skill_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    skills_dir = workspace / "orbital" / "skills"
    skills_dir.mkdir(parents=True)

    # User pre-customized learning-capture. The installer must not touch it.
    user_lc = skills_dir / "learning-capture"
    user_lc.mkdir()
    custom_body = "# my custom learning capture\n\nhand-tuned content"
    (user_lc / "SKILL.md").write_text(custom_body, encoding="utf-8")
    extra_note = user_lc / "notes.md"
    extra_note.write_text("personal notes", encoding="utf-8")

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "Preserved",
        "workspace": str(workspace),
    })

    result = install_default_skills(store, pid)
    assert result["status"] == "ok"
    assert "learning-capture" in result["skipped_existing"]
    assert set(result["installed"]) == {
        "efficient-execution", "process-capture", "task-planning",
    }

    # Byte-for-byte preservation of the customized skill.
    assert (user_lc / "SKILL.md").read_text(encoding="utf-8") == custom_body
    assert extra_note.read_text(encoding="utf-8") == "personal notes"

    # The other three bundled skills are now installed.
    for name in ("efficient-execution", "process-capture", "task-planning"):
        assert (skills_dir / name / "SKILL.md").is_file()

    project = store.get_project(pid)
    assert project["default_skills_reconciled"] is True
