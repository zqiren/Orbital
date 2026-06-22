# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy-layout skill recovery.

Before the "collapse layout to orbital/" change, a project's skills lived at
``{workspace}/skills/``. That change repointed the reader (``SkillLoader``) to
``{workspace}/orbital/skills/`` but removed the migration shim, so skills
created under the old layout became invisible in the settings registry. Worse,
``default_skills_reconciled`` was already ``True`` for those projects, so the
installer's short-circuit meant the defaults were never re-seeded either — the
registry showed nothing even though the skill files (including user-authored
ones) still existed on disk.

``install_default_skills`` must heal such projects by relocating the stranded
skills into the current location, additively, even when reconciled.
"""

from agent_os.agent.skills import SkillLoader
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


def _write_skill(root, name, body):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")
    return d


def test_installer_migrates_stranded_legacy_skills_for_reconciled_project(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # Legacy layout: skills stranded at {workspace}/skills/, NOTHING under
    # {workspace}/orbital/skills/. Two bundled defaults plus a user-authored
    # skill that must not be lost.
    legacy = workspace / "skills"
    _write_skill(
        legacy, "task-planning",
        "# Task Planning\n\nMatch planning effort to task complexity.",
    )
    _write_skill(
        legacy, "learning-capture",
        "# Learning Capture\n\nLog corrections, errors, and discoveries.",
    )
    user_body = "# Humanizer\n\nRewrite text so it sounds human."
    _write_skill(legacy, "humanizer", user_body)

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({"name": "Stranded", "workspace": str(workspace)})
    # The exact bug state: reconciled flag is True but the skills are at the
    # legacy path, so the registry shows nothing and reconcile never re-seeds.
    store.update_project(pid, {"default_skills_reconciled": True})

    install_default_skills(store, pid)

    # Skills are now at the location the scanner reads.
    new_skills = workspace / "orbital" / "skills"
    for name in ("task-planning", "learning-capture", "humanizer"):
        assert (new_skills / name / "SKILL.md").is_file(), f"{name} not migrated"

    # The user-authored skill survives byte-for-byte.
    assert (new_skills / "humanizer" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == user_body

    # The scanner the settings UI uses now returns all three.
    found = {s["dir_name"] for s in SkillLoader(str(workspace)).scan()}
    assert {"task-planning", "learning-capture", "humanizer"} <= found


def test_legacy_migration_is_additive_and_never_overwrites(tmp_path):
    """A skill the user already customized at the new path wins; the stale
    legacy copy of the same name is not moved over it."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # New-path copy the user has edited.
    new_skills = workspace / "orbital" / "skills"
    custom_body = "# Task Planning\n\nMY customized planning skill."
    _write_skill(new_skills, "task-planning", custom_body)

    # Stale legacy copy of the same skill, plus a legacy-only skill.
    legacy = workspace / "skills"
    _write_skill(
        legacy, "task-planning",
        "# Task Planning\n\nOld stale legacy version.",
    )
    _write_skill(legacy, "process-capture", "# Process Capture\n\nCapture flows.")

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({"name": "Mixed", "workspace": str(workspace)})
    store.update_project(pid, {"default_skills_reconciled": True})

    install_default_skills(store, pid)

    # The customized new-path skill is untouched.
    assert (new_skills / "task-planning" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == custom_body
    # The legacy-only skill was migrated across.
    assert (new_skills / "process-capture" / "SKILL.md").is_file()


def test_no_legacy_dir_is_a_noop(tmp_path):
    """A project born under the orbital/ layout (no legacy skills/ dir) is
    unaffected — migration must not create spurious dirs or crash."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({"name": "Fresh", "workspace": str(workspace)})

    # Not reconciled — this exercises the normal install path with no legacy dir.
    result = install_default_skills(store, pid)
    assert result["status"] == "ok"
    assert not (workspace / "skills").exists()
