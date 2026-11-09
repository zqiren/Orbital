# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Project skills reconciler — the one place a project's ``orbital/skills/``
directory is set up.

Orbital no longer ships default skills (1109 Task B): the four it bundled
(efficient-execution, learning-capture, process-capture, task-planning) were
opened in ~2% of 1,510 real sessions, while users' own skills are used
heavily. A project's skills are now only the ones the user or the agent
writes. Existing projects keep whatever copies they already have: the project
is the user's asset, so nothing here deletes or rewrites a skill.

Three consumers share this module (the function keeps its name so they do
not change):

* ``create_project`` (REST route) and the deferred onboarding seed: once,
  when a non-scratch project is created.
* ``AgentManager.start_agent``: every agent start; the persistent
  ``default_skills_reconciled`` flag short-circuits after the first run.

What is left to do is heal skills stranded at the pre-``orbital/`` layout
path and mark the project reconciled.
"""

import logging
import os
import shutil

logger = logging.getLogger(__name__)


def _migrate_legacy_skills(workspace: str) -> list[str]:
    """Relocate skills stranded at the pre-``orbital/`` layout path.

    Before the "collapse layout to orbital/" change, a project's skills lived
    at ``{workspace}/skills/``. That change repointed ``SkillLoader`` to
    ``{workspace}/orbital/skills/`` but removed the migration shim, so skills
    created under the old layout became invisible in the settings registry —
    and because ``default_skills_reconciled`` was already ``True`` for those
    projects, the installer's short-circuit never re-seeded the new location
    either.

    Move each legacy skill directory (one containing a ``SKILL.md``) into the
    current location, **additively** — a directory whose name already exists at
    the destination is left untouched, so a skill the user has since customized
    at the new path always wins over its stale legacy copy. Non-skill entries
    (no ``SKILL.md``) are never moved, so an unrelated ``skills/`` folder in a
    user's own repo is left alone.

    Returns the list of directory names moved (empty if there was nothing to do).
    """
    from agent_os.agent.project_paths import ProjectPaths

    legacy_root = os.path.join(workspace, "skills")
    if not os.path.isdir(legacy_root):
        return []

    dest_root = ProjectPaths(workspace).skills_dir
    if os.path.abspath(legacy_root) == os.path.abspath(dest_root):
        return []  # defensive: legacy and current must never resolve equal

    moved: list[str] = []
    for entry in sorted(os.listdir(legacy_root)):
        src_skill = os.path.join(legacy_root, entry)
        if not os.path.isdir(src_skill):
            continue
        if not os.path.isfile(os.path.join(src_skill, "SKILL.md")):
            continue
        dest_skill = os.path.join(dest_root, entry)
        if os.path.exists(dest_skill):
            continue  # additive — never overwrite an existing skill
        os.makedirs(dest_root, exist_ok=True)
        shutil.move(src_skill, dest_skill)
        moved.append(entry)

    # Tidy up a legacy dir we just emptied, but only if we actually moved
    # something — never disturb a dir we did not touch (keeps the
    # reconciled-with-empty-legacy short-circuit a true no-op).
    if moved and os.path.isdir(legacy_root) and not os.listdir(legacy_root):
        try:
            os.rmdir(legacy_root)
        except OSError:
            pass

    return moved


def install_default_skills(project_store, project_id: str) -> dict:
    """Reconcile ``{workspace}/orbital/skills/`` for *project_id*.

    Contract:

    * Scratch projects → ``{"status": "skipped_scratch"}``, no side effects.
    * Skills stranded at the legacy ``{workspace}/skills/`` path are relocated
      into ``{workspace}/orbital/skills/`` first, additively. This runs even
      when ``default_skills_reconciled`` is ``True`` (that is exactly the
      broken state). See ``_migrate_legacy_skills``.
    * ``default_skills_reconciled`` already ``True`` →
      ``{"status": "skipped_already_reconciled"}``.
    * Otherwise set ``default_skills_reconciled=True`` and return
      ``{"status": "ok", "installed": [], "skipped_existing": []}``. No skill
      is copied: Orbital bundles none. The flag still matters for a downgrade
      — v0.21.0's installer copies its bundled skills while the flag is
      False, so setting it keeps a project created without them that way.

    Raises:
        ValueError: the project does not exist.
    """
    project = project_store.get_project(project_id)
    if project is None:
        raise ValueError(f"project {project_id!r} not found")

    if project.get("is_scratch"):
        return {"status": "skipped_scratch"}

    # Heal pre-``orbital/`` projects whose skills are stranded at the legacy
    # ``{workspace}/skills/`` path. Must run before the reconciled
    # short-circuit below — those projects are precisely the ones with the
    # flag already set.
    workspace = project.get("workspace", "")
    if workspace:
        migrated = _migrate_legacy_skills(workspace)
        if migrated:
            logger.info(
                "migrated %d legacy skill(s) into orbital/skills for project %s: %s",
                len(migrated), project_id, migrated,
            )

    if project.get("default_skills_reconciled") is True:
        return {"status": "skipped_already_reconciled"}

    project_store.update_project(
        project_id, {"default_skills_reconciled": True}
    )
    return {"status": "ok", "installed": [], "skipped_existing": []}
