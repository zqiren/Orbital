# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Default skills installer — the single source of truth for seeding a project's
``{workspace}/skills/`` directory with the bundled default skills.

Two consumers share this module:

* ``create_project`` (REST route): called once when a non-scratch project is
  created so a fresh workspace immediately shows the built-in skills.
* ``AgentManager.start_agent``: called every time an agent is started; the
  persistent ``default_skills_reconciled`` flag short-circuits subsequent runs
  so user deletions survive daemon restarts.

The resolver mirrors ``agent_os/agent/tools/grep_tool.py::_find_ripgrep``:
PyInstaller ``_MEIPASS`` first, then walk up from ``__file__`` looking for a
directory that contains ``agent_os/default_skills/``. Do NOT count ``..``
levels — this module must keep working if it is ever relocated within the
package tree.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_default_skills_dir() -> str | None:
    """Locate the bundled ``agent_os/default_skills/`` directory.

    Resolution order:

    1. ``sys._MEIPASS`` set (PyInstaller frozen build) → check
       ``{_MEIPASS}/agent_os/default_skills/``.
    2. Otherwise, walk up from this file's parent, up to 5 levels, looking for
       a directory that contains ``agent_os/default_skills/``.

    Returns the first existing directory as an absolute path, or ``None``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "agent_os" / "default_skills"
        if candidate.is_dir():
            return str(candidate)
        # When frozen and _MEIPASS lacks the dir we do NOT fall through to the
        # walk-up — __file__ inside a frozen bundle points into the extracted
        # bundle and would just resolve to the same missing path. Return None
        # so the caller emits a WARNING naming the frozen/platform context.
        return None

    here = Path(__file__).resolve().parent
    # Walk upward up to 5 ancestors (including `here` itself).
    for ancestor in [here, *here.parents][:6]:
        candidate = ancestor / "agent_os" / "default_skills"
        if candidate.is_dir():
            return str(candidate)
    return None


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
    """Reconcile ``{workspace}/orbital/skills/`` for *project_id* with bundled
    defaults, first healing any skills stranded at the legacy layout path.

    Contract (see TASK-fix-default-skills-not-loaded.md §3.1):

    * Scratch projects → ``{"status": "skipped_scratch"}``, no side effects.
    * Skills stranded at the legacy ``{workspace}/skills/`` path are relocated
      into ``{workspace}/orbital/skills/`` first, additively. This runs even
      when ``default_skills_reconciled`` is ``True`` (that is exactly the
      broken state) and even when the bundled source is missing (a move does
      not need the source). See ``_migrate_legacy_skills``.
    * ``default_skills_reconciled`` already ``True`` →
      ``{"status": "skipped_already_reconciled"}``, no default re-seeding
      (respects user deletions).
    * Source directory not resolvable → WARNING log naming frozen state and
      platform, returns ``{"status": "source_missing"}``, flag NOT set so the
      next start retries.
    * Otherwise, copy each bundled skill whose destination directory does not
      exist; skip the ones that already exist (additive, never overwrite).
    * On success, set ``default_skills_reconciled=True``.
    * On ``shutil.copytree`` failure, log ERROR with the skill name, leave
      the flag unset, and re-raise so the caller can log/swallow.

    Raises:
        ValueError: the project does not exist.
        OSError / shutil.Error: a ``copytree`` call failed (partial progress
            preserved, flag not set, retry permitted on next start).
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

    src_dir = _resolve_default_skills_dir()
    if src_dir is None:
        logger.warning(
            "default skills source dir not found: frozen=%s platform=%s project_id=%s",
            bool(getattr(sys, "_MEIPASS", None)),
            sys.platform,
            project_id,
        )
        return {"status": "source_missing"}

    from agent_os.agent.project_paths import ProjectPaths
    dest_root = ProjectPaths(workspace).skills_dir
    os.makedirs(dest_root, exist_ok=True)

    installed: list[str] = []
    skipped_existing: list[str] = []

    for entry in sorted(os.listdir(src_dir)):
        src_skill = os.path.join(src_dir, entry)
        if not os.path.isdir(src_skill):
            continue
        if not os.path.isfile(os.path.join(src_skill, "SKILL.md")):
            continue
        dest_skill = os.path.join(dest_root, entry)
        if os.path.exists(dest_skill):
            skipped_existing.append(entry)
            continue
        try:
            shutil.copytree(src_skill, dest_skill)
        except Exception:
            logger.error(
                "failed to copy default skill %r into project %s",
                entry, project_id, exc_info=True,
            )
            # Do NOT set the reconciled flag — retry on next start.
            raise
        installed.append(entry)

    project_store.update_project(
        project_id, {"default_skills_reconciled": True}
    )

    if installed:
        logger.info(
            "installed default skills into project %s: %s (skipped existing: %s)",
            project_id, installed, skipped_existing,
        )
    return {
        "status": "ok",
        "installed": installed,
        "skipped_existing": skipped_existing,
    }
