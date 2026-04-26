# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""User-deletion-respect test.

Once ``default_skills_reconciled`` is ``True``, the installer must NOT scan
the workspace or re-install skills, even if the ``skills/`` directory is
empty. A user who deleted a default skill via the UI expects it to stay
deleted across daemon restarts."""

import os

from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


def test_installer_short_circuits_when_reconciled(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "skills").mkdir()  # empty skills dir — user deleted everything

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "ReconciledProject",
        "workspace": str(workspace),
    })
    # Simulate a previously-reconciled project.
    store.update_project(pid, {"default_skills_reconciled": True})

    result = install_default_skills(store, pid)
    assert result == {"status": "skipped_already_reconciled"}

    # No disk changes — the skills dir is still empty.
    entries = os.listdir(workspace / "skills")
    assert entries == [], (
        f"reconciled project must not get skills reinstalled, but found: {entries}"
    )
