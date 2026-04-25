# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scratch projects never receive default skills, and the reconciled flag is
NOT set for them (there is nothing to reconcile)."""

import os

from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


def test_installer_skips_scratch_project(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "scratch-workspace"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "Scratch",
        "workspace": str(workspace),
        "is_scratch": True,
    })

    result = install_default_skills(store, pid)
    assert result == {"status": "skipped_scratch"}

    # No skills/ dir created.
    assert not (workspace / "skills").exists(), (
        "scratch project should not receive skills/ dir"
    )

    # Reconciled flag must NOT be set for scratch projects (there is
    # nothing to reconcile — the scratch branch short-circuits entirely).
    project = store.get_project(pid)
    assert project["default_skills_reconciled"] is False
