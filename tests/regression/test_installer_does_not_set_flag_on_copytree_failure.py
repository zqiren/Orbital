# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""If a ``shutil.copytree`` call fails mid-install, the installer must:

* log an ERROR naming the failing skill,
* NOT set ``default_skills_reconciled``,
* leave any skills already installed in place (partial progress preserved),
* re-raise so the caller can decide how to handle.

The caller (``create_project``, ``start_agent``) wraps the call in a
try/except so agent start / project creation still succeeds."""

import logging
import os

import pytest

from agent_os.daemon_v2 import default_skills_installer
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


def test_partial_copy_failure_preserves_first_skill_and_does_not_set_flag(
    tmp_path, monkeypatch, caplog,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "PartialFail",
        "workspace": str(workspace),
    })

    import shutil as _shutil
    real_copytree = _shutil.copytree

    call_count = {"n": 0}

    def flaky_copytree(src, dst, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("synthetic disk failure on second skill")
        return real_copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(default_skills_installer.shutil, "copytree", flaky_copytree)

    with caplog.at_level(logging.ERROR, logger="agent_os.daemon_v2.default_skills_installer"):
        with pytest.raises(OSError, match="synthetic disk failure"):
            install_default_skills(store, pid)

    # Exactly one skill was successfully installed (the first), the second
    # raised before anything was created. Partial progress must remain.
    skills_dir = workspace / "orbital" / "skills"
    installed = sorted(d.name for d in skills_dir.iterdir() if d.is_dir())
    assert len(installed) == 1, (
        f"expected exactly one successfully-installed skill to remain, got: {installed}"
    )

    # Flag NOT set — retry must be allowed on next start.
    project = store.get_project(pid)
    assert project["default_skills_reconciled"] is False

    # ERROR log names the failing skill.
    error_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("default skill" in m for m in error_msgs), (
        f"expected an ERROR log mentioning the failing skill, got: {error_msgs}"
    )
