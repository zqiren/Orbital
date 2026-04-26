# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""When the resolver returns ``None`` (e.g. packaging bug), the installer
must log a WARNING naming platform+frozen state and NOT set the reconciled
flag. The next daemon start should retry once packaging is fixed."""

import logging
import sys

import pytest

from agent_os.daemon_v2 import default_skills_installer
from agent_os.daemon_v2.default_skills_installer import install_default_skills
from agent_os.daemon_v2.project_store import ProjectStore


def test_source_missing_logs_warning_and_preserves_flag(tmp_path, monkeypatch, caplog):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    store = ProjectStore(data_dir=str(data_dir))
    pid = store.create_project({
        "name": "NoSource",
        "workspace": str(workspace),
    })

    # Force the resolver to report "not found".
    monkeypatch.setattr(
        default_skills_installer, "_resolve_default_skills_dir", lambda: None
    )

    with caplog.at_level(logging.WARNING, logger="agent_os.daemon_v2.default_skills_installer"):
        result = install_default_skills(store, pid)

    assert result == {"status": "source_missing"}

    # Flag NOT set — retry must be possible on the next start.
    project = store.get_project(pid)
    assert project["default_skills_reconciled"] is False

    # WARNING log mentions frozen state and platform.
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("frozen=" in m and "platform=" in m for m in messages), (
        f"expected a WARNING log naming frozen state and platform, got: {messages}"
    )
