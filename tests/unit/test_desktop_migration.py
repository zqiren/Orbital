# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import pytest
from unittest.mock import patch


def test_fresh_install_sets_version(tmp_path):
    with patch("agent_os.desktop.migration.DATA_DIR", str(tmp_path)), \
         patch("agent_os.desktop.migration.VERSION_FILE", str(tmp_path / "version.json")):
        from agent_os.desktop.migration import run_migrations, CURRENT_DATA_VERSION
        run_migrations()
        with open(tmp_path / "version.json") as f:
            assert json.load(f)["data_version"] == CURRENT_DATA_VERSION


def test_already_current_no_op(tmp_path):
    version_file = tmp_path / "version.json"
    with patch("agent_os.desktop.migration.DATA_DIR", str(tmp_path)), \
         patch("agent_os.desktop.migration.VERSION_FILE", str(version_file)):
        from agent_os.desktop.migration import CURRENT_DATA_VERSION
        version_file.write_text(json.dumps({"data_version": CURRENT_DATA_VERSION}))
        from agent_os.desktop.migration import run_migrations
        run_migrations()
        with open(version_file) as f:
            assert json.load(f)["data_version"] == CURRENT_DATA_VERSION


def test_migration_chain_runs_in_order(tmp_path):
    version_file = tmp_path / "version.json"
    call_order = []

    def migrate_1():
        call_order.append(1)

    def migrate_2():
        call_order.append(2)

    with patch("agent_os.desktop.migration.DATA_DIR", str(tmp_path)), \
         patch("agent_os.desktop.migration.VERSION_FILE", str(version_file)), \
         patch("agent_os.desktop.migration.CURRENT_DATA_VERSION", 3), \
         patch("agent_os.desktop.migration.MIGRATIONS", {1: migrate_1, 2: migrate_2}):
        version_file.write_text(json.dumps({"data_version": 1}))
        from agent_os.desktop.migration import run_migrations
        run_migrations()
        assert call_order == [1, 2]
        with open(version_file) as f:
            assert json.load(f)["data_version"] == 3


def test_ensure_data_dir_creates_directory(tmp_path):
    target = tmp_path / "subdir" / "AgentOS"
    with patch("agent_os.desktop.migration.DATA_DIR", str(target)):
        from agent_os.desktop.migration import ensure_data_dir
        ensure_data_dir()
        assert target.is_dir()
