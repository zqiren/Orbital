# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import pytest
from agent_os.daemon_v2.project_store import ProjectStore


class TestAgentNameUniqueness:
    def test_create_project_with_agent_name(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({"name": "P1", "agent_name": "Bot1"})
        project = store.get_project(pid)
        assert project["agent_name"] == "Bot1"

    def test_agent_name_defaults_to_project_name(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({"name": "MyProject"})
        project = store.get_project(pid)
        assert project.get("agent_name") == "MyProject"

    def test_create_rejects_duplicate_agent_name(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        store.create_project({"name": "P1", "agent_name": "Bot"})
        with pytest.raises(ValueError, match="agent_name.*already in use"):
            store.create_project({"name": "P2", "agent_name": "Bot"})

    def test_update_rejects_duplicate_agent_name(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        store.create_project({"name": "P1", "agent_name": "Bot1"})
        pid2 = store.create_project({"name": "P2", "agent_name": "Bot2"})
        with pytest.raises(ValueError, match="agent_name.*already in use"):
            store.update_project(pid2, {"agent_name": "Bot1"})

    def test_update_same_agent_name_on_same_project_ok(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({"name": "P1", "agent_name": "Bot1"})
        store.update_project(pid, {"agent_name": "Bot1"})  # no error
        assert store.get_project(pid)["agent_name"] == "Bot1"


class TestIsScratch:
    def test_is_scratch_flag_persisted(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({"name": "Quick Tasks", "is_scratch": True})
        project = store.get_project(pid)
        assert project["is_scratch"] is True

    def test_find_scratch_project(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        store.create_project({"name": "Normal"})
        pid = store.create_project({"name": "Quick Tasks", "is_scratch": True})
        scratch = store.find_scratch_project()
        assert scratch is not None
        assert scratch["project_id"] == pid

    def test_find_scratch_returns_none_when_absent(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        store.create_project({"name": "Normal"})
        assert store.find_scratch_project() is None

    def test_scratch_project_cannot_be_deleted(self, tmp_path):
        store = ProjectStore(data_dir=str(tmp_path))
        pid = store.create_project({"name": "Quick Tasks", "is_scratch": True})
        with pytest.raises(ValueError, match="cannot.*delete.*scratch"):
            store.delete_project(pid)
