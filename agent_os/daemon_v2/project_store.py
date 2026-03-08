# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""JSON-file-based project store."""

import json
import os
import re
from uuid import uuid4


def project_dir_name(project_name: str, project_id: str) -> str:
    """Build a slugified directory name: ``my-web-scraper-a3f2``.

    *project_name* is the human-readable project name (e.g. "My Web Scraper").
    *project_id* is the store-level id (e.g. "proj_a3f2b1c2d4e5").

    The result is ``<slugified-name>-<short-id>`` where *short-id* is the
    first 4 hex characters after the ``proj_`` prefix.
    """
    slug = re.sub(r'[^a-z0-9]+', '-', project_name.lower()).strip('-')
    slug = slug[:40] or 'project'
    short_id = project_id.replace('proj_', '')[:4]
    return f"{slug}-{short_id}"

DEFAULT_NOTIFICATION_PREFS = {
    "task_completed": True,
    "errors": True,
    "agent_messages": True,
    "trigger_started": False,
}


class ProjectStore:
    """CRUD for projects using JSON files on disk."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._projects_file = os.path.join(data_dir, "projects.json")
        self._projects: dict[str, dict] = {}
        self._load()

    def _agent_name_taken(self, agent_name: str, exclude_pid: str | None = None) -> bool:
        """Check if agent_name is already used by another project."""
        for pid, proj in self._projects.items():
            if pid == exclude_pid:
                continue
            if proj.get("agent_name", proj.get("name", "")) == agent_name:
                return True
        return False

    def _load(self) -> None:
        if os.path.exists(self._projects_file):
            with open(self._projects_file, "r", encoding="utf-8") as f:
                self._projects = json.load(f)
        else:
            self._projects = {}

    def _save(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        with open(self._projects_file, "w", encoding="utf-8") as f:
            json.dump(self._projects, f, indent=2, ensure_ascii=False)

    def list_projects(self) -> list[dict]:
        return list(self._projects.values())

    def get_project(self, project_id: str) -> dict | None:
        project = self._projects.get(project_id)
        if project is not None:
            prefs = project.get("notification_prefs", {})
            project["notification_prefs"] = {**DEFAULT_NOTIFICATION_PREFS, **prefs}
        return project

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Strip characters that are problematic in filenames."""
        s = re.sub(r'[^\w\s-]', '', name)
        return s.strip()

    def create_project(self, config: dict) -> str:
        pid = "proj_" + uuid4().hex[:12]
        # Sanitize project name to avoid filesystem issues
        raw_name = config.get("name", "")
        if raw_name:
            config["name"] = self._sanitize_name(raw_name)
        # Default agent_name to project name if not provided
        config.setdefault("agent_name", config.get("name", ""))
        # Default is_scratch to False
        config.setdefault("is_scratch", False)
        # Validate agent_name uniqueness
        if self._agent_name_taken(config["agent_name"]):
            raise ValueError(f"agent_name '{config['agent_name']}' already in use")
        project = {
            "project_id": pid,
            **config,
        }
        self._projects[pid] = project
        self._save()
        return pid

    def update_project(self, project_id: str, updates: dict) -> None:
        project = self._projects.get(project_id)
        if project is None:
            return
        # Validate agent_name uniqueness if being changed
        if "agent_name" in updates and self._agent_name_taken(updates["agent_name"], exclude_pid=project_id):
            raise ValueError(f"agent_name '{updates['agent_name']}' already in use")
        # Partial merge for notification_prefs
        if "notification_prefs" in updates and isinstance(updates["notification_prefs"], dict):
            existing = project.get("notification_prefs", {})
            updates["notification_prefs"] = {**DEFAULT_NOTIFICATION_PREFS, **existing, **updates["notification_prefs"]}
        project.update(updates)
        self._save()

    def find_scratch_project(self) -> dict | None:
        """Return the scratch project if one exists."""
        for proj in self._projects.values():
            if proj.get("is_scratch"):
                return proj
        return None

    def update_runtime(self, project_id: str, updates: dict) -> None:
        """Update runtime-managed fields (separate from user-editable config).

        Stored under a "runtime" key in the project dict.
        """
        project = self._projects.get(project_id)
        if project is None:
            return
        runtime = project.setdefault("runtime", {})
        runtime.update(updates)
        self._save()

    def delete_project(self, project_id: str) -> None:
        if self._projects.get(project_id, {}).get("is_scratch"):
            raise ValueError("cannot delete scratch project")
        self._projects.pop(project_id, None)
        self._save()
