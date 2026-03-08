# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Skill metadata scanner for Agent OS.

Scans workspace/skills/{name}/SKILL.md files and returns metadata.
"""

import os


class SkillLoader:
    """Scan for skill definitions in the workspace."""

    def __init__(self, workspace: str):
        self.workspace = workspace
        self._skills_dir = os.path.join(workspace, "skills")

    def scan(self) -> list[dict]:
        """Scan for SKILL.md files. Return metadata only (not full content).

        Returns [{"name": str, "description": str, "path": str}, ...]
        Returns [] if no skills directory.
        """
        if not os.path.isdir(self._skills_dir):
            return []
        results = []
        for entry in os.scandir(self._skills_dir):
            if entry.is_dir():
                skill_md = os.path.join(entry.path, "SKILL.md")
                if os.path.isfile(skill_md):
                    meta = self._parse_front_matter(skill_md)
                    if meta:
                        meta["path"] = skill_md
                        meta["dir_name"] = entry.name
                        results.append(meta)
        return results

    def _parse_front_matter(self, filepath: str) -> dict | None:
        """Read first ~20 lines for name and description.

        Expected format:
        # Skill Name
        Description text here.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    lines.append(line.rstrip())
        except OSError:
            return None

        name = None
        description = None

        for line in lines:
            if line.startswith("# ") and name is None:
                name = line[2:].strip()
            elif name is not None and line.strip():
                description = line.strip()
                break

        if name and description:
            return {"name": name, "description": description}
        return None
