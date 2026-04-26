# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""In a dev checkout, the resolver must locate ``agent_os/default_skills/``
by walking up from the installer module's file path and return a directory
containing the 4 expected skill folders."""

import os
import sys

from agent_os.daemon_v2.default_skills_installer import _resolve_default_skills_dir


EXPECTED_SKILLS = {
    "efficient-execution",
    "learning-capture",
    "process-capture",
    "task-planning",
}


def test_resolver_finds_dev_source(monkeypatch):
    # Guarantee we are exercising the walk-up fallback, not a leaked _MEIPASS.
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    resolved = _resolve_default_skills_dir()
    assert resolved is not None, (
        "resolver returned None in a dev checkout — walk-up fallback failed "
        "to locate agent_os/default_skills/"
    )
    assert os.path.isdir(resolved), f"resolved path is not a directory: {resolved!r}"

    entries = {
        name for name in os.listdir(resolved)
        if os.path.isdir(os.path.join(resolved, name))
    }
    missing = EXPECTED_SKILLS - entries
    assert not missing, (
        f"resolved dir {resolved!r} is missing expected skill subdirs: {missing}"
    )
    for name in EXPECTED_SKILLS:
        skill_md = os.path.join(resolved, name, "SKILL.md")
        assert os.path.isfile(skill_md), f"missing SKILL.md in {skill_md!r}"
