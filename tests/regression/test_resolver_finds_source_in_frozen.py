# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""When running under PyInstaller, ``sys._MEIPASS`` is set to the bundle
extraction root. The resolver must check ``{_MEIPASS}/agent_os/default_skills/``
before anything else and return that path if present."""

import os
import sys

from agent_os.daemon_v2.default_skills_installer import _resolve_default_skills_dir


def test_resolver_prefers_meipass_when_bundle_has_default_skills(tmp_path, monkeypatch):
    # Construct a fake PyInstaller extraction root containing the bundled
    # default_skills tree. The resolver should find and return this path.
    bundle_root = tmp_path / "fake_meipass"
    bundled_skills = bundle_root / "agent_os" / "default_skills"
    bundled_skills.mkdir(parents=True)
    # Seed a single skill so the destination is a real dir with content.
    skill_a = bundled_skills / "alpha"
    skill_a.mkdir()
    (skill_a / "SKILL.md").write_text("# alpha\n", encoding="utf-8")

    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)

    resolved = _resolve_default_skills_dir()
    assert resolved is not None
    assert os.path.samefile(resolved, str(bundled_skills)), (
        f"resolver returned {resolved!r}, expected {str(bundled_skills)!r}"
    )
