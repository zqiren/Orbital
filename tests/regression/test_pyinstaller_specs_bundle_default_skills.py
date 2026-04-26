# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parse-level regression: every PyInstaller spec under ``agent_os/desktop/``
must bundle ``agent_os/default_skills`` as a data entry.

Deliberately text-level. The spec files are executable PyInstaller config;
running them from unit tests would require a full PyInstaller harness that
we don't want in the hot loop. A substring check is enough to catch the
regression observed in ``agentos-macos.spec`` where the entry was missing.
"""

import glob
import os

import pytest


SPEC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "agent_os", "desktop"
)


def _spec_files() -> list[str]:
    pattern = os.path.join(SPEC_DIR, "*.spec")
    return sorted(glob.glob(pattern))


def test_spec_dir_has_spec_files():
    specs = _spec_files()
    assert specs, (
        f"no .spec files found under {SPEC_DIR!r} — the per-spec regression "
        "would silently pass with an empty parametrization otherwise"
    )


@pytest.mark.parametrize("spec_path", _spec_files())
def test_spec_bundles_default_skills(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "agent_os/default_skills" in content, (
        f"{os.path.basename(spec_path)} is missing the "
        "'agent_os/default_skills' datas entry — packaged builds will ship "
        "without the default skills. Add:\n"
        "    (os.path.join(project_root, 'agent_os', 'default_skills'), 'agent_os/default_skills'),"
    )
