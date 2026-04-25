# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""If running frozen (``sys._MEIPASS`` set) but the bundle does not ship
``agent_os/default_skills/``, the resolver must return ``None`` so the
installer can emit a WARNING and leave the reconciled flag unset. The
walk-up fallback from ``__file__`` is NOT used in the frozen branch —
``__file__`` inside a PyInstaller bundle would resolve to the same missing
bundle path and produce nonsensical fallbacks."""

import sys

from agent_os.daemon_v2 import default_skills_installer
from agent_os.daemon_v2.default_skills_installer import _resolve_default_skills_dir


def test_resolver_returns_none_when_frozen_bundle_lacks_skills(tmp_path, monkeypatch):
    # _MEIPASS set, but the bundle contains no agent_os/default_skills/.
    bundle_root = tmp_path / "empty_bundle"
    bundle_root.mkdir()

    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)

    # Also ensure the walk-up fallback can't resolve to a real tree even if
    # the implementation were to try it: repoint the module's __file__ into
    # a clean tmp subtree that lacks default_skills/ at every ancestor.
    isolated_dir = tmp_path / "isolated" / "agent_os" / "daemon_v2"
    isolated_dir.mkdir(parents=True)
    fake_module_path = isolated_dir / "default_skills_installer.py"
    fake_module_path.write_text("# stub\n", encoding="utf-8")
    monkeypatch.setattr(
        default_skills_installer, "__file__", str(fake_module_path), raising=True
    )

    assert _resolve_default_skills_dir() is None
