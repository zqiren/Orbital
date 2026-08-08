# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Runtime app version (spec 046 §7).

Resolution order — first hit wins, never raises:

  1. ``agent_os/_version.py`` — written by ``scripts/build-desktop.sh`` /
     ``scripts/build-macos.sh`` from ``pyproject.toml`` at build time. The only
     authoritative source inside a frozen bundle. Not checked in.
  2. ``pyproject.toml`` — dev checkouts (``agent_os/`` sits next to it).
  3. ``importlib.metadata`` — last resort; may be a stale egg-info value
     (observed: 0.7.2 reported on a 0.8.4 install), so it ranks below the
     checkout's pyproject.
  4. ``"0.0.0"``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _read_generated() -> str | None:
    try:
        from agent_os._version import __version__  # noqa: PLC0415

        return __version__
    except Exception:
        return None


def _read_pyproject(root: Path | None = None) -> str | None:
    try:
        base = root if root is not None else Path(__file__).resolve().parent.parent
        with open(base / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        value = data.get("project", {}).get("version")
        return value if isinstance(value, str) else None
    except Exception:
        return None


def _read_metadata() -> str | None:
    try:
        from importlib.metadata import version

        return version("agent-os")
    except Exception:
        return None


def get_version() -> str:
    # Uncached: reads are rare (startup, ping assembly) and a cache would pin
    # monkeypatched test doubles.
    return _read_generated() or _read_pyproject() or _read_metadata() or "0.0.0"
