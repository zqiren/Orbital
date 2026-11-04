# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-file version history (spec 065).

A watcher per project snapshots every tracked workspace file as it settles
into a content-addressed store under ``{workspace}/orbital/history/``.
Orbital's own writers (the write/edit tools, the Files panel save, the
daemon's Layer-1 memory writes) record through ``recording()`` so their
versions carry exact attribution; everything else is credited to the
sessions running when the change happened.

Public surface:

- ``FileHistory`` — the app-level service (observer, debounce, reconcile,
  health, restore). One per daemon, owned by ``api/app.py``.
- ``recording(path, attrib)`` — context manager around a write so the
  version is recorded under the store lock with exact attribution.
- ``write_and_record(path, data, attrib)`` — bytes write + record.
- ``tool_attrib`` / ``USER_ATTRIB`` / ``ORBITAL_ATTRIB`` — attribution dicts.
"""
from agent_os.history.service import (  # noqa: F401
    FileHistory,
    ORBITAL_ATTRIB,
    USER_ATTRIB,
    active,
    recording,
    tool_attrib,
    write_and_record,
)
