# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Install identity: one random, non-fingerprinting id (spec 046 §3).

``inst_`` + 12 hex chars, persisted in ``{data_dir}/telemetry/install.json``
alongside ``first_seen`` and the lifetime milestone booleans. Deliberately NOT
the relay ``device_id`` (that identity is coupled to the relay feature and its
shared-secret auth). Deleting the file — or ``reset()`` — mints a fresh
identity and clears milestones.

Disk failures never propagate: a broken data dir degrades to an in-memory
identity for this process, mirroring the budget ledger's never-raise stance.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

_FILE = "install.json"


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _mint() -> str:
    return "inst_" + uuid.uuid4().hex[:12]


class InstallIdentity:
    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / "telemetry" / _FILE
        self._state: dict = self._load_or_create()

    @property
    def install_id(self) -> str:
        return self._state["install_id"]

    @property
    def first_seen(self) -> str:
        return self._state["first_seen"]

    @property
    def milestones(self) -> dict[str, bool]:
        return dict(self._state.get("milestones", {}))

    def latch_milestone(self, name: str) -> None:
        if self._state.get("milestones", {}).get(name):
            return
        self._state.setdefault("milestones", {})[name] = True
        self._persist()

    def reset(self) -> str:
        """Mint a new identity, clearing first_seen and milestones."""
        self._state = {"install_id": _mint(), "first_seen": _today(), "milestones": {}}
        self._persist()
        return self._state["install_id"]

    def _load_or_create(self) -> dict:
        try:
            state = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and str(state.get("install_id", "")).startswith("inst_"):
                state.setdefault("first_seen", _today())
                state.setdefault("milestones", {})
                return state
        except Exception:
            pass
        state = {"install_id": _mint(), "first_seen": _today(), "milestones": {}}
        self._state = state
        self._persist()
        return state

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            pass  # in-memory identity still serves this process
