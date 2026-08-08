# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Local event spool — append-only JSONL, never transmitted (spec 046 §4).

Rows carry an ``event`` name, a UTC ``ts``, and flat counter-safe fields only
(enums/ints/bools). No project IDs, session IDs, paths, or free text. Append
follows the budget ledger's never-raise discipline; rotation is size-capped
with one retained predecessor, like the daemon logs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class Spool:
    def __init__(self, data_dir: Path, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        self._path = Path(data_dir) / "telemetry" / "events.jsonl"
        self._max_bytes = max_bytes

    def append(self, event: str, fields: dict | None = None) -> None:
        try:
            row = {"event": event, "ts": datetime.now(timezone.utc).isoformat()}
            if fields:
                row.update(fields)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed()
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        except Exception:
            pass  # telemetry must never affect the product

    def read_day(self, day: str) -> Iterator[dict]:
        """Yield events whose UTC ts falls on ``day`` (YYYY-MM-DD), oldest
        rotation first. Corrupt lines are skipped."""
        for path in (self._path.with_name(self._path.name + ".1"), self._path):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and str(row.get("ts", "")).startswith(day):
                    yield row

    def _rotate_if_needed(self) -> None:
        try:
            if self._path.stat().st_size >= self._max_bytes:
                self._path.replace(self._path.with_name(self._path.name + ".1"))
        except FileNotFoundError:
            pass
