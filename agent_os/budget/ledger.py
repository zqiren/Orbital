# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Append-only token ledger (Budget Piece 1).

One JSONL line per LLM response, stored per-project at::

    {workspace}/orbital/ledger/usage.jsonl

The path is owned by ``ProjectPaths.ledger_file`` (``agent_os/agent/
project_paths.py``) — the single owner of all Orbital on-disk layout;
``ledger_path()`` here delegates to it.

Design rules in force:

  - **Tokens are the stored fact.** No dollars in the ledger — derived cost
    views live in a later piece and read these token fields against a rate
    table. The four token fields are DISJOINT (see ``budget/normalize.py``):
    ``uncached_input``, ``cache_read``, ``cache_write``, ``output``.
  - **Append-only.** Events are only ever appended + flushed, never rewritten.
    Low frequency (one per LLM response), so no batching.
  - **Resilient.** A write failure (unwritable dir, disk error) must never
    crash or alter the agent loop — ``append_event`` logs a warning and
    returns. The caller continues regardless.

A later task adds a ``spend()`` query over the same file; the flat per-line
schema here is chosen so reading events back is a plain ``json.loads`` per
line. This module does NOT implement querying.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only; avoids an import cycle.
    from agent_os.budget.normalize import NormalizedUsage

logger = logging.getLogger(__name__)

# Event source enum. Only "management" is captured today; the subagent values
# are RESERVED for a later piece and intentionally NOT emitted here.
SOURCE_MANAGEMENT = "management"


def ledger_path(project_dir: str) -> str:
    """Return the absolute ledger path for a project workspace.

    ``{project_dir}/orbital/ledger/usage.jsonl``. Pure path calculation — no
    I/O. Delegates to ``ProjectPaths.ledger_file`` so the layout has exactly
    one owner; a later ``spend()`` query reads the file back through this
    same function.
    """
    # Imported lazily: project_paths lives under the agent package, whose
    # __init__ eagerly imports loop -> this module. A module-level import
    # here would be a circular import (same cycle as budget.normalize).
    from agent_os.agent.project_paths import ProjectPaths

    return ProjectPaths(project_dir).ledger_file


@dataclass(frozen=True)
class LedgerEvent:
    """One LLM-response usage event, prior to serialization.

    ``ts`` is filled in at append time (ISO8601 UTC) so callers don't have to.
    ``usage`` carries the four disjoint token counts from ``NormalizedUsage``.
    """

    session_id: str
    source: str
    provider: str
    model: str
    usage: NormalizedUsage

    def to_record(self, ts: str) -> dict:
        """Flatten to the on-disk JSON record (disjoint token fields inline)."""
        return {
            "ts": ts,
            "session_id": self.session_id,
            "source": self.source,
            "provider": self.provider,
            "model": self.model,
            "uncached_input": self.usage.uncached_input,
            "cache_read": self.usage.cache_read,
            "cache_write": self.usage.cache_write,
            "output": self.usage.output,
        }


def append_event(project_dir: str, event: LedgerEvent) -> None:
    """Append one event line to the project's usage ledger.

    Creates ``{project_dir}/ledger/`` on demand, then appends a single JSON
    line and flushes. The timestamp is stamped here as ISO8601 UTC.

    Resilience contract: this function NEVER raises. Any failure (unwritable
    directory, disk error, serialization issue) is logged at WARNING and
    swallowed so the agent loop's behavior is unaffected.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        record = event.to_record(ts)
        path = ledger_path(project_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except Exception:  # noqa: BLE001 — resilience: never break the loop
        logger.warning(
            "Token-ledger append failed for project_dir=%s (session=%s); "
            "continuing without it.",
            project_dir, getattr(event, "session_id", "?"),
            exc_info=True,
        )
