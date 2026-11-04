# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which reply the user has checked, per session (spec 154 §5.1).

One user, two devices (the phone over the relay and the desktop) talk to the
same daemon and must agree on what is "waiting on you", so the checked state
lives here, not on a device:

    {data_dir}/checked.json = {project_id: {session_key: reply_at}}

``session_key`` is the list row's ``session_uuid`` (else its ``session_id``),
the key spec 108's per-device map used; ``reply_at`` is the row's
``last_reply_at`` the user has checked (a daemon-clock ISO string, so device
clocks never enter the comparison).

Rules:
  * A write keeps the later of the stored and the new time, so two devices
    checking at once can never move it backwards.
  * Entries older than the window are dropped on write; a project's key stays
    (possibly ``{}``) so it never re-seeds.
  * A project with no entry is seeded once from its current session list:
    every existing reply counts as checked, so an upgrade or an imported
    project opens on asks and running work, not history. A session that is
    live at that moment is skipped: its reply is the one being produced.
  * A corrupt or unreadable file reads as absent (every project re-seeds, so
    nothing floods) and is replaced on the next write.

App state, not a project asset: it is not under the user's ``orbital/``. A
v0.20.0 daemon never reads this file (downgrade-safe); its own per-device
localStorage map is left in place by the new client.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

CHECKED_FILENAME = "checked.json"
WINDOW_DAYS = 14
# A session in one of these states is producing its reply now; the seed
# leaves it unchecked (a run's end-of-run record step keeps it ``running``
# after the reply row lands, so a first read then would swallow that reply).
_LIVE_STATUSES = frozenset({"running", "waiting", "queued"})


def _is_live(row: dict) -> bool:
    return row.get("status") in _LIVE_STATUSES or bool(row.get("worker_running"))


def _parse(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def session_key(row: dict) -> str | None:
    """The store key of a session-list row (spec 108's ``seenKey``)."""
    key = row.get("session_uuid") or row.get("session_id")
    return key if isinstance(key, str) and key else None


class CheckedStore:
    def __init__(self, path: str, *, window_days: int = WINDOW_DAYS,
                 now_fn: Callable[[], datetime] | None = None):
        self.path = path
        self.window = timedelta(days=window_days)
        self._now = now_fn or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, str]] | None = None  # lazy-loaded mirror

    @classmethod
    def in_dir(cls, data_dir: str, **kw) -> "CheckedStore":
        return cls(os.path.join(data_dir, CHECKED_FILENAME), **kw)

    # -- disk -------------------------------------------------------------

    def _load(self) -> dict[str, dict[str, str]]:
        if self._data is not None:
            return self._data
        data: dict[str, dict[str, str]] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            for pid, entries in raw.items():
                if isinstance(pid, str) and isinstance(entries, dict):
                    data[pid] = {k: v for k, v in entries.items()
                                 if isinstance(k, str) and isinstance(v, str)}
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logger.warning("checked store %s unreadable; every project re-seeds",
                           self.path, exc_info=True)
            data = {}
        self._data = data
        return data

    def _save(self, data: dict[str, dict[str, str]]) -> None:
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            # The mirror still answers for this daemon lifetime.
            logger.warning("could not write checked store %s", self.path, exc_info=True)

    def _in_window(self, reply_at: str) -> bool:
        at = _parse(reply_at)
        return at is not None and at >= self._now() - self.window

    def _pruned(self, entries: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in entries.items() if self._in_window(v)}

    # -- API ----------------------------------------------------------------

    def project(self, project_id: str) -> dict[str, str] | None:
        """A copy of the project's checked map, or None when never seeded."""
        with self._lock:
            entries = self._load().get(project_id)
            return dict(entries) if entries is not None else None

    def seed_if_absent(self, project_id: str, rows: Iterable[dict]) -> dict[str, str]:
        """Seed the project once from its session rows (every reply checked);
        returns its map. A project already present is left alone."""
        with self._lock:
            data = self._load()
            existing = data.get(project_id)
            if existing is not None:
                return dict(existing)
            seeded: dict[str, str] = {}
            for row in rows:
                if _is_live(row):
                    continue
                key = session_key(row)
                reply_at = row.get("last_reply_at")
                if key and isinstance(reply_at, str) and self._in_window(reply_at):
                    seeded[key] = reply_at
            data[project_id] = seeded
            self._save(data)
            return dict(seeded)

    def mark(self, project_id: str, key: str, reply_at: str) -> bool:
        """Record that the user checked ``reply_at`` in this session. Keeps
        the later time; returns whether the stored value changed."""
        if _parse(reply_at) is None:
            raise ValueError("reply_at must be an ISO-8601 time")
        with self._lock:
            data = self._load()
            entries = self._pruned(data.get(project_id) or {})
            prev = entries.get(key)
            prev_at = _parse(prev)
            changed = prev_at is None or _parse(reply_at) > prev_at
            if changed:
                entries[key] = reply_at
            data[project_id] = entries
            self._save(data)
            return changed

    def annotate(self, project_id: str, rows: list[dict]) -> list[dict]:
        """Add ``checked_reply_at`` to each row (seeding the project first
        when it has no entry). One dict lookup per row; mutates and returns
        ``rows``."""
        entries = self.seed_if_absent(project_id, rows)
        for row in rows:
            key = session_key(row)
            row["checked_reply_at"] = entries.get(key) if key else None
        return rows
