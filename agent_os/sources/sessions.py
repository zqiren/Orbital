# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Where an arrival's session is now and what state it is in, for history
and for the one-at-a-time gate.

* Project: the session keeps its id when it is moved (spec 180); a project
  that no longer lists it but has ``.moved/{id}.to-{B}…`` points to B.
* State, as the Workbench reads it: ``running`` (a live status, or a worker
  turn open), ``needs_you`` (waiting for approval, or one of its asks still
  open), else ``done``.

Reads go off the loop and are cached for a few seconds: the dispatcher asks
every few seconds while something is waiting.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

LIVE_STATUSES = frozenset({"running", "waiting", "queued"})
CACHE_S = 3.0
MAX_HOPS = 5


class SessionLookup:
    def __init__(self, project_store, agent_manager, sub_agent_manager=None):
        self._projects = project_store
        self._am = agent_manager
        self._sam = sub_agent_manager
        self._rows: dict[str, tuple[float, dict]] = {}
        self._asks: dict[str, tuple[float, set]] = {}
        self._stale_before = 0.0

    def mark_stale(self) -> None:
        """Lists read before now are not trusted: a history read (the user
        looking) re-reads each project once, so a session moved a moment ago
        is seen where it is. The dispatcher's frequent checks keep the cache."""
        self._stale_before = time.monotonic()

    def _rows_for(self, project_id: str, *, fresh: bool = False) -> dict:
        hit = self._rows.get(project_id)
        now = time.monotonic()
        if hit and not fresh and now - hit[0] < CACHE_S and hit[0] >= self._stale_before:
            return hit[1]
        rows = {}
        try:
            for r in self._am.list_sessions(project_id) or []:
                for key in (r.get("session_id"), r.get("session_uuid")):
                    if isinstance(key, str):
                        rows[key] = r
        except Exception:
            logger.debug("sources: list_sessions(%s) failed", project_id, exc_info=True)
        self._rows[project_id] = (now, rows)
        return rows

    def _open_asks(self, project_id: str, workspace: str) -> set:
        hit = self._asks.get(project_id)
        now = time.monotonic()
        if hit and now - hit[0] < CACHE_S and hit[0] >= self._stale_before:
            return hit[1]
        ids: set = set()
        try:
            from agent_os.agent import asks
            from agent_os.agent.project_paths import ProjectPaths
            ids = {a.id for a in asks.read_asks(ProjectPaths(workspace).orbital_dir) if a.is_open}
        except Exception:
            logger.debug("sources: asks of %s unreadable", project_id, exc_info=True)
        self._asks[project_id] = (now, ids)
        return ids

    def _where(self, project_id: str, session_id: str) -> tuple[str | None, dict | None]:
        pid = project_id
        for _ in range(MAX_HOPS):
            row = self._rows_for(pid).get(session_id)
            if row is None:
                # A miss may only be a cache older than the session (one
                # just sent): read again before calling it gone or moved.
                row = self._rows_for(pid, fresh=True).get(session_id)
            if row is not None:
                return pid, row
            project = self._projects.get_project(pid) if self._projects else None
            workspace = (project or {}).get("workspace") or ""
            if not workspace:
                return pid, None
            try:
                from agent_os.agent.project_paths import ProjectPaths
                from agent_os.daemon_v2.session_move import moved_destination
                dest = moved_destination(ProjectPaths(workspace).sessions_dir, session_id)
            except Exception:
                dest = None
            if not dest or dest == pid:
                return pid, None
            pid = dest
        return pid, None

    def _info(self, project_id: str, session_id: str) -> tuple[str | None, str]:
        pid, row = self._where(project_id, session_id)
        if row is None:
            return pid, "done"
        running = set()
        if self._sam is not None:
            try:
                running = self._sam.running_session_ids(pid) or set()
            except Exception:
                running = set()
        if row.get("status") in LIVE_STATUSES or row.get("session_id") in running:
            return pid, "running"
        if row.get("status") == "pending_approval":
            return pid, "needs_you"
        ask_ids = [a for a in (row.get("ask_ids") or []) if isinstance(a, str)]
        if ask_ids:
            project = self._projects.get_project(pid) if self._projects else None
            if set(ask_ids) & self._open_asks(pid, (project or {}).get("workspace") or ""):
                return pid, "needs_you"
        return pid, "done"

    async def info(self, project_id: str, session_id: str) -> tuple[str | None, str]:
        return await asyncio.to_thread(self._info, project_id, session_id)
