# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for the Workbench surface (spec 089 §3.6; spec 2026-07-23 §5.3, §8).

A mirror of each project's open asks — the fold of ``orbital/ASKS.md``
(``agent/asks.py``). Nothing is derived to disk and a read never writes: every
GET re-folds the log and computes age/overdue at render time.

The two user exits append events instead of rewriting memory: Done →
``done <id> <date> by:user``, Delete → ``dropped <id> <date> by:user <reason>``.
The cards response shape is the v0.13.0 one (phones run the relay's older
frontend build); "Recently closed" — closes by the agent or the memory editor
in the last 7 days, each reopenable — is opt-in via ``?recently_closed=1``.

The empty-state CTA (``/migrate``) spawns a seeded project session through the
same dispatch seam the chat/queue uses (``agent_manager.new_session`` +
``inject_message``) that reviews PROJECT_STATE into asks.

Injected via ``configure`` (app factory): the project store, the agent manager
(session spawn), and the CalendarHub (``refresh()`` after every write so its
60s cache never serves pre-edit state).
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent_os.agent import asks, memory_entries, workbench_cards
from agent_os.agent.project_paths import ProjectPaths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/workbench")

_project_store = None
_agent_manager = None
_calendar_hub = None
_now_fn = None


def configure(project_store, agent_manager, calendar_hub, *, now_fn=None):
    """Called by the app factory to inject dependencies.

    ``now_fn`` (tz-aware ``datetime``) is overridable for deterministic tests;
    it drives age/overdue math and the dates of appended events.
    """
    global _project_store, _agent_manager, _calendar_hub, _now_fn
    _project_store = project_store
    _agent_manager = agent_manager
    _calendar_hub = calendar_hub
    _now_fn = now_fn or (lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Small I/O helpers
# --------------------------------------------------------------------------

def _now() -> datetime:
    return (_now_fn or (lambda: datetime.now(timezone.utc)))()


def _load_state(path: str) -> str | None:
    """PROJECT_STATE.md content, decode-safe (``errors="replace"``); ``None``
    when missing."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


_IS_WINDOWS = sys.platform.startswith("win")


def _atomic_write(path: str, content: str) -> None:
    """Atomic text write: tmp file in the same dir + ``os.replace``.

    The layer-1 memory convention (workspace_files.py) — a reader never sees a
    half-written file, and a crash mid-write cannot truncate the original.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    for attempt in range(5):  # Windows: target may be briefly open
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if not _IS_WINDOWS or attempt == 4:
                raise
            time.sleep(0.05)


def _write_state(path: str, content: str) -> None:
    _atomic_write(path, content)


def _state_path(workspace: str) -> str:
    return ProjectPaths(workspace).project_state


def _orbital_dir(workspace: str) -> str:
    return ProjectPaths(workspace).orbital_dir


def _require_project(project_id: str) -> dict:
    if _project_store is None:
        raise HTTPException(status_code=503, detail="Workbench not available")
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _project_tz(project: dict) -> str:
    return workbench_cards.project_timezone(project, project.get("triggers", []) or [])


def _project_today(project: dict) -> str:
    """Event dates are the user's calendar day in the project tz, not UTC."""
    return workbench_cards.today_in_tz(_project_tz(project), _now()).isoformat()


# --------------------------------------------------------------------------
# GET /api/v2/workbench
# --------------------------------------------------------------------------

def _ask_row(project_id: str, a, tz: str, now: datetime) -> dict:
    """One card, in the v0.13.0 wire shape exactly (``section`` stays null:
    asks don't live under a PROJECT_STATE heading)."""
    return {
        "project_id": project_id,
        "id": a.id,
        "text": a.text,
        "section": None,
        "due": a.due,
        "created": a.opened,
        "touched": a.updated,
        "age_days": workbench_cards.age_days(a.opened, tz, now),
        "overdue": workbench_cards.is_overdue(a.due, tz, now),
        # Project-tz days-past-due (null unless overdue) — server-authoritative
        # so the frontend never recomputes "N days late" in browser tz (§7.3).
        "days_late": workbench_cards.days_late(a.due, tz, now),
    }


def _closed_row(project_id: str, a) -> dict:
    return {
        "project_id": project_id,
        "id": a.id,
        "text": a.text,
        "due": a.due,
        "created": a.opened,
        "closed": a.updated,
        "closed_by": a.closed_by,
        "kind": a.state,
        "note": a.note,
    }


def _collect_project(project: dict, now: datetime, with_closed: bool):
    """Return ``(open card rows, recently-closed rows)`` for one project."""
    workspace = project.get("workspace", "")
    if not workspace:
        return [], []
    project_id = project.get("project_id", "")
    folded = asks.read_asks(_orbital_dir(workspace))
    tz = _project_tz(project)
    rows = [_ask_row(project_id, a, tz, now) for a in folded if a.is_open]
    closed = []
    if with_closed:
        today = workbench_cards.today_in_tz(tz, now).isoformat()
        closed = [_closed_row(project_id, a) for a in asks.recently_closed(folded, today)]
    return rows, closed


@router.get("")
async def get_workbench(
    project_id: str | None = Query(None),
    recently_closed: bool = Query(False),
):
    """Open asks. Global view respects the privacy toggle.

    Sort: overdue first, then oldest ``created`` first — the forgotten float
    up. ``project_id`` lenses to one project (and, unlike the global view,
    surfaces a project even when it is excluded from the global Workbench).
    ``recently_closed`` (opt-in) adds a ``recently_closed`` list: asks closed
    by the agent or the memory editor in the last 7 days, newest first.
    """
    now = _now()
    if project_id is not None:
        projects = [_require_project(project_id)]
    else:
        projects = [
            p for p in _project_store.list_projects()
            if not p.get("workbench_exclude_global")
        ]

    all_entries: list[dict] = []
    all_closed: list[dict] = []
    for project in projects:
        # Per-project isolation: one project failing to collect (corrupt file,
        # I/O error) must never sink the whole global view. Failed projects are
        # skipped and logged — chosen over a degraded in-band marker to keep the
        # {entries} response contract stable for the frontend.
        try:
            entries, closed = _collect_project(project, now, recently_closed)
        except Exception:
            logger.warning(
                "workbench: skipping project %s — collection failed",
                project.get("project_id"), exc_info=True,
            )
            continue
        all_entries.extend(entries)
        all_closed.extend(closed)

    all_entries.sort(key=lambda e: (not e["overdue"], e.get("created") or "9999-99-99"))
    body: dict = {"entries": all_entries}
    if recently_closed:
        all_closed.sort(key=lambda r: r.get("closed") or "", reverse=True)
        body["recently_closed"] = all_closed
    return body


# --------------------------------------------------------------------------
# Exits + reopen: each appends one event to ASKS.md
# --------------------------------------------------------------------------

class ExitRequest(BaseModel):
    kind: Literal["fulfilled", "irrelevant"]
    reason: str = ""


def _append(project: dict, kind: str, ask_id: str, note: str) -> None:
    try:
        asks.append_event(
            _orbital_dir(project.get("workspace", "")), kind, ask_id, note,
            "user", today=_project_today(project),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No ask with id {ask_id}")
    if _calendar_hub is not None:
        _calendar_hub.refresh()


@router.post("/{project_id}/entries/{mem_id}/exit")
async def exit_entry(project_id: str, mem_id: str, req: ExitRequest):
    """Done (``fulfilled``) or Delete (``irrelevant``) for an open ask.

    Appends ``done``/``dropped`` by:user. 404 if the project's log never
    opened ``mem_id``; closing an ask that is already closed is a no-op 200
    (the list the user tapped on was simply stale).
    """
    project = _require_project(project_id)
    kind = "done" if req.kind == "fulfilled" else "dropped"
    _append(project, kind, mem_id, req.reason or "")
    return {"status": "ok"}


@router.post("/{project_id}/asks/{ask_id}/reopen")
async def reopen_ask(project_id: str, ask_id: str):
    """Undo a close from the "Recently closed" list: appends ``reopen``
    by:user. 404 for an unknown ask; reopening an open ask is a no-op 200."""
    project = _require_project(project_id)
    _append(project, "reopen", ask_id, "")
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Migrate (session spawn seam)
# --------------------------------------------------------------------------

# Imperative on purpose: weaker models otherwise ANALYZE the file and ask
# for permission instead of editing (observed live, 2026-07-24 — the session
# presented an (a)/(b)/(c) menu and stalled). This instruction IS the user's
# confirmation. Since spec 089 the review lands in ASKS.md as open asks; it no
# longer flags lines in place (a `[user]` line would be moved there anyway).
_MIGRATION_MESSAGE = (
    'Review orbital/PROJECT_STATE.md NOW, in this turn, and record in '
    'orbital/ASKS.md every item that is waiting on me and must outlive this '
    'conversation: a decision I have not made, something I said I will do '
    'myself, or something with a date. For each one append a line '
    '`- open <text>` (or `- open due:YYYY-MM-DD <text>`) — Orbital stamps the '
    'id and date. Write each line self-contained, for someone who was not '
    'here: concrete names, no shorthand, no list-number references. Skip '
    'anything already listed under Open asks, never re-propose anything '
    'listed as dropped, and do not change PROJECT_STATE.md for this. Do not '
    'present findings first, do not ask which items I want, do not wait for '
    'confirmation — this message IS the confirmation. After saving, reply '
    'with one line: how many asks you opened.'
)


async def _spawn_seeded(project_id: str, content: str) -> str:
    """Mint a fresh session and inject the seed through the normal dispatch
    seam (the same path chat/queue use); return the new session id."""
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not available")
    minted = await _agent_manager.new_session(project_id)
    session_id = minted["session_id"] if isinstance(minted, dict) else minted
    await _agent_manager.inject_message(project_id, content, session_id=session_id)
    return session_id


@router.post("/{project_id}/migrate")
async def migrate_project(project_id: str):
    """Empty-state day-0 flow (spec §5.4): force-refresh the PROJECT_STATE
    format header to the current contract, then spawn a session that reviews
    PROJECT_STATE into open asks (see ``_MIGRATION_MESSAGE``)."""
    project = _require_project(project_id)
    path = _state_path(project.get("workspace", ""))
    content = _load_state(path)
    if content is not None:
        refreshed = memory_entries.force_format_header(content, "state")
        if refreshed != content:
            _write_state(path, refreshed)
    session_id = await _spawn_seeded(project_id, _MIGRATION_MESSAGE)
    return {"session_id": session_id}
