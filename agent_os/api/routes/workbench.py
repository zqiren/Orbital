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
same dispatch seam the chat/queue uses (the ``new-session`` and ``inject``
route handlers) that reviews PROJECT_STATE into asks. ``POST /tasks`` (spec
162) is the Workbench composer for scripts: route → mint → inject, the same
helpers (``pick_project``, ``_spawn_seeded``).

Injected via ``configure`` (app factory): the project store, the agent manager
(session spawn), and the CalendarHub (``refresh()`` after every write so its
60s cache never serves pre-edit state).
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, field_validator

from agent_os.agent import asks, memory_entries, workbench_cards
from agent_os.agent.project_paths import ProjectPaths
from agent_os.api.routes.agents_v2 import InjectAttachment as _InjectAttachment

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/workbench")

_project_store = None
_agent_manager = None
_calendar_hub = None
_ws_manager = None
_now_fn = None
_checked_store = None
_sub_agent_manager = None


def configure(project_store, agent_manager, calendar_hub, *, now_fn=None, ws_manager=None,
              checked_store=None, sub_agent_manager=None):
    """Called by the app factory to inject dependencies.

    ``now_fn`` (tz-aware ``datetime``) is overridable for deterministic tests;
    it drives age/overdue math and the dates of appended events.
    ``checked_store`` (spec 154 §5.1) holds which reply the user has checked
    per session; ``sub_agent_manager`` answers which sessions have a worker
    turn open (the feed's ``worker_running``).
    """
    global _project_store, _agent_manager, _calendar_hub, _now_fn, _ws_manager
    global _checked_store, _sub_agent_manager
    _ws_manager = ws_manager
    _project_store = project_store
    _agent_manager = agent_manager
    _calendar_hub = calendar_hub
    _now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    _checked_store = checked_store
    _sub_agent_manager = sub_agent_manager


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
        # Choices the agent offered with the ask ([] when none, item 160).
        "options": list(a.options),
        # The markdown context under the question ("" when none).
        "detail": a.detail,
        # Batch 1110 item 2: the goals card's "needs changes" option, when
        # this ask is that card: the UI prefills the reply with it instead of
        # sending it bare, so the correction travels in one turn.
        "change_option": _change_option(project_id, a),
    }


def _change_option(project_id: str, a) -> str | None:
    from agent_os.onboarding.import_scan import scan_ask_of
    try:
        rec = scan_ask_of(_project_store.get_project(project_id), a.id)
    except Exception:  # noqa: BLE001 — display data
        return None
    change = rec.get("change") if isinstance(rec, dict) else None
    return change if isinstance(change, str) and change in list(a.options) else None


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
    _announce_change(project)


def _announce_change(project: dict) -> None:
    if _ws_manager is not None:
        project_id = project.get("project_id", "")
        _ws_manager.broadcast(project_id, {"type": "workbench.changed", "project_id": project_id})


def close_answered_ask(project: dict, ask_id: str, words: str) -> None:
    """The user answered this ask from the composer: it closes in their words.
    An unknown or already closed ask never blocks the message itself."""
    try:
        _append(project, "done", ask_id, " ".join(words.split())[:200])
    except HTTPException:
        logger.info("answered ask %s is not in this project", ask_id)


def _session_ask_ids(workspace: str, session_id: str) -> list[str]:
    """Ids of the asks opened in this conversation, oldest first."""
    if not session_id or any(c in session_id for c in ("/", "\\", "..")):
        return []
    try:
        with open(ProjectPaths(workspace).session_file(session_id), encoding="utf-8",
                  errors="replace") as f:
            lines = [line for line in f if '"ask_opened"' in line]
    except OSError:
        return []
    ids: list[str] = []
    for line in lines:
        try:
            ask_id = (json.loads(line).get("_meta") or {}).get("ask_id")
        except ValueError:
            continue
        if ask_id and ask_id not in ids:
            ids.append(ask_id)
    return ids


@router.get("/{project_id}/session-asks")
async def session_asks(project_id: str, session_id: str = Query(...)):
    """The open asks this conversation opened: what its composer shows."""
    workspace = _require_project(project_id).get("workspace", "")
    ids = await asyncio.to_thread(_session_ask_ids, workspace, session_id)
    if not ids:
        return {"asks": []}
    open_asks = {
        a.id: a for a in await asyncio.to_thread(asks.read_asks, _orbital_dir(workspace))
        if a.is_open
    }
    return {"asks": [{"id": i, "text": open_asks[i].text, "due": open_asks[i].due,
                      "options": list(open_asks[i].options), "detail": open_asks[i].detail,
                      "change_option": _change_option(project_id, open_asks[i])}
                     for i in ids if i in open_asks]}


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


async def _spawn_seeded(project_id: str, content: str, *, agent: str | None = "orbital",
                        attachments: list | None = None) -> dict:
    """Mint a fresh session and send ``content`` into it through the same two
    route handlers the chat and the Workbench composer call
    (``POST /agents/{pid}/new-session`` then ``POST /agents/{pid}/inject``),
    so seeding, attachments, pinned dispatch and queueing behave exactly as
    they do for a typed message.

    ``agent``: ``"orbital"`` (the manager), a worker slug (a pinned send), or
    ``None`` for the composer's 自动: the minted ``pinned_target``. A pinned
    send stamps the session's pin; the project's pin is never PATCHed.

    Returns ``{"session_id", "agent", "inject"}`` (``agent`` = the target the
    send used, ``"orbital"`` for the manager; ``inject`` = the inject route's
    body). An inject error propagates after the mint, as in the composer, and
    the session keeps whatever the inject route wrote: nothing for a refused
    send (no card), the user row and a failure row for a worker that could
    not start (a failed card)."""
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not available")
    from agent_os.api.routes import agents_v2
    minted = await agents_v2.new_session(project_id)
    session_id = minted["session_id"] if isinstance(minted, dict) else minted
    if agent is None:
        agent = (minted.get("pinned_target") if isinstance(minted, dict) else None) or "orbital"
    target = None if agent == "orbital" else agent
    req = agents_v2.InjectRequest(content=content, target=target, pinned=target is not None,
                                  attachments=attachments, session_id=session_id)
    body = await agents_v2.inject_message(project_id, req)
    return {"session_id": session_id, "agent": agent, "inject": body}


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
    spawned = await _spawn_seeded(project_id, _MIGRATION_MESSAGE)
    return {"session_id": spawned["session_id"]}



# --------------------------------------------------------------------------
# Spec 154 — the checked state, the feed and routing
# --------------------------------------------------------------------------

FEED_WINDOW_DAYS = 14
# Statuses that read as "running" on the Workbench (spec 154 §5.1): a manager
# waiting on a worker or a background job is active (spec 147), and a queued
# first message is a task that started.
_LIVE_STATUSES = frozenset({"running", "waiting", "queued"})


def annotate_checked(project_id: str, rows: list[dict]) -> list[dict]:
    """Add ``checked_reply_at`` to session-list rows from the daemon's
    checked store, seeding the project once on its first read. Additive:
    without a store every row reads ``None``."""
    if _checked_store is None:
        for row in rows:
            row.setdefault("checked_reply_at", None)
        return rows
    try:
        return _checked_store.annotate(project_id, rows)
    except Exception:
        logger.warning("checked store: annotate failed for %s", project_id, exc_info=True)
        for row in rows:
            row.setdefault("checked_reply_at", None)
        return rows


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_live(row: dict) -> bool:
    return row.get("status") in _LIVE_STATUSES or bool(row.get("worker_running"))


def is_unchecked(row: dict, now: datetime) -> bool:
    """The agent spoke last, the row is resting, the reply is newer than the
    one the user checked, and it landed inside the window (spec 154 §5.1)."""
    if _is_live(row) or row.get("status") == "pending_approval":
        return False
    reply = _parse_ts(row.get("last_reply_at"))
    if reply is None or reply < now - timedelta(days=FEED_WINDOW_DAYS):
        return False
    user = _parse_ts(row.get("last_user_at"))
    if user is not None and user > reply:
        return False
    checked = _parse_ts(row.get("checked_reply_at"))
    return checked is None or reply > checked


def _feed_ask(project_id: str, a, tz: str, now: datetime) -> dict:
    return {
        "id": a.id, "text": a.text, "due": a.due, "created": a.opened,
        "overdue": workbench_cards.is_overdue(a.due, tz, now),
        "options": list(a.options),
        "detail": a.detail,
    }


def _finished_facts(workspace: str, row: dict) -> tuple[str | None, dict | None]:
    """``(result_line, stop)`` of a finished session from its JSONL (the
    left-off reader, cached by path + mtime + size)."""
    from agent_os.onboarding.history import resets_at_if_future
    from agent_os.onboarding.left_off import read_orbital_session
    uuid = row.get("session_uuid") or row.get("session_id")
    if not workspace or not isinstance(uuid, str) or any(c in uuid for c in ("/", "\\", "..")):
        return None, None
    parsed = read_orbital_session(ProjectPaths(workspace).session_file(uuid))
    if parsed is None:
        return None, None
    stop = parsed.entry.stop
    stop_json = ({"cause": stop.cause, "resets_at": resets_at_if_future(stop, _now())}
                 if stop is not None else None)
    return parsed.reply_line, stop_json


def _run_worker(workspace: str, row: dict) -> str | None:
    """The worker an automation run was handed to, from its JSONL (the
    left-off reader, cached by path + mtime + size). A trigger-started run
    never pins its session; a manager-run automation has none (Orbital)."""
    from agent_os.onboarding.left_off import read_orbital_session
    uuid = row.get("session_uuid") or row.get("session_id")
    if not workspace or not isinstance(uuid, str) or any(c in uuid for c in ("/", "\\", "..")):
        return None
    parsed = read_orbital_session(ProjectPaths(workspace).session_file(uuid))
    return parsed.worker if parsed is not None else None


def feed_rows(project: dict, sessions: list[dict], now: datetime) -> list[dict]:
    """The Workbench rows of one project (spec 154 §5.2). ``sessions`` are
    list rows already carrying ``worker_running`` and ``checked_reply_at``.
    Blocking (reads ASKS.md and finished sessions' tails): call off-loop."""
    project_id = project.get("project_id", "")
    workspace = project.get("workspace", "") or ""
    tz = _project_tz(project)
    open_asks = {}
    if workspace:
        try:
            open_asks = {a.id: a for a in asks.read_asks(_orbital_dir(workspace)) if a.is_open}
        except Exception:
            logger.warning("feed: ASKS.md unreadable in %s", project_id, exc_info=True)
    linked: set[str] = set()
    out: list[dict] = []
    for row in sessions:
        ids = [i for i in (row.get("ask_ids") or []) if isinstance(i, str)]
        linked.update(ids)
        open_ids = [i for i in ids if i in open_asks]
        ask = _feed_ask(project_id, open_asks[open_ids[-1]], tz, now) if open_ids else None
        live = _is_live(row)
        approval = row.get("status") == "pending_approval"
        if not (live or approval or ask is not None or is_unchecked(row, now)):
            continue
        result_line = stop = None
        if not live and not approval:
            try:
                result_line, stop = _finished_facts(workspace, row)
            except Exception:
                logger.warning("feed: could not read %s", row.get("session_id"), exc_info=True)
        agent = row.get("pinned_target") or None
        if agent is None and row.get("trigger_type"):
            # An automation run: the logo is the agent that ran it.
            try:
                agent = _run_worker(workspace, row)
            except Exception:
                logger.warning("feed: could not read %s", row.get("session_id"), exc_info=True)
        out.append({
            "project_id": project_id,
            "session_id": row.get("session_id"),
            "session_uuid": row.get("session_uuid"),
            "name": row.get("name"),
            "agent": agent,
            "status": row.get("status"),
            "worker_running": bool(row.get("worker_running")),
            "trigger_type": row.get("trigger_type"),
            "last_user_at": row.get("last_user_at"),
            "last_reply_at": row.get("last_reply_at"),
            "last_activity_at": row.get("last_activity_at"),
            "checked_reply_at": row.get("checked_reply_at"),
            "ask": ask,
            "result_line": result_line,
            "stop": stop,
        })
    # Asks no session opened (shell, an external agent, pre-146): their own rows.
    for ask_id, a in open_asks.items():
        if ask_id in linked:
            continue
        out.append({
            "project_id": project_id, "session_id": None, "session_uuid": None,
            "name": None, "agent": None, "status": "idle", "worker_running": False,
            "trigger_type": None, "last_user_at": None, "last_reply_at": None,
            "last_activity_at": None, "checked_reply_at": None,
            "ask": _feed_ask(project_id, a, tz, now), "result_line": None, "stop": None,
        })
    return out


def _feed_projects() -> list[dict]:
    if _project_store is None:
        return []
    return [p for p in _project_store.list_projects()
            if not p.get("workbench_exclude_global")]


async def _project_sessions(project_id: str) -> list[dict]:
    """The session list + ``worker_running`` + ``checked_reply_at``, the
    same calls the session-list route makes."""
    if _agent_manager is None:
        return []
    sessions = await asyncio.to_thread(_agent_manager.list_sessions, project_id)
    running = (_sub_agent_manager.running_session_ids(project_id)
               if _sub_agent_manager is not None else set())
    for entry in sessions:
        entry["worker_running"] = entry.get("session_id") in running
    return annotate_checked(project_id, sessions)


@router.get("/feed")
async def get_feed():
    """What can be on the Workbench (spec 154 §5.2): running, waiting for
    approval, an open ask, or a reply the user has not checked inside the
    window. Excluded projects contribute nothing; Quick Tasks is included."""
    now = _now()
    rows: list[dict] = []
    for project in _feed_projects():
        pid = project.get("project_id", "")
        try:
            sessions = await _project_sessions(pid)
            rows.extend(await asyncio.to_thread(feed_rows, project, sessions, now))
        except Exception:
            logger.warning("workbench feed: skipping project %s", pid, exc_info=True)
    return {"rows": rows, "window_days": FEED_WINDOW_DAYS}


class CheckedRequest(BaseModel):
    project_id: str
    session_id: str
    reply_at: str


@router.post("/checked")
async def post_checked(req: CheckedRequest):
    """The user checked this session's reply (✓ on a card, or viewing the
    session). Keeps the later time; broadcasts ``workbench.changed`` so the
    other device's cards, strip dot and counts update."""
    project = _require_project(req.project_id)
    if _checked_store is None:
        raise HTTPException(status_code=503, detail="Checked state not available")
    if _parse_ts(req.reply_at) is None:
        raise HTTPException(status_code=400, detail="reply_at must be an ISO-8601 time")
    if _checked_store.project(req.project_id) is None:
        # Seed first: a write to an unseeded project must not turn its whole
        # history into unchecked replies.
        await _project_sessions(req.project_id)
    changed = await asyncio.to_thread(
        _checked_store.mark, req.project_id, req.session_id, req.reply_at)
    if changed:
        _announce_change(project)
    stored = (_checked_store.project(req.project_id) or {}).get(req.session_id)
    return {"status": "ok", "checked_reply_at": stored, "changed": changed}


# ── Routing (spec 154 §5.4) ─────────────────────────────────────────────

# One budget for the whole routing step (2026-11-07): the first call may use
# all of it; one retry follows a failed call only when at least
# ROUTE_RETRY_MIN_S is left. The SDK's own retries are off for this call, so
# its backoff can't eat the budget unseen.
ROUTE_BUDGET_S = 45.0
ROUTE_RETRY_MIN_S = 10.0
ROUTE_RETRY_DELAY_S = 5.0  # an overloaded provider (429) needs a moment; 1 s failed both tries live
GOALS_CAP = 1200
STATE_LINES = 20
STATE_CAP = 1200
RECENT_SESSIONS = 3

ROUTE_INSTRUCTION = (
    "You route a user's task to one of their projects. Below are the projects, "
    "each with its id, name, goals, the head of its state file and its latest "
    "sessions; the task comes last. Pick the one project this task belongs to "
    "and reply with only a JSON object: {\"project\": \"<id>\"}, or "
    "{\"project\": null} when no project clearly fits."
)


def _read_capped(path: str, cap: int, *, lines: int | None = None) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(cap * 4 if lines is None else cap * 8)
    except OSError:
        return ""
    if lines is not None:
        text = "\n".join(text.splitlines()[:lines])
    return text[:cap].strip()


_MEM_COMMENT_LINE_RE = re.compile(r"^\s*<!--\s*mem\b.*-->\s*$")


def _state_head(path: str) -> str:
    """The head of PROJECT_STATE.md as routing context: the code-owned
    ``<!--format …-->`` header (longer than the whole cap) and the
    daemon's mem-id comment lines are dropped first, so the capped head is
    the project's real state."""
    raw = _read_capped(path, STATE_CAP * 8)
    text = memory_entries.strip_format_header(raw)
    lines = [ln for ln in text.splitlines() if not _MEM_COMMENT_LINE_RE.match(ln)]
    body = "\n".join(lines[:STATE_LINES])[:STATE_CAP].strip()
    # A file that is only its title (a new project) carries no state.
    return "" if body.lstrip("#").strip().upper() in ("", "PROJECT_STATE") else body


def route_candidates(projects: list[dict]) -> list[dict]:
    """Every project with a workspace, except Quick Tasks and projects
    excluded from the Workbench."""
    return [p for p in projects
            if p.get("workspace") and not p.get("is_scratch")
            and not p.get("workbench_exclude_global")]


def project_block(project: dict, session_names: list[str]) -> str:
    """One project's capped routing context (decided inputs, §5.4)."""
    pp = ProjectPaths(project.get("workspace", ""))
    goals = _read_capped(pp.project_goals, GOALS_CAP)
    state = _state_head(pp.project_state)
    parts = [f"## Project id: {project.get('project_id', '')}",
             f"Name: {project.get('name') or project.get('project_id', '')}"]
    if goals:
        parts.append(f"Goals:\n{goals}")
    if state:
        parts.append(f"State (head):\n{state}")
    names = [n for n in session_names if n][:RECENT_SESSIONS]
    if names:
        parts.append("Latest sessions: " + " | ".join(names))
    return "\n".join(parts)


def build_route_messages(blocks: list[str], text: str) -> list[dict]:
    """Project blocks first and the task last, so consecutive sends share a
    cacheable prefix."""
    return [
        {"role": "system", "content": ROUTE_INSTRUCTION + "\n\n" + "\n\n".join(blocks)},
        {"role": "user", "content": f"Task:\n{text}"},
    ]


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_route_reply(reply: str | None, candidate_ids: set[str]) -> tuple[str | None, str]:
    """The first JSON object in the reply (models wrap JSON in fences or a
    sentence). ``(project_id, "")`` on a match, else ``(None, reason)``."""
    if not isinstance(reply, str) or not reply.strip():
        return None, "empty reply"
    for match in _JSON_OBJ_RE.finditer(reply):
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict) or "project" not in obj:
            continue
        pid = obj.get("project")
        if pid is None:
            return None, "model chose no project"
        if isinstance(pid, str) and pid in candidate_ids:
            return pid, ""
        return None, f"unknown project id {pid!r}"
    return None, "no JSON object in the reply"


def _scratch_project() -> dict | None:
    if _project_store is None:
        return None
    find = getattr(_project_store, "find_scratch_project", None)
    if callable(find):
        return find()
    return next((p for p in _project_store.list_projects() if p.get("is_scratch")), None)


def _route_provider(scratch_pid: str):
    """Quick Tasks' utility provider when its credential has a key, else
    None (phase A: keyless routes to Quick Tasks)."""
    am = _agent_manager
    if am is None or not am.manager_ready(scratch_pid):
        return None
    config = am._build_agent_config_from_project(scratch_pid)
    _provider, _fallbacks, utility, _info = am._build_llm_providers(config)
    return utility


async def _recent_session_names(project_id: str) -> list[str]:
    if _agent_manager is None:
        return []
    try:
        rows = await asyncio.to_thread(_agent_manager.list_sessions, project_id)
    except Exception:
        return []
    # The user's own conversations only: not the Memory session (the
    # daemon's tidy pass) and not automation runs (their names are trigger
    # headers, "[Triggered by schedule…"). Both are marked in the session
    # meta (origin / trigger_type), not guessed from the title.
    rows = [r for r in rows
            if r.get("origin") != "memory" and not r.get("trigger_type")]
    rows = sorted(rows, key=lambda r: r.get("last_activity_at") or "", reverse=True)
    return [r["name"] for r in rows if isinstance(r.get("name"), str) and r["name"]][:RECENT_SESSIONS]


class RouteRequest(BaseModel):
    text: str


@router.post("/route")
async def route_task(req: RouteRequest):
    """Pick the project for a task typed on the Workbench (project chip on
    自动). The body is ``pick_project``, which ``POST /workbench/tasks``
    (spec 162) calls too: one routing path for the composer and scripts."""
    return await pick_project(req.text)


async def pick_project(text: str | None) -> dict:
    """Route a task's text to a project. Never raises.

    Answers (HTTP 200 always):
      * ``{"project_id", "by": "model"}`` — the model's pick;
      * ``{"project_id": <Quick Tasks>, "by": "fallback", "reason"}`` — a
        deliberate landing: the model chose no project, no API key, nothing
        to route between;
      * ``{"project_id": null, "by": "failed", "reason": <code>,
        "attempts"}`` (2026-11-07) — the call failed after its one retry
        (codes: ``timeout``, ``call_failed``, ``empty_reply``, ``no_json``,
        ``unknown_project``). Nothing is sent; the Workbench asks the user
        to pick a project. Additive: an older client reads a missing
        project as a send that could not start.
    """
    scratch = _scratch_project()
    fallback_pid = scratch.get("project_id") if scratch else None

    def fallback(reason: str) -> dict:
        logger.info("workbench route: Quick Tasks (%s)", reason)
        return {"project_id": fallback_pid, "by": "fallback", "reason": reason}

    text = (text or "").strip()
    if not text:
        return fallback("empty task")
    candidates = route_candidates(_project_store.list_projects() if _project_store else [])
    if not candidates:
        return fallback("no candidate projects")
    if fallback_pid is None:
        return fallback("no Quick Tasks project")
    try:
        provider = _route_provider(fallback_pid)
    except Exception as e:
        return fallback(f"no provider: {type(e).__name__}")
    if provider is None:
        return fallback("no API key")
    set_retries = getattr(provider, "set_sdk_max_retries", None)
    if callable(set_retries):
        set_retries(0)
    try:
        names = await asyncio.gather(*[_recent_session_names(p["project_id"]) for p in candidates])
        blocks = await asyncio.to_thread(
            lambda: [project_block(p, n) for p, n in zip(candidates, names)])
    except Exception as e:
        return fallback(f"context failed: {type(e).__name__}")
    messages = build_route_messages(blocks, text)
    candidate_ids = {p["project_id"] for p in candidates}

    loop = asyncio.get_running_loop()
    deadline = loop.time() + ROUTE_BUDGET_S
    attempts = 0
    code = "call_failed"
    while True:
        attempts += 1
        started = loop.time()
        try:
            response = await asyncio.wait_for(
                # Thinking left at the model's default: off saved 1-2 s and
                # broke routing whenever a model stopped accepting the
                # catalog's off-switch (glm-5.2 on OpenCode Go, 2026-11-07).
                provider.complete(messages),
                timeout=max(0.0, deadline - started))
        except asyncio.TimeoutError:
            code, detail = "timeout", f"timed out after {loop.time() - started:.0f}s"
        except Exception as e:
            code, detail = "call_failed", f"model call failed: {type(e).__name__}: {e}"[:300]
        else:
            elapsed = loop.time() - started
            pid, reason = parse_route_reply(getattr(response, "text", None), candidate_ids)
            if pid is not None:
                return _routed(pid, response, elapsed)
            if reason == "model chose no project":
                return fallback(reason)  # deliberate: never retried
            code = ("empty_reply" if reason == "empty reply"
                    else "no_json" if reason.startswith("no JSON")
                    else "unknown_project")
            detail = reason
        left = deadline - loop.time()
        logger.warning("workbench route: attempt %d failed (%s; %.0fs of the budget left)",
                       attempts, detail, max(0.0, left))
        if attempts >= 2 or left - ROUTE_RETRY_DELAY_S < ROUTE_RETRY_MIN_S:
            break
        if ROUTE_RETRY_DELAY_S:
            await asyncio.sleep(ROUTE_RETRY_DELAY_S)
    logger.warning("workbench route: no project after %d attempt(s) (%s)", attempts, code)
    return {"project_id": None, "by": "failed", "reason": code, "attempts": attempts}


def _routed(pid: str, response, elapsed: float) -> dict:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    logger.info("workbench route: %s in %.1fs (input tokens %s, output %s)",
                pid, elapsed, input_tokens, output_tokens)
    body = {"project_id": pid, "by": "model"}
    # Additive: the call's token use, when the provider reported it.
    if isinstance(input_tokens, int):
        body["input_tokens"] = input_tokens
    if isinstance(output_tokens, int):
        body["output_tokens"] = output_tokens
    return body


# --------------------------------------------------------------------------
# Spec 162 — POST /workbench/tasks: the composer for scripts
# --------------------------------------------------------------------------

def _route_text(text: str, attachments) -> str:
    """The task as routing sees it, as the composer builds it
    (``routeText`` in ``web/src/components/workbench/sendTask.ts``): the
    attached filenames listed after the text."""
    if not attachments:
        return text
    names = "\n".join(f"- {os.path.basename(a.path.replace(chr(92), '/'))}" for a in attachments)
    return f"{text}\n\n<attached_files>\n{names}\n</attached_files>"


async def _available_workers(project_id: str, project: dict) -> list[str]:
    """The workers a pinned send in this project could reach: the pin
    switcher's list (``available_workers`` on the project row, spec 153
    §4.4), the same test the dispatch gate applies. Stale-ok, off the loop."""
    try:
        return list(await asyncio.to_thread(
            _agent_manager.available_worker_slugs, project_id, project=project) or [])
    except Exception:
        logger.warning("available workers failed for %s", project_id, exc_info=True)
        return []


async def _refuse_keyless_manager(project_id: str, agent: str | None) -> None:
    """The send would go to Orbital (``agent`` ``"orbital"``, or omitted with
    no worker pin to fall back on) and Orbital has no key here: the typed
    ``missing_api_key`` 400 the onboarding routes raise, before any mint, so
    no session is left holding a message nobody can answer."""
    if _agent_manager is None:
        return
    target = agent or await asyncio.to_thread(
        _agent_manager.effective_pinned_target, project_id) or "orbital"
    if target != "orbital":
        return
    if await asyncio.to_thread(_agent_manager.manager_ready, project_id):
        return
    from agent_os.daemon_v2.provider_errors import ProviderConfigError
    err = ProviderConfigError(
        "missing_api_key", "No LLM API key configured for this project or globally")
    raise HTTPException(status_code=400, detail={"code": err.code, "message": str(err)})


class TaskRequest(BaseModel):
    text: str
    project_id: str | None = None
    agent: str | None = None
    attachments: list[_InjectAttachment] | None = None

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty")
        if "\ufffd" in v:
            # The inject route's own check, before anything is minted.
            raise ValueError("text contains invalid UTF-8 characters (U+FFFD); "
                             "send it as UTF-8")
        return v

    @field_validator("attachments")
    @classmethod
    def cap_attachments(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError("too many attachments (max 10)")
        return v


@router.post("/tasks")
async def create_task(req: TaskRequest):
    """Spec 162: put a task on the Workbench from a script, exactly as the
    composer does — route (no ``project_id``), mint a session, send the text
    through the inject route. No new auth: the same reach as every local
    route (this machine, or a registered device through the relay).

    200 ``{project_id, session_id, agent, by, inject}``, plus what
    ``/route`` adds: ``reason`` on a Quick Tasks fallback, ``input_tokens`` /
    ``output_tokens`` when the model routed it. ``by``: ``"given"`` | ``"model"`` |
    ``"fallback"``. Every refusal comes before the mint, so it leaves no
    session: 422 empty text, 404 unknown or Workbench-excluded project (or
    nowhere to route), 400 an agent that is not available or an invalid
    attachment, 400 ``{"code": "missing_api_key"}`` when the send would go
    to Orbital and Orbital has no key here, 502 routing failed after its retry (nothing sent — pass a
    ``project_id``). An error from the inject itself (after the mint) is
    returned as the inject route answered it."""
    text = req.text.strip()
    extra: dict = {}
    if req.project_id:
        project = _require_project(req.project_id)
        if project.get("workbench_exclude_global"):
            raise HTTPException(status_code=404, detail="Project is not on the Workbench")
        pid, by = req.project_id, "given"
    else:
        picked = await pick_project(_route_text(text, req.attachments))
        pid, by = picked.get("project_id"), picked.get("by")
        if pid is None and by == "failed":
            raise HTTPException(status_code=502, detail=picked)
        if pid is None:
            raise HTTPException(status_code=404,
                                detail=f"No project to send to ({picked.get('reason')})")
        project = _require_project(pid)
        # The route's own additive words: why it fell back, its token use.
        extra = {k: picked[k] for k in ("reason", "input_tokens", "output_tokens")
                 if picked.get(k) is not None}
    if req.agent and req.agent != "orbital" and req.agent not in await _available_workers(pid, project):
        raise HTTPException(status_code=400,
                            detail=f"Agent {req.agent!r} is not available in this project")
    await _refuse_keyless_manager(pid, req.agent)
    if req.attachments:
        from agent_os.api.routes._attachment_formatter import validate_attachments
        try:
            validate_attachments(project.get("workspace", ""), req.attachments)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid attachment: {e}")
    logger.info("workbench task: %s (by %s), agent %s", pid, by, req.agent or "pin")
    spawned = await _spawn_seeded(pid, text, agent=req.agent, attachments=req.attachments)
    body = {"project_id": pid, "session_id": spawned["session_id"],
            "agent": spawned["agent"], "by": by, "inject": spawned["inject"]}
    body.update(extra)
    return body
