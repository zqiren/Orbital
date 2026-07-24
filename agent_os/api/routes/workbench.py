# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for the Workbench surface (spec §5.3, §5.4, §5.6, §6, §8).

A lazy mirror of the flagged ``[user]`` entries across projects plus a small
set of daemon-computed cards (overdue dated facts, broken automations, paused
threads). Nothing is derived to disk — every GET re-parses PROJECT_STATE via
the one shared ``user_flags`` parser and computes ages/overdue at render time.

Two user exits mutate memory directly (fulfilled → unflag + ``resolved`` stamp;
irrelevant → remove + a retraction record), both OCC-guarded on the state
file's mtime. Tapping a card (``/open``) or the empty-state migration CTA
(``/migrate``) spawns a seeded project session through the same dispatch seam
the chat/queue uses (``agent_manager.new_session`` + ``inject_message``).

Injected via ``configure`` (app factory): the project store, the agent manager
(spawn + session listing), and the CalendarHub (``refresh()`` after every write
so its 60s cache never serves pre-edit state).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent_os.agent import memory_entries, retractions, user_flags, workbench_cards
from agent_os.agent.project_paths import ProjectPaths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/workbench")

_MAX_OCC_ATTEMPTS = 2  # initial write + one re-read-and-retry, then 409

_project_store = None
_agent_manager = None
_calendar_hub = None
_now_fn = None


def configure(project_store, agent_manager, calendar_hub, *, now_fn=None):
    """Called by the app factory to inject dependencies.

    ``now_fn`` (tz-aware ``datetime``) is overridable for deterministic tests;
    it drives age/overdue math and the ``resolved``/retraction/dismiss dates.
    """
    global _project_store, _agent_manager, _calendar_hub, _now_fn
    _project_store = project_store
    _agent_manager = agent_manager
    _calendar_hub = calendar_hub
    _now_fn = now_fn or (lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# Small I/O helpers (factored so tests can hook the OCC read seam)
# --------------------------------------------------------------------------

def _now() -> datetime:
    return (_now_fn or (lambda: datetime.now(timezone.utc)))()


def _today_iso() -> str:
    return _now().date().isoformat()


def _stat_mtime_ns(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _load_state(path: str) -> tuple[int | None, str | None]:
    """Return ``(mtime_ns, content)`` for the state file.

    The OCC baseline (mtime) is captured here, alongside the read, so the exit
    path has a single seam. Tests monkeypatch this to simulate a concurrent
    writer between the baseline capture and the guarded write.

    Decode-safe: read with ``errors="replace"`` so invalid UTF-8 bytes in a
    project's PROJECT_STATE.md yield replacement chars instead of a
    ``UnicodeDecodeError`` that would 500 the whole (esp. global) GET.
    """
    mtime = _stat_mtime_ns(path)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return mtime, f.read()
    except (OSError, UnicodeDecodeError):
        return mtime, None


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


def _dismissals_path(workspace: str) -> str:
    return os.path.join(_orbital_dir(workspace), "workbench_dismissals.json")


def _require_project(project_id: str) -> dict:
    if _project_store is None:
        raise HTTPException(status_code=503, detail="Workbench not available")
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _find_entry(content: str, mem_id: str):
    for e in user_flags.parse_entries(content or ""):
        if e.id == mem_id:
            return e
    return None


# --------------------------------------------------------------------------
# GET /api/v2/workbench
# --------------------------------------------------------------------------

def _entry_row(project_id: str, e, tz: str, now: datetime) -> dict:
    return {
        "project_id": project_id,
        "id": e.id,
        "text": e.text,
        "due": e.due,
        "evidence": e.evidence,
        "from_session": e.from_session,
        "confidence": e.confidence,
        "created": e.created,
        "touched": e.touched,
        "age_days": workbench_cards.age_days(e.created, tz, now),
        "overdue": workbench_cards.is_overdue(e.due, tz, now),
        # Project-tz days-past-due (null unless overdue) — server-authoritative
        # so the frontend never recomputes "N days late" in browser tz (§7.3).
        "days_late": workbench_cards.days_late(e.due, tz, now),
    }


def _read_session_tail(path: str) -> dict | None:
    """Last persisted user/assistant message of a session JSONL, or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and obj.get("role") in ("user", "assistant"):
            return obj
    return None


def _paused_cards(project: dict, project_id: str) -> list[dict]:
    if _agent_manager is None or not hasattr(_agent_manager, "list_sessions"):
        return []
    try:
        sessions = _agent_manager.list_sessions(project_id)
    except Exception:
        logger.debug("workbench: list_sessions failed for %s", project_id, exc_info=True)
        return []
    paths = ProjectPaths(project.get("workspace", ""))
    return workbench_cards.paused_thread_cards(
        project_id, sessions, lambda uuid: _read_session_tail(paths.session_file(uuid))
    )


def _load_dismissed_keys(workspace: str) -> set:
    try:
        with open(_dismissals_path(workspace), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return set()
    if not isinstance(data, list):
        return set()
    return {(d.get("type"), d.get("key")) for d in data if isinstance(d, dict)}


def _collect_project(project: dict, now: datetime) -> tuple[list[dict], list[dict]]:
    """Return ``(entry_rows, computed_cards)`` for one project."""
    workspace = project.get("workspace", "")
    if not workspace:
        return [], []
    project_id = project.get("project_id", "")
    _, content = _load_state(_state_path(workspace))
    parsed = user_flags.parse_entries(content or "")
    triggers = project.get("triggers", []) or []
    tz = workbench_cards.project_timezone(project, triggers)

    computed: list[dict] = workbench_cards.broken_automation_cards(project_id, triggers, now)
    broken_names = workbench_cards.broken_trigger_names(triggers, now)

    flagged = [e for e in parsed if e.flagged]
    kept, _suppressed = workbench_cards.suppress_entries(flagged, broken_names)
    entry_rows = [_entry_row(project_id, e, tz, now) for e in kept]

    # Overdue computed cards cover UNFLAGGED dated facts only — a flagged entry
    # that is overdue already appears in ``entry_rows`` carrying ``overdue:true``
    # (spec §3 overdue emphasis), so surfacing it again here would double it.
    # ``not e.resolved`` skips a retired entry: since the parse_entries
    # amendment now surfaces tag-less resolved facts (flagged=False), a dated
    # fact that carries a ``resolved:`` stamp must NOT resurrect as an overdue
    # card off its now-stale ``due:`` — the resolved stamp is a closed state.
    for e in parsed:
        if (not e.flagged and e.due and not e.resolved
                and workbench_cards.is_overdue(e.due, tz, now)):
            computed.append({
                "type": "overdue",
                "project_id": project_id,
                "key": e.id,
                "text": e.text,
                "since": e.due,
                # Same project-tz days-late as entry rows. Only overdue cards
                # carry it (they alone have a due-derived date); broken/paused
                # cards have no due, so no days_late field.
                "days_late": workbench_cards.days_late(e.due, tz, now),
            })

    computed.extend(_paused_cards(project, project_id))

    dismissed = _load_dismissed_keys(workspace)
    computed = [c for c in computed if (c["type"], c["key"]) not in dismissed]
    return entry_rows, computed


@router.get("")
async def get_workbench(project_id: str | None = Query(None)):
    """Flagged entries + computed cards. Global view respects the privacy toggle.

    Sort (entries): overdue first, then oldest ``created`` first — the forgotten
    float up. ``project_id`` lenses to one project (and, unlike the global view,
    surfaces a project even when it is excluded from the global Workbench).
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
    all_computed: list[dict] = []
    for project in projects:
        # Per-project isolation: one project failing to collect (corrupt state,
        # I/O error) must never sink the whole global view. Failed projects are
        # skipped and logged — chosen over a degraded in-band marker to keep the
        # {entries, computed} response contract stable for the frontend.
        try:
            entries, computed = _collect_project(project, now)
        except Exception:
            logger.warning(
                "workbench: skipping project %s — collection failed",
                project.get("project_id"), exc_info=True,
            )
            continue
        all_entries.extend(entries)
        all_computed.extend(computed)

    all_entries.sort(key=lambda e: (not e["overdue"], e.get("created") or "9999-99-99"))
    return {"entries": all_entries, "computed": all_computed}


# --------------------------------------------------------------------------
# Exits
# --------------------------------------------------------------------------

class ExitRequest(BaseModel):
    kind: Literal["fulfilled", "irrelevant"]
    reason: str = ""


def _entry_comment_fields(entry) -> dict:
    """Map a parsed Entry back to its mem-comment field dict (drop None)."""
    fields: dict[str, str] = {}
    if entry.id:
        fields["id"] = entry.id
    if entry.from_session:
        fields["from"] = entry.from_session
    if entry.evidence:
        fields["evidence"] = entry.evidence
    if entry.confidence:
        fields["confidence"] = entry.confidence
    if entry.created:
        fields["created"] = entry.created
    if entry.touched:
        fields["touched"] = entry.touched
    if entry.resolved:
        fields["resolved"] = entry.resolved
    return fields


def _apply_exit(content: str, entry, kind: str, reason: str, today: str):
    """Rewrite ONLY the target entry's lines. Returns (new_content, retraction).

    fulfilled → drop the whole bracket tag (the bullet becomes a plain fact) and
    stamp ``resolved:<today>`` into its mem-comment. irrelevant → remove the
    entry lines entirely and emit a Retraction to append.
    """
    lines = content.split("\n")
    retraction = None
    if kind == "fulfilled":
        fields = _entry_comment_fields(entry)
        fields["resolved"] = today
        replacement = [f"- {entry.text}", "  " + user_flags.render_comment(fields)]
    else:  # irrelevant (Literal already validated by pydantic)
        replacement = []
        retraction = retractions.Retraction(
            id=entry.id or user_flags.new_entry_id(),
            title=entry.text,
            reason=reason or "",
            date=today,
        )
    new_lines = lines[:entry.line_start] + replacement + lines[entry.line_end + 1:]
    return "\n".join(new_lines), retraction


@router.post("/{project_id}/entries/{mem_id}/exit")
async def exit_entry(project_id: str, mem_id: str, req: ExitRequest):
    """Fulfilled or irrelevant exit for a flagged entry (spec §5.3).

    OCC on the state file's mtime: baseline captured at read, verified
    unchanged before the write; one re-read-and-retry on conflict, then 409.
    404 if no entry carries ``mem_id``.
    """
    project = _require_project(project_id)
    workspace = project.get("workspace", "")
    path = _state_path(workspace)
    today = _today_iso()

    retraction = None
    committed = False
    for _attempt in range(_MAX_OCC_ATTEMPTS):
        baseline_mtime, content = _load_state(path)
        if content is None:
            raise HTTPException(status_code=404, detail="PROJECT_STATE.md not found")
        entry = _find_entry(content, mem_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No entry with id {mem_id}")
        new_content, retraction = _apply_exit(content, entry, req.kind, req.reason, today)
        # Guarded write: only if nothing touched the file since the baseline.
        if _stat_mtime_ns(path) != baseline_mtime:
            continue  # concurrent write — re-read and retry
        _write_state(path, new_content)
        committed = True
        break

    if not committed:
        raise HTTPException(
            status_code=409,
            detail="PROJECT_STATE.md changed concurrently; please retry",
        )

    # Append the retraction only after the state write commits (add_retraction
    # is append-only — never run it on an attempt that will be retried).
    if retraction is not None:
        retractions.add_retraction(_orbital_dir(workspace), retraction)
    if _calendar_hub is not None:
        _calendar_hub.refresh()
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Dismiss (computed cards)
# --------------------------------------------------------------------------

def _read_dismissals_list(workspace: str) -> list:
    try:
        with open(_dismissals_path(workspace), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


@router.post("/{project_id}/computed/{card_type}/{key}/dismiss")
async def dismiss_computed(project_id: str, card_type: str, key: str):
    """Persist a computed-card dismissal (spec §6); GET omits it thereafter."""
    project = _require_project(project_id)
    workspace = project.get("workspace", "")
    data = _read_dismissals_list(workspace)
    if not any(d.get("type") == card_type and d.get("key") == key
               for d in data if isinstance(d, dict)):
        data.append({"type": card_type, "key": key, "date": _today_iso()})
        _atomic_write(
            _dismissals_path(workspace),
            json.dumps(data, ensure_ascii=False, indent=2),
        )
    return {"status": "ok"}


# --------------------------------------------------------------------------
# Doorway: open + migrate (session spawn seam)
# --------------------------------------------------------------------------

# Imperative on purpose: weaker models otherwise ANALYZE the file and ask
# for permission instead of editing (observed live, 2026-07-24 — the session
# presented an (a)/(b)/(c) menu and stalled). This instruction IS the user's
# confirmation; the never-auto-decide rail covers external/irreversible acts,
# not this requested file edit.
_MIGRATION_MESSAGE = (
    "Edit orbital/PROJECT_STATE.md NOW, in this turn, and apply [user] flags "
    "per the format header's rails. Do not present findings first, do not ask "
    "which option I prefer, do not wait for confirmation — this message IS "
    "the confirmation, and tagging is a plain file edit (the never-auto-decide "
    "rail is about external or irreversible acts, not this). For each entry "
    "that needs the user: rewrite its bullet as `- [user] <one plain sentence "
    "addressed to the user>` (append ` due:YYYY-MM-DD` inside the tag only "
    "when a real deadline exists), and on the next line add "
    '`<!--mem from:<session-id> evidence:"<the user\'s own words>" '
    "confidence:stated-->` — use confidence:unconfirmed for anything you "
    "inferred rather than heard; omit fields you do not know; never write an "
    "id (ids are daemon-assigned). Do not invent obligations; do not flag "
    "agent work; leave every other line untouched. After saving, reply with "
    "one line: the number of entries you flagged."
)


def _seed_open_message(entry) -> str:
    evidence = entry.evidence or ""
    frm = f" — from {entry.from_session}" if entry.from_session else ""
    due = f"; due {entry.due}" if entry.due else ""
    receipt = f"\n(evidence: {evidence}{frm}{due})" if (evidence or frm or due) else ""
    return f'Workbench: "{entry.text}"{receipt}\nLet\'s handle this.'


async def _spawn_seeded(project_id: str, content: str) -> str:
    """Mint a fresh session and inject the seed through the normal dispatch
    seam (the same path chat/queue use); return the new session id."""
    if _agent_manager is None:
        raise HTTPException(status_code=503, detail="Agent manager not available")
    minted = await _agent_manager.new_session(project_id)
    session_id = minted["session_id"] if isinstance(minted, dict) else minted
    await _agent_manager.inject_message(project_id, content, session_id=session_id)
    return session_id


@router.post("/{project_id}/entries/{mem_id}/open")
async def open_entry(project_id: str, mem_id: str):
    """Autospawn a project session seeded with the card (spec §5.3 doorway)."""
    project = _require_project(project_id)
    _, content = _load_state(_state_path(project.get("workspace", "")))
    entry = _find_entry(content or "", mem_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No entry with id {mem_id}")
    session_id = await _spawn_seeded(project_id, _seed_open_message(entry))
    return {"session_id": session_id}


@router.post("/{project_id}/computed/{card_type}/{key}/open")
async def open_computed(project_id: str, card_type: str, key: str):
    """Seeded doorway for computed cards (spec §6 alignment, 2026-07-24):
    ``overdue`` → a session told to DO the late item now; ``broken_automation``
    → a session told to diagnose/repair the silent trigger. ``paused_thread``
    resumes client-side (its session still exists) and never calls this."""
    project = _require_project(project_id)
    if card_type == "overdue":
        _, content = _load_state(_state_path(project.get("workspace", "")))
        entry = _find_entry(content or "", key)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No entry with id {key}")
        late = f" It was due {entry.due}." if entry.due else ""
        content_msg = (
            f'Workbench: "{entry.text}" is overdue.{late} Do it now — this '
            "message is the go-ahead; when it is done, update the entry per "
            "the state file's format header."
        )
    elif card_type == "broken_automation":
        trig = next(
            (tr for tr in (project.get("triggers") or [])
             if isinstance(tr, dict) and tr.get("id") == key),
            None,
        )
        if trig is None:
            raise HTTPException(status_code=404, detail=f"No trigger {key}")
        name = trig.get("name") or key
        cron = (trig.get("schedule") or {}).get("cron", "?")
        last = trig.get("last_triggered") or "never"
        content_msg = (
            f'Workbench: the automation "{name}" (cron {cron}) has not fired '
            f"since {last}. Diagnose why and repair it if you can; if it is "
            "obsolete, say so and I will disable it."
        )
    else:
        raise HTTPException(status_code=404, detail=f"No doorway for {card_type}")
    session_id = await _spawn_seeded(project_id, content_msg)
    return {"session_id": session_id}


@router.post("/{project_id}/migrate")
async def migrate_project(project_id: str):
    """Empty-state day-0 flow (spec §5.4): force-refresh the PROJECT_STATE
    format header to the current ``[user]`` grammar rails, then spawn a
    session that flags the legacy unflagged content."""
    project = _require_project(project_id)
    path = _state_path(project.get("workspace", ""))
    _, content = _load_state(path)
    if content is not None:
        refreshed = memory_entries.force_format_header(content, "state")
        if refreshed != content:
            _write_state(path, refreshed)
    session_id = await _spawn_seeded(project_id, _MIGRATION_MESSAGE)
    return {"session_id": session_id}
