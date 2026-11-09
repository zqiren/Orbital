# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for onboarding from the user's agent history (backlog #34,
spec 138).

* ``GET /onboarding/history`` — starts the background history scan once and
  reports its progress (never blocks).
* ``GET /onboarding/importable-projects`` — folder candidates from the
  metadata-only :mod:`agent_os.onboarding.import_scanner` (available at
  once) plus, when the scan is done, the groups of scattered conversations.
* ``POST /onboarding/import`` — creates the checked folders and groups as
  projects through the one project-creation path. Spec 160: ``orbital/`` is
  written at once (keyless too), the root AGENTS.md still waits for first
  use, and each project is stamped ``import_scan: pending`` for the
  background guess-first scan (:mod:`agent_os.onboarding.import_scan`).
* ``GET /projects/{id}/left-off`` — the "Last time" / "Didn't finish" cards.
* ``POST /telemetry/event`` — the allowlisted onboarding events the web UI
  reports (counters and enums only).

Everything is local; nothing read here leaves the machine.
"""

import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from agent_os.onboarding import scan_importable_projects
from agent_os.onboarding.import_scanner import candidates_from_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/onboarding")
projects_router = APIRouter(prefix="/api/v2/projects")
telemetry_router = APIRouter(prefix="/api/v2/telemetry")

_project_store = None
_data_dir: str | None = None
_setup_engine = None
_scanner = None

# How long an import waits for the scan before groups are reported missing.
_IMPORT_SCAN_WAIT_SECONDS = 30.0


def configure(project_store=None, *, data_dir: str | None = None, setup_engine=None,
              scanner=None) -> None:
    """Inject the project store (spec 084 §3.2) so the scan can mark folders
    that are already a project's workspace, the daemon data dir (history
    cache) and the setup engine (``agents_detected``). All optional: with no
    store the scan runs without exclusion, which keeps the pure-scanner tests
    dependency-free.
    """
    global _project_store, _data_dir, _setup_engine, _scanner
    _project_store = project_store
    _data_dir = data_dir
    _setup_engine = setup_engine
    _scanner = scanner


def get_scanner():
    """The process's history scanner (created on first use)."""
    global _scanner
    if _scanner is None:
        from agent_os.onboarding.history import HistoryScanner
        _scanner = HistoryScanner(_data_dir, on_ready=_report_scan)
    return _scanner


def _report_scan(scanner) -> None:
    """Telemetry once the first scan is done: how much history there was and
    how many supported agents are installed. Counts only."""
    from agent_os import telemetry
    from agent_os.telemetry.rollup import history_bucket

    status = scanner.status()
    telemetry.emit("history_sessions_bucket",
                   {"bucket": history_bucket(status["sessions_total"])})
    # Spec 153 §4.5: the private-notes count each agent's wizard row shows.
    for agent, n in (status.get("notes_by_agent") or {}).items():
        telemetry.emit("notes_found", {"agent": agent, "n": int(n)})
    if _setup_engine is not None:
        try:
            n = sum(1 for s in _setup_engine.check_all()
                    if s.slug != "built-in" and s.installed)
            telemetry.emit("agents_detected", {"n": n})
        except Exception:
            logger.warning("agents_detected: could not check agents", exc_info=True)


def _existing_projects() -> list[dict]:
    if _project_store is None:
        return []
    try:
        return list(_project_store.list_projects())
    except Exception:
        logger.exception("importable-projects: could not list existing projects")
        return []


def _existing_workspaces() -> set[str]:
    return {p.get("workspace") for p in _existing_projects() if p.get("workspace")}


def _imported_members() -> set[tuple]:
    out = set()
    for p in _existing_projects():
        for m in p.get("imported_sessions") or []:
            if isinstance(m, dict):
                out.add((m.get("agent"), m.get("native_id")))
    return out


# --------------------------------------------------------------------------
# History scan + candidates
# --------------------------------------------------------------------------


@router.get("/history")
async def history_status():
    """Start the background scan (idempotent) and report where it is."""
    scanner = get_scanner()
    scanner.start()
    return scanner.status()


def _groups_payload(scanner) -> list[dict]:
    analysis = scanner.analysis()
    if analysis is None:
        return []
    taken = _imported_members()
    out = []
    for group in analysis.groups:
        if any((s.agent, s.native_id) in taken for s in group.members):
            continue  # already brought in as a project
        out.append(group.to_dict())
    return out


@router.get("/importable-projects")
async def importable_projects():
    """Ranked, deduped, path-verified folder candidates for link-only import
    (metadata only, available at once) plus, once the history scan is done,
    the groups of scattered conversations (spec 138 contract §3). Never
    raises: an unreadable or missing source contributes nothing."""
    scanner = get_scanner()
    scanner.start()
    try:
        candidates = await asyncio.to_thread(
            lambda: scan_importable_projects(existing_workspaces=_existing_workspaces()))
    except Exception:
        logger.exception("importable-projects scan failed")
        candidates = []
    # Once the history scan is done the folder rows come from its analysis:
    # the folder scan above counts the sessions Orbital itself ran as a
    # worker (rc DMG: 42 rows on a dev Mac, 35 of them smoke workspaces) and
    # is only the instant first answer while the history is read.
    analysis = scanner.analysis() if scanner.status()["status"] == "ready" else None
    if analysis is not None:
        try:
            candidates = candidates_from_analysis(
                analysis, vaults=candidates, existing_workspaces=_existing_workspaces(),
                notes=scanner.notes())
        except Exception:
            logger.exception("importable-projects: rows from the history scan failed")
    try:
        groups = _groups_payload(scanner)
    except Exception:
        logger.exception("importable-projects: grouping failed")
        groups = []
    return {"status": scanner.status()["status"],
            "candidates": [c.to_dict() for c in candidates],
            "groups": groups}


# --------------------------------------------------------------------------
# Batch import
# --------------------------------------------------------------------------


class ImportGroupBody(BaseModel):
    id: str
    workspace: str | None = None


class ImportBody(BaseModel):
    folders: list[str] = []
    groups: list[ImportGroupBody] = []
    # Spec 160: the UI language, so the background read replies in it (the
    # daemon only knows the system locale). Anything else is ignored.
    lang: str | None = None
    # Round 2: the sign's switch beside Import. False = don't read these
    # projects: they are created unstamped, exactly like pre-160 imports.
    import_scan: bool = True
    # Round 3: the wizard's import comes before its key step, so it stamps
    # the projects but fires no trigger; the key step's Save does, and so
    # does its Skip with the sign's switch on (spec 167; off turns them off).
    defer_scan: bool = False


def _members_json(sessions) -> list[dict]:
    return [{"agent": s.agent, "native_id": s.native_id} for s in sessions]


def _folder_members(analysis, folder: str) -> list[dict]:
    """Sessions placed in ``folder`` that its own folder match would miss
    (worktrees, and sessions attached by the files they touched): stored so
    the project's session list and left-off cards include them."""
    if analysis is None:
        return []
    from agent_os.onboarding.import_scanner import path_key
    want = path_key(folder)
    for root, project in analysis.projects.items():
        if path_key(root) == want:
            return _members_json(s for s in project.sessions
                                 if not s.cwd or path_key(s.cwd) != want)
    return []


def _last_activity(sessions) -> str | None:
    ended = [s.ended_at for s in sessions if getattr(s, "ended_at", None)]
    return max(ended) if ended else None


def _folder_activity(analysis, folder: str) -> str | None:
    """The folder's last session end in the history scan (the wizard's own
    newest-first order), for the scan queue's ordering."""
    if analysis is None:
        return None
    from agent_os.onboarding.import_scanner import path_key
    want = path_key(folder)
    for root, project in analysis.projects.items():
        if path_key(root) == want:
            return _last_activity(project.sessions)
    return None


async def _create_imported(folder: str, members: list[dict], *,
                           activity: str | None = None, lang: str | None = None,
                           scan: bool = True, deferred: bool = False) -> dict:
    from agent_os.api.routes import agents_v2
    req = agents_v2.CreateProjectRequest(
        name=os.path.basename(os.path.normpath(folder)) or folder,
        workspace=folder, auto_unique_name=True)
    from agent_os.onboarding.deferred_seed import IMPORTED_FIELD
    from agent_os.onboarding.import_scan import pending_fields

    extra = {IMPORTED_FIELD: True}
    if scan:
        extra.update(pending_fields(activity, lang=lang, deferred=deferred))
    if members:
        extra["imported_sessions"] = members
    return await agents_v2.create_project_from_request(
        req, deferred_seed=True, extra_fields=extra)


@router.post("/import")
async def import_projects(body: ImportBody):
    """Create the checked folders and groups as projects (contract §3)."""
    from agent_os import telemetry
    from agent_os.onboarding.import_scanner import path_key

    if _project_store is None:
        raise HTTPException(status_code=503, detail="project store unavailable")
    scanner = get_scanner()
    scanner.start()
    if body.groups or body.folders:
        await asyncio.to_thread(scanner.wait, _IMPORT_SCAN_WAIT_SECONDS)
    analysis = scanner.analysis()
    taken = {path_key(w) for w in _existing_workspaces()}
    created: list[dict] = []
    skipped: list[dict] = []

    async def create(path: str, origin: str, members: list[dict],
                     activity: str | None = None) -> None:
        if not path or not os.path.isabs(path):
            skipped.append({"path": path, "reason": "invalid"})
            return
        if not os.path.isdir(path):
            skipped.append({"path": path, "reason": "missing"})
            return
        if path_key(path) in taken:
            skipped.append({"path": path, "reason": "already_imported"})
            return
        try:
            project = await _create_imported(path, members, activity=activity,
                                             lang=body.lang, scan=body.import_scan,
                                             deferred=body.defer_scan)
        except HTTPException as e:
            logger.warning("onboarding import: %s refused (%s)", path, e.detail)
            skipped.append({"path": path, "reason": "invalid"})
            return
        except Exception:
            logger.exception("onboarding import: could not create a project for %s", path)
            skipped.append({"path": path, "reason": "invalid"})
            return
        taken.add(path_key(path))
        created.append({"project_id": project.get("project_id"), "name": project.get("name"),
                        "workspace": project.get("workspace"), "from": origin})

    for folder in body.folders:
        await create(folder, "folder", _folder_members(analysis, folder) if folder else [],
                     _folder_activity(analysis, folder) if folder else None)
    for g in body.groups:
        group = analysis.group(g.id) if analysis is not None else None
        if group is None:
            skipped.append({"path": g.workspace or g.id, "reason": "invalid"})
            continue
        await create(g.workspace or "", "group", _members_json(group.members),
                     _last_activity(group.members))

    # Spec 160 §5.1: write orbital/ now (skills + private-notes fold, no
    # model, keyless too); the root AGENTS.md waits for first use.
    from agent_os.onboarding.deferred_seed import seed_orbital_folder
    for c in created:
        await asyncio.to_thread(seed_orbital_folder, _project_store, c["project_id"],
                                scanner=scanner)
    # Spec 160 §4 (a): the checked store seeds a project once, on its first
    # list, counting every reply then present as checked (so history doesn't
    # flood). A new import has no Orbital reply yet: seed it now, empty, so
    # the background scan's reply — whenever the user first looks — is
    # unread through that same path.
    _seed_checked([c["project_id"] for c in created])

    n_folder = sum(1 for c in created if c["from"] == "folder")
    n_group = len(created) - n_folder
    if created:
        telemetry.emit("projects_imported", {"folder": n_folder, "group": n_group})
        telemetry.latch("first_import")
        # Spec 160 §5.2 trigger 1: with a key the scans start now; keyless,
        # the projects stay pending until a key event.
        if body.import_scan and not body.defer_scan:
            from agent_os.onboarding import import_scan
            import_scan.kick("import")
    return {"created": created, "skipped": skipped}


def _seed_checked(project_ids: list[str]) -> None:
    from agent_os.api.routes import workbench
    store = getattr(workbench, "_checked_store", None)
    if store is None:
        return
    for pid in project_ids:
        try:
            store.seed_if_absent(pid, [])
        except Exception:
            logger.warning("onboarding import: could not seed checked map for %s", pid,
                           exc_info=True)


# --------------------------------------------------------------------------
# Left-off cards
# --------------------------------------------------------------------------


@projects_router.get("/{project_id}/left-off")
async def project_left_off(project_id: str):
    if _project_store is None:
        raise HTTPException(status_code=503, detail="project store unavailable")
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    from agent_os.onboarding.left_off import left_off_for_project
    try:
        return await asyncio.to_thread(left_off_for_project, dict(project), get_scanner())
    except Exception:
        logger.exception("left-off failed for %s", project_id)
        return {"last_time": None, "unfinished": []}


async def _seed_on_open(project_id: str) -> None:
    """The first-use seed, as the session paths run it (off the event loop,
    then the queue dispatcher creation skipped); a direct seed when the
    agents routes are not configured on this same store (unit tests).
    Never raises."""
    from agent_os.api.routes import agents_v2
    from agent_os.onboarding.deferred_seed import ensure_project_seeded
    if agents_v2._project_store is not None and agents_v2._project_store is _project_store:
        await agents_v2._ensure_seeded(project_id)
        return
    await asyncio.to_thread(ensure_project_seeded, _project_store, project_id,
                            scanner=get_scanner())


@projects_router.get("/{project_id}/imported-notes")
async def project_imported_notes(project_id: str, seed: bool = False):
    """Spec 153 §4.4: how many private notes Claude Code kept about this
    project's folder (the scanner's count, or a direct read of this one
    workspace before the scan is done) and whether the fold has put them in
    the project: the seed field is cleared and the imported index exists.

    ``seed`` (the card's 打开): run the first-use seed before answering, so
    the index it opens exists. Without it the read writes nothing (spec 138)."""
    if _project_store is None:
        raise HTTPException(status_code=503, detail="project store unavailable")
    project = _project_store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    from agent_os.onboarding.deferred_seed import (
        IMPORTED_INDEX_PATH, ORBITAL_SEEDED_FIELD, is_deferred, workspace_notes,
    )
    if seed and is_deferred(project):
        await _seed_on_open(project_id)
        project = _project_store.get_project(project_id) or project
    from agent_os.onboarding.sources.claude_code import AGENT

    workspace = project.get("workspace") or ""
    try:
        notes = await asyncio.to_thread(workspace_notes, workspace, get_scanner())
    except Exception:
        logger.exception("imported-notes failed for %s", project_id)
        notes = []
    index = os.path.join(workspace, *IMPORTED_INDEX_PATH.split("/")) if workspace else ""
    # Spec 160: the fold now runs at import (``orbital_seeded``) while the
    # seed field waits for the root AGENTS.md.
    seeded = not is_deferred(project) or bool(project.get(ORBITAL_SEEDED_FIELD))
    folded = seeded and bool(index) and os.path.isfile(index)
    return {"agent": AGENT, "count": sum(1 for n in notes if n.agent == AGENT),
            "folded": folded, "index_path": IMPORTED_INDEX_PATH}


# --------------------------------------------------------------------------
# Allowlisted UI telemetry events
# --------------------------------------------------------------------------


class TelemetryEventBody(BaseModel):
    event: str
    fields: dict = {}


def _enum(fields: dict, key: str, allowed: tuple, *, required: bool = True):
    value = fields.get(key)
    if value is None and not required:
        return None
    if value not in allowed:
        raise ValueError(f"{key} must be one of {', '.join(allowed)}")
    return value


def _int(fields: dict, key: str) -> int:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100_000:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _validate_event(event: str, fields: dict) -> dict:
    """The spooled fields of an allowlisted event; ValueError otherwise.
    Only enum and integer values ever pass — never text."""
    from agent_os.telemetry.rollup import LEFTOFF_KINDS, STOP_CAUSES, WIZARD_STEPS

    if event == "wizard_step_reached":
        return {"step": _enum(fields, "step", WIZARD_STEPS)}
    if event in ("agent_located", "leftoff_shown", "agent_switch_after_open"):
        return {}
    if event == "leftoff_opened":
        out = {"kind": _enum(fields, "kind", LEFTOFF_KINDS)}
        cause = _enum(fields, "cause", STOP_CAUSES, required=False)
        if cause is not None:
            out["cause"] = cause
        return out
    if event == "projects_unchecked":
        return {"n": _int(fields, "n")}
    raise ValueError(f"unknown event {event!r}")


_MILESTONE_FOR_EVENT = {"leftoff_opened": "first_leftoff_open"}


@telemetry_router.post("/event", status_code=204)
async def telemetry_event(body: TelemetryEventBody):
    from agent_os import telemetry

    try:
        fields = _validate_event(body.event, body.fields if isinstance(body.fields, dict) else {})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    telemetry.emit(body.event, fields)
    milestone = _MILESTONE_FOR_EVENT.get(body.event)
    if milestone:
        telemetry.latch(milestone)
    return Response(status_code=204)
