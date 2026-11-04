# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""File history routes (spec 065 §5.6).

- ``GET  /projects/{id}/files/history/status`` — project health + whether any
  change was recorded (D10: nothing renders before the first one).
- ``GET  /projects/{id}/files/history?path=`` — a file's versions.
- ``GET  /projects/{id}/files/history/view?path=&sha=&from=`` — tinted lines.
- ``GET  /projects/{id}/files/history/turn?session_id=&from_ts=&to_ts=`` —
  files one turn changed.
- ``POST /projects/{id}/files/history/restore`` — ``{items:[{path, sha,
  expect_sha}]}`` → per item ``ok | conflict | busy | refused``.
- ``POST /projects/{id}/files/history/name`` — ``{path, sha, name}``.

Containment reuses ``files_v2._resolve_path`` (realpath both sides).
"""
from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent_os.api.routes import files_v2
from agent_os.history import rules, views
from agent_os.history.service import HistoryUnavailable
from agent_os.history.store import iso_to_epoch

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

_project_store = None
_agent_manager = None
_file_history = None


def configure(project_store, agent_manager=None, file_history=None):
    global _project_store, _agent_manager, _file_history
    _project_store = project_store
    _agent_manager = agent_manager
    _file_history = file_history


class RestoreItem(BaseModel):
    path: str
    sha: str | None = None
    expect_sha: str | None = None
    version: int | None = None


class RestoreRequest(BaseModel):
    items: list[RestoreItem]


class NameRequest(BaseModel):
    path: str
    sha: str
    name: str = ""


def _project(project_id: str) -> dict:
    project = _project_store.get_project(project_id) if _project_store else None
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _rel(project_id: str, path: str) -> tuple[str, str]:
    workspace, target = files_v2._resolve_path(project_id, path)
    real_ws = os.path.realpath(workspace)
    real_target = os.path.join(os.path.realpath(os.path.dirname(target)), os.path.basename(target))
    rel = os.path.relpath(real_target, real_ws)
    if rel == "." or rel.startswith(".."):
        raise HTTPException(status_code=400, detail="Path outside workspace")
    return workspace, rules.to_rel(rel)


def _proj(project_id: str):
    return _file_history.get_project(project_id) if _file_history is not None else None


def _status(project_id: str) -> dict:
    if _file_history is None:
        return {"status": "unavailable", "reason": "not_running"}
    return _file_history.status(project_id)


async def _session_index(project_id: str, workspace: str) -> views.SessionIndex:
    sessions: dict[str, dict] = {}
    if _agent_manager is not None:
        try:
            listed = await asyncio.to_thread(_agent_manager.list_sessions, project_id)
            for s in listed or []:
                sid = s.get("session_id")
                if sid:
                    sessions[sid] = {"name": s.get("name"), "session_uuid": s.get("session_uuid")}
        except Exception:  # noqa: BLE001
            logger.debug("history: session list failed for %s", project_id, exc_info=True)
    return views.SessionIndex(workspace, sessions)


@router.get("/projects/{project_id}/files/history/status")
async def history_status(project_id: str):
    _project(project_id)
    out = _status(project_id)
    proj = _proj(project_id)
    store = proj.store if proj is not None else None
    out["has_changes"] = bool(store is not None and store.has_changes())
    out["store_bytes"] = store.total_bytes if store is not None else 0
    out["tracked_files"] = len(store.index) if store is not None else 0
    skipped: dict[str, int] = {}
    if proj is not None:
        for reason in proj.skipped.values():
            skipped[reason] = skipped.get(reason, 0) + 1
    out["skipped"] = skipped
    return out


@router.get("/projects/{project_id}/files/history")
async def file_history(project_id: str, path: str):
    workspace, rel = _rel(project_id, path)
    out = {"path": rel, **_status(project_id), "whole_file_only": rules.is_office(rel)}
    proj = _proj(project_id)
    if out["status"] == "unavailable" or proj is None or proj.store is None:
        out["versions"] = []
        return out
    idx = await _session_index(project_id, workspace)
    versions = await asyncio.to_thread(views.file_versions, proj.store, rel, idx)
    out["versions"] = versions
    if not versions:
        out["skipped_reason"] = await asyncio.to_thread(_skip_reason, proj, rel)
    return out


def _skip_reason(proj, rel: str) -> str | None:
    if proj.status == "building":
        return None
    if rules.is_denied(rel, proj.nested):
        return "ignored"
    abs_path = os.path.join(proj.workspace, rel)
    if not os.path.isfile(abs_path):
        return "missing"
    if rel in proj.skipped:
        return proj.skipped[rel]
    try:
        reason = rules.classify(abs_path, rel)
    except OSError:
        return "missing"
    return reason or "ignored"


@router.get("/projects/{project_id}/files/history/view")
async def file_history_view(project_id: str, path: str, sha: str | None = None,
                            version: int | None = None,
                            from_sha: str | None = Query(None, alias="from"),
                            from_version: int | None = None):
    """The version's lines diffed against ``from`` (default: the previous
    version). ``version`` / ``from_version`` pick an exact version id when
    the same content occurs twice in a file's history."""
    workspace, rel = _rel(project_id, path)
    proj = _proj(project_id)
    if proj is None or proj.store is None or proj.status == "unavailable":
        raise HTTPException(status_code=409, detail=_status(project_id))
    idx = await _session_index(project_id, workspace)
    out = await asyncio.to_thread(views.version_view, proj.store, rel, idx, sha=sha, v=version,
                                  from_sha=from_sha, from_v=from_version)
    if out is None:
        raise HTTPException(status_code=404, detail="Version not found")
    return out


@router.get("/projects/{project_id}/files/history/turn")
async def turn_history(project_id: str, session_id: str, from_ts: str, to_ts: str | None = None):
    project = _project(project_id)
    proj = _proj(project_id)
    status = _status(project_id)
    if proj is None or proj.store is None or status["status"] == "unavailable":
        return {**status, "session_id": session_id, "files": [], "no_history": 0}
    f = iso_to_epoch(from_ts)
    if f is None:
        raise HTTPException(status_code=400, detail="from_ts must be ISO 8601 or epoch seconds")
    t = iso_to_epoch(to_ts) if to_ts else None
    idx = await _session_index(project_id, project.get("workspace", ""))
    out = await asyncio.to_thread(views.turn_files, proj.store, idx, session_id, f, t,
                                  list(proj.skip_events))
    out["from_ts"], out["to_ts"] = from_ts, to_ts
    return {**status, **out}


@router.post("/projects/{project_id}/files/history/restore")
async def restore_files(project_id: str, req: RestoreRequest):
    _project(project_id)
    if _file_history is None:
        raise HTTPException(status_code=409, detail=_status(project_id))
    items = []
    for it in req.items:
        _ws, rel = _rel(project_id, it.path)
        items.append({"path": rel, "sha": it.sha, "expect_sha": it.expect_sha, "version": it.version})
    # Attribution "at this moment" for the content a restore captures
    # first; computed here on the loop (the predicates evict adapters).
    capture = _file_history.current_attrib(project_id)
    try:
        results = await asyncio.to_thread(_file_history.restore, project_id, items,
                                          capture_attrib=capture)
    except HistoryUnavailable:
        raise HTTPException(status_code=409, detail=_status(project_id))
    return {"results": results}


@router.post("/projects/{project_id}/files/history/name")
async def name_version(project_id: str, req: NameRequest):
    _project(project_id)
    _ws, rel = _rel(project_id, req.path)
    proj = _proj(project_id)
    if proj is None or proj.store is None:
        raise HTTPException(status_code=409, detail=_status(project_id))
    if not any(v.get("sha") == req.sha for v in proj.store.versions_for(rel)):
        raise HTTPException(status_code=404, detail="Version not found")
    rec = await asyncio.to_thread(_file_history.set_name, project_id, rel, req.sha, req.name.strip())
    return {"path": rel, "sha": req.sha, "name": rec.get("name") or None}
