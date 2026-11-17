# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Global skills: Orbital's own folder every project shares, and the
user's CLI skill folders it reads (Settings → Skills).

    GET    /api/v2/skills                 four groups: global, agents, claude, codex
    POST   /api/v2/skills                 upload a .md / .zip into the global folder
    DELETE /api/v2/skills/{dir_name}      remove one from the global folder only
    POST   /api/v2/skills/from-project    copy a project's skill into it
    PUT    /api/v2/skills/enabled         switch one shared skill on or off

The global folder is ``{data_dir}/skills`` (``project_paths.global_skills_dir``),
created on the first add. The home folders (~/.claude/skills, ~/.codex/skills,
~/.agents/skills) are listed read-only and never written or deleted here.
Shapes and codes: run/1117-share/CONTRACT-skills.md.
"""

from __future__ import annotations

import asyncio
import os
import shutil

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from agent_os.agent import skills as skills_mod
from agent_os.agent.project_paths import ProjectPaths, global_skills_dir
from agent_os.agent.skill_catalog import SHARED_SOURCES, shared_skill_groups
from agent_os.agent.skills import (
    SkillInstallError, install_skill_dir, install_skill_upload, parse_skill_md,
    sanitize_unicode_dir_name, skill_id, validate_skill_text,
)

router = APIRouter(prefix="/api/v2/skills")

# Same 10 MiB as the Files upload; a zip's contents are capped separately.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UNPACKED_BYTES = 50 * 1024 * 1024
MAX_FILES = 2000

_project_store = None


def configure(data_dir: str, *, project_store=None) -> None:
    """Point every SkillLoader / SkillCatalog at ``{data_dir}/skills``."""
    global _project_store
    _project_store = project_store
    skills_mod.configure_global_skills_dir(global_skills_dir(data_dir))


def _root() -> str:
    root = skills_mod.global_skills_dir()
    if not root:
        raise HTTPException(status_code=503, detail="Global skills folder not configured")
    return root


def _limits():
    return (MAX_FILES, MAX_UNPACKED_BYTES)


def _list() -> dict:
    _root()
    return {"groups": shared_skill_groups()}


def _out(row: dict) -> dict:
    sid = skill_id("global", row["name"])
    return {**row, "source": "global", "editable": True, "id": sid,
            "enabled": sid not in skills_mod.read_disabled_skills()}


def _install(fn):
    try:
        return _out(fn())
    except SkillInstallError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)


@router.get("")
async def list_skills():
    return await asyncio.to_thread(_list)


@router.post("", status_code=201)
async def add_skill(file: UploadFile = File(...), overwrite: str = Form("")):
    root = _root()
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Skill is too large")
    filename = file.filename or ""
    return await asyncio.to_thread(_install, lambda: install_skill_upload(
        root, filename, content,
        validate=validate_skill_text, sanitize=sanitize_unicode_dir_name,
        name_from_skill=True, overwrite=overwrite.strip().lower() == "true",
        limits=_limits()))


def _delete(root: str, dir_name: str) -> dict:
    if not dir_name or dir_name.startswith(".") or "/" in dir_name or "\\" in dir_name:
        raise HTTPException(status_code=400, detail="Invalid skill name")
    path = os.path.join(root, dir_name)
    root_real = os.path.realpath(root).rstrip(os.sep) + os.sep
    if os.path.islink(path):
        # A linked skill: remove the link, never what it points at.
        sid = _switch_id(path)
        os.unlink(path)
        _forget_switch(sid)
        return {"deleted": dir_name}
    if not os.path.realpath(path).startswith(root_real):
        raise HTTPException(status_code=400, detail="Invalid skill name")
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Skill not found: {dir_name}")
    sid = _switch_id(path)
    shutil.rmtree(path)
    _forget_switch(sid)
    return {"deleted": dir_name}


def _switch_id(skill_dir: str) -> str | None:
    """The switch id of the global skill in ``skill_dir`` (read before it goes)."""
    meta = parse_skill_md(os.path.join(skill_dir, "SKILL.md"))
    if not meta:
        return None
    return skill_id("global", meta["name"] or os.path.basename(skill_dir))


def _forget_switch(sid: str | None) -> None:
    """A deleted global skill takes its switch with it, so a later skill of
    the same name starts switched on."""
    if sid and sid in skills_mod.read_disabled_skills():
        skills_mod.set_skill_enabled(sid, True)


@router.delete("/{dir_name}")
async def delete_skill(dir_name: str):
    root = _root()
    return await asyncio.to_thread(_delete, root, dir_name)


class FromProjectRequest(BaseModel):
    project_id: str
    dir_name: str
    overwrite: bool = False


@router.post("/from-project", status_code=201)
async def copy_from_project(req: FromProjectRequest):
    root = _root()
    project = _project_store.get_project(req.project_id) if _project_store else None
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    base = ProjectPaths(project.get("workspace", "")).skills_dir
    src = os.path.join(base, req.dir_name)
    base_real = os.path.realpath(base).rstrip(os.sep) + os.sep
    if not req.dir_name or not os.path.realpath(src).startswith(base_real):
        raise HTTPException(status_code=400, detail="Invalid skill name")
    if not os.path.isdir(src):
        raise HTTPException(status_code=404, detail=f"Skill not found: {req.dir_name}")
    return await asyncio.to_thread(_install, lambda: install_skill_dir(
        src, root, validate=validate_skill_text, sanitize=sanitize_unicode_dir_name,
        overwrite=req.overwrite, limits=_limits()))


class EnabledRequest(BaseModel):
    id: str
    enabled: bool


def _set_enabled(req: EnabledRequest) -> dict:
    source = req.id.split(":", 1)[0]
    if source not in SHARED_SOURCES or ":" not in req.id:
        raise HTTPException(status_code=400, detail="Invalid skill id")
    known = {r["id"] for g in shared_skill_groups() for r in g["skills"]}
    if req.id not in known:
        raise HTTPException(status_code=404, detail=f"Skill not found: {req.id}")
    skills_mod.set_skill_enabled(req.id, req.enabled)
    return {"id": req.id, "enabled": req.enabled}


@router.put("/enabled")
async def set_enabled(req: EnabledRequest):
    _root()
    return await asyncio.to_thread(_set_enabled, req)
