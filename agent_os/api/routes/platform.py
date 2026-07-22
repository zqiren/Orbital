# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for platform provider management.

Exposes sandbox setup, status, and folder access control to the desktop app (Electron).
"""

import asyncio
import inspect
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import platform as platform_mod
import string

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/platform")


# ---- Request models ----

class FolderGrantRequest(BaseModel):
    path: str
    mode: Literal["read_only", "read_write"]
    # Which project's portal scope this desktop grant attaches to. Portals are
    # per-scope (Spec 12 §2b) so a folder granted here lands in exactly one
    # project's sandbox profile. Omitted → the Quick Tasks (scratch) project,
    # the natural home for a user-initiated "let my assistant see this folder".
    project_id: str | None = None


class FolderRevokeRequest(BaseModel):
    path: str
    project_id: str | None = None


class BrowserWarmupRequest(BaseModel):
    url: str = "https://accounts.google.com"


class MkdirRequest(BaseModel):
    parent: str
    name: str


# ---- Dependency holder ----

_platform_provider = None
_agent_manager = None
_browser_manager = None


def configure(platform_provider, agent_manager=None, browser_manager=None):
    """Called by app factory to inject the platform provider."""
    global _platform_provider, _agent_manager, _browser_manager
    _platform_provider = platform_provider
    _agent_manager = agent_manager
    _browser_manager = browser_manager


# ---- Folder-name validation (New Project workspace picker's "New folder") ----
#
# Enforced on ALL platforms, not just Windows: a folder created here may later
# sync to or be shared with a Windows machine, so names must stay valid there
# regardless of which OS the daemon happens to run on.

_INVALID_NAME_CHARS = '<>:"/\\|?*'
_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
# ~240 leaves headroom under Windows' ~260-char MAX_PATH for the rest of a
# real filesystem path (drive letter, parent segments) built on top of this.
_MAX_FOLDER_PATH_LENGTH = 240


def _validate_folder_name(name: str) -> Optional[str]:
    """Return None if `name` is a valid cross-platform folder name, else a
    user-legible reason string."""
    if not name or not name.strip():
        return "Folder name cannot be empty"
    if name != name.strip():
        return "Folder name cannot start or end with a space"
    if name.startswith('.') or name.endswith('.'):
        return "Folder name cannot start or end with a dot"
    for ch in name:
        if ch in _INVALID_NAME_CHARS or ord(ch) < 0x20:
            return f'Folder name cannot contain any of: {_INVALID_NAME_CHARS}'
    stem = name.split('.', 1)[0].upper()
    if stem in _RESERVED_NAMES:
        return f'"{name}" is a reserved name on Windows'
    return None


# ---- Endpoints ----

@router.get("/status")
async def platform_status():
    """Return platform capabilities and setup status."""
    caps = _platform_provider.get_capabilities()
    return {"status": "ok", **asdict(caps)}


@router.post("/setup")
async def platform_setup():
    """Trigger first-run sandbox setup (may require UAC elevation)."""
    result = await _platform_provider.setup()
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Setup failed")
    return {"status": "ok", **asdict(result)}


@router.post("/skip")
async def platform_skip():
    """Disable sandbox isolation at runtime, switching to NullProvider."""
    global _platform_provider
    from agent_os.platform.null import NullProvider
    null_provider = NullProvider()
    _platform_provider = null_provider
    if _agent_manager is not None:
        _agent_manager._platform_provider = null_provider
        if hasattr(_agent_manager, '_sub_agent_manager') and _agent_manager._sub_agent_manager is not None:
            _agent_manager._sub_agent_manager._platform_provider = null_provider
    return {"status": "ok"}


@router.post("/teardown")
async def platform_teardown():
    """Remove sandbox user and clean up."""
    result = await _platform_provider.teardown()
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Teardown failed")
    return {"status": "ok", **asdict(result)}


@router.get("/folders")
async def platform_folders():
    """Return list of grantable folders with access status."""
    folders = _platform_provider.get_available_folders()
    return {"status": "ok", "folders": [asdict(f) for f in folders]}


def _resolve_grant_scope(project_id: str | None) -> str:
    """Resolve the portal scope (owning workspace realpath) for a desktop grant.

    A named project uses its own workspace; otherwise the grant attaches to the
    Quick Tasks (scratch) workspace. Raises 400 if no target can be resolved.
    """
    store = getattr(_agent_manager, "_project_store", None) if _agent_manager else None
    project = None
    if store is not None:
        if project_id:
            project = store.get_project(project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
        else:
            project = store.find_scratch_project()
    workspace = (project or {}).get("workspace")
    if not workspace:
        raise HTTPException(
            status_code=400, detail="No target project workspace for folder grant"
        )
    return os.path.realpath(workspace)


@router.post("/folders/grant")
async def platform_folders_grant(req: FolderGrantRequest):
    """Grant sandbox access to a folder within a project's portal scope."""
    scope = _resolve_grant_scope(req.project_id)
    result = _platform_provider.grant_folder_access(req.path, req.mode, scope=scope)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Grant failed")
    return {"status": "ok", **asdict(result)}


@router.post("/folders/revoke")
async def platform_folders_revoke(req: FolderRevokeRequest):
    """Revoke sandbox access to a folder within a project's portal scope."""
    scope = _resolve_grant_scope(req.project_id)
    result = _platform_provider.revoke_folder_access(req.path, scope=scope)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Revoke failed")
    return {"status": "ok", **asdict(result)}


@router.get("/browse")
async def platform_browse(path: Optional[str] = None):
    """List subdirectories of a given path for the folder picker UI."""
    home = Path.home()
    target = Path(path) if path else home

    # On Windows, "/" is not a real directory — list available drives instead
    if platform_mod.system() == "Windows" and path == "/":
        drives = []
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                drives.append({
                    "name": f"{letter}:",
                    "path": str(drive),
                    "has_children": True,
                })
        return {
            "path": "/",
            "parent": None,
            "display_name": "This PC",
            "entries": drives,
        }

    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    entries = []
    try:
        for item in sorted(target.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                try:
                    has_children = any(
                        c.is_dir() for c in item.iterdir()
                        if not c.name.startswith('.')
                    )
                except PermissionError:
                    has_children = False
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "has_children": has_children,
                })
    except PermissionError:
        pass

    resolved = target.resolve()
    parent = str(resolved.parent) if resolved.parent != resolved else None

    return {
        "path": str(target),
        "parent": parent,
        "display_name": target.name or str(target),
        "entries": entries,
    }


@router.post("/mkdir")
async def platform_mkdir(req: MkdirRequest):
    """Create a single leaf folder inside an existing parent, for the New
    Project workspace picker's "New folder" affordance.

    Creates ONLY the leaf (`Path.mkdir()`, never `parents=True`) so a typo'd
    parent path fails loudly (400) instead of materializing a deep wrong tree.
    """
    invalid_reason = _validate_folder_name(req.name)
    if invalid_reason:
        raise HTTPException(status_code=400, detail=invalid_reason)

    parent = Path(req.parent)
    if not parent.is_dir():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")

    new_path = parent / req.name
    if len(str(new_path)) > _MAX_FOLDER_PATH_LENGTH:
        raise HTTPException(status_code=400, detail="Resulting path is too long")

    if new_path.exists():
        raise HTTPException(status_code=409, detail="A folder with that name already exists")

    try:
        new_path.mkdir()
    except FileExistsError:
        raise HTTPException(status_code=409, detail="A folder with that name already exists")
    except PermissionError:
        raise HTTPException(
            status_code=403, detail="Permission denied creating this folder"
        )
    except OSError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok", "path": str(new_path)}


@router.post("/browser/warmup")
async def browser_warmup(req: BrowserWarmupRequest = BrowserWarmupRequest()):
    """Launch headed browser for cookie warmup (non-blocking).

    Starts the warmup browser in a background task and returns immediately.
    The frontend polls GET /browser/warmup/status to detect when the user
    closes the browser.

    Guarded on RUNNING SESSIONS, not on context existence: the daemon's
    headless context holds the profile lock once lazily launched, so we
    close it (close_for_handoff) before launching the headed warmup. While
    any session is running that close would yank the browser out from under
    it — refuse with 409 instead.
    """
    if _browser_manager is None:
        raise HTTPException(status_code=503, detail="Browser manager not available")
    if _browser_manager.warmup_active:
        return {"status": "already_active"}

    if _agent_manager is not None:
        project_ids = {pid for (pid, _sid) in _agent_manager._handles.keys()}
        for pid in project_ids:
            for sess in _agent_manager.list_sessions(pid):
                if sess["status"] != "idle":
                    raise HTTPException(
                        status_code=409,
                        detail="A session is currently running. Stop running sessions, then try again.",
                    )

    # Release the profile lock held by the idle daemon context (no-op if
    # never launched). isawaitable: tolerates sync doubles in route tests.
    close_result = _browser_manager.close_for_handoff()
    if inspect.isawaitable(close_result):
        await close_result

    async def _run_warmup():
        try:
            await _browser_manager.launch_warmup(req.url)
        except Exception:
            # warmup_active is reset in launch_warmup's finally path, but a
            # dead warmup must never be silent — an orphaned sign-in browser
            # blocks every agent browser launch until the user quits it.
            logger.exception("Browser warmup task failed")

    asyncio.create_task(_run_warmup())

    # Give the browser a moment to launch so we can catch immediate errors
    await asyncio.sleep(0.5)
    if not _browser_manager.warmup_active:
        raise HTTPException(status_code=500, detail="Browser failed to launch")
    return {"status": "launched"}


@router.get("/browser/warmup/status")
async def browser_warmup_status():
    """Check whether the warmup browser is still open."""
    if _browser_manager is None:
        raise HTTPException(status_code=503, detail="Browser manager not available")
    return {"active": _browser_manager.warmup_active}
