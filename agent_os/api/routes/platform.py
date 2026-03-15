# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for platform provider management.

Exposes sandbox setup, status, and folder access control to the desktop app (Electron).
"""

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import platform as platform_mod
import string

router = APIRouter(prefix="/api/v2/platform")


# ---- Request models ----

class FolderGrantRequest(BaseModel):
    path: str
    mode: Literal["read_only", "read_write"]


class FolderRevokeRequest(BaseModel):
    path: str


class BrowserWarmupRequest(BaseModel):
    url: str = "https://accounts.google.com"


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


@router.post("/folders/grant")
async def platform_folders_grant(req: FolderGrantRequest):
    """Grant sandbox user access to a folder."""
    result = _platform_provider.grant_folder_access(req.path, req.mode)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Grant failed")
    return {"status": "ok", **asdict(result)}


@router.post("/folders/revoke")
async def platform_folders_revoke(req: FolderRevokeRequest):
    """Revoke sandbox user access to a folder."""
    result = _platform_provider.revoke_folder_access(req.path)
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


@router.post("/browser/warmup")
async def browser_warmup(req: BrowserWarmupRequest = BrowserWarmupRequest()):
    """Launch headed browser for cookie warmup (non-blocking).

    Starts the warmup browser in a background task and returns immediately.
    The frontend polls GET /browser/warmup/status to detect when the user
    closes the browser.
    """
    if _browser_manager is None:
        raise HTTPException(status_code=503, detail="Browser manager not available")
    if _browser_manager.warmup_active:
        return {"status": "already_active"}

    async def _run_warmup():
        try:
            await _browser_manager.launch_warmup(req.url)
        except Exception:
            pass  # warmup_active is reset in launch_warmup's finally path

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
