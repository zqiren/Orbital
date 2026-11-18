# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""In-app upgrade routes (``agent_os/updater.py``).

``GET  /api/v2/update/install``  the install state; no network (the pill
                                 polls it while a run is in flight).
``POST /api/v2/update/install``  the pill click: start (or continue) the
                                 one-click upgrade. 409 ``{code}`` when this
                                 app cannot update itself or nothing newer is
                                 offered; the pill then keeps the manual
                                 download.
``POST /api/v2/update/dev-state`` dev daemons only (``AGENT_OS_UPDATE_DEV=1``,
                                 never frozen): force a state for UI review.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_os import updater

router = APIRouter(prefix="/api/v2")

_IDLE = {"state": updater.IDLE, "version": None, "progress": 0, "error": None,
         "source": None, "can_install": False}


class InstallRequest(BaseModel):
    region: Literal["global", "china"] | None = None


class DevStateRequest(BaseModel):
    state: Literal["idle", "downloading", "verifying", "ready", "installing", "failed"] | None = None
    version: str | None = None
    progress: int = 0
    error: str | None = None
    can_install: bool = True
    clear: bool = False


def install_snapshot() -> dict:
    u = updater.get_updater()
    return u.snapshot() if u is not None else dict(_IDLE)


@router.get("/update/install")
async def get_install_state():
    return install_snapshot()


@router.post("/update/install")
async def start_install(req: InstallRequest):
    u = updater.get_updater()
    if u is None:
        raise HTTPException(status_code=409, detail={"code": "unsupported"})
    try:
        return await u.start(req.region)
    except updater.UpdateError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code}) from None


def _dev_hook_enabled() -> bool:
    return not getattr(sys, "frozen", False) and os.environ.get("AGENT_OS_UPDATE_DEV") == "1"


@router.post("/update/dev-state")
async def force_dev_state(req: DevStateRequest):
    u = updater.get_updater()
    if not _dev_hook_enabled() or u is None:
        raise HTTPException(status_code=404)
    if req.clear:
        return u.force_dev_state(None)
    return u.force_dev_state(req.model_dump())
