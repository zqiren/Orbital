# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Credential CRUD endpoints — website credential management."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2")

_credential_store = None
_agent_manager = None


def configure(credential_store, agent_manager=None):
    global _credential_store, _agent_manager
    _credential_store = credential_store
    _agent_manager = agent_manager


class StoreCredentialRequest(BaseModel):
    name: str
    domain: str
    fields: dict[str, str]
    project_id: str | None = None  # If set, resume paused session


@router.post("/credentials")
async def store_credential(req: StoreCredentialRequest):
    """Store a credential. Session lifecycle (resume) is handled by the
    separate ``/approve`` endpoint which the frontend calls after storing."""
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    try:
        _credential_store.store(req.name, req.domain, req.fields)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "stored", "name": req.name}


@router.get("/credentials")
async def list_credentials():
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    return _credential_store.list_all()


@router.delete("/credentials/{name}")
async def delete_credential(name: str):
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    _credential_store.delete(name)
    return {"status": "deleted", "name": name}


@router.post("/credentials/{name}/revoke")
async def revoke_credential(name: str):
    """Clear browser cookies for this credential's domain.
    Stub: browser cookie clearing deferred to browser-tool merge."""
    if _credential_store is None:
        raise HTTPException(status_code=501, detail="Credential store not available")
    # TODO: when BrowserManager is available, call browser_manager.clear_cookies(domain)
    return {"status": "revoked", "name": name, "note": "Cookie clearing pending browser-tool integration"}
