# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for connector management (Spec 011 §2, §6; Task B3).

Global catalog + OAuth live here; per-project enablement rides the existing
project-update route via ``enabled_connectors`` (not exposed here). The
ConnectorManager singleton is injected by the app factory via ``configure``.
"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_os.connectors.manager import AuthStart, ConnectorError
from agent_os.connectors.oauth import OAuthError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/connectors")

_manager = None


def configure(connector_manager):
    """Called by the app factory to inject the ConnectorManager singleton."""
    global _manager
    _manager = connector_manager


class CustomServerRequest(BaseModel):
    name: str
    url: str
    auth_type: Literal["none", "oauth2"] = "none"


def _require_manager():
    if _manager is None:
        raise HTTPException(status_code=503, detail="Connector manager not available")
    return _manager


def _serialize(entry: dict) -> dict:
    """Flatten a ``list_connectors`` entry into a JSON card for the settings UI."""
    manifest = entry["manifest"]
    return {
        **manifest.to_dict(),
        "connected": entry["connected"],
        "account": entry["account"],
        "enabled_error": entry["enabled_error"],
    }


@router.get("")
async def list_connectors():
    """Catalog + custom servers with per-connector connection status."""
    mgr = _require_manager()
    return {"connectors": [_serialize(e) for e in mgr.list_connectors()]}


@router.post("/custom")
async def add_custom(req: CustomServerRequest):
    """Add a Tier-0 user-supplied MCP server (Spec 011 §3)."""
    mgr = _require_manager()
    try:
        manifest = await mgr.add_custom(req.name, req.url, req.auth_type)
    except (ConnectorError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return manifest.to_dict()


@router.delete("/custom/{connector_id}")
async def remove_custom(connector_id: str):
    mgr = _require_manager()
    try:
        await mgr.remove_custom(connector_id)
    except ConnectorError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "ok"}


@router.post("/{connector_id}/connect")
async def connect(connector_id: str):
    """Run the OAuth handshake (or no-op for ``none``). Blocking loopback flow:
    returns ``{connected: true, account}`` on success, or ``{auth_url}`` if a
    connector defers to a client-driven flow."""
    mgr = _require_manager()
    if mgr.get_manifest(connector_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown connector: {connector_id}")
    try:
        result = await mgr.connect(connector_id)
    except ConnectorError as exc:
        # pending_verification / not-connectable
        raise HTTPException(status_code=409, detail=str(exc))
    except OAuthError as exc:
        # client not configured, state mismatch, timeout, token failure
        raise HTTPException(status_code=400, detail=str(exc))
    if isinstance(result, AuthStart):
        return {"auth_url": result.auth_url}
    return {"connected": True, "account": result.account}


@router.post("/{connector_id}/disconnect")
async def disconnect(connector_id: str):
    mgr = _require_manager()
    if mgr.get_manifest(connector_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown connector: {connector_id}")
    await mgr.disconnect(connector_id)
    return {"status": "ok"}
