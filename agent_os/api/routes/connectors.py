# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""REST endpoints for connector management (Spec 011 §2, §6; Task B3).

Global catalog + OAuth live here; per-project enablement rides the existing
project-update route via ``enabled_connectors`` (not exposed here). The
ConnectorManager singleton is injected by the app factory via ``configure``.

Connectors shared across agents: each connector carries its agent ``access``
(``PUT /{id}/access`` sets it) and ``GET /agents`` lists who it can be shared
with. Contract: run/1117-share/CONTRACT-connectors.md.
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
_agent_registry = None
_setup_engine = None


def configure(connector_manager, *, agent_registry=None, setup_engine=None):
    """Called by the app factory to inject the ConnectorManager singleton
    (and the agent registry + setup engine the agents list reads)."""
    global _manager, _agent_registry, _setup_engine
    _manager = connector_manager
    _agent_registry = agent_registry
    _setup_engine = setup_engine


class CustomServerRequest(BaseModel):
    name: str
    url: str
    auth_type: Literal["none", "oauth2"] = "none"


class AccessRequest(BaseModel):
    mode: Literal["all", "selected"]
    agents: list[str] = []


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
        "access": entry.get("access") or {"mode": "all", "agents": []},
    }


def _agents() -> list[dict]:
    """The manager first, then every worker manifest by name (``built-in`` is
    the legacy manager manifest, not a worker)."""
    from agent_os.daemon_v2.connector_sharing import (
        MANAGER_AGENT_ID, MANAGER_NAME, support_of)
    out = [{"id": MANAGER_AGENT_ID, "name": MANAGER_NAME, "kind": "manager",
            "installed": True, "supported": True, "reason": None}]
    manifests = _agent_registry.list_all() if _agent_registry is not None else []
    workers = []
    for m in manifests:
        if m.slug in ("built-in", MANAGER_AGENT_ID) or getattr(
                m.runtime, "adapter", None) == "built_in":
            continue
        supported, reason = support_of(m)
        installed = None
        cached = getattr(_setup_engine, "cached_status", None)
        if cached is not None:
            try:
                status = cached(m.slug)
            except Exception:  # noqa: BLE001 — a status read never fails the list
                status = None
            if status is not None:
                installed = bool(getattr(status, "installed", False))
        workers.append({"id": m.slug, "name": m.name, "kind": "worker",
                        "installed": installed, "supported": supported,
                        "reason": reason})
    workers.sort(key=lambda a: a["name"].lower())
    return out + workers


@router.get("")
async def list_connectors():
    """Catalog + custom servers with per-connector connection status."""
    mgr = _require_manager()
    return {"connectors": [_serialize(e) for e in mgr.list_connectors()]}


@router.get("/agents")
async def list_agents():
    """The agents a connector can be shared with (CONTRACT-connectors.md §3)."""
    _require_manager()
    return {"agents": _agents()}


@router.put("/{connector_id}/access")
async def set_access(connector_id: str, req: AccessRequest):
    """Set which agents may use a connector (CONTRACT-connectors.md §2)."""
    mgr = _require_manager()
    if mgr.get_manifest(connector_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown connector: {connector_id}")
    known = {a["id"] for a in _agents()}
    for agent in req.agents:
        if agent not in known:
            raise HTTPException(status_code=400, detail=f"unknown agent: {agent}")
    try:
        access = mgr.set_access(connector_id, req.mode, req.agents)
    except ConnectorError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"id": connector_id, "access": access}


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
