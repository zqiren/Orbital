# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pairing endpoints for cloud relay phone pairing.

Allows starting a pairing flow, listing paired devices,
and revoking paired devices.  Paired device state is persisted
to ~/orbital/paired_devices.json.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v2")

_relay_client = None


def configure(relay_client):
    """Called by app factory to inject the RelayClient (may be None)."""
    global _relay_client
    _relay_client = relay_client


def _devices_file() -> Path:
    p = Path.home() / "orbital" / "paired_devices.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_devices() -> list[dict]:
    f = _devices_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save_devices(devices: list[dict]):
    _devices_file().write_text(json.dumps(devices, indent=2))


def add_paired_device(phone_id: str):
    """Record a newly paired phone (called by RelayClient on pairing.complete)."""
    devices = _load_devices()
    # Avoid duplicates
    if any(d.get("phone_id") == phone_id for d in devices):
        return
    devices.append({
        "phone_id": phone_id,
        "paired_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_devices(devices)


@router.post("/pairing/start")
async def start_pairing():
    """Initiate pairing: sends pairing.create through relay, returns code."""
    if _relay_client is None:
        raise HTTPException(status_code=503, detail="Relay not configured")

    try:
        result = await _relay_client.send_pairing_create(timeout=30.0)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    code = result.get("code")
    if not code:
        raise HTTPException(status_code=502, detail="No pairing code received")

    return {"code": code, "relay_url": _relay_client.relay_url}


@router.get("/pairing/devices")
async def list_paired_devices():
    """Return all currently paired phone devices."""
    return _load_devices()


@router.delete("/pairing/devices/{phone_id}")
async def revoke_paired_device(phone_id: str):
    """Revoke a paired phone and remove from local storage."""
    devices = _load_devices()
    updated = [d for d in devices if d.get("phone_id") != phone_id]

    if len(updated) == len(devices):
        raise HTTPException(status_code=404, detail="Device not found")

    _save_devices(updated)

    # Notify relay if connected
    if _relay_client is not None:
        try:
            await _relay_client.send_pairing_revoke(phone_id)
        except Exception:
            pass  # best-effort; local state already updated

    return {"status": "revoked"}
