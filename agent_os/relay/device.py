# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Device identity management for cloud relay pairing.

On first run, generates a unique device_id and device_secret and persists
them to ~/orbital/device.json.  Subsequent calls return the same identity.
"""

import json
import uuid
import secrets
from pathlib import Path

import httpx


def get_or_create_device_identity(config_dir=None):
    """Return {device_id, device_secret}. Creates file on first run."""
    if config_dir is None:
        config_dir = Path.home() / "orbital"
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    device_file = config_dir / "device.json"

    if device_file.exists():
        return json.loads(device_file.read_text())

    identity = {
        "device_id": f"dev_{uuid.uuid4().hex[:12]}",
        "device_secret": secrets.token_hex(32),
    }
    device_file.write_text(json.dumps(identity, indent=2))
    return identity


async def register_device(relay_url, device_id, device_secret):
    """POST /relay/devices to register this device with the relay server."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{relay_url}/relay/devices",
            json={"device_id": device_id, "secret": device_secret},
        )
        resp.raise_for_status()
        return resp.json()
