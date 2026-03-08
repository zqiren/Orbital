# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: GET /api/v2/network/lan-url returns the machine's LAN IP.

The endpoint discovers the machine's LAN IP via a UDP socket trick and
returns {"ip": "<ip>"}.  On machines without network it returns
ip=null with an error message.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def client(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    app = create_app(data_dir=str(data_dir))
    return TestClient(app)


def test_lan_url_returns_ip(client):
    """GET /api/v2/network/lan-url must return 200 with an ip key."""
    resp = client.get("/api/v2/network/lan-url")
    assert resp.status_code == 200
    body = resp.json()
    assert "ip" in body
    ip = body["ip"]
    if ip is not None:
        # Must not be a loopback address
        assert not ip.startswith("127.")


def test_lan_ip_format(client):
    """If a LAN IP is returned, it must be a valid IPv4 address."""
    resp = client.get("/api/v2/network/lan-url")
    body = resp.json()
    ip = body.get("ip")
    if ip is not None:
        assert re.match(r"\d+\.\d+\.\d+\.\d+$", ip), (
            f"IP does not match expected format: {ip}"
        )
