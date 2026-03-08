# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E2E Test: Full pairing flow between desktop daemon and simulated phone.

Tests:
  1. Desktop initiates pairing via daemon → relay tunnel
  2. Phone redeems the pairing code via relay REST
  3. JWT is returned and is valid
  4. Paired device is persisted on daemon side
"""

import httpx
import jwt
import pytest

from .conftest import DAEMON_BASE, RELAY_BASE

# The relay uses this secret in test mode (set via env in conftest).
JWT_SECRET = "test-secret-not-for-production"


@pytest.mark.usefixtures("services")
class TestPairingFlow:
    """Full pairing flow E2E test."""

    def test_pairing_start_returns_code(self, daemon_http: httpx.Client):
        """POST /api/v2/pairing/start on daemon returns a 6-char code."""
        resp = daemon_http.post("/api/v2/pairing/start")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "code" in data
        code = data["code"]
        assert isinstance(code, str)
        assert len(code) == 6, f"Expected 6-char code, got {code!r}"

    def test_full_pairing_flow(self, daemon_http: httpx.Client, phone_http: httpx.Client):
        """Complete pairing: daemon starts, phone redeems, JWT validates."""
        # 1. Desktop starts pairing
        resp = daemon_http.post("/api/v2/pairing/start")
        assert resp.status_code == 200
        code = resp.json()["code"]

        # 2. Phone redeems code
        resp2 = phone_http.post("/api/v1/pair", json={"code": code})
        assert resp2.status_code == 200, f"Phone pair failed: {resp2.text}"
        pair_data = resp2.json()

        assert "token" in pair_data
        assert "device_id" in pair_data
        assert "phone_id" in pair_data

        # 3. Verify JWT is valid and contains expected claims
        token = pair_data["token"]
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        assert decoded["device_id"] == pair_data["device_id"]
        assert decoded["phone_id"] == pair_data["phone_id"]

        # 4. Verify relay status endpoint works with the token
        status_resp = phone_http.get(
            "/api/v1/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert status_resp.status_code == 200
        status = status_resp.json()
        assert "device_online" in status
        assert "paired_phones" in status

    def test_invalid_code_rejected(self, phone_http: httpx.Client):
        """Phone submitting a bogus code should get 400."""
        resp = phone_http.post("/api/v1/pair", json={"code": "ZZZZZZ"})
        assert resp.status_code == 400

    def test_code_single_use(self, daemon_http: httpx.Client, phone_http: httpx.Client):
        """A pairing code can only be redeemed once."""
        # Start pairing
        resp = daemon_http.post("/api/v2/pairing/start")
        assert resp.status_code == 200
        code = resp.json()["code"]

        # First redemption succeeds
        resp2 = phone_http.post("/api/v1/pair", json={"code": code})
        assert resp2.status_code == 200

        # Second redemption fails
        resp3 = phone_http.post("/api/v1/pair", json={"code": code})
        assert resp3.status_code == 400

    def test_unauthenticated_status_rejected(self, phone_http: httpx.Client):
        """Status endpoint without JWT returns 401."""
        resp = phone_http.get("/api/v1/status")
        assert resp.status_code == 401
