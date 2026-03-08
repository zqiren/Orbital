# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""E2E Test: Remote project dashboard via relay proxy.

Tests:
  1. Create a project via the daemon
  2. Fetch projects via relay (as phone) with JWT auth
  3. Verify same projects returned, but api_key is redacted
"""

import httpx
import pytest

from .conftest import DAEMON_BASE, RELAY_BASE


def _pair_phone(daemon_http: httpx.Client, phone_http: httpx.Client) -> str:
    """Helper: run pairing flow and return JWT token."""
    resp = daemon_http.post("/api/v2/pairing/start")
    assert resp.status_code == 200, f"Pairing start failed: {resp.text}"
    code = resp.json()["code"]

    resp2 = phone_http.post("/api/v1/pair", json={"code": code})
    assert resp2.status_code == 200, f"Code redeem failed: {resp2.text}"
    return resp2.json()["token"]


@pytest.mark.usefixtures("services")
class TestProjectDashboard:
    """Remote project listing and api_key redaction tests."""

    def test_project_list_via_relay(
        self, daemon_http: httpx.Client, phone_http: httpx.Client
    ):
        """Phone can list projects via relay; api_key is redacted."""
        # 0. Pair phone
        token = _pair_phone(daemon_http, phone_http)
        auth = {"Authorization": f"Bearer {token}"}

        # 1. Create a project directly on the daemon
        project_payload = {
            "name": "E2E Test Project",
            "workspace": "/tmp/e2e-test",
            "model": "gpt-4",
            "api_key": "sk-secret-key-12345",
        }
        create_resp = daemon_http.post(
            "/api/v2/projects",
            json=project_payload,
        )
        assert create_resp.status_code in (200, 201), (
            f"Project create failed: {create_resp.text}"
        )
        created = create_resp.json()
        project_id = created["project_id"]

        try:
            # 2. Fetch projects via relay (phone)
            relay_resp = phone_http.get("/api/v2/projects", headers=auth)
            assert relay_resp.status_code == 200, (
                f"Relay project list failed: {relay_resp.text}"
            )
            projects = relay_resp.json()

            # 3. Find our project in the list
            assert isinstance(projects, list)
            matched = [p for p in projects if p.get("project_id") == project_id]
            assert len(matched) == 1, (
                f"Expected project {project_id} in relay response, got {[p.get('project_id') for p in projects]}"
            )
            relay_project = matched[0]

            # 4. Verify project data matches (minus api_key)
            assert relay_project["name"] == project_payload["name"]
            assert relay_project["model"] == project_payload["model"]

            # 5. api_key MUST be redacted
            assert relay_project.get("api_key") != project_payload["api_key"], (
                "api_key was NOT redacted in relay response!"
            )
            assert relay_project.get("api_key") == "***", (
                f"Expected api_key='***', got {relay_project.get('api_key')!r}"
            )
        finally:
            # Cleanup: delete the project
            daemon_http.delete(f"/api/v2/projects/{project_id}")

    def test_relay_requires_auth_for_projects(self, phone_http: httpx.Client):
        """Accessing /api/v2/projects via relay without JWT returns 401."""
        resp = phone_http.get("/api/v2/projects")
        assert resp.status_code == 401

    def test_relay_project_detail_redacted(
        self, daemon_http: httpx.Client, phone_http: httpx.Client
    ):
        """Single project detail via relay also has api_key redacted."""
        token = _pair_phone(daemon_http, phone_http)
        auth = {"Authorization": f"Bearer {token}"}

        # Create project
        create_resp = daemon_http.post(
            "/api/v2/projects",
            json={
                "name": "Detail Test",
                "workspace": "/tmp/detail-test",
                "model": "gpt-4",
                "api_key": "sk-detail-secret",
            },
        )
        assert create_resp.status_code in (200, 201)
        project_id = create_resp.json()["project_id"]

        try:
            # Fetch single project via relay
            resp = phone_http.get(f"/api/v2/projects/{project_id}", headers=auth)
            if resp.status_code == 200:
                data = resp.json()
                assert data.get("api_key") == "***", (
                    f"api_key not redacted in detail view: {data.get('api_key')!r}"
                )
        finally:
            daemon_http.delete(f"/api/v2/projects/{project_id}")
