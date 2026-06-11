# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: budget field naming and wiring.

P3-F coupled removal: the legacy ``budget_spent_usd`` /
``runtime_budget_spent_usd`` surface is GONE on both request and response sides.

Covers:
- GET / PUT project responses contain NO ``budget_spent_usd`` key (the ledger
  is the single source of recorded spend; the frontend reads it via GET /cost)
- the legacy ``budget_spent_usd`` / ``runtime_budget_spent_usd`` PUT *request*
  aliases no longer trigger a spend-window reset (the only reset surface is
  ``reset_budget_anchor: true``); they are ignored as bogus config keys
- budget_limit_usd accepted on POST /api/v2/projects (creation) and GET
"""

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


def _create_project(client, workspace):
    resp = client.post("/api/v2/projects", json={
        "name": "BudgetTest",
        "workspace": workspace,
        "model": "gpt-4o",
        "api_key": "sk-test",
    })
    assert resp.status_code == 201
    return resp.json()["project_id"]


class TestBudgetSpentUsdRemoved:
    """P3-F: ``budget_spent_usd`` is removed from BOTH the GET and PUT response
    shapes, and the legacy request aliases no longer reset the spend window."""

    def test_get_response_has_no_budget_spent_usd_key(self, client, workspace):
        pid = _create_project(client, workspace)
        detail = client.get(f"/api/v2/projects/{pid}").json()
        assert "budget_spent_usd" not in detail
        assert "runtime_budget_spent_usd" not in detail

    def test_put_response_has_no_budget_spent_usd_key(self, client, workspace):
        pid = _create_project(client, workspace)
        body = client.put(
            f"/api/v2/projects/{pid}", json={"budget_limit_usd": 10.0},
        ).json()
        assert "budget_spent_usd" not in body
        assert "runtime_budget_spent_usd" not in body

    def test_legacy_spent_alias_does_not_reset_window(self, client, workspace):
        """Sending the dead ``budget_spent_usd`` field is accepted (200) but is
        NOT a reset surface anymore — no ``budget_reset`` outcome is produced."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "budget_spent_usd": 2.25,
        })
        assert resp.status_code == 200
        assert "budget_reset" not in resp.json()
        assert "budget_spent_usd" not in resp.json()

    def test_legacy_runtime_alias_does_not_reset_window(self, client, workspace):
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "runtime_budget_spent_usd": 3.50,
        })
        assert resp.status_code == 200
        assert "budget_reset" not in resp.json()

    def test_reset_only_via_reset_budget_anchor(self, client, workspace):
        """The ONLY reset surface is ``reset_budget_anchor: true`` (total mode)."""
        pid = _create_project(client, workspace)
        # Put the project in total mode first (anchor only bounds that window).
        client.put(f"/api/v2/projects/{pid}", json={"budget_period": "total"})
        resp = client.put(f"/api/v2/projects/{pid}",
                          json={"reset_budget_anchor": True})
        assert resp.status_code == 200
        assert resp.json()["budget_reset"]["applied"] is True
        assert resp.json()["budget_reset"]["code"] is None


class TestBudgetLimitOnCreate:
    """budget_limit_usd accepted on project creation."""

    def test_create_project_with_budget_limit(self, client, workspace):
        """POST /api/v2/projects with budget_limit_usd stores it."""
        resp = client.post("/api/v2/projects", json={
            "name": "BudgetLimitTest",
            "workspace": workspace,
            "model": "gpt-4o",
            "api_key": "sk-test",
            "budget_limit_usd": 10.00,
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        detail = client.get(f"/api/v2/projects/{pid}").json()
        assert detail["budget_limit_usd"] == 10.00
        # P3-F: no spend field on the response.
        assert "budget_spent_usd" not in detail

    def test_create_project_without_budget_limit(self, client, workspace):
        """POST without budget_limit_usd defaults to no limit."""
        resp = client.post("/api/v2/projects", json={
            "name": "NoBudgetTest",
            "workspace": workspace,
            "model": "gpt-4o",
            "api_key": "sk-test",
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        detail = client.get(f"/api/v2/projects/{pid}").json()
        assert detail.get("budget_limit_usd") is None

    def test_get_returns_limit_without_spent_field(self, client, workspace):
        """GET /api/v2/projects/{pid} returns the set budget_limit_usd and NO
        spend field."""
        pid = _create_project(client, workspace)
        client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 7.0})
        detail = client.get(f"/api/v2/projects/{pid}").json()
        assert detail["budget_limit_usd"] == 7.0
        assert "budget_spent_usd" not in detail
        assert "runtime_budget_spent_usd" not in detail
