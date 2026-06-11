# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: budget field naming and wiring.

Covers:
- the legacy budget_spent_usd / runtime_budget_spent_usd PUT fields are now
  INERT (Budget Piece 2 deleted the spend-write); they are accepted but do not
  persist a dollar value
- budget_limit_usd accepted on POST /api/v2/projects (creation)
- GET returns both budget_spent_usd (always 0.0 now) and budget_limit_usd
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


class TestBudgetFieldNaming:
    """Budget Piece 2: the legacy spend-write PUT fields are accepted but INERT
    — they no longer persist a dollar accumulator (the ledger owns spend)."""

    def test_runtime_budget_spent_usd_is_inert(self, client, workspace):
        """'runtime_budget_spent_usd' is accepted (200) but does not persist a
        dollar value — spend stays 0.0 (the legacy write is gone)."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "runtime_budget_spent_usd": 3.50,
        })
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0

    def test_budget_spent_usd_alias_is_inert(self, client, workspace):
        """The 'budget_spent_usd' alias is likewise accepted but inert."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "budget_spent_usd": 2.25,
        })
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0

    def test_both_fields_together_still_inert(self, client, workspace):
        """Sending both legacy fields is accepted without error and persists
        nothing (no dollar accumulator)."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "runtime_budget_spent_usd": 5.00,
            "budget_spent_usd": 1.00,
        })
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0


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
        assert detail["budget_spent_usd"] == 0.0

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

    def test_get_returns_consistent_field_names(self, client, workspace):
        """GET /api/v2/projects/{pid} returns budget_spent_usd (not runtime_ prefix)."""
        pid = _create_project(client, workspace)
        detail = client.get(f"/api/v2/projects/{pid}").json()
        assert "budget_spent_usd" in detail
        assert "runtime_budget_spent_usd" not in detail
