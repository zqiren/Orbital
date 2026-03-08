# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: budget field naming and wiring.

Covers:
- budget_spent_usd vs runtime_budget_spent_usd alignment (PUT accepts both)
- budget_limit_usd accepted on POST /api/v2/projects (creation)
- GET returns both budget_spent_usd and budget_limit_usd
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
    """Both field names work for setting budget spend."""

    def test_runtime_budget_spent_usd_still_works(self, client, workspace):
        """Original 'runtime_budget_spent_usd' field continues to work."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "runtime_budget_spent_usd": 3.50,
        })
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 3.50

    def test_budget_spent_usd_alias_works(self, client, workspace):
        """New 'budget_spent_usd' alias also works for PUT."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "budget_spent_usd": 2.25,
        })
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 2.25

    def test_runtime_takes_precedence(self, client, workspace):
        """If both fields sent, runtime_budget_spent_usd takes precedence."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={
            "runtime_budget_spent_usd": 5.00,
            "budget_spent_usd": 1.00,
        })
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 5.00


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
