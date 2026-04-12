# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for budget spend persistence across sessions.

Root cause: budget_spent_usd was reset to 0.0 every session because it was
only tracked in-memory in AgentLoop._budget_spent_usd. A project with a $5
budget effectively had $5 per session, not $5 total.

Fix: persist cumulative spend in project JSON under "runtime.budget_spent_usd",
load on session start, flush after each LLM call via on_cost_update callback.
"""
import json
import os
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.agent.pricing import get_cost_rates, _pricing_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def project_store(tmp_path):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return ProjectStore(data_dir)


@pytest.fixture(autouse=True)
def _clear_pricing_cache():
    """Reset the pricing module cache between tests."""
    import agent_os.agent.pricing as pricing_mod
    pricing_mod._pricing_cache = None
    yield
    pricing_mod._pricing_cache = None


def _create_project(client, workspace, **overrides):
    payload = {
        "name": "TestProj",
        "workspace": workspace,
        "model": "gpt-4o",
        "api_key": "sk-test",
        **overrides,
    }
    resp = client.post("/api/v2/projects", json=payload)
    assert resp.status_code == 201
    return resp.json()["project_id"]


# ---------------------------------------------------------------------------
# Project store: runtime budget persistence
# ---------------------------------------------------------------------------

class TestProjectBudgetPersistence:
    """Budget spend persists across ProjectStore instances (simulating restart)."""

    def test_project_spend_persists(self, tmp_path):
        """update_runtime() writes to disk; a fresh store instance reads it back."""
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)

        store1 = ProjectStore(data_dir)
        pid = store1.create_project({
            "name": "P1", "workspace": "/tmp", "model": "m", "api_key": "k",
        })
        store1.update_runtime(pid, {"budget_spent_usd": 3.45})

        # Simulate daemon restart: new store instance reads from disk
        store2 = ProjectStore(data_dir)
        project = store2.get_project(pid)
        assert project["runtime"]["budget_spent_usd"] == 3.45

    def test_project_spend_defaults_zero(self, project_store):
        """A new project has no runtime section; default spend is 0."""
        pid = project_store.create_project({
            "name": "P2", "workspace": "/tmp", "model": "m", "api_key": "k",
        })
        project = project_store.get_project(pid)
        assert project.get("runtime", {}).get("budget_spent_usd", 0.0) == 0.0

    def test_update_runtime_accumulates(self, project_store):
        """Multiple update_runtime calls overwrite, not accumulate."""
        pid = project_store.create_project({
            "name": "P3", "workspace": "/tmp", "model": "m", "api_key": "k",
        })
        project_store.update_runtime(pid, {"budget_spent_usd": 1.0})
        project_store.update_runtime(pid, {"budget_spent_usd": 2.5})
        project = project_store.get_project(pid)
        assert project["runtime"]["budget_spent_usd"] == 2.5

    def test_update_runtime_ignores_unknown_project(self, project_store):
        """update_runtime with non-existent project_id does nothing."""
        project_store.update_runtime("proj_nonexistent", {"budget_spent_usd": 1.0})
        # No exception raised


# ---------------------------------------------------------------------------
# Agent loop: cost callback and cumulative init
# ---------------------------------------------------------------------------

class TestLoopCostTracking:
    """AgentLoop calls on_cost_update after each LLM response."""

    @pytest.mark.asyncio
    async def test_loop_cost_callback_invoked(self):
        """on_cost_update receives (delta, total) after each response."""
        from agent_os.agent.loop import AgentLoop

        session = MagicMock()
        session.messages = [{"role": "user", "content": "hello"}]
        session.is_stopped = MagicMock(return_value=False)
        session.pop_queued_messages = MagicMock(return_value=[])
        session.append = MagicMock()
        session.append_system = MagicMock()
        session.notify_stream = MagicMock()
        session.pause = MagicMock()

        # Track callback invocations
        cost_updates = []
        def on_cost(delta, total):
            cost_updates.append((delta, total))

        provider = MagicMock()
        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 500
        response = MagicMock()
        response.usage = usage
        response.has_tool_calls = False
        response.text = "done"

        loop = AgentLoop(
            session, provider, MagicMock(), MagicMock(),
            budget_spent_usd=0.0,
            on_cost_update=on_cost,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )

        # Monkey-patch _stream_response to return our mock response
        loop._stream_response = AsyncMock(return_value=response)

        await loop.run("hello")

        assert len(cost_updates) == 1
        delta, total = cost_updates[0]
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015
        assert abs(delta - expected) < 0.0001
        assert abs(total - expected) < 0.0001

    @pytest.mark.asyncio
    async def test_loop_starts_with_persisted_spend(self):
        """Loop accumulates from persisted budget_spent_usd, not from zero."""
        from agent_os.agent.loop import AgentLoop

        session = MagicMock()
        session.messages = [{"role": "user", "content": "hello"}]
        session.is_stopped = MagicMock(return_value=False)
        session.pop_queued_messages = MagicMock(return_value=[])
        session.append = MagicMock()
        session.append_system = MagicMock()
        session.notify_stream = MagicMock()
        session.pause = MagicMock()

        cost_updates = []
        def on_cost(delta, total):
            cost_updates.append((delta, total))

        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 0
        response = MagicMock()
        response.usage = usage
        response.has_tool_calls = False
        response.text = "done"

        loop = AgentLoop(
            session, MagicMock(), MagicMock(), MagicMock(),
            budget_spent_usd=1.5,  # persisted from previous session
            on_cost_update=on_cost,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        )
        loop._stream_response = AsyncMock(return_value=response)

        await loop.run("hello")

        assert len(cost_updates) == 1
        delta, total = cost_updates[0]
        # total should be 1.5 (persisted) + delta (new)
        expected_delta = (1000 / 1000) * 0.003
        assert abs(total - (1.5 + expected_delta)) < 0.0001


# ---------------------------------------------------------------------------
# Pricing lookup
# ---------------------------------------------------------------------------

class TestPricingLookup:
    """get_cost_rates resolves per-model pricing from providers.json."""

    def test_exact_match(self):
        """Known model returns exact pricing."""
        input_rate, output_rate = get_cost_rates("gpt-5.2", "openai")
        # providers.json: gpt-5.2 => 1.75/14.00 per 1M => 0.00175/0.014 per 1K
        assert abs(input_rate - 0.00175) < 0.00001
        assert abs(output_rate - 0.014) < 0.00001

    def test_prefix_match(self):
        """Model with date suffix matches by prefix."""
        input_rate, output_rate = get_cost_rates(
            "claude-opus-4-6-20260301", "anthropic",
        )
        # Should match "claude-opus-4-6" => 5.00/25.00 per 1M
        assert abs(input_rate - 0.005) < 0.00001
        assert abs(output_rate - 0.025) < 0.00001

    def test_provider_default_fallback(self):
        """Unknown model uses provider _default."""
        input_rate, output_rate = get_cost_rates("unknown-model-xyz", "openai")
        # openai _default => 2.00/8.00 per 1M
        assert abs(input_rate - 0.002) < 0.00001
        assert abs(output_rate - 0.008) < 0.00001

    def test_unknown_provider_global_fallback(self):
        """Unknown provider falls back to global default."""
        input_rate, output_rate = get_cost_rates("some-model", "nonexistent_provider")
        # Global fallback: 3.00/15.00 per 1M
        assert abs(input_rate - 0.003) < 0.00001
        assert abs(output_rate - 0.015) < 0.00001


# ---------------------------------------------------------------------------
# API: budget fields roundtrip
# ---------------------------------------------------------------------------

class TestBudgetAPI:
    """Budget fields flow through project CRUD endpoints."""

    def test_budget_limit_roundtrip(self, client, workspace):
        """PUT budget_limit_usd -> GET returns it."""
        pid = _create_project(client, workspace)
        resp = client.put(f"/api/v2/projects/{pid}", json={"budget_limit_usd": 5.0})
        assert resp.status_code == 200

        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["budget_limit_usd"] == 5.0
        assert data["budget_spent_usd"] == 0.0

    def test_reset_spend_via_api(self, client, workspace):
        """PUT runtime_budget_spent_usd=0 resets spend."""
        pid = _create_project(client, workspace)

        # Set some spend via API first
        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"runtime_budget_spent_usd": 2.75},
        )
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 2.75

        # Reset via API
        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"runtime_budget_spent_usd": 0},
        )
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0

    def test_budget_spent_in_project_detail(self, client, workspace):
        """GET /projects/:id includes budget_spent_usd from runtime."""
        pid = _create_project(client, workspace)

        # Set spend via API (PUT runtime_budget_spent_usd)
        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"runtime_budget_spent_usd": 1.23},
        )
        assert resp.status_code == 200

        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 1.23

    def test_new_project_has_zero_spent(self, client, workspace):
        """New project starts with budget_spent_usd=0."""
        pid = _create_project(client, workspace)
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0
