# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for budget config + the new ledger-based spend model.

History: this file originally pinned the LEGACY cumulative dollar accumulator
(``runtime.budget_spent_usd``, written after each LLM call via the loop's
``on_cost_update`` callback). Budget Piece 2 DELETED that accumulator entirely
— the token ledger is now the single source of recorded spend, and the
pre-flight guard (``budget/guard.py``) is the single enforcement point. The
loop no longer writes ``budget_spent_usd`` at all.

The old-contract tests have been rewritten to the NEW invariant:
  * the loop emits a token-ledger event per response and writes NO dollar
    accumulator (``TestLoopLedgerEmission``);
  * the PUT route's legacy spend-reset write is gone — resetting spend no longer
    mutates a persisted counter (``TestBudgetAPI``).
The generic ProjectStore runtime-batching tests stay (the field is still a
tolerated runtime key, just never written by the budget path now).
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
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None
    yield
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None


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
# Agent loop: ledger emission, NO dollar accumulator (Budget Piece 2)
# ---------------------------------------------------------------------------

class TestLoopLedgerEmission:
    """The loop records spend via the token LEDGER only — never a dollar
    accumulator. The legacy on_cost_update / budget_spent_usd constructor
    params are GONE; passing them raises TypeError."""

    def test_loop_rejects_legacy_budget_kwargs(self):
        """The deleted legacy budget constructor params no longer exist."""
        from agent_os.agent.loop import AgentLoop

        for kw in (
            {"on_cost_update": lambda *a: None},
            {"budget_spent_usd": 1.5},
            {"cost_per_1k_input": 0.003},
            {"budget_limit_usd": 5.0},
            {"budget_action": "stop"},
        ):
            with pytest.raises(TypeError):
                AgentLoop(
                    MagicMock(), MagicMock(), MagicMock(), MagicMock(), **kw,
                )

    @pytest.mark.asyncio
    async def test_loop_emits_ledger_event_and_writes_no_dollar_accumulator(
        self, tmp_path,
    ):
        """One LLM response → one ledger line; project_store gets NO
        budget_spent_usd write (the dollar accumulator is gone)."""
        from agent_os.agent.loop import AgentLoop
        from agent_os.agent.session import Session
        from agent_os.budget.ledger import ledger_path
        from agent_os.agent.providers.types import TokenUsage

        workspace = str(tmp_path / "ws")
        os.makedirs(workspace, exist_ok=True)
        session = Session.new("proj_x_0001", workspace, session_id="proj_x_0001",
                              provider="moonshot", model="kimi", sdk="openai")

        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        response = MagicMock()
        response.usage = usage
        response.has_tool_calls = False
        response.text = "done"
        response.raw_message = {}

        provider = MagicMock()
        provider.sdk = "openai"
        provider.provider = "moonshot"
        provider.model = "kimi"

        registry = MagicMock()
        registry.reset_run_state = MagicMock()

        context_manager = MagicMock()

        loop = AgentLoop(
            session, provider, registry, context_manager,
            project_dir=workspace,
        )
        loop._stream_response = AsyncMock(return_value=response)

        await loop.run("hello")

        # Exactly one ledger line was written for the management response.
        path = ledger_path(workspace)
        assert os.path.exists(path), "ledger file should exist after a response"
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1
        rec = lines[0]
        assert rec["source"] == "management"
        assert rec["provider"] == "moonshot"
        # uncached_input = input - cache_read (0) = 1000; output = 500
        assert rec["uncached_input"] == 1000
        assert rec["output"] == 500


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
        # openai _default => 2.5/15.0 per 1M (providers.json, verified 2026-06-11)
        assert abs(input_rate - 0.0025) < 0.00001
        assert abs(output_rate - 0.015) < 0.00001

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

    def test_legacy_spend_write_is_retired(self, client, workspace):
        """Budget Piece 2: PUT runtime_budget_spent_usd no longer WRITES a
        persisted dollar counter. The field is accepted (200) but inert — the
        endpoint's spend-reset repurpose (anchor reset) is P2-D's job. Until
        then, a PUT carrying it does not mutate budget_spent_usd, which stays 0."""
        pid = _create_project(client, workspace)

        # PUT the legacy field — must NOT 500, must NOT persist a dollar value.
        resp = client.put(
            f"/api/v2/projects/{pid}",
            json={"runtime_budget_spent_usd": 2.75},
        )
        assert resp.status_code == 200
        # The legacy write is gone — spend stays at the default 0.0.
        assert resp.json()["budget_spent_usd"] == 0.0

        # Confirm via GET as well (no persisted runtime accumulator).
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0

    def test_budget_spent_read_shape_preserved(self, client, workspace):
        """GET /projects/:id still surfaces budget_spent_usd (tolerated READ of
        the persisted field for back-compat); it defaults to 0.0 since nothing
        writes it anymore."""
        pid = _create_project(client, workspace)
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert "budget_spent_usd" in resp.json()
        assert resp.json()["budget_spent_usd"] == 0.0

    def test_new_project_has_zero_spent(self, client, workspace):
        """New project starts with budget_spent_usd=0."""
        pid = _create_project(client, workspace)
        resp = client.get(f"/api/v2/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["budget_spent_usd"] == 0.0
