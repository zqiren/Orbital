# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression + integration tests for the pricing API (Task P3-D).

Boots the real ``pricing_v2`` router in a FastAPI app via TestClient and drives
``GET /api/v2/pricing`` and ``PUT /api/v2/pricing/overrides`` against real files
on a temp override path (injected the same way ``resolve_rates`` resolves it —
the module-level ``pricing._OVERRIDE_FILE`` set from ``AGENT_OS_DATA_DIR``). The
ROUTE reads ``pricing.override_file_path()`` at request time, so it picks up the
injected path with no hardcoding.

Coverage:
- PUT validation rejects malformed overrides (negative, non-numeric, currency
  non-string) with codes-only 400 bodies.
- override → revert round-trip: PUT a field → GET origin "override" → PUT again
  WITHOUT the field → GET origin "default" AND resolve_rates returns the
  default value (the per-field revert the frontend performs by omission).
- PUT then ``resolve_rates`` reflects the new value WITHOUT a restart (the
  mtime-invalidation path from Piece 1).
- GET exposes the resolved override_path for debuggability.
- Atomic write round-trips: GET-after-PUT reports the written field as
  origin "override".
- Full-replace semantics: a second PUT that omits an earlier provider drops it.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import agent_os.agent.pricing as pricing_mod
from agent_os.agent.pricing import resolve_rates
from agent_os.api.routes import pricing_v2


@pytest.fixture(autouse=True)
def _reset_pricing_caches():
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None
    yield
    pricing_mod._pricing_cache = None
    pricing_mod._full_providers_cache = None
    pricing_mod._override_cache = None
    pricing_mod._override_mtime = None


@pytest.fixture
def client(tmp_path):
    """Real app with the real pricing router; override path on tmp_path."""
    override_path = str(tmp_path / "pricing-overrides.json")
    pricing_mod._OVERRIDE_FILE = override_path

    app = FastAPI()
    pricing_v2.configure()
    app.include_router(pricing_v2.router)
    c = TestClient(app)
    c._override_path = override_path  # type: ignore[attr-defined]
    yield c
    pricing_mod._OVERRIDE_FILE = pricing_mod._default_override_path()


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------

class TestGetPricing:
    def test_get_returns_table_with_origin_and_path(self, client):
        r = client.get("/api/v2/pricing")
        assert r.status_code == 200
        body = r.json()
        assert "providers" in body
        assert body["override_path"] == client._override_path
        anthropic = body["providers"]["anthropic"]
        assert anthropic["currency"] == "USD"
        assert anthropic["currency_origin"] == "default"
        cell = anthropic["models"]["claude-sonnet-4-6"]
        assert cell["input_per_1m"]["value"] == pytest.approx(3.0)
        assert cell["input_per_1m"]["origin"] == "default"
        assert cell["output_per_1m"]["value"] == pytest.approx(15.0)

    def test_get_covers_every_default_model(self, client):
        r = client.get("/api/v2/pricing")
        body = r.json()
        full = pricing_mod._load_full_providers()
        for provider, data in full.items():
            assert provider in body["providers"]
            models = body["providers"][provider]["models"]
            for model in data.get("pricing", {}):
                assert model in models, f"{provider}/{model}"


# ---------------------------------------------------------------------------
# PUT validation — codes-only 400 bodies
# ---------------------------------------------------------------------------

class TestPutValidation:
    def test_negative_rate_rejected(self, client):
        r = client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"input_per_1m": -1.0}}
        })
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_rate"}

    def test_non_numeric_rate_rejected(self, client):
        r = client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"input_per_1m": "lots"}}
        })
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_rate"}

    def test_bool_rate_rejected(self, client):
        """bool is an int subclass — must not slip through as a rate."""
        r = client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"input_per_1m": True}}
        })
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_rate"}

    def test_non_string_currency_rejected(self, client):
        r = client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"_currency": 5}
        })
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_currency"}

    def test_non_dict_provider_block_rejected(self, client):
        r = client.put("/api/v2/pricing/overrides", json={"anthropic": "nope"})
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_shape"}

    def test_non_dict_model_entry_rejected(self, client):
        r = client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": 3.0}
        })
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_shape"}

    def test_non_dict_top_level_rejected(self, client):
        r = client.put("/api/v2/pricing/overrides", json=[1, 2, 3])
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_shape"}

    def test_unparseable_json_rejected(self, client):
        r = client.put(
            "/api/v2/pricing/overrides",
            content="{not json",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert r.json() == {"code": "invalid_json"}

    def test_unknown_provider_and_model_allowed(self, client):
        """Forward-compat: unknown keys are accepted, not rejected."""
        r = client.put("/api/v2/pricing/overrides", json={
            "future-provider": {"future-model": {"input_per_1m": 1.0}}
        })
        assert r.status_code == 200
        body = r.json()
        assert body["providers"]["future-provider"]["models"][
            "future-model"]["input_per_1m"]["origin"] == "override"

    def test_zero_rate_allowed(self, client):
        r = client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"input_per_1m": 0}}
        })
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Round-trip: PUT writes, GET reports origin "override"
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_put_then_get_reports_override(self, client):
        client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"output_per_1m": 50.0}}
        })
        body = client.get("/api/v2/pricing").json()
        cell = body["providers"]["anthropic"]["models"]["claude-sonnet-4-6"]
        assert cell["output_per_1m"]["origin"] == "override"
        assert cell["output_per_1m"]["value"] == pytest.approx(50.0)
        # Untouched field still default.
        assert cell["input_per_1m"]["origin"] == "default"

    def test_put_response_matches_subsequent_get(self, client):
        put_body = client.put("/api/v2/pricing/overrides", json={
            "openai": {"gpt-5.2": {"input_per_1m": 9.0}}
        }).json()
        get_body = client.get("/api/v2/pricing").json()
        assert put_body["providers"] == get_body["providers"]

    def test_file_written_atomically_on_disk(self, client):
        client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"output_per_1m": 50.0}}
        })
        with open(client._override_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk == {
            "anthropic": {"claude-sonnet-4-6": {"output_per_1m": 50.0}}
        }


# ---------------------------------------------------------------------------
# override → revert round-trip (per-field revert by omission)
# ---------------------------------------------------------------------------

class TestOverrideRevertRoundTrip:
    def test_override_then_revert_field(self, client):
        path = client._override_path

        # 1. Override a field.
        client.put("/api/v2/pricing/overrides", json={
            "anthropic": {"claude-sonnet-4-6": {"output_per_1m": 50.0}}
        })
        cell = client.get("/api/v2/pricing").json()[
            "providers"]["anthropic"]["models"]["claude-sonnet-4-6"]
        assert cell["output_per_1m"]["origin"] == "override"
        assert cell["output_per_1m"]["value"] == pytest.approx(50.0)
        # resolve_rates agrees, without restart.
        assert resolve_rates(
            "anthropic", "claude-sonnet-4-6", override_path=path
        ).output_per_1m == pytest.approx(50.0)

        # 2. PUT again WITHOUT that field (the frontend's revert = omission).
        client.put("/api/v2/pricing/overrides", json={})

        # 3. GET now shows origin "default" and the default value is back.
        cell2 = client.get("/api/v2/pricing").json()[
            "providers"]["anthropic"]["models"]["claude-sonnet-4-6"]
        assert cell2["output_per_1m"]["origin"] == "default"
        assert cell2["output_per_1m"]["value"] == pytest.approx(15.0)
        # resolve_rates returns the default value, no restart.
        assert resolve_rates(
            "anthropic", "claude-sonnet-4-6", override_path=path
        ).output_per_1m == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# PUT then resolve_rates reflects the new value WITHOUT restart (mtime path)
# ---------------------------------------------------------------------------

class TestNoRestartMtimePath:
    def test_resolve_rates_sees_put_without_restart(self, client):
        path = client._override_path

        # Baseline default before any override.
        assert resolve_rates(
            "openai", "gpt-5.2", override_path=path
        ).input_per_1m == pytest.approx(1.75)

        # PUT a new value.
        client.put("/api/v2/pricing/overrides", json={
            "openai": {"gpt-5.2": {"input_per_1m": 42.0}}
        })

        # resolve_rates (same process, no restart) sees it via mtime invalidation.
        assert resolve_rates(
            "openai", "gpt-5.2", override_path=path
        ).input_per_1m == pytest.approx(42.0)

        # A second PUT updates again, still no restart.
        client.put("/api/v2/pricing/overrides", json={
            "openai": {"gpt-5.2": {"input_per_1m": 7.0}}
        })
        assert resolve_rates(
            "openai", "gpt-5.2", override_path=path
        ).input_per_1m == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Full-replace semantics (documented choice)
# ---------------------------------------------------------------------------

class TestFullReplaceSemantics:
    def test_second_put_replaces_not_merges(self, client):
        # First PUT sets two providers.
        client.put("/api/v2/pricing/overrides", json={
            "openai": {"gpt-5.2": {"input_per_1m": 9.0}},
            "anthropic": {"claude-sonnet-4-6": {"output_per_1m": 50.0}},
        })
        # Second PUT only mentions openai → anthropic override is dropped.
        client.put("/api/v2/pricing/overrides", json={
            "openai": {"gpt-5.2": {"input_per_1m": 9.0}},
        })
        body = client.get("/api/v2/pricing").json()
        assert body["providers"]["openai"]["models"]["gpt-5.2"][
            "input_per_1m"]["origin"] == "override"
        # anthropic reverted to default (full-replace dropped it).
        assert body["providers"]["anthropic"]["models"]["claude-sonnet-4-6"][
            "output_per_1m"]["origin"] == "default"
