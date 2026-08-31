# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Route tests for GET /api/v2/agents/{pid}/sessions/{sid}/context.

Feeds the composer's context line. The endpoint owns no state: it reads the
last management row out of the token ledger and pairs it with the model's
window from the provider registry. The one thing it MUST get right is that the
compaction mark it reports is the same number the agent actually triggers on —
a meter that predicts the wrong moment is worse than no meter.
"""

import json
import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.routes.agents_v2 import router, configure
from agent_os.budget.ledger import ledger_path


@pytest.fixture
def store(tmp_path):
    from agent_os.daemon_v2.project_store import ProjectStore
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    return ProjectStore(data_dir)


class _FakeRegistry:
    """Only the one method the route uses."""

    def __init__(self, windows: dict):
        self._windows = windows

    def get_context_window(self, provider: str, model: str) -> int:
        return self._windows.get((provider, model), 128_000)


@pytest.fixture
def make_client():
    def _make(store, registry=None):
        app = FastAPI()
        configure(
            project_store=store,
            agent_manager=MagicMock(),
            ws_manager=MagicMock(),
            provider_registry=registry,
        )
        app.include_router(router)
        return TestClient(app)
    return _make


def _new_project(store, tmp_path, **extra):
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace, exist_ok=True)
    config = {
        "name": "proj", "workspace": workspace, "model": "claude-x",
        "api_key": "k", "provider": "anthropic",
    }
    config.update(extra)
    return store.create_project(config), workspace


def _write_event(workspace, *, session_id="s1", source="management",
                 provider="anthropic", model="claude-x",
                 uncached=0, read=0, write=0, output=0):
    path = ledger_path(workspace)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": "2026-08-21T12:00:00+00:00", "session_id": session_id,
            "source": source, "provider": provider, "model": model,
            "uncached_input": uncached, "cache_read": read,
            "cache_write": write, "output": output,
        }) + "\n")


def _url(pid, sid="s1"):
    return f"/api/v2/agents/{pid}/sessions/{sid}/context"


class TestContextRoute:
    def test_unknown_project_404s(self, store, tmp_path, make_client):
        c = make_client(store)
        assert c.get(_url("nope")).status_code == 404

    def test_session_with_no_calls_reports_null_usage(self, store, tmp_path, make_client):
        """A fresh session has no measured prompt. The UI must render nothing
        rather than a confident 0% — those are different claims."""
        pid, _ws = _new_project(store, tmp_path)
        c = make_client(store)
        body = c.get(_url(pid)).json()
        assert body["used"] is None

    def test_reports_used_window_and_threshold(self, store, tmp_path, make_client):
        pid, ws = _new_project(store, tmp_path)
        _write_event(ws, uncached=100_000, read=40_000, write=4_000, output=900)
        c = make_client(store, _FakeRegistry({("anthropic", "claude-x"): 200_000}))
        body = c.get(_url(pid)).json()
        assert body["used"] == 144_000          # output is not context
        assert body["window"] == 200_000
        assert body["threshold"] == 160_000     # 80% of the raw window
        assert body["provider"] == "anthropic"
        assert body["model"] == "claude-x"

    def test_threshold_matches_the_agents_actual_trigger(self, store, tmp_path, make_client):
        """The mark the UI draws and the point the loop compacts at come from
        ONE function. If this drifts, the meter lies about the future."""
        from agent_os.agent.context import compaction_threshold_tokens

        for window in (32_768, 128_000, 200_000, 1_000_000):
            pid, ws = _new_project(
                store, tmp_path / f"w{window}",
                name=f"proj-{window}", agent_name=f"proj-{window}",
            )
            _write_event(ws, uncached=10)
            c = make_client(store, _FakeRegistry({("anthropic", "claude-x"): window}))
            body = c.get(_url(pid)).json()
            assert body["window"] == window
            assert body["threshold"] == compaction_threshold_tokens(window), window

    def test_threshold_is_80_percent_for_normal_models(self, store, tmp_path, make_client):
        """The whole point of the trigger change: a mark users can read."""
        pid, ws = _new_project(store, tmp_path)
        _write_event(ws, uncached=10)
        c = make_client(store, _FakeRegistry({("anthropic", "claude-x"): 1_000_000}))
        body = c.get(_url(pid)).json()
        assert body["threshold"] / body["window"] == pytest.approx(0.80)

    def test_window_follows_the_model_that_served(self, store, tmp_path, make_client):
        """Fallback rotation can swap the model mid-session. The window has to
        follow the row, not the project's pinned config."""
        pid, ws = _new_project(store, tmp_path)  # project pins claude-x
        _write_event(ws, uncached=1_000, provider="moonshot", model="kimi-x")
        c = make_client(store, _FakeRegistry({
            ("anthropic", "claude-x"): 200_000,
            ("moonshot", "kimi-x"): 128_000,
        }))
        body = c.get(_url(pid)).json()
        assert body["window"] == 128_000
        assert body["model"] == "kimi-x"

    def test_scoped_to_the_requested_session(self, store, tmp_path, make_client):
        pid, ws = _new_project(store, tmp_path)
        _write_event(ws, session_id="s1", uncached=1_000)
        _write_event(ws, session_id="s2", uncached=90_000)
        c = make_client(store, _FakeRegistry({("anthropic", "claude-x"): 200_000}))
        assert c.get(_url(pid, "s1")).json()["used"] == 1_000
        assert c.get(_url(pid, "s2")).json()["used"] == 90_000

    def test_no_registry_still_answers(self, store, tmp_path, make_client):
        """The daemon can run without a registry wired; fall back rather than
        500, so the composer just shows a conservative window."""
        pid, ws = _new_project(store, tmp_path)
        _write_event(ws, uncached=1_000)
        c = make_client(store, None)
        body = c.get(_url(pid)).json()
        assert body["used"] == 1_000
        assert body["window"] > 0
