# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GET /api/v2/settings never waits for the agent check.

v0.21.0 regression (Windows user, 2026-11-09): ``/settings`` gained
``import_scan_agent`` (spec 160), which asked ``available_worker_slugs`` and
so ran the whole agent check (every CLI's version + sign-in) on a cold cache
before answering. ``/settings`` is also the desktop shell's boot health
check (2 s per try, 15 s in all): on a machine where the check takes longer
than that, Orbital showed "Failed to start daemon" with a healthy daemon
behind it. The route now reads only the last check's result and starts a
check in the background when there is none.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agents.setup_engine import SetupEngine
from tests.unit.keyless_harness import agent_status, build_client

SLOW = 1.5  # seconds per agent: the check is far slower than the route may take


def _slow_engine(tmp_path, slugs=("claude-code", "codex"), delay=SLOW):
    registry = MagicMock()
    registry.list_all.return_value = [SimpleNamespace(slug=s) for s in slugs]
    registry.get.side_effect = lambda slug: SimpleNamespace(
        name={"claude-code": "Claude Code", "codex": "Codex"}.get(slug, slug))
    engine = SetupEngine(registry, data_dir=str(tmp_path / "engine"))
    calls: list[str] = []
    lock = threading.Lock()

    def check_agent(slug):
        with lock:
            calls.append(slug)
        time.sleep(delay)
        return agent_status(slug)

    engine.check_agent = check_agent
    return engine, calls


def _wait_for(pred, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# SetupEngine.check_all_cached
# ---------------------------------------------------------------------------

def test_cold_cache_returns_none_at_once_and_checks_in_the_background(tmp_path):
    engine, calls = _slow_engine(tmp_path)
    started = time.monotonic()
    assert engine.check_all_cached() is None
    assert time.monotonic() - started < 0.2
    assert _wait_for(lambda: engine.check_all_cached() is not None)
    assert [s.slug for s in engine.check_all_cached()] == ["claude-code", "codex"]
    assert sorted(calls) == ["claude-code", "codex"]  # one check, not one per call


def test_expired_cache_is_served_while_one_refresh_runs(tmp_path):
    engine, calls = _slow_engine(tmp_path, delay=0.0)
    engine.check_all()
    results, _ = engine._check_all_cache
    engine._check_all_cache = (results, time.monotonic() - 1)  # expire it
    engine.check_agent = lambda slug: (time.sleep(SLOW), agent_status(slug))[1]
    started = time.monotonic()
    assert [s.slug for s in engine.check_all_cached()] == ["claude-code", "codex"]
    assert time.monotonic() - started < 0.2
    assert _wait_for(lambda: engine._check_all_cache[1] > time.monotonic())


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------

def _settings_client(tmp_path, monkeypatch, engine):
    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import settings as settings_routes

    _tc, deps = build_client(tmp_path, keyed=False)
    mgr = deps["agent_manager"]
    monkeypatch.setattr(mgr, "_setup_engine", engine)
    monkeypatch.setattr(mgr, "_registry", engine._registry, raising=False)
    monkeypatch.setattr(agents_v2, "_agent_manager", mgr)
    fake = MagicMock()
    fake.get_masked = MagicMock(return_value={"llm": {"api_key_set": False}})
    settings_routes.configure(fake, project_store=deps["project_store"])
    app = FastAPI()
    app.include_router(settings_routes.router)
    return TestClient(app), deps


def _timed_get(client):
    started = time.monotonic()
    resp = client.get("/api/v2/settings")
    return resp, time.monotonic() - started


def test_settings_answers_before_the_first_agent_check(tmp_path, monkeypatch):
    engine, _calls = _slow_engine(tmp_path)
    client, _deps = _settings_client(tmp_path, monkeypatch, engine)

    resp, elapsed = _timed_get(client)
    assert resp.status_code == 200
    assert elapsed < 1.0, f"/settings waited {elapsed:.1f}s for the agent check"
    assert resp.json()["import_scan_agent"] is None  # not known yet

    # The check it started finishes in the background; the next read has it.
    assert _wait_for(lambda: engine.check_all_cached() is not None)
    resp, elapsed = _timed_get(client)
    assert elapsed < 1.0
    assert resp.json()["import_scan_agent"] == {"slug": "claude-code", "name": "Claude Code"}


def test_settings_with_an_imported_project_waiting_does_not_wait_either(tmp_path, monkeypatch):
    from agent_os.onboarding.import_scan import PENDING, STATE_FIELD

    engine, _calls = _slow_engine(tmp_path)
    client, deps = _settings_client(tmp_path, monkeypatch, engine)
    ws = tmp_path / "imported"
    ws.mkdir()
    pid = deps["project_store"].create_project({"name": "imp", "workspace": str(ws)})
    deps["project_store"].update_project(pid, {STATE_FIELD: PENDING})

    resp, elapsed = _timed_get(client)
    assert resp.status_code == 200
    assert elapsed < 1.0, f"/settings waited {elapsed:.1f}s for the agent check"
    assert resp.json()["unscanned_imports"] == 1

    assert _wait_for(lambda: engine.check_all_cached() is not None)
    assert client.get("/api/v2/settings").json()["import_scan_agent"] == {
        "slug": "claude-code", "name": "Claude Code"}
