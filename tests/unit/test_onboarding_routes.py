# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/onboarding/importable-projects route (backlog #34).

Mounts only the onboarding router over an in-process ASGI transport and stubs
the scanner so the route contract (shape + best-effort error handling) is tested
without touching real home directories."""

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import onboarding as onboarding_routes
from agent_os.onboarding.import_scanner import ImportCandidate


@pytest.fixture
def client():
    app = FastAPI()
    onboarding_routes.configure()
    app.include_router(onboarding_routes.router)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_returns_ranked_candidates(client, monkeypatch):
    fake = [
        ImportCandidate(
            source="claude-code", name="alpha", path="/p/alpha",
            session_count=3, last_activity="2026-07-01T00:00:00+00:00",
        ),
        ImportCandidate(
            source="obsidian", name="notes", path="/p/notes",
            session_count=0, last_activity="2026-08-01T00:00:00+00:00",
        ),
    ]
    monkeypatch.setattr(onboarding_routes, "scan_importable_projects", lambda: fake)

    async with client:
        r = await client.get("/api/v2/onboarding/importable-projects")
    assert r.status_code == 200
    body = r.json()
    assert [c["name"] for c in body["candidates"]] == ["alpha", "notes"]
    first = body["candidates"][0]
    assert set(first) == {"source", "name", "path", "session_count", "last_activity"}
    assert first["source"] == "claude-code"
    assert first["session_count"] == 3


async def test_scanner_failure_yields_empty_list_not_500(client, monkeypatch):
    def boom():
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(onboarding_routes, "scan_importable_projects", boom)

    async with client:
        r = await client.get("/api/v2/onboarding/importable-projects")
    assert r.status_code == 200
    assert r.json() == {"candidates": []}
