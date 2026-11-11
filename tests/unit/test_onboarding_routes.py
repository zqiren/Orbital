# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the /api/v2/onboarding/importable-projects route (backlog #34).

Mounts only the onboarding router over an in-process ASGI transport and stubs
the scanner so the route contract (shape + best-effort error handling) is tested
without touching real home directories. Spec 138 extended the payload with
``status`` / ``groups`` and each candidate with ``kind`` / ``minor`` /
``is_git`` / ``by_agent``; the history scanner is stubbed too."""

import httpx
import pytest
from fastapi import FastAPI

from agent_os.api.routes import onboarding as onboarding_routes
from agent_os.onboarding.import_scanner import ImportCandidate

CANDIDATE_KEYS = {"source", "name", "path", "session_count", "last_activity",
                  "already_imported", "kind", "minor", "is_git", "by_agent", "notes"}


class _StubScanner:
    """A history scan that is still running: no groups yet."""

    def start(self):
        pass

    def refresh(self):
        return False  # still scanning: nothing to re-read yet

    def status(self):
        return {"status": "scanning", "sessions_total": 0, "by_agent": {}}

    def analysis(self):
        return None


@pytest.fixture
def client():
    app = FastAPI()
    onboarding_routes.configure(scanner=_StubScanner())
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
    monkeypatch.setattr(onboarding_routes, "scan_importable_projects",
                        lambda **_kw: fake)

    async with client:
        r = await client.get("/api/v2/onboarding/importable-projects")
    assert r.status_code == 200
    body = r.json()
    assert [c["name"] for c in body["candidates"]] == ["alpha", "notes"]
    first = body["candidates"][0]
    assert set(first) == CANDIDATE_KEYS
    assert body["status"] == "scanning" and body["groups"] == []
    assert first["source"] == "claude-code"
    assert first["session_count"] == 3


async def test_scanner_failure_yields_empty_list_not_500(client, monkeypatch):
    def boom(**_kw):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(onboarding_routes, "scan_importable_projects", boom)

    async with client:
        r = await client.get("/api/v2/onboarding/importable-projects")
    assert r.status_code == 200
    assert r.json() == {"status": "scanning", "candidates": [], "groups": []}


async def test_route_passes_store_workspaces_and_payload_carries_flag(monkeypatch):
    """Spec 084 §3.2: the route hands the project store's workspaces to the
    scanner and the candidate payload carries ``already_imported``."""
    class FakeStore:
        def list_projects(self):
            return [{"workspace": "/p/alpha"}, {"workspace": "/p/zeta"}]

    seen = {}

    def fake_scan(*, existing_workspaces=()):
        seen["existing"] = set(existing_workspaces)
        return [ImportCandidate(source="codex", name="alpha", path="/p/alpha",
                                already_imported=True)]

    app = FastAPI()
    onboarding_routes.configure(project_store=FakeStore(), scanner=_StubScanner())
    app.include_router(onboarding_routes.router)
    monkeypatch.setattr(onboarding_routes, "scan_importable_projects", fake_scan)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.get("/api/v2/onboarding/importable-projects")
    assert r.status_code == 200
    assert seen["existing"] == {"/p/alpha", "/p/zeta"}
    cand = r.json()["candidates"][0]
    assert cand["already_imported"] is True
    assert set(cand) == CANDIDATE_KEYS


async def test_route_without_a_store_scans_without_exclusion(client, monkeypatch):
    seen = {}

    def fake_scan(*, existing_workspaces=()):
        seen["existing"] = list(existing_workspaces)
        return []

    monkeypatch.setattr(onboarding_routes, "scan_importable_projects", fake_scan)
    async with client:
        r = await client.get("/api/v2/onboarding/importable-projects")
    assert r.status_code == 200
    assert seen["existing"] == []
