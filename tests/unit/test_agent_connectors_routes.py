# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""``GET /api/v2/connectors/agent-held``: each agent's own connectors, as it
last reported them (CONTRACT-connectors.md v1.3). Names and status only; an
agent never seen running is not listed."""

from __future__ import annotations

import os

import httpx
import pytest
from fastapi import FastAPI

from agent_os.agents.registry import AgentRegistry
from agent_os.api.routes import connectors as connector_routes
from agent_os.daemon_v2 import agent_connectors as ac

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_MANIFESTS = os.path.join(_REPO, "agent_os", "agents", "manifests")


@pytest.fixture
async def client(tmp_path):
    ac.configure(str(tmp_path))
    registry = AgentRegistry()
    registry.load_directory(_MANIFESTS)
    app = FastAPI()
    connector_routes.configure(None, agent_registry=registry)
    app.include_router(connector_routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as c:
        yield c
    connector_routes.configure(None)
    ac.configure(None)


async def test_nothing_seen_lists_no_agents(client):
    r = await client.get("/api/v2/connectors/agent-held")
    assert r.status_code == 200
    assert r.json() == {"agents": []}


async def test_lists_each_agents_own_connectors_by_agent_name(client):
    ac.record("codex", [{"name": "node_repl", "status": "connected"},
                        {"name": "codex_apps", "status": "connected"},
                        {"name": "orbital", "status": "connected"}])
    ac.record("claude-code", [{"name": "claude.ai Gmail", "status": "pending"},
                              {"name": "figma", "status": "needs-auth"}])
    ac.record("aider", [])                       # seen, holds nothing
    body = (await client.get("/api/v2/connectors/agent-held")).json()
    assert [a["id"] for a in body["agents"]] == ["claude-code", "codex"]
    claude = body["agents"][0]
    assert claude["name"] == "Claude Code"
    assert claude["connectors"] == [
        {"name": "figma", "raw_name": "figma", "status": "needs-auth"},
        {"name": "Gmail", "raw_name": "claude.ai Gmail", "status": "pending"}]
    assert claude["last_seen"].endswith("Z")
    assert body["agents"][1]["connectors"] == [
        {"name": "node_repl", "raw_name": "node_repl", "status": "connected"}]


async def test_an_agent_unknown_to_the_registry_shows_its_id(client):
    ac.record("someday-agent", [{"name": "x", "status": "connected"}])
    body = (await client.get("/api/v2/connectors/agent-held")).json()
    assert body["agents"][0]["id"] == "someday-agent"
    assert body["agents"][0]["name"] == "someday-agent"


async def test_works_without_the_connector_manager(client):
    # Agent-held connectors are not Orbital's: no 503 when its manager is absent.
    ac.record("codex", [{"name": "node_repl", "status": "connected"}])
    assert (await client.get("/api/v2/connectors/agent-held")).status_code == 200


def test_create_app_points_the_registry_at_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    from agent_os.api.app import create_app
    create_app(data_dir=str(tmp_path / "data"))
    try:
        ac.record("codex", [{"name": "node_repl", "status": "connected"}])
        assert os.path.isfile(tmp_path / "data" / ac.FILE_NAME)
    finally:
        ac.configure(None)
