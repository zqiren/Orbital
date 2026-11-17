# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The connector routes carry each connector's agent access, set it, and list
the agents a connector can be shared with (CONTRACT-connectors.md v1)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.api.routes import connectors as connector_routes
from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorManager

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_MANIFESTS = os.path.join(_REPO, "agent_os", "agents", "manifests")


class _Setup:
    def cached_status(self, slug):
        return {"claude-code": SimpleNamespace(installed=True),
                "aider": SimpleNamespace(installed=False)}.get(slug)


@pytest.fixture
async def client(tmp_path):
    cm = ConnectorManager(catalog=load_catalog(), credential_store=None,
                          data_dir=str(tmp_path),
                          session_opener=in_memory_opener(build_mock_server()))
    await cm.add_custom("Notes", "https://notes/mcp", "none")
    registry = AgentRegistry()
    registry.load_directory(_MANIFESTS)
    app = FastAPI()
    connector_routes.configure(cm, agent_registry=registry, setup_engine=_Setup())
    app.include_router(connector_routes.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://test") as c:
        yield c
    connector_routes.configure(None)


async def test_every_connector_carries_its_access(client):
    body = (await client.get("/api/v2/connectors")).json()
    for c in body["connectors"]:
        assert c["access"] == {"mode": "all", "agents": []}


async def test_put_access_sets_and_returns_it(client):
    r = await client.put("/api/v2/connectors/custom-notes/access",
                         json={"mode": "selected", "agents": ["claude-code", "orbital", "claude-code"]})
    assert r.status_code == 200
    assert r.json() == {"id": "custom-notes",
                        "access": {"mode": "selected", "agents": ["claude-code", "orbital"]}}
    listed = {c["id"]: c for c in (await client.get("/api/v2/connectors")).json()["connectors"]}
    assert listed["custom-notes"]["access"]["mode"] == "selected"
    r = await client.put("/api/v2/connectors/custom-notes/access", json={"mode": "all"})
    assert r.json()["access"] == {"mode": "all", "agents": []}


async def test_put_access_errors(client):
    r = await client.put("/api/v2/connectors/nope/access", json={"mode": "all"})
    assert r.status_code == 404
    r = await client.put("/api/v2/connectors/custom-notes/access",
                         json={"mode": "selected", "agents": ["foo"]})
    assert r.status_code == 400 and r.json()["detail"] == "unknown agent: foo"
    r = await client.put("/api/v2/connectors/custom-notes/access", json={"mode": "some"})
    assert r.status_code == 422
    r = await client.put("/api/v2/connectors/custom-notes/access",
                         json={"mode": "selected", "agents": "codex"})
    assert r.status_code == 422


async def test_agents_lists_the_manager_first_then_workers_with_support(client):
    agents = (await client.get("/api/v2/connectors/agents")).json()["agents"]
    assert agents[0] == {"id": "orbital", "name": "Orbital", "kind": "manager",
                         "installed": True, "supported": True, "reason": None}
    by_id = {a["id"]: a for a in agents}
    assert "built-in" not in by_id
    assert by_id["claude-code"] == {"id": "claude-code", "name": "Claude Code",
                                    "kind": "worker", "installed": True,
                                    "supported": True, "reason": None}
    assert by_id["codex"]["supported"] is True and by_id["codex"]["installed"] is None
    assert by_id["pi"]["supported"] is True
    assert by_id["minimax-code"]["supported"] is True
    assert by_id["aider"] == {"id": "aider", "name": "Aider", "kind": "worker",
                              "installed": False, "supported": False,
                              "reason": "no_mcp_mount"}
    workers = [a["name"] for a in agents[1:]]
    assert workers == sorted(workers, key=str.lower)


async def test_an_unsupported_agent_can_still_be_selected(client):
    r = await client.put("/api/v2/connectors/custom-notes/access",
                         json={"mode": "selected", "agents": ["aider"]})
    assert r.status_code == 200
