# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: /agents/available must not block the async event loop."""

from types import SimpleNamespace
import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.ws import WebSocketManager
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.activity_translator import ActivityTranslator
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.browser_manager import BrowserManager


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "orbital-data"
    d.mkdir()
    return str(d)


@pytest.fixture
def workspace(tmp_path):
    w = tmp_path / "workspace"
    w.mkdir()
    return str(w)


@pytest.fixture
def api_app(data_dir, workspace):
    """Build a FastAPI app with real stores and mock external deps."""
    app = FastAPI()

    project_store = ProjectStore(data_dir=data_dir)
    ws_manager = WebSocketManager()
    activity_translator = ActivityTranslator(ws_manager)
    process_manager = ProcessManager(ws_manager, activity_translator)

    mock_settings_store = MagicMock()
    mock_settings_store.get.return_value = MagicMock(
        llm=MagicMock(provider="anthropic", model="claude-sonnet-4-20250514",
                       api_key="", base_url="")
    )
    mock_credential_store = MagicMock()
    mock_credential_store.get_api_key.return_value = "sk-test-key"

    mock_platform = MagicMock()
    mock_platform.get_capabilities.return_value = MagicMock(
        platform="windows", setup_complete=True
    )

    mock_registry = MagicMock()
    mock_registry.list_manifests.return_value = []
    mock_setup_engine = MagicMock()
    mock_setup_engine.check_all.return_value = []
    mock_provider_registry = MagicMock()
    mock_trigger_manager = MagicMock()

    browser_manager = BrowserManager(
        profile_dir=os.path.join(data_dir, "browser-profile"),
        headless=True,
    )

    sub_agent_manager = SubAgentManager(
        process_manager=process_manager,
        registry=mock_registry,
        setup_engine=mock_setup_engine,
        platform_provider=mock_platform,
        project_store=project_store,
    )

    agent_manager = AgentManager(
        project_store=project_store,
        settings_store=mock_settings_store,
        credential_store=mock_credential_store,
        ws_manager=ws_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
        platform_provider=mock_platform,
        sub_agent_manager=sub_agent_manager,
        browser_manager=browser_manager,
        provider_registry=mock_provider_registry,
    )

    agents_v2.configure(
        project_store=project_store,
        agent_manager=agent_manager,
        ws_manager=ws_manager,
        sub_agent_manager=sub_agent_manager,
        setup_engine=mock_setup_engine,
        settings_store=mock_settings_store,
        credential_store=mock_credential_store,
        trigger_manager=mock_trigger_manager,
        provider_registry=mock_provider_registry,
    )

    app.include_router(agents_v2.router)

    return app, {
        "project_store": project_store,
        "agent_manager": agent_manager,
        "ws_manager": ws_manager,
        "sub_agent_manager": sub_agent_manager,
        "mock_setup_engine": mock_setup_engine,
        "workspace": workspace,
    }


# ======================================================================
# Test 1: /agents/available does not block concurrent requests
# ======================================================================

@pytest.mark.asyncio
async def test_available_does_not_block_chat(api_app):
    """A slow check_all() in /agents/available must not block other endpoints."""
    app, deps = api_app

    # Create a project so the chat endpoint has something to query
    project_store = deps["project_store"]
    workspace = deps["workspace"]
    pid = project_store.create_project({"name": "test", "workspace": workspace})

    # Make check_all sleep 2 seconds to simulate a slow subprocess
    def slow_check_all(**_kw):
        import time as _time
        _time.sleep(2)
        return []
    deps["mock_setup_engine"].check_all = slow_check_all

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        t0 = time.time()
        # Fire both requests concurrently
        available_task = asyncio.create_task(
            client.get("/api/v2/agents/available")
        )
        chat_task = asyncio.create_task(
            client.get(f"/api/v2/agents/{pid}/chat?limit=50")
        )
        chat_resp = await chat_task
        chat_time = time.time() - t0
        available_resp = await available_task

    assert chat_resp.status_code == 200
    assert chat_time < 1.0, (
        f"Chat endpoint took {chat_time:.2f}s — it should not be blocked by "
        f"the 2s sleep in check_all(). The to_thread() offload is broken."
    )
    assert available_resp.status_code == 200


# ======================================================================
# Test 2: Cache prevents repeated subprocess calls
# ======================================================================

def test_available_shows_what_the_engine_knows_at_once(api_app):
    """The caching lives in SetupEngine.check_all, whose cache a login, a
    locate or an install drops. A second one in the route kept the pin menu
    stale for up to a minute after those."""
    app, deps = api_app

    statuses: list = []
    deps["mock_setup_engine"].check_all = lambda **_kw: list(statuses)

    client = TestClient(app)
    assert client.get("/api/v2/agents/available").json() == []

    statuses.append(SimpleNamespace(
        slug="codex", name="Codex", installed=True, binary_path="/x/codex",
        version="1", dependencies_met=True, missing_dependencies=[],
        credentials_configured=True, missing_credentials=[], setup_actions=[],
    ))
    assert [e["slug"] for e in client.get("/api/v2/agents/available").json()] == ["codex"]


def test_reading_available_never_starts_a_scan():
    """Opening the pin menu reads the last sweep, however old. Expired, it is
    still served: no probe runs and no background sweep starts."""
    from agent_os.agents.registry import AgentRegistry
    from agent_os.agents.setup_engine import SetupEngine

    engine = SetupEngine(AgentRegistry())
    last_sweep = [SimpleNamespace(slug="codex")]
    engine._check_all_cache = (last_sweep, 0.0)  # expired long ago
    engine._probe_all = lambda **_kw: pytest.fail("the menu read started a scan")
    engine._refresh_in_background = lambda: pytest.fail(
        "the menu read started a background scan")

    assert engine.check_all(allow_stale=True, refresh_if_stale=False) is last_sweep
