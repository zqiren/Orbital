# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests: /agents/available must not block the async event loop."""

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


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the available-agents cache before and after each test."""
    agents_v2._available_cache["result"] = None
    agents_v2._available_cache["expires_at"] = 0.0
    yield
    agents_v2._available_cache["result"] = None
    agents_v2._available_cache["expires_at"] = 0.0


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
    def slow_check_all():
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

def test_cache_prevents_repeated_calls(api_app):
    """After the first call, subsequent calls within the TTL must use cached results."""
    app, deps = api_app

    call_count = 0

    def counting_check_all():
        nonlocal call_count
        call_count += 1
        return []
    deps["mock_setup_engine"].check_all = counting_check_all

    client = TestClient(app)

    # Three sequential calls
    r1 = client.get("/api/v2/agents/available")
    r2 = client.get("/api/v2/agents/available")
    r3 = client.get("/api/v2/agents/available")

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 200
    assert call_count == 1, (
        f"check_all() was called {call_count} times — expected 1. "
        f"The cache is not preventing repeated subprocess invocations."
    )


def test_cache_expires_after_ttl(api_app):
    """After manually expiring the cache, the next call must invoke check_all() again."""
    app, deps = api_app

    call_count = 0

    def counting_check_all():
        nonlocal call_count
        call_count += 1
        return []
    deps["mock_setup_engine"].check_all = counting_check_all

    client = TestClient(app)

    # First call — populates cache
    resp = client.get("/api/v2/agents/available")
    assert resp.status_code == 200
    assert call_count == 1

    # Second call — should hit cache
    resp = client.get("/api/v2/agents/available")
    assert resp.status_code == 200
    assert call_count == 1

    # Expire the cache
    agents_v2._available_cache["expires_at"] = 0.0

    # Third call — cache expired, must call check_all() again
    resp = client.get("/api/v2/agents/available")
    assert resp.status_code == 200
    assert call_count == 2, (
        f"check_all() was called {call_count} times — expected 2 after cache expiry."
    )
