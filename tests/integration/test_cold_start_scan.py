# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration: cold-start workspace scan journey (in-process FastAPI app).

Covers the deterministic, automatable surface:
  - is_empty_workspace gating on the project payload
  - POST /cold-start-scan mints + starts the project's first session
  - the onboarding gates flip once the confirm-time files are written

The end-to-end "agent actually writes the files from a real conversation" leg
is verified by the live-daemon smoke (CLAUDE.md §3), not a scripted LLM stub —
that flow needs a real model and scripting a multi-turn provider here would be
disproportionate for a one-time path.
"""
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from agent_os.api.ws import WebSocketManager
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.browser_manager import BrowserManager
from agent_os.daemon_v2.activity_translator import ActivityTranslator
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.agent.project_paths import ProjectPaths


@pytest.fixture
def workspace(tmp_path):
    w = tmp_path / "workspace"
    w.mkdir()
    return str(w)


@pytest.fixture
def client(tmp_path, workspace):
    data_dir = tmp_path / "orbital-data"
    data_dir.mkdir()
    app = FastAPI()

    project_store = ProjectStore(data_dir=str(data_dir))
    ws_manager = WebSocketManager()
    activity_translator = ActivityTranslator(ws_manager)
    process_manager = ProcessManager(ws_manager, activity_translator)

    mock_settings_store = MagicMock()
    mock_settings_store.get.return_value = MagicMock(
        llm=MagicMock(provider="anthropic", model="claude-sonnet-4-20250514", api_key="", base_url="")
    )
    mock_credential_store = MagicMock()
    mock_credential_store.get_api_key.return_value = "sk-test-key"
    mock_platform = MagicMock()
    mock_platform.get_capabilities.return_value = MagicMock(platform="macos", setup_complete=True)
    browser_manager = BrowserManager(profile_dir=str(data_dir / "bp"), headless=True)

    mock_registry = MagicMock()
    mock_registry.list_manifests.return_value = []
    sub_agent_manager = SubAgentManager(
        process_manager=process_manager, registry=mock_registry,
        setup_engine=MagicMock(), platform_provider=mock_platform,
        project_store=project_store,
    )
    agent_manager = AgentManager(
        project_store=project_store, settings_store=mock_settings_store,
        credential_store=mock_credential_store, ws_manager=ws_manager,
        activity_translator=activity_translator, process_manager=process_manager,
        platform_provider=mock_platform, sub_agent_manager=sub_agent_manager,
        browser_manager=browser_manager, provider_registry=MagicMock(),
    )
    agents_v2.configure(
        project_store=project_store, agent_manager=agent_manager,
        ws_manager=ws_manager, sub_agent_manager=sub_agent_manager,
        setup_engine=MagicMock(), settings_store=mock_settings_store,
        credential_store=mock_credential_store, trigger_manager=MagicMock(),
        provider_registry=MagicMock(),
    )
    app.include_router(agents_v2.router)
    return TestClient(app), {"workspace": workspace, "agent_manager": agent_manager}


def _create_project(tc, workspace, name="Imported"):
    resp = tc.post("/api/v2/projects", json={
        "name": name, "workspace": workspace, "model": "claude-sonnet-4-20250514",
        "api_key": "sk-test-key",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def test_empty_workspace_has_no_scan_flag(client):
    tc, deps = client
    pid = _create_project(tc, deps["workspace"])
    proj = tc.get(f"/api/v2/projects/{pid}").json()
    assert proj["is_empty_workspace"] is True


def test_imported_workspace_flag_and_scan_starts_session(client):
    tc, deps = client
    open(os.path.join(deps["workspace"], "README.md"), "w").write("# Real project")
    pid = _create_project(tc, deps["workspace"])

    proj = tc.get(f"/api/v2/projects/{pid}").json()
    assert proj["is_empty_workspace"] is False
    assert tc.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"] == []

    r = tc.post(f"/api/v2/agents/{pid}/cold-start-scan")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "started" and body["session_id"]

    sessions = tc.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
    assert any(s.get("session_id") == body["session_id"] for s in sessions)


def test_confirmation_files_flip_onboarding_gates(client):
    """Deterministic gate check: once the confirm-time files exist, both the
    dispatcher gate (PROJECT_STATE.md) and the prompt off-switch (project_goals.md)
    flip. This is what the agent's Stage-3 writes produce."""
    tc, deps = client
    ws = deps["workspace"]
    open(os.path.join(ws, "README.md"), "w").write("# Real project")
    pid = _create_project(tc, ws)

    am = deps["agent_manager"]
    assert am.is_onboarding_complete(pid) is False  # nothing written yet

    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    open(pp.project_goals, "w").write("Mission: ship the thing")
    open(pp.project_state, "w").write("FastAPI backend + React frontend")

    # Dispatcher gate flips on PROJECT_STATE.md.
    assert am.is_onboarding_complete(pid) is True
    # Prompt off-switch flips on project_goals.md → directive, not onboarding.
    from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
    section = PromptBuilder()._onboarding_or_directive(PromptContext(
        workspace=ws, model="m", autonomy=Autonomy.HANDS_OFF, enabled_agents=[],
        tool_names=["read"], os_type="macos", datetime_now="2026-06-08T00:00:00",
        cold_start=True,
    ))
    assert "PROJECT DIRECTIVE" in section
