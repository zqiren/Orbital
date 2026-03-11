# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for Agent 3: Daemon Wiring — manifest registry + setup engine integration.

Tests cover:
1. create_app() loads registry with seed manifests
2. GET /api/v2/agents/available returns agent list
3. GET /api/v2/agents/{slug}/status returns status object
4. SubAgentManager.start() uses manifest registry path
5. SubAgentManager.start() with unknown slug returns error
6. Project creation accepts agent_slug
7. Project creation accepts enabled_sub_agents
8. Prompt context includes agent capabilities from registry
"""

import asyncio
import os
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. create_app loads registry
# ---------------------------------------------------------------------------


class TestCreateAppLoadsRegistry:
    """Verify registry is loaded with seed manifests during app creation."""

    def test_create_app_loads_registry(self, tmp_path):
        from agent_os.api.app import create_app

        app = create_app(data_dir=str(tmp_path))

        # The app should have been created without errors.
        # Verify the registry was loaded by hitting the available endpoint.
        from starlette.testclient import TestClient
        client = TestClient(app)
        resp = client.get("/api/v2/agents/available")
        assert resp.status_code == 200
        data = resp.json()
        # Should have at least built-in and claude-code
        slugs = [a["slug"] for a in data]
        assert "built-in" in slugs
        assert "claude-code" in slugs
        assert len(slugs) >= 2


# ---------------------------------------------------------------------------
# 2. GET /api/v2/agents/available
# ---------------------------------------------------------------------------


class TestAvailableAgentsEndpoint:
    """Tests for the available agents endpoint."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def test_returns_list(self, app_client):
        resp = app_client.get("/api/v2/agents/available")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2

    def test_built_in_is_installed(self, app_client):
        resp = app_client.get("/api/v2/agents/available")
        data = resp.json()
        built_in = [a for a in data if a["slug"] == "built-in"][0]
        assert built_in["installed"] is True
        assert built_in["ready"] is True
        assert built_in["name"] == "Orbital Assistant"

    def test_response_has_expected_fields(self, app_client):
        resp = app_client.get("/api/v2/agents/available")
        data = resp.json()
        for agent in data:
            assert "slug" in agent
            assert "name" in agent
            assert "installed" in agent
            assert "ready" in agent
            assert "setup_actions" in agent


# ---------------------------------------------------------------------------
# 3. GET /api/v2/agents/{slug}/status
# ---------------------------------------------------------------------------


class TestAgentStatusEndpoint:
    """Tests for the single agent status endpoint."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def test_built_in_status(self, app_client):
        resp = app_client.get("/api/v2/agents/built-in/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "built-in"
        assert data["installed"] is True
        assert data["ready"] is True

    def test_unknown_slug_returns_404(self, app_client):
        resp = app_client.get("/api/v2/agents/nonexistent-agent/status")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. SubAgentManager.start() uses manifest (registry path)
# ---------------------------------------------------------------------------


class TestSubAgentStartUsesManifest:
    """SubAgentManager with registry + setup_engine uses manifest path."""

    @pytest.mark.asyncio
    async def test_start_uses_manifest(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.registry import AgentRegistry
        from agent_os.agents.manifest import (
            AgentManifest, ManifestRuntime, ManifestSetup,
            ManifestCapabilities, ManifestPermissions,
        )

        # Build a minimal registry with a CLI agent
        registry = AgentRegistry()
        manifest = AgentManifest(
            manifest_version="1",
            name="Test Agent",
            slug="test-agent",
            description="A test agent",
            author="test",
            version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="test-cmd"),
            capabilities=ManifestCapabilities(skills=["testing"]),
            permissions=ManifestPermissions(network_domains=["example.com"]),
        )
        registry.register(manifest)

        # Mock setup engine
        setup_engine = MagicMock()
        setup_engine.get_adapter_config.return_value = {
            "command": "/usr/bin/test-cmd",
            "args": [],
            "workspace": "/tmp",
            "approval_patterns": [],
            "env": {},
            "network_domains": ["example.com"],
            "interactive": False,
        }

        pm = MagicMock()
        pm.start = AsyncMock()

        mgr = SubAgentManager(
            process_manager=pm,
            registry=registry,
            setup_engine=setup_engine,
        )

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as MockAdapter:
            mock_instance = AsyncMock()
            MockAdapter.return_value = mock_instance

            result = await mgr.start("proj_1", "test-agent")

        assert "Started" in result
        assert "Test Agent" in result
        setup_engine.get_adapter_config.assert_called_once_with(
            slug="test-agent",
            project_workspace="",
        )


# ---------------------------------------------------------------------------
# 5. SubAgentManager.start() unknown slug
# ---------------------------------------------------------------------------


class TestSubAgentStartUnknownSlug:
    """SubAgentManager.start() with unknown slug returns error."""

    @pytest.mark.asyncio
    async def test_start_unknown_slug(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.registry import AgentRegistry

        registry = AgentRegistry()
        setup_engine = MagicMock()
        pm = MagicMock()

        mgr = SubAgentManager(
            process_manager=pm,
            registry=registry,
            setup_engine=setup_engine,
        )

        result = await mgr.start("proj_1", "foobar")
        assert "Error" in result
        assert "unknown" in result.lower() or "foobar" in result

    @pytest.mark.asyncio
    async def test_start_built_in_returns_error(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        from agent_os.agents.registry import AgentRegistry
        from agent_os.agents.manifest import AgentManifest, ManifestRuntime

        registry = AgentRegistry()
        manifest = AgentManifest(
            manifest_version="1",
            name="Built-In",
            slug="built-in",
            description="Built-in agent",
            author="test",
            version="1.0.0",
            runtime=ManifestRuntime(adapter="built_in"),
        )
        registry.register(manifest)

        setup_engine = MagicMock()
        pm = MagicMock()

        mgr = SubAgentManager(
            process_manager=pm,
            registry=registry,
            setup_engine=setup_engine,
        )

        result = await mgr.start("proj_1", "built-in")
        assert "Error" in result
        assert "built-in" in result.lower() or "built_in" in result.lower()


# ---------------------------------------------------------------------------
# 6. Project creation with agent_slug
# ---------------------------------------------------------------------------


class TestProjectCreateWithAgentSlug:
    """Project creation accepts and stores agent_slug field."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def test_create_with_agent_slug(self, app_client, tmp_path):
        resp = app_client.post("/api/v2/projects", json={
            "name": "Test Project",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
            "agent_slug": "claude-code",
        })
        assert resp.status_code == 201
        data = resp.json()
        pid = data["project_id"]

        # Verify it was stored
        resp2 = app_client.get(f"/api/v2/projects/{pid}")
        assert resp2.status_code == 200
        project = resp2.json()
        assert project.get("agent_slug") == "claude-code"


# ---------------------------------------------------------------------------
# 7. Project creation with enabled_sub_agents
# ---------------------------------------------------------------------------


class TestProjectCreateWithEnabledSubAgents:
    """Project creation accepts and stores enabled_sub_agents list."""

    @pytest.fixture
    def app_client(self, tmp_path):
        from agent_os.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=str(tmp_path))
        return TestClient(app)

    def test_create_with_enabled_sub_agents(self, app_client, tmp_path):
        resp = app_client.post("/api/v2/projects", json={
            "name": "Test Project",
            "workspace": str(tmp_path),
            "model": "gpt-4",
            "api_key": "sk-test",
            "enabled_sub_agents": ["claude-code"],
        })
        assert resp.status_code == 201
        data = resp.json()
        pid = data["project_id"]

        resp2 = app_client.get(f"/api/v2/projects/{pid}")
        assert resp2.status_code == 200
        project = resp2.json()
        assert project.get("enabled_sub_agents") == ["claude-code"]


# ---------------------------------------------------------------------------
# 8. Prompt context includes agent capabilities
# ---------------------------------------------------------------------------


class TestPromptContextIncludesCapabilities:
    """Agent manager uses registry to build rich enabled_agents_detail."""

    @pytest.mark.asyncio
    async def test_prompt_context_uses_registry(self):
        from agent_os.daemon_v2.agent_manager import AgentManager
        from agent_os.daemon_v2.models import AgentConfig
        from agent_os.agents.registry import AgentRegistry
        from agent_os.agents.manifest import (
            AgentManifest, ManifestRuntime, ManifestCapabilities,
        )

        # Build registry with a test agent
        registry = AgentRegistry()
        manifest = AgentManifest(
            manifest_version="1",
            name="Claude Code",
            slug="claude-code",
            description="Coding agent",
            author="anthropic",
            version="1.0.0",
            runtime=ManifestRuntime(adapter="cli", command="claude"),
            capabilities=ManifestCapabilities(
                skills=["code_generation", "debugging"],
                routing_hint="Use for coding tasks",
            ),
        )
        registry.register(manifest)

        # Create manager with mocks
        project_store = MagicMock()
        ws = MagicMock()
        ws.broadcast = MagicMock()
        sub_agent_mgr = MagicMock()
        sub_agent_mgr.list_active = MagicMock(return_value=[])
        sub_agent_mgr.stop = AsyncMock()
        sub_agent_mgr.stop_all = AsyncMock()
        activity_translator = MagicMock()
        process_manager = MagicMock()
        process_manager.set_session = MagicMock()

        mgr = AgentManager(
            project_store=project_store,
            ws_manager=ws,
            sub_agent_manager=sub_agent_mgr,
            activity_translator=activity_translator,
            process_manager=process_manager,
            registry=registry,
        )

        config = AgentConfig(
            workspace="/tmp/ws",
            model="gpt-4",
            api_key="sk-test",
            enabled_sub_agents=["claude-code"],
        )

        # Capture the PromptContext that gets created
        captured_context = {}

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
             patch("agent_os.daemon_v2.agent_manager.ContextManager") as MockCM, \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):

            mock_session = MagicMock()
            MockSession.new.return_value = mock_session
            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read", "write"]
            MockReg.return_value = mock_reg

            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            MockLoop.return_value = mock_loop

            await mgr.start_agent("proj_1", config)

            # ContextManager is called with (session, prompt_builder, prompt_context)
            cm_call_args = MockCM.call_args
            prompt_context = cm_call_args[0][2]  # third positional arg

        # Verify enabled_agents has rich info from manifest
        assert len(prompt_context.enabled_agents) == 1
        agent_info = prompt_context.enabled_agents[0]
        assert agent_info["handle"] == "claude-code"
        assert agent_info["display_name"] == "Claude Code"
        assert agent_info["type"] == "cli"
        assert "code_generation" in agent_info["skills"]
        assert "debugging" in agent_info["skills"]
        assert agent_info["routing_hint"] == "Use for coding tasks"

    @pytest.mark.asyncio
    async def test_prompt_context_fallback_without_registry(self):
        """When no registry, falls back to simple enabled_agents format."""
        from agent_os.daemon_v2.agent_manager import AgentManager
        from agent_os.daemon_v2.models import AgentConfig

        project_store = MagicMock()
        ws = MagicMock()
        ws.broadcast = MagicMock()
        sub_agent_mgr = MagicMock()
        sub_agent_mgr.list_active = MagicMock(return_value=[])
        sub_agent_mgr.stop = AsyncMock()
        sub_agent_mgr.stop_all = AsyncMock()
        activity_translator = MagicMock()
        process_manager = MagicMock()
        process_manager.set_session = MagicMock()

        mgr = AgentManager(
            project_store=project_store,
            ws_manager=ws,
            sub_agent_manager=sub_agent_mgr,
            activity_translator=activity_translator,
            process_manager=process_manager,
            # No registry
        )

        config = AgentConfig(
            workspace="/tmp/ws",
            model="gpt-4",
            api_key="sk-test",
            enabled_agents=["claudecode"],
        )

        with patch("agent_os.daemon_v2.agent_manager.LLMProvider"), \
             patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
             patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
             patch("agent_os.daemon_v2.agent_manager.Session") as MockSession, \
             patch("agent_os.daemon_v2.agent_manager.ContextManager") as MockCM, \
             patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
             patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):

            mock_session = MagicMock()
            MockSession.new.return_value = mock_session
            mock_reg = MagicMock()
            mock_reg.tool_names.return_value = ["read", "write"]
            MockReg.return_value = mock_reg

            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            MockLoop.return_value = mock_loop

            await mgr.start_agent("proj_1", config)

            cm_call_args = MockCM.call_args
            prompt_context = cm_call_args[0][2]

        # Fallback: simple format
        assert len(prompt_context.enabled_agents) == 1
        agent_info = prompt_context.enabled_agents[0]
        assert agent_info["handle"] == "claudecode"
        assert agent_info["display_name"] == "claudecode"
        assert agent_info["type"] == "cli"


# ---------------------------------------------------------------------------
# AgentConfig new fields
# ---------------------------------------------------------------------------


class TestAgentConfigNewFields:
    """Verify new fields on AgentConfig dataclass."""

    def test_default_values(self):
        from agent_os.daemon_v2.models import AgentConfig

        config = AgentConfig(workspace="/tmp", model="gpt-4", api_key="sk-test")
        assert config.agent_slug == "built-in"
        assert config.enabled_sub_agents == []
        assert config.agent_credentials == {}
        assert config.network_extra_domains == []

    def test_custom_values(self):
        from agent_os.daemon_v2.models import AgentConfig

        config = AgentConfig(
            workspace="/tmp",
            model="gpt-4",
            api_key="sk-test",
            agent_slug="claude-code",
            enabled_sub_agents=["claude-code"],
            agent_credentials={"ANTHROPIC_API_KEY": "sk-ant-123"},
            network_extra_domains=["custom.api.com"],
        )
        assert config.agent_slug == "claude-code"
        assert config.enabled_sub_agents == ["claude-code"]
        assert config.agent_credentials == {"ANTHROPIC_API_KEY": "sk-ant-123"}
        assert config.network_extra_domains == ["custom.api.com"]


# ---------------------------------------------------------------------------
# 9. PromptBuilder._sub_agents() produces directive output
# ---------------------------------------------------------------------------


class TestPromptBuilderSubAgentsSection:
    """PromptBuilder._sub_agents() should produce explicit agent_message instructions."""

    def test_sub_agents_section_content(self, tmp_path):
        from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy

        builder = PromptBuilder(workspace=str(tmp_path))
        context = PromptContext(
            workspace=str(tmp_path),
            model="gpt-4",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[
                {
                    "handle": "claude-code",
                    "display_name": "Claude Code",
                    "type": "cli",
                    "skills": ["code generation", "debugging"],
                    "routing_hint": "Use for coding tasks",
                },
            ],
            tool_names=["read", "write", "shell", "agent_message"],
            os_type="linux",
            datetime_now="2026-01-01T00:00:00",
        )

        _cached, dynamic = builder.build(context)

        # Section header
        assert "Sub-Agents Available" in dynamic
        # Agent handle appears
        assert "claude-code" in dynamic
        # Tool name referenced
        assert "agent_message" in dynamic
        # Skills listed
        assert "code generation" in dynamic
        assert "debugging" in dynamic
        # Routing hint
        assert "Use for coding tasks" in dynamic
        # Prohibition against shell
        assert "Do NOT" in dynamic

    def test_sub_agents_section_absent_when_no_agents(self, tmp_path):
        from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy

        builder = PromptBuilder(workspace=str(tmp_path))
        context = PromptContext(
            workspace=str(tmp_path),
            model="gpt-4",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[],
            tool_names=["read", "write"],
            os_type="linux",
            datetime_now="2026-01-01T00:00:00",
        )

        _cached, dynamic = builder.build(context)
        assert "Sub-Agents Available" not in dynamic

    def test_sub_agents_section_without_optional_fields(self, tmp_path):
        """Agent without skills or routing_hint should still render correctly."""
        from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy

        builder = PromptBuilder(workspace=str(tmp_path))
        context = PromptContext(
            workspace=str(tmp_path),
            model="gpt-4",
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[
                {
                    "handle": "my-agent",
                    "display_name": "My Agent",
                    "type": "cli",
                },
            ],
            tool_names=["read", "write", "agent_message"],
            os_type="linux",
            datetime_now="2026-01-01T00:00:00",
        )

        _cached, dynamic = builder.build(context)

        assert "Sub-Agents Available" in dynamic
        assert "my-agent" in dynamic
        assert "agent_message" in dynamic
        assert "Do NOT" in dynamic
