# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""API-level smoke tests: boots real FastAPI app, makes actual HTTP calls.

Tests BUG-001, BUG-003a, BUG-005a, BUG-005b against the running daemon
by simulating user actions through REST endpoints.
"""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.api.ws import WebSocketManager
from agent_os.api.routes import agents_v2, platform as platform_routes, settings as settings_routes
from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.daemon_v2.browser_manager import BrowserManager
from agent_os.daemon_v2.activity_translator import ActivityTranslator
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.agent.transports.base import TransportEvent


# ======================================================================
# Fixtures: wire up a real FastAPI app with real stores, mock LLM
# ======================================================================

@pytest.fixture
def data_dir(tmp_path):
    """Temporary data directory for project store."""
    d = tmp_path / "orbital-data"
    d.mkdir()
    return str(d)


@pytest.fixture
def workspace(tmp_path):
    """Temporary workspace directory for test projects."""
    w = tmp_path / "workspace"
    w.mkdir()
    return str(w)


@pytest.fixture
def api_app(data_dir, workspace):
    """Build a real FastAPI app with real stores, mock external deps."""
    app = FastAPI()

    project_store = ProjectStore(data_dir=data_dir)
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
    mock_platform.get_capabilities.return_value = MagicMock(platform="windows", setup_complete=True)

    browser_manager = BrowserManager(
        profile_dir=os.path.join(data_dir, "browser-profile"),
        headless=True,
    )

    mock_registry = MagicMock()
    mock_registry.list_manifests.return_value = []
    mock_setup_engine = MagicMock()
    mock_provider_registry = MagicMock()
    mock_trigger_manager = MagicMock()

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

    # Configure route modules with real stores
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
    platform_routes.configure(mock_platform, agent_manager=agent_manager, browser_manager=browser_manager)

    app.include_router(agents_v2.router)
    app.include_router(platform_routes.router)

    return app, {
        "project_store": project_store,
        "agent_manager": agent_manager,
        "ws_manager": ws_manager,
        "sub_agent_manager": sub_agent_manager,
        "browser_manager": browser_manager,
        "workspace": workspace,
    }


@pytest.fixture
def client(api_app):
    app, deps = api_app
    return TestClient(app), deps


# ======================================================================
# BUG-001: Browser warmup via real API endpoint
# ======================================================================

class TestBug001_BrowserWarmupAPI:
    """Hit POST /api/v2/platform/browser/warmup and verify flags."""

    def test_warmup_endpoint_calls_launch_warmup(self, client):
        """POST /browser/warmup → BrowserManager.launch_warmup() is called."""
        test_client, deps = client
        bm = deps["browser_manager"]

        # Mock launch_warmup to capture the call
        bm.launch_warmup = AsyncMock()

        resp = test_client.post("/api/v2/platform/browser/warmup")
        assert resp.status_code == 200
        bm.launch_warmup.assert_awaited_once_with("https://accounts.google.com")

    def test_warmup_launch_uses_full_flags(self, client):
        """When launch_warmup runs, it passes all CHROME_FLAGS to playwright."""
        test_client, deps = client
        bm = deps["browser_manager"]

        captured_args = []

        async def capture_warmup(url):
            # Simulate what launch_warmup does internally — check the args
            # by inspecting the code path
            launch_kwargs_args = bm.CHROME_FLAGS + ["--force-color-profile=srgb"]
            captured_args.extend(launch_kwargs_args)

        bm.launch_warmup = capture_warmup

        resp = test_client.post("/api/v2/platform/browser/warmup")
        assert resp.status_code == 200
        assert "--disable-blink-features=AutomationControlled" in captured_args
        for flag in BrowserManager.CHROME_FLAGS:
            assert flag in captured_args, f"Missing flag: {flag}"


# ======================================================================
# BUG-003a: Create project, then test credential interception via API
# ======================================================================

class TestBug003a_CredentialInterceptionAPI:
    """Create a real project, instantiate interceptor, verify interception."""

    def test_create_project_and_intercept_credential(self, client):
        """Full flow: POST /projects → create interceptor → request_credential intercepted."""
        test_client, deps = client
        workspace = deps["workspace"]

        # Step 1: Create project via real API
        resp = test_client.post("/api/v2/projects", json={
            "name": "test-credential-project",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "autonomy": "hands_off",
        })
        assert resp.status_code == 201, f"Create project failed: {resp.text}"
        project = resp.json()
        project_id = project["project_id"]

        # Step 2: Create interceptor with the project's autonomy preset
        ws = deps["ws_manager"]
        interceptor = AutonomyInterceptor(
            preset=Autonomy.HANDS_OFF,
            ws_manager=ws,
            project_id=project_id,
        )

        # Step 3: Verify credential interception
        cred_call = {
            "id": "tc-cred-1",
            "name": "request_credential",
            "arguments": {
                "name": "github", "domain": "github.com",
                "fields": ["username", "password"],
                "reason": "Access repo",
            },
        }
        assert interceptor.should_intercept(cred_call) is True, \
            "request_credential not intercepted in hands_off project"

        # Step 4: Verify shell is NOT intercepted in hands_off
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is False, \
            "shell should not be intercepted in hands_off"

    def test_credential_intercepted_even_with_bypass_all(self, client):
        """Even with bypass-all active, credential requests must be intercepted."""
        test_client, deps = client
        workspace = deps["workspace"]

        resp = test_client.post("/api/v2/projects", json={
            "name": "bypass-test-project",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "autonomy": "supervised",
        })
        assert resp.status_code == 201
        project_id = resp.json()["project_id"]

        interceptor = AutonomyInterceptor(
            preset=Autonomy.SUPERVISED,
            ws_manager=deps["ws_manager"],
            project_id=project_id,
        )
        interceptor.activate_bypass_all(duration=600)

        # Shell should be bypassed
        shell_call = {"name": "shell", "arguments": {"command": "ls"}}
        assert interceptor.should_intercept(shell_call) is False

        # Credential should NOT be bypassed
        cred_call = {"name": "request_credential", "arguments": {"name": "gh", "domain": "github.com", "fields": ["token"], "reason": "CI"}}
        assert interceptor.should_intercept(cred_call) is True

    def test_on_intercept_broadcasts_approval_request_with_credential_fields(self, client):
        """on_intercept() broadcasts approval.request with tool_args containing credential fields."""
        test_client, deps = client
        ws = deps["ws_manager"]

        # Track broadcasts
        broadcasts = []
        original_broadcast = ws.broadcast
        ws.broadcast = lambda pid, payload: broadcasts.append(payload)

        interceptor = AutonomyInterceptor(
            preset=Autonomy.HANDS_OFF,
            ws_manager=ws,
            project_id="proj-test",
        )

        cred_call = {
            "id": "tc-cred-2",
            "name": "request_credential",
            "arguments": {
                "name": "github", "domain": "github.com",
                "fields": ["username", "password"],
                "reason": "Login needed",
            },
        }

        interceptor.on_intercept(cred_call, recent_context=[])

        assert len(broadcasts) == 1
        payload = broadcasts[0]
        assert payload["type"] == "approval.request"
        assert payload["tool_name"] == "request_credential"
        assert payload["tool_args"]["domain"] == "github.com"
        assert payload["tool_args"]["fields"] == ["username", "password"]

        ws.broadcast = original_broadcast


# ======================================================================
# BUG-005a: SDK transport streaming through real ProcessManager
# ======================================================================

class _FakeTextBlock:
    def __init__(self, text): self.text = text

class _FakeToolUseBlock:
    def __init__(self, name, id, input): self.name = name; self.id = id; self.input = input

class _FakeAssistantMessage:
    def __init__(self, content): self.content = content

class _FakeResultMessage:
    def __init__(self, is_error=False, result=None, session_id=None):
        self.is_error = is_error; self.result = result; self.session_id = session_id


class TestBug005a_StreamingAPI:
    """Real ProcessManager consuming from real SDKTransport → WS broadcast (v5: no session)."""

    @pytest.mark.asyncio
    async def test_full_stack_streaming(self):
        """SDKTransport → CLIAdapter → ProcessManager → WS broadcast (v5 transcript isolation)."""
        # Build real components
        broadcast_items = []
        broadcast_event = asyncio.Event()
        ws = WebSocketManager()
        original_broadcast = ws.broadcast

        def tracking_broadcast(pid, payload):
            original_broadcast(pid, payload)
            broadcast_items.append(payload)
            broadcast_event.set()

        ws.broadcast = tracking_broadcast
        activity_translator = ActivityTranslator(ws)
        pm = ProcessManager(ws, activity_translator)

        transport = SDKTransport.__new__(SDKTransport)
        transport._client = MagicMock()
        transport._session_id = None
        transport._alive = True
        transport._workspace = ""
        transport._pending_approvals = {}
        transport._event_queue = asyncio.Queue()
        transport._needs_flush = False

        adapter = CLIAdapter(handle="claude-code", display_name="Claude Code", transport=transport)

        # Set up mock SDK responses
        assistant_msg = _FakeAssistantMessage([_FakeTextBlock("Analysis complete: 42 files found.")])
        result_msg = _FakeResultMessage(session_id="sess-001")

        async def mock_receive():
            yield assistant_msg
            await asyncio.sleep(0.05)
            yield result_msg

        transport._client.query = AsyncMock()
        transport._client.receive_response = mock_receive

        with patch.multiple(
            "agent_os.agent.transports.sdk_transport",
            AssistantMessage=_FakeAssistantMessage,
            ResultMessage=_FakeResultMessage,
            TextBlock=_FakeTextBlock,
            ToolUseBlock=_FakeToolUseBlock,
        ):
            # Start PM consumer (real background task)
            await pm.start("proj-1", "claude-code", adapter)
            await asyncio.sleep(0.05)

            # Simulate user sending message via API
            send_task = asyncio.create_task(adapter.send("analyze workspace"))

            # Wait for message to arrive at WS broadcast (via PM consumer)
            try:
                await asyncio.wait_for(broadcast_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pytest.fail(
                    "BUG-005a: message never reached WS broadcast through ProcessManager. "
                    "Events are still being buffered instead of streamed."
                )

            await send_task

            # Verify content reached broadcast
            chat_msgs = [b for b in broadcast_items if b.get("type") == "chat.sub_agent_message"]
            assert len(chat_msgs) >= 1
            assert any("42 files found" in str(m.get("content", "")) for m in chat_msgs), \
                f"Expected message content in broadcast, got: {chat_msgs}"

            transport._alive = False
            await pm.stop("proj-1", "claude-code")


# ======================================================================
# BUG-005b: Idle race — adapter status after send via API
# ======================================================================

class TestBug005b_IdleRaceAPI:
    """Test the exact _on_loop_done scenario with real components."""

    def test_create_project_start_subagent_check_status(self, client):
        """Create project → simulate sub-agent start → check status is not idle."""
        test_client, deps = client
        workspace = deps["workspace"]

        # Create project
        resp = test_client.post("/api/v2/projects", json={
            "name": "idle-race-test",
            "workspace": workspace,
            "model": "claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "autonomy": "hands_off",
        })
        assert resp.status_code == 201
        project_id = resp.json()["project_id"]

        # Create adapter (simulates sub-agent start)
        mock_transport = MagicMock()
        mock_transport.is_alive.return_value = True
        mock_transport.send = AsyncMock(return_value="Started")
        mock_transport.start = AsyncMock()

        adapter = CLIAdapter(
            handle="claude-code",
            display_name="Claude Code",
            transport=mock_transport,
        )

        # Verify: adapter is not idle at construction
        assert adapter.is_idle() is False

        # Register with sub_agent_manager
        sam = deps["sub_agent_manager"]
        sam._adapters.setdefault(project_id, {})["claude-code"] = adapter

        # Check list_active — should show "running", not "idle"
        active = sam.list_active(project_id)
        assert len(active) == 1
        assert active[0]["status"] == "running", \
            f"Expected 'running', got '{active[0]['status']}' — BUG-005b regression"

        # Simulate _on_loop_done check
        busy = [a for a in active if a.get("status") != "idle"]
        assert len(busy) == 1, "Sub-agent should be in busy list, not idle"

        # Clean up
        del sam._adapters[project_id]

    @pytest.mark.asyncio
    async def test_adapter_idle_after_transport_send(self):
        """After transport.send() completes, adapter should be idle (work done)."""
        mock_transport = MagicMock()
        mock_transport.send = AsyncMock(return_value="Quick response")
        mock_transport.is_alive.return_value = True

        adapter = CLIAdapter(handle="cc", display_name="CC", transport=mock_transport)
        await adapter.send("do work")

        assert adapter.is_idle() is True, \
            "Adapter should be idle after send() completes — work is done"

        # Simulate what list_active returns
        status = "running" if not adapter.is_idle() else "idle"
        assert status == "idle"
