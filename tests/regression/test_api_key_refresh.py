# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: API key refresh on hot-resume and project creation.

Root cause: the global API key was snapshotted into project JSON at creation
time, so rotating the key in global settings had no effect until projects were
manually re-saved. Also, hot-resume (Case 2 inject) reused the stale
LLMProvider without re-resolving the key from the credential store.

Fix: _start_loop() now re-resolves the API key from the credential store
before each hot-resume. create_project stores empty string when the provided
key matches the global key, so projects inherit the live global key at runtime.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle
from agent_os.daemon_v2.models import AgentConfig
from agent_os.daemon_v2.credential_store import ApiKeyStore
from agent_os.config.provider_registry import ProviderRegistry


# ── helpers ─────────────────────────────────────────────────────────────


class FakeWsManager:
    """Minimal WS hub that accepts broadcasts silently."""
    def broadcast(self, project_id, payload):
        pass
    def add_broadcast_hook(self, hook):
        pass


class FakeProjectStore:
    """In-memory project store stub."""
    def __init__(self, projects=None):
        self._projects = projects or {}

    def get_project(self, pid):
        return self._projects.get(pid)

    def create_project(self, config):
        pid = "proj_fake"
        config["id"] = pid
        self._projects[pid] = config
        return pid

    def list_projects(self):
        return list(self._projects.values())

    def find_scratch_project(self):
        return None

    def update_runtime(self, pid, data):
        proj = self._projects.get(pid)
        if proj:
            proj.setdefault("runtime", {}).update(data)


class FakeCredentialStore:
    """Stub credential store that returns a configurable key."""
    def __init__(self, key=None):
        self._key = key

    def get_api_key(self):
        return self._key


class FakeSettingsStore:
    """Stub settings store with an LLM sub-object."""
    def __init__(self, api_key=None, model="gpt-4o", base_url=None):
        self._settings = MagicMock()
        self._settings.llm.api_key = api_key
        self._settings.llm.model = model
        self._settings.llm.base_url = base_url

    def get(self):
        return self._settings


class FakeActivityTranslator:
    def on_message(self, msg, pid):
        pass
    def on_stream_chunk(self, chunk, pid, source):
        pass


class FakeProcessManager:
    def set_session(self, pid, session):
        pass


def _make_config(workspace, api_key="test-key"):
    """Build a minimal AgentConfig for testing."""
    return AgentConfig(
        workspace=workspace,
        model="gpt-4o",
        api_key=api_key,
        project_name="test-project",
    )


def _make_manager(project_store=None, credential_store=None, settings_store=None):
    """Build an AgentManager with all required stubs."""
    return AgentManager(
        project_store=project_store or FakeProjectStore(),
        ws_manager=FakeWsManager(),
        sub_agent_manager=MagicMock(),
        activity_translator=FakeActivityTranslator(),
        process_manager=FakeProcessManager(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=None,
        settings_store=settings_store,
        credential_store=credential_store,
        browser_manager=None,
        user_credential_store=None,
        provider_registry=ProviderRegistry(),
    )


# ── Test 1: hot resume picks up updated key ────────────────────────────


@pytest.mark.asyncio
async def test_hot_resume_uses_updated_key(tmp_path):
    """After the global key changes, _start_loop() should refresh the provider."""
    project_id = "proj_001"
    cred_store = FakeCredentialStore(key="new-key")
    project_store = FakeProjectStore({
        project_id: {
            "id": project_id,
            "workspace": str(tmp_path),
            "api_key": "",  # empty -> inherits global key
            "name": "test",
        },
    })
    mgr = _make_manager(
        project_store=project_store,
        credential_store=cred_store,
        settings_store=FakeSettingsStore(),
    )

    # Start agent with old key
    config = _make_config(str(tmp_path), api_key="old-key")
    await mgr.start_agent(project_id, config)

    handle = mgr._handles[project_id]
    assert handle.provider.api_key == "old-key"

    # Simulate loop finishing (cancel the real task)
    if handle.task and not handle.task.done():
        handle.task.cancel()
        try:
            await handle.task
        except (asyncio.CancelledError, Exception):
            pass

    # Mock loop.run to be a no-op coroutine so _start_loop doesn't actually run the LLM
    handle.loop.run = AsyncMock(return_value=None)

    await mgr._start_loop(project_id)

    assert handle.provider.api_key == "new-key"


# ── Test 2: no-op when key unchanged ───────────────────────────────────


@pytest.mark.asyncio
async def test_hot_resume_no_op_when_key_unchanged(tmp_path):
    """When the key hasn't changed, the internal client object should be preserved."""
    project_id = "proj_002"
    cred_store = FakeCredentialStore(key="same-key")
    project_store = FakeProjectStore({
        project_id: {
            "id": project_id,
            "workspace": str(tmp_path),
            "api_key": "",
            "name": "test",
        },
    })
    mgr = _make_manager(
        project_store=project_store,
        credential_store=cred_store,
        settings_store=FakeSettingsStore(),
    )

    config = _make_config(str(tmp_path), api_key="same-key")
    await mgr.start_agent(project_id, config)

    handle = mgr._handles[project_id]
    original_client = handle.provider._openai_client

    # Cancel the running task
    if handle.task and not handle.task.done():
        handle.task.cancel()
        try:
            await handle.task
        except (asyncio.CancelledError, Exception):
            pass

    handle.loop.run = AsyncMock(return_value=None)
    await mgr._start_loop(project_id)

    # Key unchanged -> client object identity preserved (no reconstruction)
    assert handle.provider._openai_client is original_client
    assert handle.provider.api_key == "same-key"


# ── Test 3: update_api_key reconstructs client ─────────────────────────


def test_update_api_key_reconstructs_client():
    """LLMProvider.update_api_key() must reconstruct the underlying client."""
    provider = LLMProvider(model="gpt-4o", api_key="old", sdk="openai")
    old_client = provider._openai_client

    provider.update_api_key("new")

    assert provider.api_key == "new"
    assert provider._openai_client is not old_client


def test_update_api_key_no_op_when_same():
    """LLMProvider.update_api_key() with same key should preserve client."""
    provider = LLMProvider(model="gpt-4o", api_key="same", sdk="openai")
    old_client = provider._openai_client

    provider.update_api_key("same")

    assert provider.api_key == "same"
    assert provider._openai_client is old_client


# ── Test 4: create_project strips global key ────────────────────────────


def test_create_project_does_not_copy_global_key(tmp_path):
    """POST /api/v2/projects with the global key should store empty string."""
    from agent_os.api.app import create_app
    from fastapi.testclient import TestClient

    data_dir = str(tmp_path / "data")
    app = create_app(data_dir=data_dir)
    client = TestClient(app)

    # Inject a mock credential store that returns "global-key"
    from agent_os.api.routes import agents_v2
    original_cred = agents_v2._credential_store
    mock_cred = FakeCredentialStore(key="global-key")
    agents_v2._credential_store = mock_cred

    workspace = str(tmp_path / "ws")
    (tmp_path / "ws").mkdir()

    try:
        resp = client.post("/api/v2/projects", json={
            "name": "TestProj",
            "workspace": workspace,
            "model": "gpt-4o",
            "api_key": "global-key",  # same as global
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        # Read directly from the project store
        from agent_os.daemon_v2.project_store import ProjectStore
        store = ProjectStore(data_dir=data_dir)
        project = store.get_project(pid)
        assert project is not None
        assert project["api_key"] == "", \
            f"Expected empty string but got '{project['api_key']}'"
    finally:
        agents_v2._credential_store = original_cred


# ── Test 5: create_project preserves different key ──────────────────────


def test_create_project_preserves_different_key(tmp_path):
    """POST with a BYOK key (different from global) should store it as-is."""
    from agent_os.api.app import create_app
    from fastapi.testclient import TestClient

    data_dir = str(tmp_path / "data")
    app = create_app(data_dir=data_dir)
    client = TestClient(app)

    from agent_os.api.routes import agents_v2
    original_cred = agents_v2._credential_store
    mock_cred = FakeCredentialStore(key="global-key")
    agents_v2._credential_store = mock_cred

    workspace = str(tmp_path / "ws")
    (tmp_path / "ws").mkdir()

    try:
        resp = client.post("/api/v2/projects", json={
            "name": "TestBYOK",
            "workspace": workspace,
            "model": "gpt-4o",
            "api_key": "my-byok-key",  # different from global
        })
        assert resp.status_code == 201
        pid = resp.json()["project_id"]

        from agent_os.daemon_v2.project_store import ProjectStore
        store = ProjectStore(data_dir=data_dir)
        project = store.get_project(pid)
        assert project is not None
        assert project["api_key"] == "my-byok-key"
    finally:
        agents_v2._credential_store = original_cred


# ── Test 6: inject_message Case 3 falls through to keychain ────────────


@pytest.mark.asyncio
async def test_inject_message_case3_falls_through_to_keychain(tmp_path):
    """When project has api_key='', inject_message Case 3 should pick up
    the current global key from the credential store."""
    project_id = "proj_006"
    cred_store = FakeCredentialStore(key="current-global-key")
    project_store = FakeProjectStore({
        project_id: {
            "id": project_id,
            "workspace": str(tmp_path),
            "name": "Test Fall-through",
            "api_key": "",  # empty -> must inherit global
            "model": "gpt-4o",
            "autonomy": "hands_off",
            "sdk": "openai",
            "provider": "custom",
        },
    })
    mgr = _make_manager(
        project_store=project_store,
        credential_store=cred_store,
        settings_store=FakeSettingsStore(),
    )

    # Patch start_agent to capture the config it receives
    captured_configs = []
    original_start = mgr.start_agent

    async def capture_start(pid, config, **kwargs):
        captured_configs.append(config)
        # Don't actually start — just record the config
        return

    mgr.start_agent = capture_start

    await mgr.inject_message(project_id, "hello")

    assert len(captured_configs) == 1
    assert captured_configs[0].api_key == "current-global-key", \
        f"Expected 'current-global-key' but got '{captured_configs[0].api_key}'"
