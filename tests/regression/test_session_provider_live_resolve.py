# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: sessions live-resolve their LLM provider from project config
at each turn start (TASK-session-provider-live-resolve).

DIAGNOSIS E1: a session created under provider A kept calling A forever, even
after the project switched to provider B — surfacing as "I changed the key and
it still fails (401)". Root cause: ``_start_loop`` (hot resume) reused the
``LLMProvider`` objects constructed at session start and refreshed only the
API key onto the STALE provider (old model / base_url / sdk).

These tests pin the live-resolve contract:
  1. A turn started on an existing session AFTER the project switched to
     provider B constructs the B client (asserted on the patched LLMProvider
     factory) and never touches the A client again.
  2. With unchanged config, the provider object is preserved (no client churn
     per turn — keeps the prefix cache warm and pins the cheap path).
  3. The out-of-turn-start readers (session-end / state-refresh callbacks →
     memory flush, PROJECT_STATE refresh) also use the live provider, not the
     closure snapshot from session start.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.config.provider_registry import ProviderRegistry
from agent_os.daemon_v2.agent_manager import AgentManager


# ── stubs (mirrors tests/regression/test_api_key_refresh.py) ────────────


class FakeWsManager:
    def broadcast(self, project_id, payload):
        pass

    def add_broadcast_hook(self, hook):
        pass


class FakeProjectStore:
    def __init__(self, projects=None):
        self._projects = projects or {}

    def get_project(self, pid):
        return self._projects.get(pid)

    def list_projects(self):
        return list(self._projects.values())

    def find_scratch_project(self):
        return None

    def update_runtime(self, pid, data):
        proj = self._projects.get(pid)
        if proj:
            proj.setdefault("runtime", {}).update(data)


class FakeCredentialStore:
    def __init__(self, key=None):
        self._key = key

    def get_api_key(self):
        return self._key


class FakeSettingsStore:
    def __init__(self, api_key=None, model=None, base_url=None, provider=None):
        self._settings = MagicMock()
        self._settings.llm.api_key = api_key
        self._settings.llm.model = model
        self._settings.llm.base_url = base_url
        self._settings.llm.provider = provider

    def get(self):
        return self._settings


class FakeActivityTranslator:
    def on_message(self, msg, pid, session_id=None):
        pass

    def on_stream_chunk(self, chunk, pid, source, session_id=None):
        pass


class FakeProcessManager:
    def set_session(self, pid, session):
        pass


class RecordingProviderFactory:
    """Patched in place of LLMProvider: records every construction and returns
    a MagicMock 'client' carrying the construction args as attributes."""

    def __init__(self):
        self.instances = []

    def __call__(self, model, api_key=None, base_url=None, sdk="openai",
                 max_output=16384, capabilities=None, reasoning=None,
                 provider="unknown"):
        inst = MagicMock(name=f"LLMProvider({provider}/{model})")
        inst.model = model
        inst.api_key = api_key
        inst.base_url = base_url
        inst.sdk = sdk
        inst.provider = provider
        inst.capabilities = capabilities
        inst.reasoning = reasoning
        def _update_key(new_key, _inst=inst):
            _inst.api_key = new_key
        inst.update_api_key = MagicMock(side_effect=_update_key)
        self.instances.append(inst)
        return inst

    def constructed(self, *, provider=None, model=None):
        return [
            i for i in self.instances
            if (provider is None or i.provider == provider)
            and (model is None or i.model == model)
        ]


PROJECT_A = {
    "id": "proj_live",
    "name": "live-resolve",
    "api_key": "key-a",
    "model": "kimi-k2",
    "provider": "moonshot",
    "base_url": "https://api.moonshot.cn/v1",
    "sdk": "openai",
    "autonomy": "hands_off",
}

SESSION_ID = "sess-live-resolve"


def _make_manager(project_store):
    return AgentManager(
        project_store=project_store,
        ws_manager=FakeWsManager(),
        sub_agent_manager=MagicMock(),
        activity_translator=FakeActivityTranslator(),
        process_manager=FakeProcessManager(),
        platform_provider=None,
        registry=MagicMock(),
        setup_engine=None,
        settings_store=FakeSettingsStore(),
        credential_store=FakeCredentialStore(),
        browser_manager=None,
        user_credential_store=None,
        provider_registry=ProviderRegistry(),
    )


async def _start_session(mgr, project_id):
    """Start an agent session and quiesce its initial loop task."""
    config = mgr._build_agent_config_from_project(project_id)
    await mgr.start_agent(project_id, config, session_id=SESSION_ID)
    handle = mgr._handles[(project_id, SESSION_ID)]
    if handle.task and not handle.task.done():
        handle.task.cancel()
        try:
            await handle.task
        except (asyncio.CancelledError, Exception):
            pass
    return handle


def _switch_project_to_b(project_store, project_id):
    project_store._projects[project_id].update({
        "provider": "deepseek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": "key-b",
    })


@pytest.mark.asyncio
async def test_next_turn_after_provider_switch_constructs_b_client(tmp_path):
    """Provider A session + project switches to B → next turn on the SAME
    session constructs the B client and never touches A again."""
    project_id = "proj_live"
    project_store = FakeProjectStore({
        project_id: dict(PROJECT_A, workspace=str(tmp_path)),
    })
    factory = RecordingProviderFactory()

    with patch("agent_os.daemon_v2.agent_manager.LLMProvider", factory):
        mgr = _make_manager(project_store)
        handle = await _start_session(mgr, project_id)

        # Turn 1 ran under provider A.
        provider_a = handle.provider
        assert provider_a.provider == "moonshot"
        assert provider_a.model == "kimi-k2"

        # The project switches to provider B.
        _switch_project_to_b(project_store, project_id)

        # Next turn on the SAME session.
        provider_a.reset_mock()
        handle.loop.run = AsyncMock(return_value=None)
        await mgr._start_loop(project_id, session_id=SESSION_ID)

    # The B client was constructed from CURRENT project config...
    b_instances = factory.constructed(provider="deepseek", model="deepseek-chat")
    assert b_instances, (
        "next turn after the provider switch must construct the provider-B "
        f"client; factory saw only: {[(i.provider, i.model) for i in factory.instances]}"
    )
    provider_b = b_instances[-1]
    assert provider_b.api_key == "key-b"
    assert provider_b.base_url == "https://api.deepseek.com"

    # ...and is what the session now routes through.
    assert handle.provider is provider_b
    assert handle.loop._provider is provider_b

    # A is never touched again: not re-keyed, not streamed, not rotated into.
    provider_a.update_api_key.assert_not_called()
    provider_a.stream.assert_not_called()
    assert handle.loop._provider is not provider_a
    assert provider_a not in handle.loop._fallback_providers
    assert handle.loop._utility_provider is not provider_a


@pytest.mark.asyncio
async def test_unchanged_config_preserves_provider_object(tmp_path):
    """No config change → no client churn on hot resume (cheap path stays)."""
    project_id = "proj_live"
    project_store = FakeProjectStore({
        project_id: dict(PROJECT_A, workspace=str(tmp_path)),
    })
    factory = RecordingProviderFactory()

    with patch("agent_os.daemon_v2.agent_manager.LLMProvider", factory):
        mgr = _make_manager(project_store)
        handle = await _start_session(mgr, project_id)
        provider_a = handle.provider

        handle.loop.run = AsyncMock(return_value=None)
        constructed_before = len(factory.instances)
        await mgr._start_loop(project_id, session_id=SESSION_ID)

    assert handle.provider is provider_a, "unchanged config must not rebuild the provider"
    assert len(factory.instances) == constructed_before, (
        "unchanged config must not construct new clients on hot resume"
    )


@pytest.mark.asyncio
async def test_session_end_callbacks_use_live_provider(tmp_path):
    """Out-of-turn-start readers (session-end routine → memory flush /
    PROJECT_STATE refresh) must also live-resolve, not keep the closure
    snapshot of provider A."""
    project_id = "proj_live"
    project_store = FakeProjectStore({
        project_id: dict(PROJECT_A, workspace=str(tmp_path)),
    })
    factory = RecordingProviderFactory()

    with patch("agent_os.daemon_v2.agent_manager.LLMProvider", factory):
        mgr = _make_manager(project_store)
        handle = await _start_session(mgr, project_id)
        provider_a = handle.provider

        _switch_project_to_b(project_store, project_id)
        handle.loop.run = AsyncMock(return_value=None)
        await mgr._start_loop(project_id, session_id=SESSION_ID)

        with patch(
            "agent_os.daemon_v2.agent_manager.run_session_end_routine",
            new_callable=AsyncMock,
        ) as routine:
            await handle.loop._on_session_end()

    assert routine.await_count == 1
    kwargs = routine.await_args.kwargs
    assert kwargs["provider"] is not provider_a, (
        "session-end routine still received the snapshotted provider A"
    )
    assert kwargs["provider"].provider == "deepseek"
    assert kwargs["utility_provider"] is not provider_a
