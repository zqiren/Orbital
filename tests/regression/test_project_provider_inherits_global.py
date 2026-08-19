# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a project that inherits its model from global settings must
inherit the global *provider* too.

Bug: `_build_agent_config_from_project` fell `model` and `base_url` back to
global settings when the project left them empty, but `provider` stayed at the
project's own value (default "custom"). A project left at provider="custom" with
model="" therefore ran as provider=custom + model=<global model> — and the
registry lookup `get_model_info("custom", "MiniMax-M3")` missed the real model
entry, silently bypassing model-specific behavior (e.g. MiniMax inline-<think>
reasoning separation). Provider must track the model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent_os.daemon_v2.agent_manager import AgentManager


def _manager(project: dict, global_llm):
    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value=project)
    settings_store = MagicMock()
    settings_store.get = MagicMock(return_value=SimpleNamespace(llm=global_llm))
    return AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        settings_store=settings_store,
    )


GLOBAL = SimpleNamespace(
    provider="minimax", model="MiniMax-M3",
    base_url="https://api.minimaxi.com/v1", api_key=None,
)


def test_project_inherits_global_provider_when_model_unset():
    # Mirrors the real orbital-marketing project: provider left at "custom",
    # model/base_url empty so they inherit global.
    mgr = _manager(
        {"workspace": "/tmp/x", "provider": "custom", "model": "",
         "base_url": None, "sdk": "openai", "api_key": ""},
        GLOBAL,
    )
    cfg = mgr._build_agent_config_from_project("proj")
    assert cfg.model == "MiniMax-M3"
    assert cfg.base_url == "https://api.minimaxi.com/v1"
    assert cfg.provider == "minimax"  # <-- the fix (was "custom")


def test_project_keeps_own_provider_when_it_pins_a_model():
    # A project that specifies its own model keeps its own provider — global
    # is not allowed to override an explicitly self-hosted setup.
    mgr = _manager(
        {"workspace": "/tmp/x", "provider": "custom", "model": "my-local-model",
         "base_url": "http://localhost:1234/v1", "sdk": "openai", "api_key": "k"},
        GLOBAL,
    )
    cfg = mgr._build_agent_config_from_project("proj")
    assert cfg.model == "my-local-model"
    assert cfg.provider == "custom"


def test_cross_provider_project_does_not_inherit_global_base_url():
    # Project pinned to Moonshot with no saved base_url, global on MiniMax:
    # falling back to the global base_url pairs a Moonshot key with MiniMax's
    # endpoint. The fallback must stay within the project's provider — the
    # registry default, not another provider's URL.
    mgr = _manager(
        {"workspace": "/tmp/x", "provider": "moonshot", "model": "kimi-k2.5",
         "base_url": None, "sdk": "openai", "api_key": "sk-moonshot"},
        GLOBAL,
    )
    cfg = mgr._build_agent_config_from_project("proj")
    assert cfg.provider == "moonshot"
    assert cfg.base_url != GLOBAL.base_url
    assert cfg.base_url and "moonshot" in cfg.base_url


def test_cross_provider_project_does_not_inherit_global_api_key():
    # Same shape, key side: the global key belongs to the global provider.
    # Sending it to another provider guarantees a misleading 401 — an empty
    # key (clean "no API key configured" error) is the honest failure.
    global_with_key = SimpleNamespace(
        provider="minimax", model="MiniMax-M3",
        base_url="https://api.minimaxi.com/v1", api_key="minimax-key",
    )
    mgr = _manager(
        {"workspace": "/tmp/x", "provider": "moonshot", "model": "kimi-k2.5",
         "base_url": "https://api.moonshot.cn/v1", "sdk": "openai", "api_key": ""},
        global_with_key,
    )
    cfg = mgr._build_agent_config_from_project("proj")
    assert cfg.api_key == ""


def test_same_provider_project_still_inherits_global_base_url_and_key():
    # A project pinning a model on the SAME provider as global keeps the
    # old inheritance — overriding just the model (or just the key) against
    # the shared endpoint is a legitimate setup.
    mgr = _manager(
        {"workspace": "/tmp/x", "provider": "minimax", "model": "MiniMax-M2.7",
         "base_url": None, "sdk": "openai", "api_key": ""},
        SimpleNamespace(
            provider="minimax", model="MiniMax-M3",
            base_url="https://api.minimaxi.com/v1", api_key="shared-key",
        ),
    )
    cfg = mgr._build_agent_config_from_project("proj")
    assert cfg.base_url == "https://api.minimaxi.com/v1"
    assert cfg.api_key == "shared-key"


# --- Same invariant, second entry point: POST /agents/start -----------------
#
# The rule above is enforced in `_build_agent_config_from_project`, but the
# start route builds its own AgentConfig inline (it must: the canonical builder
# omits llm_fallback_models, agent_slug, agent_credentials and
# global_preferences_path). Its base_url/api_key fallback had no notion of
# provider, so a project pinned to one provider with an empty endpoint field
# silently borrowed the GLOBAL provider's endpoint — sending the project's key
# to someone else's API. Caught live: a project on opencode-zen/hy3-free with
# base_url unset reached tokendance.space and came back
# `401 API 密钥不存在`.

import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def start_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from agent_os.api.routes import agents_v2
    from agent_os.config.provider_registry import ProviderRegistry

    project = {
        "project_id": "p1", "id": "p1", "workspace": "/tmp/ws",
        "provider": "opencode-zen", "model": "hy3-free",
        "api_key": "sk-project-own-key", "base_url": None,
        "name": "p1", "autonomy": "hands_off",
    }
    project_store = MagicMock()
    project_store.get_project = MagicMock(return_value=project)
    global_llm = SimpleNamespace(
        provider="tokendance", model="deepseek-v4-flash",
        base_url="https://tokendance.space/gateway/v1",
        api_key="sk-global-tokendance-key", fallback_models=[],
    )
    settings_store = MagicMock()
    settings_store.get = MagicMock(return_value=SimpleNamespace(llm=global_llm))
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value="sk-global-tokendance-key")

    manager = MagicMock()
    manager.start_agent = AsyncMock(return_value=None)

    monkeypatch.setattr(agents_v2, "_project_store", project_store)
    monkeypatch.setattr(agents_v2, "_settings_store", settings_store)
    monkeypatch.setattr(agents_v2, "_credential_store", credential_store)
    monkeypatch.setattr(agents_v2, "_agent_manager", manager)
    monkeypatch.setattr(agents_v2, "_provider_registry", ProviderRegistry())

    app = FastAPI()
    app.include_router(agents_v2.router)
    return TestClient(app, raise_server_exceptions=False), manager, project


def _started_config(manager):
    assert manager.start_agent.await_count == 1, "agent never started"
    return manager.start_agent.await_args.args[1]


def test_start_route_resolves_endpoint_from_the_pinned_provider(start_client):
    client, manager, _ = start_client
    resp = client.post("/api/v2/agents/start", json={"project_id": "p1"})
    assert resp.status_code == 200, resp.text
    config = _started_config(manager)
    assert config.provider == "opencode-zen"
    assert config.base_url == "https://opencode.ai/zen/v1"
    assert "tokendance" not in (config.base_url or "")


def test_start_route_does_not_lend_the_global_key_across_providers(start_client):
    """A key belongs to the provider it was issued for. Pairing the global key
    with another provider's endpoint is the classic wrong-provider 401."""
    client, manager, project = start_client
    project["api_key"] = ""
    resp = client.post("/api/v2/agents/start", json={"project_id": "p1"})
    assert resp.status_code in (200, 400), resp.text
    if manager.start_agent.await_count:
        config = _started_config(manager)
        assert config.api_key != "sk-global-tokendance-key"


def test_start_route_still_inherits_within_the_same_provider(start_client):
    """Same provider: inheriting the global endpoint and key is correct and
    is how a project left on defaults works at all."""
    client, manager, project = start_client
    project["provider"] = "tokendance"
    project["model"] = "deepseek-v4-flash"
    project["api_key"] = ""
    resp = client.post("/api/v2/agents/start", json={"project_id": "p1"})
    assert resp.status_code == 200, resp.text
    config = _started_config(manager)
    assert config.base_url == "https://tokendance.space/gateway/v1"
    assert config.api_key == "sk-global-tokendance-key"
