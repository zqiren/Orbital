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
