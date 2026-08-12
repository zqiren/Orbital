# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue-0 regression (2026-07-05): the inject-Case-3 auto-start path built
AgentConfig without is_scratch/agent_name, silently disabling the whole
scratch scope plane (prompt section, multi-root tools, portals) for every
chat session that auto-starts an agent. Both AgentConfig construction sites
(``_build_agent_config_from_project`` here, and the ``/agents/start`` route
in agents_v2.py) must agree on these fields.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager


def _manager_with(projects: dict) -> AgentManager:
    """Mirror the construction idiom in tests/unit/test_build_worker_deps.py
    (``_make_manager``): a real AgentManager with a MagicMock project store
    plus MagicMock settings/credential stores whose lookups return None so
    the fallback chains in ``_build_agent_config_from_project`` don't blow up
    on missing attributes.
    """
    ws = MagicMock()
    project_store = MagicMock()
    project_store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
    project_store.list_projects = MagicMock(return_value=list(projects.values()))
    sub_agent_manager = MagicMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    provider_registry = MagicMock()
    provider_registry.get_model_info.return_value = MagicMock(
        max_output=16384, capabilities=None, reasoning=None,
    )
    settings_store = MagicMock()
    settings_store.get = MagicMock(return_value=None)
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
        provider_registry=provider_registry,
        settings_store=settings_store,
        credential_store=credential_store,
    )
    return mgr


SCRATCH = {"project_id": "p_s", "name": "Quick Tasks", "agent_name": "Assistant",
           "workspace": "/tmp/s", "is_scratch": True}
NORMAL = {"project_id": "p_n", "name": "Hn-daily", "workspace": "/tmp/n",
          "is_scratch": False,
          "sub_agent_deployment_instructions": "Use Codex for implementation."}


def test_auto_start_config_carries_is_scratch_and_agent_name():
    mgr = _manager_with({"p_s": SCRATCH})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.is_scratch is True
    assert cfg.agent_name == "Assistant"


def test_auto_start_config_non_scratch_stays_false():
    mgr = _manager_with({"p_n": NORMAL})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.is_scratch is False
    # agent_name falls back to the project name, matching the /agents/start route
    assert cfg.agent_name == "Hn-daily"
    assert cfg.sub_agent_deployment_instructions == "Use Codex for implementation."


# ---- Stale project base_url vs inherited global provider (Spec 47 fallout) ----
#
# Project rows snapshot base_url verbatim at creation, so a project created
# under an earlier global provider carries a stale endpoint. Observed on a real
# install: a scratch project with an api.openai.com snapshot inherited the
# freshly-provisioned global TokenDance key and sent it to OpenAI → 401.
# Invariant (same as the crosses_provider comment in the source): base_url and
# api_key must stay within the resolved provider.


def _manager_with_global(projects: dict) -> AgentManager:
    """_manager_with, plus real-looking global settings + a stored global key
    (the state after the TokenDance one-click flow + wizard save)."""
    mgr = _manager_with(projects)
    gs = MagicMock()
    gs.llm.provider = "tokendance"
    gs.llm.base_url = "https://tokendance.space/gateway/v1"
    gs.llm.model = "deepseek-v4-flash"
    gs.llm.api_key = None
    mgr._settings_store.get = MagicMock(return_value=gs)
    mgr._credential_store.get_api_key = MagicMock(return_value="sk-td-global")
    return mgr


def test_stale_project_base_url_ignored_when_inheriting_global_key():
    """No model pin + no own key = full inherit: the stale base_url snapshot
    must not pair the global key with the old provider's endpoint."""
    stale = {**SCRATCH, "model": "", "api_key": "",
             "base_url": "https://api.openai.com/v1"}
    mgr = _manager_with_global({"p_s": stale})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.api_key == "sk-td-global"
    assert cfg.base_url == "https://tokendance.space/gateway/v1"
    assert cfg.provider == "tokendance"
    assert cfg.model == "deepseek-v4-flash"


def test_byok_project_keeps_its_own_base_url():
    """A project with its OWN key keeps its own endpoint — that pairing is
    deliberate (BYOK against a specific endpoint), not a stale snapshot."""
    byok = {**NORMAL, "model": "", "api_key": "sk-own-key",
            "base_url": "https://proxy.example/v1"}
    mgr = _manager_with_global({"p_n": byok})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.api_key == "sk-own-key"
    assert cfg.base_url == "https://proxy.example/v1"


def test_cross_provider_pinned_project_unchanged():
    """A model-pinned project (crosses_provider branch) keeps its own trio —
    guard that the inherit-branch fix didn't leak into it."""
    pinned = {**NORMAL, "model": "deepseek-chat", "provider": "deepseek",
              "api_key": "sk-ds-key", "base_url": "https://api.deepseek.com"}
    mgr = _manager_with_global({"p_n": pinned})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.api_key == "sk-ds-key"
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.provider == "deepseek"
    assert cfg.model == "deepseek-chat"
