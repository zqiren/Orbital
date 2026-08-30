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
    gs.llm.fallback_models = []
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


# ---- Three-way start-path parity (2026-08-19) --------------------------------
#
# Live-daemon bug: every scheduled trigger on the Orbital-marketing project
# fired into a 401 while typing in chat worked, minutes apart, same session.
# Cause: three AgentConfig construction sites had drifted apart, and the
# trigger one (``TriggerManager._fire_trigger``) had none of the
# provider/endpoint invariants the canonical builder grew. It resolved each
# field independently — project provider ``minimax`` + project base_url
# ``api.minimaxi.com`` + global model ``deepseek-v4-flash`` + the global
# OpenCode Go key — a combination no single provider can serve.
#
# The invariant these tests pin is not any one rule but the *absence of a
# second implementation*: every path that starts an agent for a project
# (chat/inject, queue, trigger, /agents/start) must derive its config from
# ``_build_agent_config_from_project`` so a rule written once holds
# everywhere. Fallback models included — they were previously resolved only
# in the /agents/start route, so a trigger or queue auto-start silently ran
# with no fallback chain at all.

import asyncio
from unittest.mock import AsyncMock

from agent_os.daemon_v2.trigger_manager import TriggerManager


# The exact live shape that broke: project pins a provider + endpoint it no
# longer has a key for, leaves model/key empty to inherit the global provider.
STALE_PINNED = {
    "project_id": "p_m", "name": "Orbital-marketing", "workspace": "/tmp/m",
    "provider": "minimax", "model": "", "api_key": "",
    "base_url": "https://api.minimaxi.com/v1",
    "triggers": [{
        "id": "trg_test", "name": "Daily scan", "enabled": True,
        "type": "schedule", "task": "Scan the repo.",
        "schedule": {"cron": "0 9 * * *", "human": "Every day at 9:00 AM"},
    }],
}


def _manager_with_opencode_global(projects: dict) -> AgentManager:
    """Global settings on OpenCode Go with the key in the credential store —
    the state of the install where the trigger 401s were observed."""
    mgr = _manager_with(projects)
    gs = MagicMock()
    gs.llm.provider = "opencode-go"
    gs.llm.base_url = "https://opencode.ai/zen/go/v1"
    gs.llm.model = "deepseek-v4-flash"
    gs.llm.api_key = None
    gs.llm.fallback_models = []
    mgr._settings_store.get = MagicMock(return_value=gs)
    mgr._credential_store.get_api_key = MagicMock(return_value="sk-opencode-global")
    return mgr


def _llm_fields(cfg) -> dict:
    """The subset that decides which endpoint the request actually reaches."""
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "sdk": cfg.sdk,
        "fallbacks": [(fb.provider, fb.model, fb.base_url, fb.api_key)
                      for fb in cfg.llm_fallback_models],
        # Spec 072: every start path must derive the same auth-fallback rung.
        "auth_fallback": (
            (cfg.auth_fallback.provider, cfg.auth_fallback.model,
             cfg.auth_fallback.base_url, cfg.auth_fallback.api_key,
             cfg.auth_fallback.sdk)
            if cfg.auth_fallback is not None else None
        ),
    }


def _config_passed_to_start_agent(mgr) -> "AgentConfig":
    """Pull the AgentConfig out of the mocked start_agent call."""
    assert mgr.start_agent.await_count == 1, "start_agent was not called exactly once"
    args, kwargs = mgr.start_agent.await_args
    return kwargs.get("config") or args[1]


def test_trigger_start_uses_canonical_config():
    """A fired trigger must start the agent with exactly the config the
    canonical builder derives — not its own re-derivation."""
    mgr = _manager_with_opencode_global({"p_m": STALE_PINNED})
    expected = _llm_fields(mgr._build_agent_config_from_project("p_m"))

    mgr.start_agent = AsyncMock()
    mgr.is_running = MagicMock(return_value=False)
    tm = TriggerManager(mgr._project_store, mgr)
    asyncio.run(tm._fire_trigger("p_m", "trg_test"))

    assert _llm_fields(_config_passed_to_start_agent(mgr)) == expected


def test_trigger_start_does_not_pair_global_key_with_stale_endpoint():
    """The concrete failure: an OpenCode Go key sent to api.minimaxi.com."""
    mgr = _manager_with_opencode_global({"p_m": STALE_PINNED})
    mgr.start_agent = AsyncMock()
    mgr.is_running = MagicMock(return_value=False)
    tm = TriggerManager(mgr._project_store, mgr)
    asyncio.run(tm._fire_trigger("p_m", "trg_test"))

    cfg = _config_passed_to_start_agent(mgr)
    assert cfg.base_url == "https://opencode.ai/zen/go/v1"
    assert cfg.provider == "opencode-go"
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.api_key == "sk-opencode-global"


def test_start_route_uses_canonical_config():
    """``POST /agents/start`` must derive config the same way."""
    from agent_os.api.routes import agents_v2

    mgr = _manager_with_opencode_global({"p_m": STALE_PINNED})
    expected = _llm_fields(mgr._build_agent_config_from_project("p_m"))

    mgr.start_agent = AsyncMock()
    agents_v2.configure(
        project_store=mgr._project_store, agent_manager=mgr,
        ws_manager=MagicMock(), settings_store=mgr._settings_store,
        credential_store=mgr._credential_store,
        provider_registry=mgr._provider_registry,
    )
    asyncio.run(agents_v2.start_agent(
        agents_v2.StartAgentRequest(project_id="p_m")))

    assert _llm_fields(_config_passed_to_start_agent(mgr)) == expected


def test_canonical_config_carries_project_fallback_models():
    """Fallback chains were resolved only in the /agents/start route, so a
    trigger or queue auto-start ran with no fallback at all."""
    with_fb = {**STALE_PINNED, "llm_fallback_models": [
        {"provider": "deepseek", "model": "deepseek-chat",
         "base_url": "https://api.deepseek.com", "api_key": "", "sdk": "openai"},
    ]}
    mgr = _manager_with_opencode_global({"p_m": with_fb})
    cfg = mgr._build_agent_config_from_project("p_m")

    assert [fb.model for fb in cfg.llm_fallback_models] == ["deepseek-chat"]
    # An entry with no key of its own inherits the resolved primary key.
    assert cfg.llm_fallback_models[0].api_key == "sk-opencode-global"


def test_canonical_config_falls_back_to_global_fallback_models():
    """No project-level chain = inherit the global one."""
    mgr = _manager_with_opencode_global({"p_m": STALE_PINNED})
    gs = mgr._settings_store.get()
    fb = MagicMock()
    fb.model_dump = MagicMock(return_value={
        "provider": "deepseek", "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com", "api_key": "sk-fb", "sdk": "openai",
    })
    gs.llm.fallback_models = [fb]

    cfg = mgr._build_agent_config_from_project("p_m")
    assert [f.model for f in cfg.llm_fallback_models] == ["deepseek-reasoner"]
    assert cfg.llm_fallback_models[0].api_key == "sk-fb"


def test_canonical_config_carries_agent_slug_and_credentials():
    """Fields the /agents/start route used to add on its own."""
    pinned = {**NORMAL, "agent_slug": "claude-code",
              "agent_credentials": {"claude-code": {"token": "t"}}}
    mgr = _manager_with_opencode_global({"p_n": pinned})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.agent_slug == "claude-code"
    assert cfg.agent_credentials == {"claude-code": {"token": "t"}}


# ---- Spec 072: auth-fallback rung derivation --------------------------------
#
# The canonical builder attaches a wholesale GLOBAL-default snapshot as
# ``auth_fallback`` — the rung the loop rotates to after a 401/403 rejects the
# project key. Because every start path goes through the canonical builder
# (pinned above), deriving it here and only here IS the parity guarantee; the
# ``auth_fallback`` entry in ``_llm_fields`` makes the existing three-way
# parity tests cover it too.


def _manager_with_global_sdk(projects: dict) -> AgentManager:
    """_manager_with_global plus an explicit global sdk (MagicMock would
    otherwise leak a truthy mock into the sdk assertion)."""
    mgr = _manager_with_global(projects)
    mgr._settings_store.get().llm.sdk = "anthropic"
    return mgr


def test_auth_fallback_attached_when_project_key_differs():
    """BYOK project + usable global default = a rung built wholesale from
    global settings (provider/model/key/base_url/sdk all global)."""
    byok = {**NORMAL, "model": "", "api_key": "sk-own-key",
            "base_url": "https://proxy.example/v1"}
    mgr = _manager_with_global_sdk({"p_n": byok})
    cfg = mgr._build_agent_config_from_project("p_n")

    af = cfg.auth_fallback
    assert af is not None
    assert af.provider == "tokendance"
    assert af.model == "deepseek-v4-flash"
    assert af.base_url == "https://tokendance.space/gateway/v1"
    assert af.api_key == "sk-td-global"
    assert af.sdk == "anthropic"


def test_auth_fallback_absent_when_project_inherits_global_key():
    """Nothing different to fall back to: the resolved primary key IS the
    global key."""
    inheriting = {**SCRATCH, "model": "", "api_key": "", "base_url": None}
    mgr = _manager_with_global_sdk({"p_s": inheriting})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.api_key == "sk-td-global"
    assert cfg.auth_fallback is None


def test_auth_fallback_absent_when_global_lacks_model_or_key():
    byok = {**NORMAL, "model": "", "api_key": "sk-own-key"}

    mgr = _manager_with_global_sdk({"p_n": byok})
    mgr._settings_store.get().llm.model = ""
    assert mgr._build_agent_config_from_project("p_n").auth_fallback is None

    mgr = _manager_with_global_sdk({"p_n": byok})
    mgr._settings_store.get().llm.api_key = None
    mgr._credential_store.get_api_key = MagicMock(return_value=None)
    assert mgr._build_agent_config_from_project("p_n").auth_fallback is None


def test_auth_fallback_provider_built_like_the_primary():
    """_build_auth_fallback_provider resolves registry model info (base_url /
    sdk overrides, max_output) and the provider's static headers — the same
    construction the primary gets in _build_llm_providers."""
    byok = {**NORMAL, "model": "", "api_key": "sk-own-key"}
    mgr = _manager_with_global_sdk({"p_n": byok})
    model_info = MagicMock(base_url=None, sdk=None, max_output=16384,
                           capabilities=None, reasoning=None)
    mgr._provider_registry.get_model_info = MagicMock(return_value=model_info)
    mgr._provider_registry.get_provider_data = MagicMock(
        return_value={"extra_headers": {"X-App-Name": "orbital"}})

    cfg = mgr._build_agent_config_from_project("p_n")
    # sdk="anthropic" would construct a real Anthropic client; the wire
    # protocol is asserted via the provider attrs, so keep the client cheap.
    cfg.auth_fallback.sdk = "openai"
    rung = mgr._build_auth_fallback_provider(cfg)

    assert rung is not None
    mgr._provider_registry.get_model_info.assert_any_call(
        "tokendance", "deepseek-v4-flash")
    assert rung.model == "deepseek-v4-flash"
    assert rung.provider == "tokendance"
    assert rung.api_key == "sk-td-global"
    # No registry override → the rung's own (global) endpoint and sdk win.
    assert rung.base_url == "https://tokendance.space/gateway/v1"
    assert rung.sdk == "openai"
    assert rung.extra_headers == {"X-App-Name": "orbital"}


def test_auth_fallback_provider_none_when_config_has_no_rung():
    mgr = _manager_with_global_sdk({"p_s": SCRATCH})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.auth_fallback is None
    assert mgr._build_auth_fallback_provider(cfg) is None
