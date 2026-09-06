# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Issue-0 regression (2026-07-05): the inject-Case-3 auto-start path built
AgentConfig without is_scratch/agent_name, silently disabling the whole
scratch scope plane (prompt section, multi-root tools, portals) for every
chat session that auto-starts an agent. Both AgentConfig construction sites
(``_build_agent_config_from_project`` here, and the ``/agents/start`` route
in agents_v2.py) must agree on these fields.

Spec 082 rewrote the LLM half of the builder as a single credential-card
lookup, so the credential assertions below are expressed in cards. The
invariants they pin are the SAME ones the merge rules used to state, and
several of them are now true by construction rather than by branch: a card
carries provider, endpoint, key and model together, so no path can pair one
provider's key with another's endpoint.
"""

from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from tests.card_doubles import FakeCardStore


def _manager_with(projects: dict, settings_store=None) -> AgentManager:
    """A real AgentManager with a MagicMock project store and a card-aware
    settings store double.

    The provider registry is the REAL one: since spec 082 the endpoint comes
    from the card's provider + region through the registry, so a mock registry
    would assert nothing about the pairing these tests exist to pin.
    """
    ws = MagicMock()
    project_store = MagicMock()
    project_store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
    project_store.list_projects = MagicMock(return_value=list(projects.values()))
    sub_agent_manager = MagicMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    credential_store = MagicMock()
    credential_store.get_api_key = MagicMock(return_value=None)
    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_manager,
        activity_translator=activity_translator,
        process_manager=process_manager,
        settings_store=settings_store if settings_store is not None else FakeCardStore(),
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


# ---- The card is the whole setup (spec 082 §3.4) -----------------------------
#
# Project rows used to snapshot base_url/provider/model/api_key independently,
# so a project created under an earlier global provider carried a stale
# endpoint. Observed on a real install: a scratch project with an
# api.openai.com snapshot inherited the freshly-provisioned global TokenDance
# key and sent it to OpenAI → 401; and on 2026-09-03 an OpenCode Go model was
# paired with an OpenRouter endpoint and key. Neither shape is representable
# now: a project stores a card id, and the card carries all four together.


def _manager_with_global(projects: dict) -> AgentManager:
    """Global default card = TokenDance (the state after the one-click flow)."""
    store = FakeCardStore.with_default(
        card_id="card_global", provider="tokendance",
        model="deepseek-v4-flash", key="sk-td-global",
    )
    return _manager_with(projects, settings_store=store)


def test_project_on_the_default_card_runs_that_card():
    """No card of its own = follow the global default, wholesale. The stale
    per-project endpoint that used to live on the row is simply gone."""
    mgr = _manager_with_global({"p_s": SCRATCH})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.api_key == "sk-td-global"
    assert cfg.base_url == "https://tokendance.space/gateway/v1"
    assert cfg.provider == "tokendance"
    assert cfg.model == "deepseek-v4-flash"
    assert cfg.card_id == "card_global"


def test_custom_card_keeps_its_own_base_url():
    """A Custom card carries its endpoint and key together — the BYOK case,
    now expressed as one object instead of four project fields."""
    mgr = _manager_with_global({"p_n": {**NORMAL, "card_id": "card_byok"}})
    mgr._settings_store.add(
        card_id="card_byok", provider="custom", model="local-model",
        key="sk-own-key", base_url="https://proxy.example/v1", sdk="openai",
    )
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.api_key == "sk-own-key"
    assert cfg.base_url == "https://proxy.example/v1"
    assert cfg.provider == "custom"
    assert cfg.card_id == "card_byok"


def test_registry_card_resolves_its_provider_endpoint():
    """A registry provider's endpoint comes from the registry by region, so a
    card can never carry a URL belonging to a different provider."""
    mgr = _manager_with_global({"p_n": {**NORMAL, "card_id": "card_ds"}})
    mgr._settings_store.add(card_id="card_ds", provider="deepseek",
                            model="deepseek-chat", key="sk-ds-key")
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.api_key == "sk-ds-key"
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.provider == "deepseek"
    assert cfg.model == "deepseek-chat"


def test_china_region_card_resolves_the_china_endpoint():
    mgr = _manager_with_global({"p_n": {**NORMAL, "card_id": "card_mm"}})
    mgr._settings_store.add(card_id="card_mm", provider="minimax",
                            model="MiniMax-M3", key="sk-mm", region="china")
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.base_url == "https://api.minimaxi.com/v1"


def test_stale_card_id_without_a_default_raises_missing_card():
    """Spec 082 §3.4: a project pointing at a deleted card with no default to
    fall back on is a typed error, never a silently keyless run."""
    from agent_os.daemon_v2.provider_errors import ProviderConfigError

    mgr = _manager_with({"p_n": {**NORMAL, "card_id": "card_gone"}})
    with pytest.raises(ProviderConfigError) as exc:
        mgr._build_agent_config_from_project("p_n")
    assert exc.value.code == "missing_card"


def test_stale_card_id_falls_back_to_the_default_card():
    mgr = _manager_with_global({"p_n": {**NORMAL, "card_id": "card_gone"}})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.card_id == "card_global"
    assert cfg.api_key == "sk-td-global"


def test_no_cards_at_all_leaves_the_key_empty():
    """A fresh install with nothing configured still produces a config; the
    empty key is what surfaces the typed missing_api_key on first start."""
    mgr = _manager_with({"p_n": NORMAL})
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.api_key == ""
    assert cfg.card_id == ""


# ---- Three-way start-path parity (2026-08-19) --------------------------------
#
# Live-daemon bug: every scheduled trigger on the Orbital-marketing project
# fired into a 401 while typing in chat worked, minutes apart, same session.
# Cause: three AgentConfig construction sites had drifted apart, and the
# trigger one (``TriggerManager._fire_trigger``) had none of the
# provider/endpoint invariants the canonical builder grew.
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


# The live shape that broke, in card terms: the project follows the global
# default card and has a trigger.
STALE_PINNED = {
    "project_id": "p_m", "name": "Orbital-marketing", "workspace": "/tmp/m",
    "card_id": None,
    "triggers": [{
        "id": "trg_test", "name": "Daily scan", "enabled": True,
        "type": "schedule", "task": "Scan the repo.",
        "schedule": {"cron": "0 9 * * *", "human": "Every day at 9:00 AM"},
    }],
}


def _manager_with_opencode_global(projects: dict) -> AgentManager:
    """Default card on OpenCode Go — the install where the 401s were seen."""
    store = FakeCardStore.with_default(
        card_id="card_go", provider="opencode-go", model="deepseek-v4-flash",
        key="sk-opencode-global",
    )
    return _manager_with(projects, settings_store=store)


def _llm_fields(cfg) -> dict:
    """The subset that decides which endpoint the request actually reaches."""
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "sdk": cfg.sdk,
        "card_id": cfg.card_id,
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


def test_trigger_start_resolves_the_whole_card():
    """The concrete failure: an OpenCode Go key sent to api.minimaxi.com. The
    trigger path now gets provider, endpoint, key and model from one card."""
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
    assert cfg.card_id == "card_go"


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
    trigger or queue auto-start ran with no fallback at all. Entries are card
    references now — the rung's key comes from ITS card, never from the
    primary's, so a rung can no longer inherit a key from another provider."""
    with_fb = {**STALE_PINNED,
               "llm_fallback_models": [{"card_id": "card_ds"}]}
    mgr = _manager_with_opencode_global({"p_m": with_fb})
    mgr._settings_store.add(card_id="card_ds", provider="deepseek",
                            model="deepseek-chat", key="sk-ds")
    cfg = mgr._build_agent_config_from_project("p_m")

    assert [fb.model for fb in cfg.llm_fallback_models] == ["deepseek-chat"]
    assert cfg.llm_fallback_models[0].api_key == "sk-ds"
    assert cfg.llm_fallback_models[0].base_url == "https://api.deepseek.com"
    assert cfg.llm_fallback_models[0].card_id == "card_ds"


def test_fallback_entry_for_a_deleted_card_is_skipped():
    """A dangling reference drops the rung rather than guessing a setup for
    it — the rotation is short one entry, not pointed at a stranger."""
    with_fb = {**STALE_PINNED,
               "llm_fallback_models": [{"card_id": "card_gone"}]}
    mgr = _manager_with_opencode_global({"p_m": with_fb})
    cfg = mgr._build_agent_config_from_project("p_m")
    assert cfg.llm_fallback_models == []


def test_canonical_config_falls_back_to_global_fallback_models():
    """No project-level chain = inherit the global one."""
    mgr = _manager_with_opencode_global({"p_m": STALE_PINNED})
    mgr._settings_store.add(card_id="card_dsr", provider="deepseek",
                            model="deepseek-reasoner", key="sk-fb")
    mgr._settings_store.set_global_fallbacks(["card_dsr"])

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


# ---- Spec 072 auth-fallback rung, re-keyed on card identity (082 D8) --------
#
# The canonical builder attaches a wholesale DEFAULT-CARD snapshot as
# ``auth_fallback`` — the rung the loop rotates to after a 401/403 rejects the
# project's card. The old test was key-string inequality, which reads two
# cards that legitimately share one key (D2) as "nothing to fall back to";
# it is card-id inequality now.


def _manager_with_byok(project_extra=None) -> AgentManager:
    mgr = _manager_with_global({"p_n": {**NORMAL, "card_id": "card_byok",
                                        **(project_extra or {})}})
    mgr._settings_store.add(card_id="card_byok", provider="custom",
                            model="local-model", key="sk-own-key",
                            base_url="https://proxy.example/v1", sdk="openai")
    return mgr


def test_auth_fallback_attached_when_the_project_is_on_another_card():
    """A project on its own card + a usable default = a rung built wholesale
    from the DEFAULT card (provider/model/key/endpoint/sdk all its own)."""
    mgr = _manager_with_byok()
    cfg = mgr._build_agent_config_from_project("p_n")

    af = cfg.auth_fallback
    assert af is not None
    assert af.provider == "tokendance"
    assert af.model == "deepseek-v4-flash"
    assert af.base_url == "https://tokendance.space/gateway/v1"
    assert af.api_key == "sk-td-global"
    assert af.card_id == "card_global"


def test_auth_fallback_absent_when_the_project_is_on_the_default_card():
    """Nothing different to fall back to: the resolved card IS the default."""
    mgr = _manager_with_global({"p_s": SCRATCH})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.api_key == "sk-td-global"
    assert cfg.auth_fallback is None


def test_auth_fallback_attached_even_when_the_two_cards_share_one_key():
    """D2: the same key may back several cards (one per model). Card identity
    is the test, so the rung is still offered — the OLD key-string comparison
    would have called this "nothing to fall back to"."""
    mgr = _manager_with_global({"p_n": {**NORMAL, "card_id": "card_twin"}})
    mgr._settings_store.add(card_id="card_twin", provider="tokendance",
                            model="deepseek-v4-pro", key="sk-td-global")
    cfg = mgr._build_agent_config_from_project("p_n")
    assert cfg.auth_fallback is not None
    assert cfg.auth_fallback.model == "deepseek-v4-flash"


def test_auth_fallback_absent_when_the_default_card_lacks_model_or_key():
    mgr = _manager_with_byok()
    mgr._settings_store.resolve_card("card_global").model = ""
    assert mgr._build_agent_config_from_project("p_n").auth_fallback is None

    mgr = _manager_with_byok()
    mgr._settings_store.keys["card_global"] = ""
    assert mgr._build_agent_config_from_project("p_n").auth_fallback is None


def test_auth_fallback_provider_built_like_the_primary():
    """_build_auth_fallback_provider resolves registry model info (base_url /
    sdk overrides, max_output) and the provider's static headers — the same
    construction the primary gets in _build_llm_providers."""
    from agent_os.config.provider_registry import ModelInfo

    mgr = _manager_with_byok()
    mgr._provider_registry = MagicMock()
    mgr._provider_registry.get_model_info = MagicMock(
        return_value=ModelInfo(max_output=16384))
    mgr._provider_registry.get_provider_data = MagicMock(
        return_value={"base_url": "https://tokendance.space/gateway/v1",
                      "sdk": "openai",
                      "extra_headers": {"X-App-Name": "orbital"}})

    cfg = mgr._build_agent_config_from_project("p_n")
    rung = mgr._build_auth_fallback_provider(cfg)

    assert rung is not None
    mgr._provider_registry.get_model_info.assert_any_call(
        "tokendance", "deepseek-v4-flash")
    assert rung.model == "deepseek-v4-flash"
    assert rung.provider == "tokendance"
    assert rung.api_key == "sk-td-global"
    # No registry override → the rung's own (card) endpoint and sdk win.
    assert rung.base_url == "https://tokendance.space/gateway/v1"
    assert rung.sdk == "openai"
    assert rung.extra_headers == {"X-App-Name": "orbital"}


def test_auth_fallback_provider_none_when_config_has_no_rung():
    mgr = _manager_with_global({"p_s": SCRATCH})
    cfg = mgr._build_agent_config_from_project("p_s")
    assert cfg.auth_fallback is None
    assert mgr._build_auth_fallback_provider(cfg) is None
