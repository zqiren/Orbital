# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pi as a first-class CLI sub-agent (spec 087 §5.1, §5.4, §5.5, §11b).

Manifest contract, daemon-level params, credential readiness (the credential
card key for the configured model's provider, else Pi's own
``pi auth check``), the settings wire shape, and SubAgentManager's pi-rpc
wiring. No Pi binary and no provider is touched.
"""

import json
import os
import subprocess
from unittest.mock import MagicMock

import pytest

import agent_os.agents.setup_engine as setup_engine_module
from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.agents.setup_types import AgentSetupStatus
from agent_os.daemon_v2.sub_agent_config_store import (
    SCHEMA,
    SubAgentConfigError,
    SubAgentConfigStore,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_MANIFESTS_DIR = os.path.join(_REPO, "agent_os", "agents", "manifests")
_PI_YAML = os.path.join(_MANIFESTS_DIR, "pi.yaml")
_BINARY = "/fake/bin/pi"
_PI_MODEL_KEY = "PI_MODEL_PROVIDER_KEY"


def _pi_manifest():
    return ManifestLoader.load(_PI_YAML)


class _Cards:
    """Stand-in for SettingsStore.key_for_provider."""

    def __init__(self, keys):
        self.keys = keys
        self.calls = []

    def key_for_provider(self, provider, region=None):
        self.calls.append((provider, region))
        return self.keys.get((provider, region), "")


def _fake_run(stdout="", returncode=1, raises=None):
    def run(argv, **kwargs):
        run.calls.append((argv, kwargs))
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    run.calls = []
    return run


def _engine(tmp_path, *, model=None, cards=None):
    registry = AgentRegistry()
    registry.register(_pi_manifest())
    store = SubAgentConfigStore(str(tmp_path / "sub_agent_config.json"))
    if model:
        store.set("pi", {"model": model})
    engine = SetupEngine(registry, credential_store=None,
                         sub_agent_config_store=store, data_dir=str(tmp_path),
                         card_store=cards)
    engine._resolved_paths["pi"] = _BINARY
    engine._run_check_command = lambda cmd: "0.85.1"
    engine.check_dependencies = lambda manifest: (True, [])
    return engine


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestPiManifest:
    def test_runtime_is_native_rpc_ignoring_untrusted_project_resources(self):
        m = _pi_manifest()
        assert (m.slug, m.name, m.runtime.adapter, m.runtime.command) == (
            "pi", "Pi", "cli", "pi")
        assert m.runtime.transport == "pi-rpc"
        assert m.runtime.mode == "pipe"
        assert m.runtime.args == ["--mode", "rpc", "--no-approve"]

    def test_bring_your_own_install_and_no_unenforced_network_claim(self):
        m = _pi_manifest()
        assert m.setup.orbital_install.platforms == []
        assert m.setup.install_command == (
            "npm install -g --ignore-scripts @earendil-works/pi-coding-agent")
        assert m.setup.check_command == "pi --version"
        assert m.permissions.network_domains == []
        assert m.permissions.workspace_access == "read_write"
        assert m.permissions.autonomy_recommended == "hands_off"

    def test_detects_the_pi_executable_on_every_platform(self):
        detect = _pi_manifest().setup.auto_detect
        assert detect["windows"] and all(
            p.lower().endswith("\\pi.cmd") for p in detect["windows"])
        assert "/opt/homebrew/bin/pi" in detect["macos"]
        assert "$HOME/.npm-global/bin/pi" in detect["macos"]
        assert "/usr/local/bin/pi" in detect["linux"]
        for os_name in ("macos", "linux"):
            assert all(p.endswith("/pi") for p in detect[os_name])

    def test_registry_loads_pi_beside_the_other_agents(self):
        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)
        assert registry.get("pi") is not None
        assert registry.get("codex") is not None

    def test_provider_key_map_covers_only_native_pi_providers(self):
        [cred] = _pi_manifest().setup.credentials
        assert (cred.key, cred.type, cred.required, cred.env_var) == (
            _PI_MODEL_KEY, "model_provider", True, "")
        pairs = {(e["provider"], e["card_provider"], e.get("card_region")): e["env_var"]
                 for e in cred.provider_keys}
        assert pairs[("openrouter", "openrouter", None)] == "OPENROUTER_API_KEY"
        assert pairs[("opencode-go", "opencode-go", None)] == "OPENCODE_API_KEY"
        assert pairs[("opencode", "opencode-zen", None)] == "OPENCODE_API_KEY"
        assert pairs[("deepseek", "deepseek", None)] == "DEEPSEEK_API_KEY"
        assert pairs[("minimax", "minimax", "global")] == "MINIMAX_API_KEY"
        assert pairs[("minimax-cn", "minimax", "china")] == "MINIMAX_CN_API_KEY"
        assert pairs[("moonshotai-cn", "moonshot", "china")] == "MOONSHOT_API_KEY"
        with open(os.path.join(_REPO, "agent_os", "config", "providers.json"),
                  encoding="utf-8") as f:
            data = json.load(f)
        known = set(data.get("providers", data))
        card_providers = {e["card_provider"] for e in cred.provider_keys}
        assert card_providers <= known
        # Gateways and custom endpoints need Pi models.json — user config
        # Orbital must not write (spec §11a).
        assert not {"custom", "tokendance", "hunyuan"} & card_providers
        assert len({e["provider"] for e in cred.provider_keys}) == len(cred.provider_keys)

    def test_auth_probe_never_refreshes_or_prints_credentials(self):
        [cred] = _pi_manifest().setup.credentials
        assert cred.check_command == "pi auth check --model {model} --json --no-refresh"
        assert (cred.check_field, cred.check_value) == ("status", "ready")
        assert "--credentials" not in cred.check_command
        assert cred.setup_command == ""


# ---------------------------------------------------------------------------
# Daemon-level params
# ---------------------------------------------------------------------------


class TestPiParams:
    def test_model_is_free_text_passed_as_the_model_flag(self):
        assert SCHEMA["pi"]["model"].allowed is None
        assert SCHEMA["pi"]["model"].flag_template == "--model {value}"

    def test_effort_is_the_pi_thinking_level(self):
        schema = SCHEMA["pi"]["effort"]
        assert schema.allowed == ("off", "minimal", "low", "medium", "high", "xhigh", "max")
        assert schema.flag_template == "--thinking {value}"
        assert schema.default is None

    def test_params_render_as_separate_argv_tokens(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "c.json"))
        store.set("pi", {"model": "openrouter/deepseek/deepseek-chat-v3.1",
                         "effort": "high"})
        assert store.build_extra_args("pi") == [
            "--model", "openrouter/deepseek/deepseek-chat-v3.1", "--thinking", "high"]

    def test_unknown_thinking_level_is_rejected(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "c.json"))
        with pytest.raises(SubAgentConfigError):
            store.set("pi", {"effort": "ultra"})


# ---------------------------------------------------------------------------
# Readiness + key injection
# ---------------------------------------------------------------------------


class TestPiReadiness:
    def test_unknown_without_a_configured_model(self, tmp_path, monkeypatch):
        run = _fake_run()
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        status = _engine(tmp_path, cards=_Cards({})).check_agent("pi")
        assert status.credential_state == "unknown"
        assert status.credentials_configured is True
        assert status.missing_credentials == []
        assert run.calls == []

    def test_matching_card_is_configured_and_injects_only_that_key(
            self, tmp_path, monkeypatch):
        run = _fake_run()
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        cards = _Cards({("openrouter", None): "sk-or-card-key",
                        ("deepseek", None): "sk-ds-card-key"})
        engine = _engine(tmp_path, model="openrouter/deepseek/deepseek-chat-v3.1",
                         cards=cards)
        status = engine.check_agent("pi")
        assert status.credential_state == "configured"
        assert status.credentials_configured is True
        assert run.calls == []
        config = engine.get_adapter_config("pi", str(tmp_path))
        assert config["env"] == {"OPENROUTER_API_KEY": "sk-or-card-key"}
        assert config["args"] == ["--mode", "rpc", "--no-approve",
                                  "--model", "openrouter/deepseek/deepseek-chat-v3.1"]
        assert not any("sk-or-card-key" in a for a in config["args"])
        assert {provider for provider, _ in cards.calls} == {"openrouter"}

    def test_card_region_selects_the_regional_pi_provider(self, tmp_path, monkeypatch):
        monkeypatch.setattr(setup_engine_module.subprocess, "run", _fake_run())
        cards = _Cards({("minimax", "china"): "mm-cn-key",
                        ("minimax", "global"): "mm-intl-key"})
        engine = _engine(tmp_path, model="minimax-cn/MiniMax-M2.7", cards=cards)
        assert engine.get_adapter_config("pi", str(tmp_path))["env"] == {
            "MINIMAX_CN_API_KEY": "mm-cn-key"}

    def test_native_pi_auth_ready_is_configured(self, tmp_path, monkeypatch):
        run = _fake_run('{"status":"ready","provider":"anthropic","authType":"oauth"}\n', 0)
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        engine = _engine(tmp_path, model="anthropic/claude-sonnet-5", cards=_Cards({}))
        status = engine.check_agent("pi")
        assert status.credential_state == "configured"
        [(argv, kwargs)] = run.calls
        assert argv == [_BINARY, "auth", "check", "--model", "anthropic/claude-sonnet-5",
                        "--json", "--no-refresh"]
        assert not kwargs.get("shell")
        assert engine.get_adapter_config("pi", str(tmp_path))["env"] == {}

    def test_native_not_ready_is_missing_and_blocks_readiness(self, tmp_path, monkeypatch):
        run = _fake_run('{"status":"not_ready","provider":"openrouter",'
                        '"reason":"credentials_not_configured"}\n', 1)
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        engine = _engine(tmp_path, model="openrouter/deepseek/deepseek-chat-v3.1",
                         cards=_Cards({}))
        status = engine.check_agent("pi")
        assert status.credential_state == "missing"
        assert status.credentials_configured is False
        assert status.missing_credentials == [_PI_MODEL_KEY]

    @pytest.mark.parametrize("stdout", [
        "",
        "Unable to resolve model",
        '{"status":"invalid","provider":"x","reason":"invalid_state"}',
    ])
    def test_unverifiable_probe_is_unknown(self, tmp_path, monkeypatch, stdout):
        monkeypatch.setattr(setup_engine_module.subprocess, "run", _fake_run(stdout, 1))
        status = _engine(tmp_path, model="anthropic/x", cards=_Cards({})).check_agent("pi")
        assert status.credential_state == "unknown"
        assert status.credentials_configured is True

    def test_probe_timeout_is_unknown(self, tmp_path, monkeypatch):
        run = _fake_run(raises=subprocess.TimeoutExpired(["pi"], 10))
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        status = _engine(tmp_path, model="anthropic/x", cards=_Cards({})).check_agent("pi")
        assert status.credential_state == "unknown"

    def test_model_text_never_reaches_a_shell(self, tmp_path, monkeypatch):
        run = _fake_run('{"status":"ready"}', 0)
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        model = "x/y; touch /tmp/pwned"
        _engine(tmp_path, model=model, cards=_Cards({})).check_agent("pi")
        [(argv, kwargs)] = run.calls
        assert argv[4] == model
        assert not kwargs.get("shell")

    def test_without_a_card_store_the_native_probe_still_answers(
            self, tmp_path, monkeypatch):
        run = _fake_run('{"status":"ready"}', 0)
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        status = _engine(tmp_path, model="openrouter/a/b", cards=None).check_agent("pi")
        assert status.credential_state == "configured"
        assert len(run.calls) == 1

    def test_not_installed_never_probes(self, tmp_path, monkeypatch):
        run = _fake_run('{"status":"ready"}', 0)
        monkeypatch.setattr(setup_engine_module.subprocess, "run", run)
        engine = _engine(tmp_path, model="openrouter/a/b", cards=_Cards({}))
        engine._resolved_paths.clear()
        monkeypatch.setattr(engine, "resolve_binary", lambda manifest: None)
        status = engine.check_agent("pi")
        assert status.installed is False
        assert status.credential_state == "unknown"
        assert run.calls == []

    def test_other_agents_carry_no_credential_state(self, tmp_path):
        registry = AgentRegistry()
        registry.load_directory(_MANIFESTS_DIR)
        engine = SetupEngine(registry, data_dir=str(tmp_path))
        engine.resolve_binary = lambda manifest: None
        engine.check_dependencies = lambda manifest: (True, [])
        assert engine.check_agent("codebuddy").credential_state is None


# ---------------------------------------------------------------------------
# Credential-card lookup
# ---------------------------------------------------------------------------


class _KeyStore:
    def __init__(self):
        self.keys = {}

    def get(self, card_id):
        return self.keys.get(card_id)

    def set(self, card_id, key):
        self.keys[card_id] = key
        return {}

    def delete(self, card_id):
        self.keys.pop(card_id, None)

    def source(self, card_id):
        return "keychain" if card_id in self.keys else "none"


class TestCardKeyForProvider:
    def _store(self, tmp_path):
        from agent_os.daemon_v2.settings_store import SettingsStore
        return SettingsStore(data_dir=str(tmp_path), card_key_store=_KeyStore(),
                             migrate=False)

    def test_default_then_recency_and_custom_endpoints_are_skipped(self, tmp_path):
        s = self._store(tmp_path)
        s.create_card(provider="deepseek", model="deepseek-v4-flash", api_key="ds-default")
        s.create_card(provider="openrouter", model="a",
                      base_url="https://gateway.example/v1", api_key="or-gateway")
        s.create_card(provider="openrouter", model="b", api_key="or-direct")
        assert s.key_for_provider("openrouter") == "or-direct"
        assert s.key_for_provider("deepseek") == "ds-default"
        assert s.key_for_provider("anthropic") == ""

    def test_default_card_wins_over_a_more_recent_one(self, tmp_path):
        s = self._store(tmp_path)
        first = s.create_card(provider="openrouter", model="a", api_key="or-first")
        s.create_card(provider="openrouter", model="b", api_key="or-second")
        s.set_default_card(first.id)
        assert s.key_for_provider("openrouter") == "or-first"

    def test_region_must_match_when_asked(self, tmp_path):
        s = self._store(tmp_path)
        s.create_card(provider="minimax", model="MiniMax-M3", region="china",
                      api_key="mm-cn")
        assert s.key_for_provider("minimax", "china") == "mm-cn"
        assert s.key_for_provider("minimax", "global") == ""
        assert s.key_for_provider("minimax") == "mm-cn"

    def test_card_without_a_key_is_passed_over(self, tmp_path):
        s = self._store(tmp_path)
        s.create_card(provider="openrouter", model="a")
        s.create_card(provider="openrouter", model="b", api_key="or-keyed")
        assert s.key_for_provider("openrouter") == "or-keyed"


# ---------------------------------------------------------------------------
# Settings wire shape
# ---------------------------------------------------------------------------


class TestSettingsSurface:
    def _status(self, **overrides):
        fields = dict(slug="pi", name="Pi", installed=True, binary_path=_BINARY,
                      version="0.85.1", dependencies_met=True, missing_dependencies=[],
                      credentials_configured=True, missing_credentials=[])
        fields.update(overrides)
        return AgentSetupStatus(**fields)

    def test_status_entry_carries_credential_state(self, monkeypatch):
        from agent_os.api.routes import settings as settings_routes
        monkeypatch.setattr(settings_routes, "_setup_engine", None)
        monkeypatch.setattr(settings_routes, "_sub_agent_config_store", None)
        entry = settings_routes._build_sub_agent_status_entry(
            self._status(credential_state="unknown"))
        assert entry["credential_state"] == "unknown"
        assert entry["ready"] is True

    def test_missing_state_is_not_ready(self, monkeypatch):
        from agent_os.api.routes import settings as settings_routes
        monkeypatch.setattr(settings_routes, "_setup_engine", None)
        monkeypatch.setattr(settings_routes, "_sub_agent_config_store", None)
        entry = settings_routes._build_sub_agent_status_entry(self._status(
            credential_state="missing", credentials_configured=False,
            missing_credentials=[_PI_MODEL_KEY]))
        assert entry["credential_state"] == "missing"
        assert entry["ready"] is False

    def test_agents_without_the_state_report_none(self, monkeypatch):
        from agent_os.api.routes import settings as settings_routes
        monkeypatch.setattr(settings_routes, "_setup_engine", None)
        monkeypatch.setattr(settings_routes, "_sub_agent_config_store", None)
        entry = settings_routes._build_sub_agent_status_entry(
            self._status(slug="codex", name="Codex"))
        assert entry["credential_state"] is None


# ---------------------------------------------------------------------------
# SubAgentManager wiring
# ---------------------------------------------------------------------------


class TestManagerWiring:
    def _manager(self, **kwargs):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        return SubAgentManager(process_manager=MagicMock(), **kwargs)

    def test_pi_rpc_resolves_the_native_transport(self, tmp_path):
        from agent_os.agent.transports.pi_rpc_transport import PiRPCTransport
        transport = self._manager()._resolve_transport(
            _pi_manifest(), {"workspace": str(tmp_path), "args": []},
            system_prompt="brief", resume_record={"session_id": "S1"})
        assert isinstance(transport, PiRPCTransport)
        assert transport._system_prompt == "brief"
        assert transport._resume_session_id == "S1"
        assert transport._session_dir == os.path.join(
            str(tmp_path), "orbital", "sub_agents", "pi", "pi-sessions")

    def test_resume_is_provider_confirmed_not_file_prechecked(self, tmp_path):
        registry = AgentRegistry()
        registry.register(_pi_manifest())
        mgr = self._manager(registry=registry)
        record = {"session_id": "S1"}
        session = MagicMock()
        session.get_sub_agent_thread.return_value = record
        mgr._session_resolver = lambda project_id, session_id: session
        assert mgr._determine_resume(str(tmp_path), "p1", "pi", "sess") == (
            record, "resumed", None)

    async def test_dispatch_forwards_the_briefing_and_creates_memory(self, tmp_path):
        from unittest.mock import AsyncMock, patch

        from agent_os.agent.transports.pi_rpc_transport import PiRPCTransport
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

        registry = AgentRegistry()
        registry.register(_pi_manifest())
        ws = tmp_path / "ws"
        ws.mkdir()
        setup_engine = MagicMock()
        setup_engine.get_adapter_config.return_value = {
            "command": _BINARY, "args": ["--mode", "rpc", "--no-approve"],
            "workspace": str(ws), "approval_patterns": [], "env": {},
            "network_domains": [], "interactive": False,
        }
        setup_engine.check_all.return_value = []
        project_store = MagicMock()
        project_store.get_project.return_value = {"workspace": str(ws)}
        process_manager = MagicMock()
        process_manager.start = AsyncMock()
        mgr = SubAgentManager(process_manager=process_manager, registry=registry,
                              setup_engine=setup_engine, project_store=project_store)

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as adapter_cls:
            adapter_cls.return_value = AsyncMock()
            result = await mgr.start("p1", "pi", session_id="s1")

        assert result.startswith("Started Pi")
        transport = adapter_cls.call_args.kwargs["transport"]
        assert isinstance(transport, PiRPCTransport)
        assert transport._system_prompt
        assert os.path.isfile(os.path.join(str(ws), "orbital", "sub_agents", "pi", "MEMORY.md"))

    def test_a_failed_resume_is_reported_fresh_after_start(self):
        from agent_os.agent.transports.pi_rpc_transport import PiRPCTransport
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        transport = PiRPCTransport(resume_record={"session_id": "S1"})
        transport._resume_outcome = ("fresh", "resume_failed")
        assert SubAgentManager._provider_confirmed_resume_outcome(
            {"session_id": "S1"}, "resumed", None, transport) == (
            "fresh", "resume_failed")
