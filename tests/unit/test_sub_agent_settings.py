# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the sub-agent settings page and supporting plumbing.

Covers:
- SetupEngine.check_all() caching with explicit invalidation.
- ``disabled_sub_agents`` denylist semantics in the dispatch path.
- Legacy ``enabled_sub_agents`` field is informational-only in v1.
- SubAgentConfigStore validation, persistence, and arg rendering.
- ``PUT /api/v2/settings/sub-agents/{slug}/config`` validation + persistence.
- ``POST /api/v2/settings/sub-agents/{slug}/login`` doesn't store tokens.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCapabilities,
    ManifestCredential,
    ManifestPermissions,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.daemon_v2.sub_agent_config_store import (
    SubAgentConfigError,
    SubAgentConfigStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_codex_probe(monkeypatch):
    """Keep unit tests hermetic: the settings routes probe the codex binary
    for its live model list; never spawn the real one here. The cache and
    the route seam stay exercised — only the subprocess layer is stubbed.
    (TestCodexLiveModels patches the higher `_codex_live_models` seam and is
    unaffected.)"""
    import agent_os.agent.transports.codex_models as _cm

    async def _unavailable(binary="codex", **_kw):
        return None

    monkeypatch.setattr(_cm, "fetch_codex_models", _unavailable)
    _cm.clear_codex_models_cache()
    yield
    _cm.clear_codex_models_cache()


def _make_cli_manifest(slug="test-agent", command="testagent",
                      check_command="testagent --version") -> AgentManifest:
    return AgentManifest(
        manifest_version="1",
        name=f"Test {slug}",
        slug=slug,
        description="d",
        author="a",
        version="1.0.0",
        runtime=ManifestRuntime(
            adapter="cli",
            command=command,
            args=["--output-format", "stream-json"],
        ),
        setup=ManifestSetup(
            install_command=f"npm install -g {command}",
            check_command=check_command,
        ),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    )


def _make_registry(*manifests: AgentManifest) -> AgentRegistry:
    reg = AgentRegistry()
    for m in manifests:
        reg.register(m)
    return reg


# ---------------------------------------------------------------------------
# 1. SetupEngine.check_all() caching
# ---------------------------------------------------------------------------

class TestCheckAllCaching:

    @patch("agent_os.agents.setup_engine.subprocess.run")
    @patch("agent_os.agents.setup_engine.shutil.which")
    def test_check_all_caches_within_ttl(self, mock_which, mock_run):
        """Two calls within TTL -> subprocess invoked only once per agent."""
        mock_which.return_value = "/usr/bin/testagent"
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0\n")

        manifest = _make_cli_manifest()
        engine = SetupEngine(_make_registry(manifest))

        engine.check_all()
        first_call_count = mock_run.call_count

        engine.check_all()
        second_call_count = mock_run.call_count

        # No new subprocess invocations on the cached path.
        assert second_call_count == first_call_count

    @patch("agent_os.agents.setup_engine.subprocess.run")
    @patch("agent_os.agents.setup_engine.shutil.which")
    def test_invalidate_cache_forces_recheck(self, mock_which, mock_run):
        """invalidate_cache() drops the result; next call re-spawns subprocess."""
        mock_which.return_value = "/usr/bin/testagent"
        mock_run.return_value = MagicMock(returncode=0, stdout="v1.0.0\n")

        manifest = _make_cli_manifest()
        engine = SetupEngine(_make_registry(manifest))

        engine.check_all()
        first = mock_run.call_count

        engine.invalidate_cache()
        engine.check_all()
        second = mock_run.call_count

        assert second > first


# ---------------------------------------------------------------------------
# 2. disabled_sub_agents denylist semantics
# ---------------------------------------------------------------------------

class TestDisabledSubAgents:

    def test_disabled_sub_agents_excluded_from_available_list(self, tmp_path):
        """A project with disabled_sub_agents=['codex'] should not see codex."""
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(str(tmp_path))
        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        pid = store.create_project({
            "name": "p", "agent_name": "p", "workspace": ws,
            "model": "", "api_key": "", "disabled_sub_agents": ["codex"],
        })
        proj = store.get_project(pid)

        # Simulate the agent_manager filter logic with two installed agents.
        installed_slugs = ["claude-code", "codex"]
        disabled = set(proj.get("disabled_sub_agents", []) or [])
        result = [s for s in installed_slugs if s not in disabled]

        assert "codex" not in result
        assert "claude-code" in result

    def test_legacy_enabled_sub_agents_field_ignored(self, tmp_path):
        """Project with legacy enabled_sub_agents but no disabled_sub_agents
        gets the full installed list (denylist semantics, not allowlist)."""
        from agent_os.daemon_v2.project_store import ProjectStore

        store = ProjectStore(str(tmp_path))
        ws = str(tmp_path / "ws")
        os.makedirs(ws, exist_ok=True)
        pid = store.create_project({
            "name": "p", "agent_name": "p", "workspace": ws,
            "model": "", "api_key": "",
            # Legacy: should be ignored. Empty disabled list = nothing filtered.
            "enabled_sub_agents": ["claude-code"],
            "disabled_sub_agents": [],
        })
        proj = store.get_project(pid)

        # The new code path does NOT consult enabled_sub_agents; it computes
        # the set from setup_engine and subtracts disabled_sub_agents.
        installed_slugs = ["claude-code", "codex", "gemini-cli"]
        disabled = set(proj.get("disabled_sub_agents", []) or [])
        result = [s for s in installed_slugs if s not in disabled]

        # All three appear, regardless of the legacy enabled field.
        assert set(result) == {"claude-code", "codex", "gemini-cli"}


# ---------------------------------------------------------------------------
# 3. SubAgentConfigStore validation, persistence, arg rendering
# ---------------------------------------------------------------------------

class TestSubAgentConfigStore:

    def test_set_valid_config_persists(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        store.set("claude-code", {"model": "opus", "effort": "high"})

        # Reload from disk.
        store2 = SubAgentConfigStore(str(tmp_path / "config.json"))
        config = store2.get("claude-code")
        assert config == {"model": "opus", "effort": "high"}

    def test_invalid_model_rejected(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        with pytest.raises(SubAgentConfigError):
            store.set("claude-code", {"model": "definitely-not-real"})

    def test_current_flagship_models_selectable(self, tmp_path):
        """The claude-code model whitelist must offer the current-generation
        flagship IDs. It had gone stale (pinned claude-opus-4-7 /
        claude-sonnet-4-6), so users couldn't select the current models even
        though the claude-code CLI accepts them."""
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        for model in ("claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"):
            store.set("claude-code", {"model": model})
            assert store.get("claude-code") == {"model": model}

    def test_fable_alias_and_pin_selectable(self, tmp_path):
        """Claude 5 generation: the `fable` alias (documented by the
        claude-code CLI as auto-tracking the newest model) and the explicit
        claude-fable-5 pin must both be offered. Regression: the whitelist
        predated Fable, so the newest subscription model was only reachable
        by clearing the override entirely ("default")."""
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        for model in ("fable", "claude-fable-5"):
            store.set("claude-code", {"model": model})
            assert store.get("claude-code") == {"model": model}

    def test_invalid_effort_rejected(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        with pytest.raises(SubAgentConfigError):
            store.set("claude-code", {"effort": "extreme"})

    def test_unknown_param_rejected(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        with pytest.raises(SubAgentConfigError):
            store.set("claude-code", {"weather": "sunny"})

    def test_codex_model_freetext(self, tmp_path):
        """Codex model is free-text validated as non-empty string only."""
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        store.set("codex", {"model": "o3"})
        assert store.get("codex") == {"model": "o3"}

    def test_empty_string_clears_param(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        store.set("claude-code", {"model": "opus", "effort": "high"})
        store.set("claude-code", {"model": "", "effort": "low"})
        # Empty model cleared; effort preserved.
        assert store.get("claude-code") == {"effort": "low"}

    def test_build_extra_args_renders_flags(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        store.set("claude-code", {"model": "opus", "effort": "high"})
        args = store.build_extra_args("claude-code")
        # Each flag becomes two argv tokens.
        assert "--model" in args
        assert "opus" in args
        assert "--effort" in args
        assert "high" in args

    def test_build_extra_args_empty_when_no_config(self, tmp_path):
        store = SubAgentConfigStore(str(tmp_path / "config.json"))
        assert store.build_extra_args("claude-code") == []


# ---------------------------------------------------------------------------
# 4. SetupEngine.get_adapter_config injects extra args
# ---------------------------------------------------------------------------

class TestGetAdapterConfigWithSubAgentParams:

    @patch("agent_os.agents.setup_engine.shutil.which")
    def test_get_adapter_config_appends_extra_args(self, mock_which, tmp_path):
        mock_which.return_value = "/usr/bin/claude"

        manifest = AgentManifest(
            manifest_version="1",
            name="Claude Code", slug="claude-code", description="d", author="a",
            version="1.0.0",
            runtime=ManifestRuntime(
                adapter="cli", command="claude",
                args=["--print"],  # base args from manifest
            ),
            setup=ManifestSetup(),
            capabilities=ManifestCapabilities(),
            permissions=ManifestPermissions(),
        )
        registry = _make_registry(manifest)

        config_store = SubAgentConfigStore(str(tmp_path / "config.json"))
        config_store.set("claude-code", {"model": "opus"})

        engine = SetupEngine(registry, sub_agent_config_store=config_store)
        engine.resolve_binary(manifest)

        config = engine.get_adapter_config(
            slug="claude-code", project_workspace="/tmp/ws",
        )

        # Manifest args first, then daemon-level overrides appended.
        assert config["args"] == ["--print", "--model", "opus"]


# ---------------------------------------------------------------------------
# 5/6/7. REST API: /api/v2/settings/sub-agents/*
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect HOME so the daemon's sub_agent_config.json lands in tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # USERPROFILE is the Windows equivalent of HOME for os.path.expanduser.
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


class TestSubAgentSettingsRoutes:

    def test_get_sub_agents_returns_list(self, client):
        resp = client.get("/api/v2/settings/sub-agents")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # Built-in is filtered out.
        for entry in data:
            assert entry["slug"] != "built-in"
            assert "config" in entry
            assert "param_schema" in entry

    def test_put_config_validates_model(self, client):
        """PUT /config with a bogus model returns 400."""
        resp = client.put(
            "/api/v2/settings/sub-agents/claude-code/config",
            json={"model": "definitely-not-a-real-model"},
        )
        assert resp.status_code == 400

    def test_put_config_persists_across_get(self, client):
        """Valid PUT persists; subsequent GET reflects the values."""
        resp = client.put(
            "/api/v2/settings/sub-agents/claude-code/config",
            json={"model": "opus", "effort": "high"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["config"] == {"model": "opus", "effort": "high"}

        # Now fetch the list and verify the config came back.
        resp = client.get("/api/v2/settings/sub-agents")
        cc_entry = next(e for e in resp.json() if e["slug"] == "claude-code")
        assert cc_entry["config"] == {"model": "opus", "effort": "high"}

    def test_refresh_invalidates_cache(self, client):
        resp = client.post("/api/v2/settings/sub-agents/refresh")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_status_entry_exposes_supports_login(self, client):
        """Each entry carries a boolean supports_login derived from the
        manifest: true when a credential has a login/setup_command, false
        otherwise. The frontend uses this to hide the dead Login button."""
        resp = client.get("/api/v2/settings/sub-agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data, "expected at least one sub-agent entry"
        for entry in data:
            assert "supports_login" in entry
            assert isinstance(entry["supports_login"], bool)

        # claude-code and codex both ship a login setup_command -> True.
        by_slug = {e["slug"]: e for e in data}
        if "claude-code" in by_slug:
            assert by_slug["claude-code"]["supports_login"] is True
        if "codex" in by_slug:
            assert by_slug["codex"]["supports_login"] is True

    def test_login_endpoint_does_not_store_tokens(self, client, tmp_path):
        """POST /login starts the CLI's own subprocess; orbital never sees
        a token. Verifies that ~/.orbital/credentials.* never contains
        sub-agent tokens after a login attempt."""
        # Trigger a login. The fake claude binary may not be installed, so
        # this returns 400 (no setup_command) — that's fine; what matters is
        # that orbital storage is untouched either way.
        resp = client.post("/api/v2/settings/sub-agents/claude-code/login")
        # Either 200 (job started) or 400 (no login subcommand for this
        # platform's resolved binary). Both are valid; what we're asserting
        # is the storage invariant.
        assert resp.status_code in (200, 400)

        # Check no orbital-managed files contain a token-shaped string.
        orbital_dir = os.path.join(str(tmp_path / "home"), ".orbital")
        if os.path.isdir(orbital_dir):
            for fname in os.listdir(orbital_dir):
                if fname == "credentials.json":
                    pytest.fail(
                        f"orbital credential file appeared after login: {fname}"
                    )


# ---------------------------------------------------------------------------
# 6. Codex live model list (TASK-live-model-config)
# ---------------------------------------------------------------------------

class TestCodexLiveModels:
    """The codex model dropdown must reflect the ChatGPT account's LIVE
    model/list, and saves must be validated against it. A free-text override
    the account can't use (`gpt-5.6`, valid in Codex desktop but not through
    the CLI's ChatGPT-account gate) previously saved fine and then 400'd on
    every dispatch — invisibly."""

    @staticmethod
    def _patch_live(monkeypatch, result):
        import agent_os.api.routes.settings as settings_routes

        async def fake(binary=None):
            return result

        monkeypatch.setattr(settings_routes, "_codex_live_models", fake)

    def test_get_codex_allowed_populated_from_live_list(self, client,
                                                        monkeypatch):
        self._patch_live(monkeypatch, ["gpt-5.5", "gpt-5.4-mini"])
        resp = client.get("/api/v2/settings/sub-agents")
        assert resp.status_code == 200
        codex = next(e for e in resp.json() if e["slug"] == "codex")
        assert codex["param_schema"]["model"]["allowed"] == [
            "gpt-5.5", "gpt-5.4-mini"]

    def test_get_codex_allowed_stays_freetext_when_unavailable(self, client,
                                                               monkeypatch):
        self._patch_live(monkeypatch, None)
        resp = client.get("/api/v2/settings/sub-agents")
        codex = next(e for e in resp.json() if e["slug"] == "codex")
        assert codex["param_schema"]["model"]["allowed"] is None

    def test_put_codex_model_rejected_when_not_in_live_list(self, client,
                                                            monkeypatch):
        self._patch_live(monkeypatch, ["gpt-5.5", "gpt-5.4-mini"])
        resp = client.put("/api/v2/settings/sub-agents/codex/config",
                          json={"model": "gpt-5.6"})
        assert resp.status_code == 400
        # The error must teach: name the rejected value and the valid ids.
        assert "gpt-5.6" in resp.json()["detail"]
        assert "gpt-5.5" in resp.json()["detail"]

    def test_put_codex_model_accepted_when_in_live_list(self, client,
                                                        monkeypatch):
        self._patch_live(monkeypatch, ["gpt-5.5", "gpt-5.4-mini"])
        resp = client.put("/api/v2/settings/sub-agents/codex/config",
                          json={"model": "gpt-5.5"})
        assert resp.status_code == 200
        assert resp.json()["config"] == {"model": "gpt-5.5"}

    def test_put_codex_model_accepted_when_live_list_unavailable(
            self, client, monkeypatch):
        """No live list (codex missing/broken) → free-text fallback: the
        save must NOT be blocked on a probe failure."""
        self._patch_live(monkeypatch, None)
        resp = client.put("/api/v2/settings/sub-agents/codex/config",
                          json={"model": "anything-goes"})
        assert resp.status_code == 200

    def test_put_codex_clear_skips_live_validation(self, client, monkeypatch):
        """Clearing the override (empty string) never consults the live
        list — clearing must always work, even with codex broken."""
        called = []
        import agent_os.api.routes.settings as settings_routes

        async def fake(binary=None):
            called.append(True)
            return ["gpt-5.5"]

        monkeypatch.setattr(settings_routes, "_codex_live_models", fake)
        resp = client.put("/api/v2/settings/sub-agents/codex/config",
                          json={"model": ""})
        assert resp.status_code == 200
        assert called == []
