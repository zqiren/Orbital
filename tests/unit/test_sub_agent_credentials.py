# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for sub-agent API credential storage (SubAgentCredentialStore).

Orbital normally delegates sub-agent auth to the CLI's own credential store
(``~/.claude/``, ``~/.codex/``). Agents with no such store — dsh — are the
documented exception: their key lives in a dedicated keychain service, is
injected into the spawn env by SetupEngine, and is never enumerated by the
credentials UI nor reachable through the agent's ``<secret:...>`` path.

Covers:
- Store round-trip against a real keyring backend (in-memory, hermetic).
- SetupEngine credential resolution: check_credentials + spawn env injection.
- ``credential_overrides`` precedence over the store.
- The key never lands in sub_agent_config_store or UserCredentialStore.
- ``POST``/``DELETE /api/v2/settings/sub-agents/{slug}/credential`` validation,
  masking, and the "copy my DeepSeek provider key" path.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import keyring
import keyring.backend
import keyring.core
import keyring.errors
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
from agent_os.daemon_v2.credential_store import (
    SubAgentCredentialStore,
    UserCredentialStore,
)

# The slug used for route tests: gemini-cli is the shipped manifest that
# declares a plain ``secret`` credential (dsh lands in a later task and has
# the same shape).
ROUTE_SLUG = "gemini-cli"
ROUTE_KEY = "GEMINI_API_KEY"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _InMemoryKeyring(keyring.backend.KeyringBackend):
    """Hermetic keyring backend so tests never touch the OS keychain.

    ``PYTHON_KEYRING_BACKEND=in-memory`` (the documented test env) does not
    resolve to a real backend — it makes ``get_keyring()`` raise, which the
    store swallows into "no keyring". Installing this explicitly is what lets
    the round-trip actually be exercised.
    """

    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise keyring.errors.PasswordDeleteError(username)
        del self.store[(service, username)]


@pytest.fixture(autouse=True)
def memory_keyring():
    """Install the in-memory backend for every test in this module."""
    previous = keyring.core._keyring_backend
    fake = _InMemoryKeyring()
    keyring.set_keyring(fake)
    with patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", True):
        yield fake
    keyring.core._keyring_backend = previous


@pytest.fixture(autouse=True)
def _no_real_codex_probe(monkeypatch):
    """Keep the route tests hermetic — never spawn the real codex binary."""
    import agent_os.agent.transports.codex_models as _cm

    async def _unavailable(binary="codex", **_kw):
        return None

    monkeypatch.setattr(_cm, "fetch_codex_models", _unavailable)
    _cm.clear_codex_models_cache()
    yield
    _cm.clear_codex_models_cache()


def _dsh_like_manifest(slug="dsh", command="dsh-acp-demo") -> AgentManifest:
    """A manifest shaped like the dsh one: required secret + optional secret."""
    return AgentManifest(
        manifest_version="1",
        name="DeepSeek Harness",
        slug=slug,
        description="d",
        author="a",
        version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", command=command, args=[]),
        setup=ManifestSetup(
            credentials=[
                ManifestCredential(
                    key="DEEPSEEK_API_KEY",
                    label="DeepSeek API Key",
                    type="secret",
                    required=True,
                    env_var="DEEPSEEK_API_KEY",
                ),
                ManifestCredential(
                    key="DEEPSEEK_BASE_URL",
                    label="DeepSeek API Base URL",
                    type="secret",
                    required=False,
                    env_var="DEEPSEEK_BASE_URL",
                ),
            ],
        ),
        capabilities=ManifestCapabilities(),
        permissions=ManifestPermissions(),
    )


def _registry(*manifests: AgentManifest) -> AgentRegistry:
    reg = AgentRegistry()
    for m in manifests:
        reg.register(m)
    return reg


@pytest.fixture(autouse=True)
def _clean_deepseek_env(monkeypatch):
    """The store, not the ambient environment, must satisfy the credential."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)


# ---------------------------------------------------------------------------
# 1. The store itself
# ---------------------------------------------------------------------------

class TestSubAgentCredentialStore:

    def test_round_trip(self):
        store = SubAgentCredentialStore()
        assert store.get("DEEPSEEK_API_KEY") is None

        store.set("DEEPSEEK_API_KEY", "sk-round-trip-0001")
        assert store.get("DEEPSEEK_API_KEY") == "sk-round-trip-0001"

        store.delete("DEEPSEEK_API_KEY")
        assert store.get("DEEPSEEK_API_KEY") is None

    def test_entries_are_keyed_by_credential_key_in_a_dedicated_service(
            self, memory_keyring):
        """Service name is the isolation boundary: not ``agent-os`` (the LLM
        key) and not ``agent-os-creds`` (agent-reachable website creds)."""
        SubAgentCredentialStore().set("DEEPSEEK_API_KEY", "sk-svc-0001")

        assert ("agent-os-subagent-creds", "DEEPSEEK_API_KEY") in memory_keyring.store
        services = {service for service, _ in memory_keyring.store}
        assert services == {"agent-os-subagent-creds"}

    def test_set_rejects_empty_value(self):
        store = SubAgentCredentialStore()
        with pytest.raises(ValueError):
            store.set("DEEPSEEK_API_KEY", "   ")

    def test_delete_of_absent_key_is_a_noop(self):
        SubAgentCredentialStore().delete("NEVER_STORED")  # must not raise

    def test_get_returns_none_without_a_keyring(self):
        store = SubAgentCredentialStore()
        store.set("DEEPSEEK_API_KEY", "sk-no-keyring")
        with patch("agent_os.daemon_v2.credential_store._KEYRING_AVAILABLE", False):
            assert store.get("DEEPSEEK_API_KEY") is None

    def test_is_not_reachable_through_the_user_credential_store(self, tmp_path):
        """UserCredentialStore is enumerated by GET /credentials and readable
        by the agent's <secret:name.field> substitution — sub-agent keys must
        not be visible there."""
        SubAgentCredentialStore().set("DEEPSEEK_API_KEY", "sk-not-here-0001")

        user_store = UserCredentialStore(meta_path=str(tmp_path / "meta.json"))
        assert user_store.list_all() == []
        assert user_store.get_value("DEEPSEEK_API_KEY", "value") is None


# ---------------------------------------------------------------------------
# 2. SetupEngine resolution
# ---------------------------------------------------------------------------

class TestSetupEngineResolution:

    def test_check_credentials_satisfied_by_the_store(self):
        manifest = _dsh_like_manifest()
        store = SubAgentCredentialStore()
        engine = SetupEngine(_registry(manifest), credential_store=store)

        ok, missing = engine.check_credentials(manifest)
        assert ok is False
        assert missing == ["DEEPSEEK_API_KEY"]

        store.set("DEEPSEEK_API_KEY", "sk-configured-0001")

        ok, missing = engine.check_credentials(manifest)
        assert ok is True
        assert missing == []

    @patch("agent_os.agents.setup_engine.shutil.which",
           return_value="/opt/dsh/dsh-acp-demo")
    def test_spawn_env_carries_the_stored_key(self, _mock_which):
        manifest = _dsh_like_manifest()
        store = SubAgentCredentialStore()
        store.set("DEEPSEEK_API_KEY", "sk-spawn-env-0001")
        engine = SetupEngine(_registry(manifest), credential_store=store)
        engine.resolve_binary(manifest)

        config = engine.get_adapter_config(slug="dsh", project_workspace="/tmp/ws")

        assert config["env"]["DEEPSEEK_API_KEY"] == "sk-spawn-env-0001"
        # The optional credential was never stored — no empty entry for it.
        assert "DEEPSEEK_BASE_URL" not in config["env"]

    @patch("agent_os.agents.setup_engine.shutil.which",
           return_value="/opt/dsh/dsh-acp-demo")
    def test_credential_overrides_beat_the_store(self, _mock_which):
        """Unit-level contract only — ``credential_overrides`` is never passed
        in production (sub_agent_manager.py never supplies it). Do not build
        features on this path."""
        manifest = _dsh_like_manifest()
        store = SubAgentCredentialStore()
        store.set("DEEPSEEK_API_KEY", "sk-from-store")
        engine = SetupEngine(_registry(manifest), credential_store=store)
        engine.resolve_binary(manifest)

        config = engine.get_adapter_config(
            slug="dsh",
            project_workspace="/tmp/ws",
            credential_overrides={"DEEPSEEK_API_KEY": "sk-from-override"},
        )

        assert config["env"]["DEEPSEEK_API_KEY"] == "sk-from-override"

    def test_app_wires_the_store_into_the_setup_engine(self, client):
        """The constructor param has existed unwired since it was written —
        the point of this task is that a conforming store is actually passed."""
        import agent_os.api.routes.settings as settings_routes

        wired = settings_routes._setup_engine._credential_store
        assert isinstance(wired, SubAgentCredentialStore)


# ---------------------------------------------------------------------------
# 3. Routes: POST / DELETE /settings/sub-agents/{slug}/credential
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)

    from agent_os.api.app import create_app
    app = create_app(data_dir=str(tmp_path / "data"))
    with TestClient(app) as c:
        yield c


def _post(client, slug=ROUTE_SLUG, **body):
    return client.post(f"/api/v2/settings/sub-agents/{slug}/credential", json=body)


class TestCredentialRouteValidation:

    def test_unknown_slug_404(self, client):
        resp = _post(client, slug="not-an-agent", key=ROUTE_KEY, value="sk-x")
        assert resp.status_code == 404

    def test_unknown_key_400(self, client):
        resp = _post(client, key="TOTALLY_MADE_UP", value="sk-x")
        assert resp.status_code == 400
        assert ROUTE_KEY in resp.json()["detail"]

    def test_neither_value_nor_flag_400(self, client):
        assert _post(client, key=ROUTE_KEY).status_code == 400

    def test_both_value_and_flag_400(self, client):
        resp = _post(client, key=ROUTE_KEY, value="sk-x",
                     use_llm_provider_key=True)
        assert resp.status_code == 400

    def test_blank_value_400(self, client):
        assert _post(client, key=ROUTE_KEY, value="   ").status_code == 400


class TestCredentialRouteStorage:

    def test_stores_and_returns_only_a_mask(self, client):
        raw = "sk-abcdefgh12345678"
        resp = _post(client, key=ROUTE_KEY, value=raw)
        assert resp.status_code == 200

        body = resp.json()
        assert body["key"] == ROUTE_KEY
        assert body["set"] is True
        assert body["masked"] == "sk-a...5678"
        # The hard invariant of the whole settings surface.
        assert raw not in resp.text

        assert SubAgentCredentialStore().get(ROUTE_KEY) == raw

    def test_short_value_masks_without_leaking_it(self, client):
        resp = _post(client, key=ROUTE_KEY, value="short")
        assert resp.status_code == 200
        assert "short" not in resp.text

    def test_post_invalidates_the_setup_cache(self, client):
        import agent_os.api.routes.settings as settings_routes

        engine = settings_routes._setup_engine
        engine._check_all_cache = ([], time.monotonic() + 300)

        assert _post(client, key=ROUTE_KEY, value="sk-invalidate-0001").status_code == 200
        assert engine._check_all_cache is None

    def test_delete_removes_and_invalidates(self, client):
        import agent_os.api.routes.settings as settings_routes

        _post(client, key=ROUTE_KEY, value="sk-delete-me-0001")
        engine = settings_routes._setup_engine
        engine._check_all_cache = ([], time.monotonic() + 300)

        resp = client.delete(
            f"/api/v2/settings/sub-agents/{ROUTE_SLUG}/credential/{ROUTE_KEY}")
        assert resp.status_code == 200
        assert resp.json()["set"] is False
        assert SubAgentCredentialStore().get(ROUTE_KEY) is None
        assert engine._check_all_cache is None

    def test_delete_validates_slug_and_key(self, client):
        assert client.delete(
            f"/api/v2/settings/sub-agents/nope/credential/{ROUTE_KEY}"
        ).status_code == 404
        assert client.delete(
            f"/api/v2/settings/sub-agents/{ROUTE_SLUG}/credential/MADE_UP"
        ).status_code == 400

    def test_key_never_lands_in_plaintext_orbital_storage(self, client, tmp_path):
        """sub_agent_config.json is world-readable plaintext; the credential
        metadata file is enumerated by the credentials UI. Neither may ever
        contain the raw key."""
        raw = "sk-plaintext-canary-0001"
        assert _post(client, key=ROUTE_KEY, value=raw).status_code == 200
        # Touch the config store too, so its file definitely exists on disk.
        client.put(f"/api/v2/settings/sub-agents/{ROUTE_SLUG}/config",
                   json={"model": "gemini-2.5-pro"})

        offenders = []
        scanned = 0
        for root in (tmp_path / "data", tmp_path / "home"):
            for dirpath, _dirnames, filenames in os.walk(root):
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    except OSError:
                        continue
                    scanned += 1
                    if raw in content:
                        offenders.append(path)
        assert scanned > 0, "scanned nothing — the sweep would pass vacuously"
        assert offenders == [], f"raw key found in {offenders}"

        config_path = os.path.join(
            str(tmp_path / "home"), ".orbital", "sub_agent_config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                assert ROUTE_KEY not in json.dumps(json.load(f))


class TestUseLlmProviderKey:
    """Server-side copy of the global LLM key. The client never receives key
    text — that is why this is a flag and not a prefill."""

    @staticmethod
    def _set_provider(client, provider):
        resp = client.put("/api/v2/settings", json={"llm_provider": provider})
        assert resp.status_code == 200

    def test_409_when_provider_is_not_deepseek(self, client, monkeypatch):
        monkeypatch.setenv("AGENT_OS_API_KEY", "sk-provider-1234567890")
        self._set_provider(client, "custom")

        resp = _post(client, key=ROUTE_KEY, use_llm_provider_key=True)
        assert resp.status_code == 409
        assert SubAgentCredentialStore().get(ROUTE_KEY) is None

    def test_409_when_no_provider_key_is_set(self, client, monkeypatch):
        monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)
        self._set_provider(client, "deepseek")

        resp = _post(client, key=ROUTE_KEY, use_llm_provider_key=True)
        assert resp.status_code == 409
        assert SubAgentCredentialStore().get(ROUTE_KEY) is None

    def test_copies_the_provider_key_without_returning_it(self, client,
                                                          monkeypatch):
        raw = "sk-provider-1234567890"
        monkeypatch.setenv("AGENT_OS_API_KEY", raw)
        self._set_provider(client, "deepseek")

        resp = _post(client, key=ROUTE_KEY, use_llm_provider_key=True)
        assert resp.status_code == 200
        assert resp.json()["set"] is True
        assert raw not in resp.text
        assert SubAgentCredentialStore().get(ROUTE_KEY) == raw
