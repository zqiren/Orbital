# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 136 — the Settings readers serve the last sweep at once.

Global Settings → Sub-agents waited 9–10 s for a cold sweep plus the live
model probes run back to back; Project Settings waited on the same sweep
twice. These tests pin: stale-then-update (the read returns the last result,
one background sweep refreshes it, and ``sub_agents.updated`` goes out only
when something changed), Refresh still a blocking fresh sweep, model probes
off the critical path, one request for Project Settings, and the per-agent
re-probe after a single-agent change.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from types import SimpleNamespace

import pytest

from agent_os.agents.manifest import (
    AgentManifest,
    ManifestCredential,
    ManifestPermissions,
    ManifestRuntime,
    ManifestSetup,
)
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.agents.setup_types import AgentSetupStatus
from agent_os.api.routes import agents_v2 as agents_routes
from agent_os.api.routes import settings as routes


def _manifest(slug, *, command=None, auto_detect=None, credentials=None) -> AgentManifest:
    command = command or f"orbital-test-no-such-{slug}"
    return AgentManifest(
        manifest_version="1", name=slug.title(), slug=slug, description="",
        author="tester", version="1.0.0",
        runtime=ManifestRuntime(adapter="cli", command=command, args=[],
                                interactive=False, output_format="stream-json"),
        setup=ManifestSetup(dependencies=[], install_command=None,
                            check_command=f"{command} --version",
                            auto_detect=auto_detect or {},
                            credentials=credentials or []),
        permissions=ManifestPermissions(network_domains=[]),
    )


def _status(slug, version="1", installed=True, **extra) -> AgentSetupStatus:
    return AgentSetupStatus(
        slug=slug, name=slug.title(), installed=installed,
        binary_path=f"/bin/{slug}" if installed else None, version=version,
        dependencies_met=True, missing_dependencies=[],
        credentials_configured=True, missing_credentials=[], **extra)


class _Ws:
    def __init__(self):
        self.events: list[dict] = []

    def broadcast_global(self, payload: dict) -> None:
        self.events.append(payload)

    def updates(self):
        return [e for e in self.events if e.get("type") == "sub_agents.updated"]


class _Agents:
    """A real engine whose per-agent probe is a counted stub."""

    def __init__(self, tmp_path, slugs=("alpha", "beta")):
        self.registry = AgentRegistry()
        for slug in slugs:
            self.registry.register(_manifest(slug))
        self.engine = SetupEngine(self.registry, data_dir=str(tmp_path / "data"))
        self.version = "1"
        self.delay = 0.0
        self.installed = {s: True for s in slugs}
        self.probes: list[str] = []
        self.threads: list[str] = []
        self._lock = threading.Lock()
        self.engine.check_agent = self._check

    def _check(self, slug):
        version = self.version
        with self._lock:
            self.probes.append(slug)
        time.sleep(self.delay)
        return _status(slug, version, installed=self.installed[slug])

    def expire(self):
        results, _ = self.engine._check_all_cache
        self.engine._check_all_cache = (results, time.monotonic() - 1)


@pytest.fixture
def env(monkeypatch, tmp_path):
    agents = _Agents(tmp_path)
    ws = _Ws()
    monkeypatch.setattr(routes, "_setup_engine", agents.engine)
    monkeypatch.setattr(routes, "_ws_manager", ws)
    monkeypatch.setattr(routes, "_sub_agent_config_store", None)
    monkeypatch.setattr(routes, "_installer", None)
    monkeypatch.setattr(agents_routes, "_setup_engine", agents.engine)

    async def no_models(*_a, **_kw):
        return None

    monkeypatch.setattr(routes, "_codex_live_models", no_models)
    monkeypatch.setattr(routes, "_claude_live_models", no_models)
    monkeypatch.setattr(routes, "_acp_live_options", no_models)
    routes._late_model_digests.clear()
    return SimpleNamespace(agents=agents, ws=ws, engine=agents.engine)


async def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


# ---------------------------------------------------------------------------
# A.1 / A.2 stale-then-update
# ---------------------------------------------------------------------------

def test_list_serves_the_last_sweep_at_once_and_updates_in_the_background(env):
    async def scenario():
        routes.start_sub_agent_status_updates(env.engine, env.ws)
        await _wait_for(lambda: env.engine._check_all_cache is not None)
        env.agents.version = "2"
        env.agents.delay = 0.4
        env.agents.expire()
        started = time.monotonic()
        stale = await routes.list_sub_agent_settings()
        elapsed = time.monotonic() - started
        assert elapsed < 0.3, f"read waited {elapsed:.2f}s for the sweep"
        assert {e["version"] for e in stale} == {"1"}
        assert await _wait_for(lambda: env.ws.updates())
        assert env.ws.updates() == [{"type": "sub_agents.updated", "reason": "status"}]
        fresh = await routes.list_sub_agent_settings()
        assert {e["version"] for e in fresh} == {"2"}

    asyncio.run(scenario())


def test_an_unchanged_background_sweep_says_nothing(env):
    async def scenario():
        routes.start_sub_agent_status_updates(env.engine, env.ws)
        await _wait_for(lambda: env.engine._check_all_cache is not None)
        probes = len(env.agents.probes)
        env.agents.expire()
        await routes.list_sub_agent_settings()
        await _wait_for(lambda: len(env.agents.probes) >= probes + 2)
        await asyncio.sleep(0.1)
        assert env.ws.updates() == []

    asyncio.run(scenario())


def test_startup_warm_up_fills_the_cache(env):
    async def scenario():
        assert env.engine._check_all_cache is None
        routes.start_sub_agent_status_updates(env.engine, env.ws)
        assert await _wait_for(lambda: env.engine._check_all_cache is not None)

    asyncio.run(scenario())


def test_entries_are_built_off_the_event_loop(env, monkeypatch):
    """The keychain reads behind optional credentials ran on the loop."""
    threads = []
    original = routes._build_sub_agent_status_entry

    def recording(status):
        threads.append(threading.current_thread() is threading.main_thread())
        return original(status)

    monkeypatch.setattr(routes, "_build_sub_agent_status_entry", recording)
    asyncio.run(routes.list_sub_agent_settings())
    assert threads and not any(threads)


# ---------------------------------------------------------------------------
# A.4 Refresh stays a real re-probe
# ---------------------------------------------------------------------------

def test_refresh_is_a_blocking_fresh_sweep_and_asks_versions_again(monkeypatch, tmp_path):
    """A binary rewritten in place with the same mtime and size: only
    Refresh, which also drops the version cache, reports the new version."""
    binary = tmp_path / "gamma"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    registry = AgentRegistry()
    registry.register(_manifest("gamma", auto_detect={
        os_name: [str(binary)] for os_name in ("macos", "linux", "windows")}))
    engine = SetupEngine(registry, data_dir=str(tmp_path / "data"))
    answer = {"v": "gamma 1.0.0"}
    engine._run_check_command = lambda cmd: answer["v"]
    monkeypatch.setattr(routes, "_setup_engine", engine)
    monkeypatch.setattr(routes, "_sub_agent_config_store", None)
    monkeypatch.setattr(routes, "_installer", None)
    monkeypatch.setattr(routes, "_ws_manager", _Ws())
    import agent_os.platform.shell_path as shell_path
    monkeypatch.setattr(shell_path, "merge_shell_path_into_environ", lambda: None)

    first = asyncio.run(routes.list_sub_agent_settings())
    assert first[0]["version"] == "gamma 1.0.0"
    answer["v"] = "gamma 2.0.0"
    engine.invalidate_cache()
    assert asyncio.run(routes.list_sub_agent_settings())[0]["version"] == "gamma 1.0.0"
    refreshed = asyncio.run(routes.refresh_sub_agent_status())
    assert refreshed[0]["version"] == "gamma 2.0.0"


# ---------------------------------------------------------------------------
# C.1 / C.2 model probes off the critical path
# ---------------------------------------------------------------------------

def _model_env(monkeypatch, tmp_path, delay):
    """Real manifests (codex, claude-code) and config store; fake probes."""
    from agent_os.daemon_v2.sub_agent_config_store import SubAgentConfigStore
    registry = AgentRegistry()
    registry.load_directory(os.path.join(os.path.dirname(routes.__file__),
                                         "..", "..", "agents", "manifests"))
    engine = SetupEngine(registry, data_dir=str(tmp_path / "data"))
    engine.check_agent = lambda slug: _status(slug)
    # The real ACP binary lookups (`which` + a check command each) cost
    # ~0.4 s on a cold Windows sweep and are not what these tests time.
    engine._start_handshakes = lambda: None
    store = SubAgentConfigStore(str(tmp_path / "cfg.json"))
    engine.set_sub_agent_config_store(store)
    ws = _Ws()
    monkeypatch.setattr(routes, "_setup_engine", engine)
    monkeypatch.setattr(routes, "_sub_agent_config_store", store)
    monkeypatch.setattr(routes, "_installer", None)
    monkeypatch.setattr(routes, "_ws_manager", ws)
    routes._late_model_digests.clear()
    started: dict[str, float] = {}
    cache: dict[str, object] = {}

    async def codex(binary=None):
        if "codex" in cache:
            return cache["codex"]
        started["codex"] = time.monotonic()
        await asyncio.sleep(delay)
        cache["codex"] = ["gpt-live-1"]
        return cache["codex"]

    async def claude(binary=None):
        if "claude" in cache:
            return cache["claude"]
        started["claude"] = time.monotonic()
        await asyncio.sleep(delay)
        cache["claude"] = [{"value": "claude-live-1", "label": "Live 1"}]
        return cache["claude"]

    async def acp(slug):
        return None

    monkeypatch.setattr(routes, "_codex_live_models", codex)
    monkeypatch.setattr(routes, "_claude_live_models", claude)
    monkeypatch.setattr(routes, "_acp_live_options", acp)
    return SimpleNamespace(ws=ws, started=started)


def _allowed(entries, slug):
    entry = next(e for e in entries if e["slug"] == slug)
    return (entry["param_schema"].get("model") or {}).get("allowed")


def test_cold_model_probes_do_not_hold_the_list(monkeypatch, tmp_path):
    m = _model_env(monkeypatch, tmp_path, delay=0.6)

    async def scenario():
        t0 = time.monotonic()
        entries = await routes.list_sub_agent_settings()
        assert time.monotonic() - t0 < 0.5
        assert _allowed(entries, "codex") != ["gpt-live-1"]
        assert await _wait_for(lambda: m.ws.updates(), timeout=3)
        assert m.ws.updates()[0] == {"type": "sub_agents.updated", "reason": "models"}
        # Both probes ran at once, not one after the other.
        assert abs(m.started["codex"] - m.started["claude"]) < 0.1
        entries = await routes.list_sub_agent_settings()
        assert _allowed(entries, "codex") == ["gpt-live-1"]
        assert "claude-live-1" in _allowed(entries, "claude-code")

    asyncio.run(scenario())


def test_warm_model_lists_land_in_the_same_response(monkeypatch, tmp_path):
    _model_env(monkeypatch, tmp_path, delay=0.0)
    entries = asyncio.run(routes.list_sub_agent_settings())
    assert _allowed(entries, "codex") == ["gpt-live-1"]
    assert "claude-live-1" in _allowed(entries, "claude-code")


def test_a_late_model_list_is_announced_once(monkeypatch, tmp_path):
    """A refetch that again misses the budget with the same answer must not
    announce again, or the page and the daemon would loop."""
    m = _model_env(monkeypatch, tmp_path, delay=0.3)

    async def scenario():
        await routes.list_sub_agent_settings()
        assert await _wait_for(lambda: m.ws.updates())
        await asyncio.sleep(0.4)
        count = len(m.ws.updates())
        slow = {"n": 0}

        async def codex_again(binary=None):
            slow["n"] += 1
            await asyncio.sleep(0.3)
            return ["gpt-live-1"]

        monkeypatch.setattr(routes, "_codex_live_models", codex_again)
        await routes.list_sub_agent_settings()
        await asyncio.sleep(0.5)
        assert slow["n"] == 1
        assert len(m.ws.updates()) == count

    asyncio.run(scenario())


def test_a_read_during_a_rerun_probe_serves_the_last_live_list(monkeypatch, tmp_path):
    """Windows smoke 2026-11-03, decision 4: the pin menu opened after the probe
    cache expired showed raw ids (`opus`) until the late list landed. A read
    that misses the budget now serves the list the probe gave last time, and
    the same answer arriving late announces nothing."""
    m = _model_env(monkeypatch, tmp_path, delay=0.3)

    async def scenario():
        await routes.list_sub_agent_settings()
        assert await _wait_for(lambda: m.ws.updates())
        await asyncio.sleep(0.4)
        count = len(m.ws.updates())

        async def codex_again(binary=None):
            await asyncio.sleep(0.3)
            return ["gpt-live-1"]

        async def claude_again(binary=None):
            await asyncio.sleep(0.3)
            return [{"value": "claude-live-1", "label": "Live 1"}]

        monkeypatch.setattr(routes, "_codex_live_models", codex_again)
        monkeypatch.setattr(routes, "_claude_live_models", claude_again)
        t0 = time.monotonic()
        entries = await routes.list_sub_agent_settings()
        assert time.monotonic() - t0 < 0.25
        assert _allowed(entries, "codex") == ["gpt-live-1"]
        assert "claude-live-1" in _allowed(entries, "claude-code")
        claude = next(e for e in entries if e["slug"] == "claude-code")
        assert claude["param_schema"]["model"]["labels"]["claude-live-1"] == "Live 1"
        await asyncio.sleep(0.5)
        assert len(m.ws.updates()) == count

    asyncio.run(scenario())


def test_a_model_saved_since_the_last_live_list_stays_listed(monkeypatch, tmp_path):
    m = _model_env(monkeypatch, tmp_path, delay=0.3)

    async def scenario():
        await routes.list_sub_agent_settings()
        assert await _wait_for(lambda: m.ws.updates())
        await asyncio.sleep(0.4)
        routes._sub_agent_config_store.set("codex", {"model": "gpt-saved"}, extra_allowed={"model": ["gpt-saved"]})

        async def codex_again(binary=None):
            await asyncio.sleep(0.3)
            return ["gpt-live-1"]

        monkeypatch.setattr(routes, "_codex_live_models", codex_again)
        entries = await routes.list_sub_agent_settings()
        assert _allowed(entries, "codex") == ["gpt-live-1", "gpt-saved"]

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# C.3 Project Settings asks once; nothing project-specific in the sweep
# ---------------------------------------------------------------------------

class _Projects:
    def __init__(self, projects):
        self._projects = projects

    def get_project(self, pid):
        return self._projects.get(pid)


def test_memory_listing_carries_ready_and_is_served_stale(env, monkeypatch, tmp_path):
    env.agents.installed["beta"] = True
    monkeypatch.setattr(agents_routes, "_project_store", _Projects({
        "p1": {"workspace": str(tmp_path / "ws1")}}))
    env.engine.check_all()
    env.agents.expire()
    env.agents.delay = 0.5
    started = time.monotonic()
    entries = asyncio.run(agents_routes.list_sub_agent_memory("p1"))
    assert time.monotonic() - started < 0.3
    assert [(e["agent_slug"], e["agent_name"], e["ready"]) for e in entries] == [
        ("alpha", "Alpha", True), ("beta", "Beta", True)]


def test_two_projects_get_their_own_lists_from_one_sweep(env):
    env.engine.check_all()
    probes = len(env.agents.probes)
    one = agents_routes._resolve_available_sub_agents({"disabled_sub_agents": ["alpha"]})
    two = agents_routes._resolve_available_sub_agents({"disabled_sub_agents": ["beta"]})
    assert [e["slug"] for e in one] == ["beta"]
    assert [e["slug"] for e in two] == ["alpha"]
    assert len(env.agents.probes) == probes  # served from the one sweep


def test_available_sub_agents_serve_a_stale_sweep(env):
    env.engine.check_all()
    env.agents.expire()
    env.agents.delay = 0.5
    started = time.monotonic()
    agents_routes._resolve_available_sub_agents({})
    assert time.monotonic() - started < 0.3


# ---------------------------------------------------------------------------
# D.2 pin validation never probes
# ---------------------------------------------------------------------------

def test_pin_validation_asks_the_registry_not_the_sweep(env):
    from fastapi import HTTPException
    agents_routes._validate_pin_slug("alpha")  # registered: fine, nothing probed
    assert env.agents.probes == []
    with pytest.raises(HTTPException):
        agents_routes._validate_pin_slug("nope")
    with pytest.raises(HTTPException):
        agents_routes._validate_pin_slug("built-in")
    assert env.agents.probes == []


# ---------------------------------------------------------------------------
# A.5 one agent re-probed after a single-agent change
# ---------------------------------------------------------------------------

class _CredStore:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)


def test_saving_a_credential_reprobes_that_agent_only(env, monkeypatch):
    cred = ManifestCredential(key="BETA_KEY", label="Beta key", type="secret",
                              required=True, env_var="BETA_KEY")
    env.agents.registry.register(_manifest("beta", credentials=[cred]))
    env.engine._credential_store = _CredStore()
    import agent_os.daemon_v2.quota_store as quota_store
    monkeypatch.setattr(quota_store, "get_quota_store", lambda: None)
    env.engine.check_all()
    expires = env.engine._check_all_cache[1]
    env.agents.probes.clear()
    env.agents.version = "2"
    asyncio.run(routes.set_sub_agent_credential(
        "beta", routes.SubAgentCredentialRequest(key="BETA_KEY", value="sk-test-value")))
    assert env.agents.probes == ["beta"]
    results, expires_after = env.engine._check_all_cache
    assert expires_after == expires
    assert [(r.slug, r.version) for r in results] == [("alpha", "1"), ("beta", "2")]
    asyncio.run(routes.delete_sub_agent_credential("beta", "BETA_KEY"))
    assert env.agents.probes == ["beta", "beta"]
