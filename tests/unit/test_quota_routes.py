# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 099 §3.2 surface: the additive ``quota`` field on
``GET /api/v2/agents/available``, the codex read-through refresh, the
``agent.quota_updated`` WS event, and the account-switch clear."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_os.api.routes import agents_v2
from agent_os.daemon_v2 import quota_store as qs
from agent_os.daemon_v2.quota_store import (
    QuotaStore,
    normalize_claude_rate_limit,
    normalize_codex_snapshot,
)

CODEX_SNAPSHOT = {
    "limitId": "codex",
    "primary": {"usedPercent": 18, "windowDurationMins": 300,
                "resetsAt": 1790517805},
    "secondary": {"usedPercent": 3, "windowDurationMins": 10080,
                  "resetsAt": 1791104605},
    "planType": "plus",
    "rateLimitReachedType": None,
}
CODEX_READ_RESULT = {"rateLimits": CODEX_SNAPSHOT,
                     "rateLimitsByLimitId": {"codex": CODEX_SNAPSHOT}}
CLAUDE_RAW = {"status": "allowed", "unifiedWindows": {
    "five_hour": {"utilization": 0.79, "resetsAt": 1790517600},
    "seven_day": {"utilization": 0.08, "resetsAt": 1790672400}}}


def _status(slug, *, ready=True, installed=True):
    return SimpleNamespace(
        slug=slug, name=slug.title(), installed=installed,
        binary_path=f"/bin/{slug}" if installed else None, version="1",
        dependencies_met=True, missing_dependencies=[],
        credentials_configured=ready,
        missing_credentials=[] if ready else ["auth"], setup_actions=[])


class _FakeEngine:
    def __init__(self, statuses):
        self.statuses = statuses

    def check_all(self, **_kw):
        return self.statuses


@pytest.fixture
def route_env(monkeypatch, tmp_path):
    store = QuotaStore(str(tmp_path / "quota.json"))
    prev = qs.get_quota_store()
    qs.install(store)
    monkeypatch.setattr(agents_v2, "_sub_agent_manager", None)
    probes = []

    async def fake_probe(binary="codex", *, timeout=10.0):
        probes.append(binary)
        return CODEX_READ_RESULT

    from agent_os.agent.transports import codex_transport
    monkeypatch.setattr(codex_transport, "fetch_codex_rate_limits", fake_probe)
    yield SimpleNamespace(store=store, probes=probes, monkeypatch=monkeypatch)
    qs.install(prev)


def _by_slug(entries):
    return {e["slug"]: e for e in entries}


class TestAvailableQuota:
    @pytest.mark.asyncio
    async def test_quota_is_additive_and_absent_without_snapshot(self, route_env):
        route_env.monkeypatch.setattr(agents_v2, "_setup_engine", _FakeEngine([
            _status("claude-code"), _status("cursor")]))
        route_env.store.record("claude-code",
                               normalize_claude_rate_limit(CLAUDE_RAW))
        out = _by_slug(await agents_v2.available_agents())
        assert out["claude-code"]["quota"]["agent"] == "claude-code"
        assert "quota" not in out["cursor"]
        # Every pre-existing field is still there.
        assert out["cursor"]["ready"] is True

    @pytest.mark.asyncio
    async def test_codex_read_through_uses_short_lived_probe(self, route_env):
        route_env.monkeypatch.setattr(agents_v2, "_setup_engine", _FakeEngine([
            _status("codex")]))
        first = _by_slug(await agents_v2.available_agents())
        # Non-blocking: the first response does not wait for the read.
        assert "quota" not in first["codex"]
        task = route_env.store._codex_task
        assert task is not None
        await task
        assert route_env.probes == ["/bin/codex"]
        second = _by_slug(await agents_v2.available_agents())
        assert second["codex"]["quota"]["plan"] == "plus"
        # Fresh now: no second probe.
        assert route_env.probes == ["/bin/codex"]

    @pytest.mark.asyncio
    async def test_not_ready_codex_is_never_probed(self, route_env):
        route_env.monkeypatch.setattr(agents_v2, "_setup_engine", _FakeEngine([
            _status("codex", ready=False)]))
        await agents_v2.available_agents()
        assert route_env.store._codex_task is None
        assert route_env.probes == []

    @pytest.mark.asyncio
    async def test_live_codex_handle_is_preferred(self, route_env):
        from agent_os.agent.transports.codex_transport import CodexTransport
        live = CodexTransport()
        live._alive = True
        reads = []

        async def fake_request(method, params=None, timeout=30.0):
            reads.append(method)
            return CODEX_READ_RESULT

        live._request = fake_request
        manager = SimpleNamespace(live_transports=lambda: iter([live]))
        route_env.monkeypatch.setattr(agents_v2, "_sub_agent_manager", manager)
        route_env.monkeypatch.setattr(agents_v2, "_setup_engine", _FakeEngine([
            _status("codex")]))
        await agents_v2.available_agents()
        await route_env.store._codex_task
        assert reads == ["account/rateLimits/read"]
        assert route_env.probes == []

    @pytest.mark.asyncio
    async def test_cached_list_still_reflects_the_current_store(self, route_env):
        route_env.monkeypatch.setattr(agents_v2, "_setup_engine", _FakeEngine([
            _status("claude-code")]))
        route_env.store.record("claude-code",
                               normalize_claude_rate_limit(CLAUDE_RAW))
        assert "quota" in _by_slug(await agents_v2.available_agents())["claude-code"]
        route_env.store.clear("claude-code")
        # The status list is cached for 60 s; the quota overlay is not.
        assert "quota" not in _by_slug(await agents_v2.available_agents())["claude-code"]


class TestLiveTransports:
    def test_yields_only_live_adapters_transports(self):
        from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
        mgr = SubAgentManager.__new__(SubAgentManager)
        alive = SimpleNamespace(_transport="T-alive", is_alive=lambda: True)
        dead = SimpleNamespace(_transport="T-dead", is_alive=lambda: False)
        bare = SimpleNamespace(is_alive=lambda: True)
        mgr._adapters = {("p", "s1"): {"codex": alive, "x": dead},
                         ("p", "s2"): {"y": bare}}
        assert list(mgr.live_transports()) == ["T-alive"]


class TestAppWiring:
    def test_create_app_installs_a_persistent_store_that_broadcasts(self, tmp_path):
        from agent_os.api.app import create_app
        prev = qs.get_quota_store()
        try:
            create_app(data_dir=str(tmp_path))
            store = qs.get_quota_store()
            assert store is not None and store is not prev
            sent = []
            agents_v2._ws_manager.broadcast_global = sent.append
            snap = normalize_codex_snapshot(CODEX_SNAPSHOT)
            store.record("codex", snap)
            assert sent == [{"type": "agent.quota_updated", "agent": "codex",
                             "snapshot": snap}]
            assert (tmp_path / "quota.json").is_file()
        finally:
            qs.install(prev)


class TestAccountSwitchClears:
    @pytest.mark.asyncio
    async def test_logout_clears_that_agents_snapshot(self, tmp_path, monkeypatch):
        from agent_os.api.routes import settings as settings_routes
        store = QuotaStore(str(tmp_path / "q.json"))
        prev = qs.get_quota_store()
        qs.install(store)
        try:
            store.record("codex", normalize_codex_snapshot(CODEX_SNAPSHOT))
            store.record("claude-code", normalize_claude_rate_limit(CLAUDE_RAW))
            settings_routes._forget_quota("codex")
            assert store.get("codex") is None
            assert store.get("claude-code") is not None
        finally:
            qs.install(prev)

    def test_forget_without_store_is_a_noop(self):
        from agent_os.api.routes import settings as settings_routes
        prev = qs.get_quota_store()
        qs.install(None)
        try:
            settings_routes._forget_quota("codex")
        finally:
            qs.install(prev)
