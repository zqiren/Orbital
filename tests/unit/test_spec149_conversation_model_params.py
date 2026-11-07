# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 149 — the pin menu's per-conversation model / thinking.

The user picks a worker's model and thinking level for ONE conversation; the
pick is stored on the session (beside its pin), layered over the Settings
value at spawn, and a live worker is rebuilt with resume on its next message
so the conversation continues on the new model.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.session import Session
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_config_store import (
    SubAgentConfigError,
    SubAgentConfigStore,
    resolve_params,
    validate_session_params,
)
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


# ---------------------------------------------------------------------------
# Store layering
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return SubAgentConfigStore(str(tmp_path / "sub_agent_config.json"))


def test_override_replaces_the_settings_value_with_one_flag(store):
    store.set("claude-code", {"model": "sonnet", "effort": "low"})
    args = store.build_extra_args("claude-code",
                                  overrides={"model": "opus", "effort": "high"})
    assert args == ["--model", "opus", "--effort", "high"]


def test_no_override_keeps_settings(store):
    store.set("claude-code", {"model": "sonnet"})
    assert store.build_extra_args("claude-code") == ["--model", "sonnet"]
    assert store.build_extra_args("claude-code", overrides={}) == ["--model", "sonnet"]


def test_partial_override_keeps_the_other_settings_param(store):
    store.set("claude-code", {"model": "sonnet", "effort": "low"})
    assert resolve_params("claude-code", store, {"effort": "max"}) == {
        "model": "sonnet", "effort": "max"}


def test_override_never_reaches_non_session_params(store):
    store.set("claude-code", {"permission-mode": "plan"})
    params = resolve_params("claude-code", store, {"permission-mode": "auto"})
    assert params["permission-mode"] == "plan"


def test_codex_effort_renders_as_config_override(store):
    assert store.build_extra_args("codex", overrides={"model": "gpt-5.6", "effort": "low"}) == [
        "-m", "gpt-5.6", "-c", "model_reasoning_effort=low"]


def test_codex_effort_levels_are_low_medium_high(store):
    assert store.schema_for("codex")["effort"]["allowed"] == ["low", "medium", "high"]


def test_dsh_effort_override_goes_to_acp_options(store):
    store.set("dsh", {"effort": "low"})
    assert store.build_acp_config_options("dsh", overrides={"effort": "max"}) == {
        "reasoning_effort": "max"}


def test_validate_session_params_accepts_model_and_effort_only():
    assert validate_session_params("claude-code", {"model": "opus", "effort": "high"}) == {
        "model": "opus", "effort": "high"}
    # Empty means "Default": dropped, not stored.
    assert validate_session_params("claude-code", {"model": "", "effort": "high"}) == {
        "effort": "high"}
    with pytest.raises(SubAgentConfigError):
        validate_session_params("claude-code", {"permission-mode": "plan"})
    with pytest.raises(SubAgentConfigError):
        validate_session_params("claude-code", {"effort": "ultra"})
    with pytest.raises(SubAgentConfigError):
        validate_session_params("codex", {"effort": "xhigh"})
    # An agent without thinking levels refuses one.
    with pytest.raises(SubAgentConfigError):
        validate_session_params("minimax-code", {"effort": "high"})
    # Free-text model ids pass as typed.
    assert validate_session_params(
        "minimax-code", {"model": "m:minimax:MiniMax-M3:v:thinking"}) == {
        "model": "m:minimax:MiniMax-M3:v:thinking"}


def test_validate_session_params_widened_by_live_list():
    assert validate_session_params(
        "claude-code", {"model": "claude-opus-6"},
        extra_allowed={"model": ["claude-opus-6"]}) == {"model": "claude-opus-6"}


# ---------------------------------------------------------------------------
# Session persistence (session_start meta, beside pinned_target)
# ---------------------------------------------------------------------------

def _materialized_session(tmp_path) -> Session:
    s = Session.new("sess", str(tmp_path / "ws"))
    s.append({"role": "user", "content": "hi", "source": "user"})
    return s


def test_session_params_round_trip_through_the_meta_line(tmp_path):
    s = _materialized_session(tmp_path)
    s.set_pinned_target("claude-code")
    s.set_sub_agent_params("claude-code", {"model": "opus", "effort": "high"})
    s.set_sub_agent_params("codex", {"effort": "low"})
    loaded = Session.load(s._filepath)
    assert loaded.sub_agent_params == {
        "claude-code": {"model": "opus", "effort": "high"},
        "codex": {"effort": "low"}}
    # Beside the pin, which is untouched.
    assert loaded.pinned_target == "claude-code"
    s.set_sub_agent_params("claude-code", {})
    assert Session.load(s._filepath).sub_agent_params == {"codex": {"effort": "low"}}


def test_session_without_params_reads_empty(tmp_path):
    s = _materialized_session(tmp_path)
    assert Session.load(s._filepath).sub_agent_params == {}


# ---------------------------------------------------------------------------
# Manager: storage, layering, rebuild decision
# ---------------------------------------------------------------------------

def _mgr_with_session(session):
    mgr = SubAgentManager(process_manager=MagicMock())
    mgr._session_resolver = lambda pid, sid: session
    return mgr


def test_manager_reads_and_writes_through_the_session(tmp_path):
    s = _materialized_session(tmp_path)
    mgr = _mgr_with_session(s)
    assert mgr.session_params("p", "sess", "claude-code") == {}
    assert mgr.set_session_params("p", "sess", "claude-code",
                                  {"model": "opus"}) is True
    assert Session.load(s._filepath).sub_agent_params == {
        "claude-code": {"model": "opus"}}
    # A fresh manager (daemon restart) reads it back from the file.
    assert _mgr_with_session(Session.load(s._filepath)).session_params(
        "p", "sess", "claude-code") == {"model": "opus"}


def test_pick_before_the_session_exists_is_kept_and_written_on_first_send(tmp_path):
    holder = {"session": None}
    mgr = SubAgentManager(process_manager=MagicMock())
    mgr._session_resolver = lambda pid, sid: holder["session"]
    assert mgr.set_session_params("p", "sess", "claude-code",
                                  {"effort": "high"}) is False
    assert mgr.session_params("p", "sess", "claude-code") == {"effort": "high"}
    holder["session"] = _materialized_session(tmp_path)
    mgr._flush_unsaved_session_params("p", "sess")
    assert Session.load(holder["session"]._filepath).sub_agent_params == {
        "claude-code": {"effort": "high"}}
    assert ("p", "sess") not in mgr._session_params_unsaved


def test_layering_rebuilds_argv_from_the_manifest_args(store):
    store.set("claude-code", {"model": "sonnet"})
    mgr = SubAgentManager(process_manager=MagicMock())
    mgr._setup_engine = SimpleNamespace(sub_agent_config_store=store)
    manifest = SimpleNamespace(runtime=SimpleNamespace(args=["--verbose"], config_mirror=None))
    cfg = {"command": "claude", "args": ["--verbose", "--model", "sonnet"],
           "acp_config_options": {}}
    out = mgr._layer_session_params(manifest, "claude-code", cfg,
                                    {"model": "opus", "effort": "high"})
    assert out["args"] == ["--verbose", "--model", "opus", "--effort", "high"]
    # The original dict is not mutated (the Settings fingerprint was taken from it).
    assert cfg["args"] == ["--verbose", "--model", "sonnet"]
    assert SubAgentManager._argv_value(out["args"], "--model") == "opus"
    assert SubAgentManager._argv_value(out["args"], "--effort") == "high"


class _Adapter:
    def __init__(self, params):
        self._spawn_fingerprint = "fp"
        self._session_params = params

    def is_alive(self):
        return True


def _stale_mgr(tmp_path, params):
    s = _materialized_session(tmp_path)
    if params:
        s.set_sub_agent_params("claude-code", params)
    mgr = _mgr_with_session(s)
    mgr._setup_engine = MagicMock()
    mgr._setup_engine.spawn_fingerprint.return_value = "fp"
    mgr._project_store = MagicMock()
    mgr._project_store.get_project.return_value = {"workspace": "/ws"}
    return mgr


def test_changed_conversation_params_make_the_spawn_stale(tmp_path):
    mgr = _stale_mgr(tmp_path, {"model": "opus", "effort": "high"})
    adapter = _Adapter({})
    assert asyncio.run(mgr._spawn_config_stale(
        "p", "claude-code", adapter, session_id="sess")) is True


def test_unchanged_conversation_params_are_not_stale(tmp_path):
    mgr = _stale_mgr(tmp_path, {"model": "opus"})
    adapter = _Adapter({"model": "opus"})
    assert asyncio.run(mgr._spawn_config_stale(
        "p", "claude-code", adapter, session_id="sess")) is False


def test_back_to_default_is_stale_too(tmp_path):
    mgr = _stale_mgr(tmp_path, {})
    adapter = _Adapter({"model": "opus"})
    assert asyncio.run(mgr._spawn_config_stale(
        "p", "claude-code", adapter, session_id="sess")) is True


def test_idle_worker_is_rebuilt_with_resume_after_a_pick(tmp_path):
    mgr = _stale_mgr(tmp_path, {"model": "opus"})
    calls = {"stop": [], "start": []}

    class _T:
        def __init__(self):
            self.dispatched = []

        async def dispatch(self, m):
            self.dispatched.append(m)

    old = _Adapter({})
    old._transport = _T()
    old._idle = True
    mgr._adapters[make_session_key("p", "sess")] = {"claude-code": old}

    async def fake_stop(project_id, handle, *, session_id=None):
        calls["stop"].append(handle)
        mgr._adapters[make_session_key(project_id, session_id)].pop(handle, None)

    async def fake_start(project_id, handle, depth=0, *, session_id=None,
                         announce=True, fresh=False, pinned=False, initiator="management_agent"):
        calls["start"].append(fresh)
        new = _Adapter({"model": "opus"})
        new._transport = _T()
        new._idle = True
        new._resume_status = ("resumed", None)
        mgr._adapters[make_session_key(project_id, session_id)][handle] = new
        return f"Started {handle}"

    mgr.stop = fake_stop
    mgr.start = fake_start
    asyncio.run(mgr.send("p", "claude-code", "next", session_id="sess"))
    assert calls["stop"] == ["claude-code"]
    # Rebuilt with resume (fresh=False): the conversation continues.
    assert calls["start"] == [False]


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------

def test_sdk_transport_passes_effort_to_the_sdk():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    t = SDKTransport(model="opus", effort="high")
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client:
        client.return_value.connect = AsyncMock()
        with patch.object(SDKTransport, "_capture_process_handle"), \
             patch.object(SDKTransport, "_ensure_reader"):
            captured = {}
            from agent_os.agent.transports import sdk_transport as mod
            real = mod.ClaudeAgentOptions

            def _opts(**kw):
                captured.update(kw)
                return real(**{k: v for k, v in kw.items()})
            with patch.object(mod, "ClaudeAgentOptions", side_effect=_opts) as m:
                m.__dataclass_fields__ = real.__dataclass_fields__
                asyncio.run(t.start("claude", [], "/tmp/ws"))
    assert captured["model"] == "opus"
    assert captured["effort"] == "high"


def test_sdk_transport_without_effort_leaves_cli_default():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    t = SDKTransport()
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client, \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as opts, \
         patch.object(SDKTransport, "_capture_process_handle"), \
         patch.object(SDKTransport, "_ensure_reader"):
        client.return_value.connect = AsyncMock()
        asyncio.run(t.start("claude", [], "/tmp/ws"))
    assert "effort" not in opts.call_args[1]
    assert "model" not in opts.call_args[1]


def test_manager_hands_argv_effort_to_the_sdk_transport():
    from agent_os.agent.transports import sdk_transport as mod
    if not mod.HAS_SDK:
        pytest.skip("claude-agent-sdk not installed")
    mgr = SubAgentManager(process_manager=MagicMock())
    manifest = SimpleNamespace(slug="claude-code", runtime=SimpleNamespace(
        transport="sdk", mode="pipe"))
    t = mgr._resolve_transport(manifest, {"args": ["--model", "opus", "--effort", "high"]})
    assert t._model == "opus" and t._effort == "high"


def test_codex_argv_effort():
    from agent_os.agent.transports.codex_transport import CodexTransport
    assert CodexTransport._argv_effort(["-m", "x", "-c", "model_reasoning_effort=low"]) == "low"
    assert CodexTransport._argv_effort(["-c", "other=1"]) is None
    assert CodexTransport._argv_effort([]) is None


@pytest.mark.parametrize("effort", ["low", None])
def test_codex_turn_start_carries_effort(effort):
    from agent_os.agent.transports.codex_transport import CodexTransport
    t = CodexTransport()
    t._alive = True
    t._thread_id = "T1"
    t._effort = effort
    sent = {}

    async def fake_request(method, params, **kw):
        sent[method] = params
        return {"turn": {"id": "turn-1"}}

    t._request = fake_request
    asyncio.run(t.dispatch("hi"))
    if effort:
        assert sent["turn/start"]["effort"] == "low"
    else:
        assert "effort" not in sent["turn/start"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    os.makedirs(str(tmp_path / "home"), exist_ok=True)
    from fastapi.testclient import TestClient
    from agent_os.api.app import create_app
    from agent_os.api.routes import settings as settings_routes
    app = create_app(data_dir=str(tmp_path / "data"))

    async def _no_live(*_a, **_k):
        return None
    monkeypatch.setattr(settings_routes, "_claude_live_models", _no_live)
    monkeypatch.setattr(settings_routes, "_codex_live_models", _no_live)
    return TestClient(app)


def _project(client, tmp_path):
    ws = str(tmp_path / "ws149")
    os.makedirs(ws, exist_ok=True)
    resp = client.post("/api/v2/projects", json={
        "name": "p149", "workspace": ws, "model": "gpt-4", "api_key": "test-key"})
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def test_route_set_read_and_clear(client, tmp_path):
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sessions/sess149/sub-agent-params"
    assert client.get(base).json()["params"] == {}
    r = client.patch(base, json={"slug": "claude-code", "model": "opus", "effort": "high"})
    assert r.status_code == 200, r.text
    assert r.json()["params"] == {"model": "opus", "effort": "high"}
    # No session file yet: held in memory, written by the first send.
    assert r.json()["persisted"] is False
    assert client.get(base).json()["params"] == {
        "claude-code": {"model": "opus", "effort": "high"}}
    # Absent field untouched; null clears to Default.
    r = client.patch(base, json={"slug": "claude-code", "model": None})
    assert r.json()["params"] == {"effort": "high"}
    r = client.patch(base, json={"slug": "claude-code", "effort": ""})
    assert r.json()["params"] == {}
    assert client.get(base).json()["params"] == {}


def test_route_rejects_bad_values(client, tmp_path):
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sessions/sess149/sub-agent-params"
    assert client.patch(base, json={"slug": "claude-code", "effort": "ultra"}).status_code == 400
    assert client.patch(base, json={"slug": "codex", "effort": "max"}).status_code == 400
    assert client.patch(base, json={"slug": "minimax-code", "effort": "high"}).status_code == 400
    assert client.get(
        "/api/v2/agents/nope/sessions/x/sub-agent-params").status_code == 404


def test_route_codex_model_checked_against_the_live_list(client, tmp_path, monkeypatch):
    from agent_os.api.routes import settings as settings_routes

    async def _live(*_a, **_k):
        return ["gpt-5.6", "gpt-5.5"]
    monkeypatch.setattr(settings_routes, "_codex_live_models", _live)
    pid = _project(client, tmp_path)
    base = f"/api/v2/agents/{pid}/sessions/sess149/sub-agent-params"
    assert client.patch(base, json={"slug": "codex", "model": "gpt-9"}).status_code == 400
    r = client.patch(base, json={"slug": "codex", "model": "gpt-5.6", "effort": "low"})
    assert r.status_code == 200 and r.json()["params"] == {"model": "gpt-5.6", "effort": "low"}
