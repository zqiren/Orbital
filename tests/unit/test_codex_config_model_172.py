# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 172 B: Codex runs on the model the user configured in Codex.

A fresh thread with no Orbital override used to be opened on the ``isDefault``
entry of ``model/list`` (OpenAI's catalog default) even when the user's
``config.toml`` names a model on a custom provider, so the provider refused a
model the user never chose (fresh-install run 2026-11-11: OpenCode Go answered
"Model is unavailable" for gpt-6-astra while config.toml said gpt-6-luna).

The transport now asks the app-server for its effective config
(``config/read`` with the thread's cwd — the app-server resolves
``$CODEX_HOME``, project layers and profiles itself, read-only). When it names
a model, ``thread/start`` omits ``model`` and Codex uses its own config. The
auto-pick stays for a config that names none (the ChatGPT cold-start 400 it
was built for), and when the read fails for any reason (older CLI, a broken
config.toml) — exactly today's behaviour. An Orbital Settings override still
wins over both. No codex process here: ``_request`` is faked.
"""

import logging

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.codex_transport import (
    CodexRpcError,
    CodexTransport,
)


def _fake(script: dict, calls: list):
    async def fake(method, params=None, timeout=30.0):
        calls.append((method, dict(params or {})))
        outcome = script[method]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return fake


def _config(model=None, provider=None):
    return {"config": {"model": model, "model_provider": provider},
            "origins": {}, "layers": None}


_LIST = {"data": [{"id": "gpt-6-astra", "isDefault": True}, {"id": "gpt-6-luna"}]}
_STARTED = {"thread": {"id": "T1", "path": "/r/T1.jsonl"}, "model": "gpt-6-luna"}


def _fresh():
    return CodexTransport(autonomy=Autonomy.CHECK_IN)


def _start_params(calls):
    [(_, params)] = [c for c in calls if c[0] == "thread/start"]
    return params


@pytest.mark.asyncio
async def test_a_model_in_the_users_codex_config_is_left_to_codex():
    t, calls = _fresh(), []
    t._request = _fake({"config/read": _config("gpt-6-luna", "go"),
                        "model/list": _LIST, "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert [c[0] for c in calls] == ["config/read", "thread/start"]
    assert calls[0][1] == {"cwd": "/ws"}           # project layers resolve from the cwd
    assert "model" not in _start_params(calls)
    assert t._model is None and t._model_auto is False
    # What Codex actually used is what Orbital shows and records.
    assert t._turn_meta("success")["model"] == "gpt-6-luna"


@pytest.mark.asyncio
async def test_a_profile_model_resolved_by_codex_counts_too():
    # Codex resolves the active profile (and $CODEX_HOME) itself; config/read
    # returns the effective model, whichever layer set it.
    t, calls = _fresh(), []
    t._request = _fake({"config/read": {"config": {"model": "prof-model"},
                                        "origins": {"model": {"name": {
                                            "type": "user", "file": "/h/work.config.toml"}}}},
                        "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert "model" not in _start_params(calls)


@pytest.mark.asyncio
async def test_no_model_in_the_config_keeps_the_auto_pick():
    t, calls = _fresh(), []
    t._request = _fake({"config/read": _config(None),
                        "model/list": _LIST, "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert [c[0] for c in calls] == ["config/read", "model/list", "thread/start"]
    assert _start_params(calls)["model"] == "gpt-6-astra"
    assert t._model_auto is True


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["", "   "])
async def test_a_blank_model_is_no_model(model):
    t, calls = _fresh(), []
    t._request = _fake({"config/read": _config(model),
                        "model/list": _LIST, "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert _start_params(calls)["model"] == "gpt-6-astra"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    # A broken config.toml (observed on codex 0.157.1 with a scratch CODEX_HOME).
    CodexRpcError("config/read", {"code": -32603, "message":
        "failed to read configuration layers: config.toml:1:16: invalid basic string"}),
    # An older CLI without the method.
    CodexRpcError("config/read", {"code": -32601, "message": "method not found"}),
    TimeoutError(),
    RuntimeError("transport down"),
])
async def test_a_failed_config_read_falls_back_to_todays_auto_pick(failure, caplog):
    t, calls = _fresh(), []
    t._request = _fake({"config/read": failure,
                        "model/list": _LIST, "thread/start": _STARTED}, calls)
    with caplog.at_level(logging.INFO):
        await t._open_thread("/ws")
    assert _start_params(calls)["model"] == "gpt-6-astra"
    assert t._model_auto is True
    assert any("config/read" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_malformed_answer_falls_back_to_todays_auto_pick():
    t, calls = _fresh(), []
    t._request = _fake({"config/read": {"config": "nonsense"},
                        "model/list": _LIST, "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert _start_params(calls)["model"] == "gpt-6-astra"


@pytest.mark.asyncio
async def test_an_orbital_settings_override_wins_over_the_codex_config():
    t, calls = _fresh(), []
    t._model = "gpt-6-sol"                       # from argv (-m), set in start()
    t._request = _fake({"config/read": _config("gpt-6-luna"),
                        "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert [c[0] for c in calls] == ["thread/start"]
    assert _start_params(calls)["model"] == "gpt-6-sol"


@pytest.mark.asyncio
async def test_resume_never_reads_the_config_and_omits_the_model():
    t, calls = CodexTransport(autonomy=Autonomy.CHECK_IN,
                              resume_record={"session_id": "T0"}), []
    t._request = _fake({"thread/resume": _STARTED}, calls)
    await t._open_thread("/ws")
    assert [c[0] for c in calls] == ["thread/resume"]
    assert "model" not in calls[0][1]


@pytest.mark.asyncio
async def test_a_config_model_is_never_repicked_on_an_error():
    # The re-pick is only for a model Orbital chose; this one is the user's.
    t, calls = _fresh(), []
    t._request = _fake({"config/read": _config("gpt-6-luna"),
                        "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    t._begin_turn()
    await t._route_server_message({"jsonrpc": "2.0", "method": "error", "params": {
        "error": {"message": '{"code":"model_not_found"}'}, "willRetry": False}})
    events = []
    while not t._event_queue.empty():
        events.append(t._event_queue.get_nowait())
    assert [e.event_type for e in events] == ["error"]


@pytest.mark.asyncio
async def test_a_fork_failure_fresh_thread_also_leaves_a_config_model_to_codex():
    held = "01a0a825-7f33-75d0-b789-bb62c9c3e8be"
    t = CodexTransport(autonomy=Autonomy.CHECK_IN,
                       resume_record={"session_id": held})
    calls = []
    writer = CodexRpcError("thread/resume", {
        "code": -32600, "message": f"thread {held} already has an active writer"})
    t._request = _fake({
        "thread/resume": writer,
        "thread/fork": CodexRpcError("thread/fork", {"code": -32603, "message": "boom"}),
        "config/read": _config("gpt-6-luna"),
        "thread/start": _STARTED}, calls)
    await t._open_thread("/ws")
    assert [c[0] for c in calls] == ["thread/resume", "thread/fork",
                                     "config/read", "thread/start"]
    assert calls[2][1] == {"cwd": "/ws"}
    assert "model" not in _start_params(calls)
    assert "threadId" not in _start_params(calls)
