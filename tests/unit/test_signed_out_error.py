# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""One-page Welcome §6 (2026-11-23): an honest "signed out" error.

A keyless send whose only agent is installed but signed out used to be
refused as ``no_runner`` ("…no agent is enabled for this project…"), though
the agent IS enabled. The pre-flight now names it: ``agent_signed_out`` with
the agent's slug and display name, before any row is written, so the UI can
offer that agent's sign-in. ``no_runner`` keeps its meaning when nothing is
installed or every agent is turned off; a send with a key never sees either.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import AgentConfig
from agent_os.daemon_v2.provider_errors import ProviderConfigError

SESSION_ID = "proj_deadbeef"


def _config(workspace, key="k"):
    return AgentConfig(workspace=workspace, model="stub-model", api_key=key,
                       provider="stub-provider", sdk="openai", project_name="proj")


def _status(slug, name, *, installed=True, signed_in=True, expired=False):
    return SimpleNamespace(slug=slug, name=name, installed=installed,
                           dependencies_met=True, credentials_configured=signed_in,
                           update_needed=False, handshake=None, auth_expired=expired)


def _manifest(slug, *, login=True):
    cred = SimpleNamespace(setup_command=f"{slug} login" if login else "", key="k")
    return SimpleNamespace(slug=slug, setup=SimpleNamespace(credentials=[cred]))


def _manager(workspace, statuses, *, manifests=None, disabled=()):
    engine = MagicMock()
    engine.check_all.return_value = statuses
    engine.check_all_cached.return_value = statuses
    registry = MagicMock()
    manifests = manifests or {s.slug: _manifest(s.slug) for s in statuses}
    registry.get.side_effect = manifests.get
    mgr = AgentManager(project_store=MagicMock(), ws_manager=MagicMock(),
                       sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
                       process_manager=MagicMock(), registry=registry, setup_engine=engine)
    mgr._ws.broadcast = MagicMock()
    mgr._project_store.get_project.return_value = {
        "name": "proj", "workspace": workspace, "disabled_sub_agents": list(disabled)}
    mgr._platform_provider = None
    keyless = _config(workspace, key="")
    mgr._build_agent_config_from_project = lambda pid: keyless
    mgr.start_agent = MagicMock(side_effect=AssertionError("must not start"))
    return mgr


def _rows(workspace):
    path = os.path.join(workspace, "orbital", "sessions", f"{SESSION_ID}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _refusal(mgr):
    with pytest.raises(ProviderConfigError) as info:
        await mgr.inject_message("proj", "hello", nonce="n-1", session_id=SESSION_ID)
    return info.value


@pytest.mark.asyncio
async def test_a_signed_out_agent_is_named(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws, [_status("claude-code", "Claude Code", signed_in=False),
                        _status("codex", "Codex", installed=False, signed_in=False)])
    err = await _refusal(mgr)
    assert err.code == "agent_signed_out"
    assert err.fields == {"agent": "claude-code", "agent_name": "Claude Code"}
    assert "Claude Code" in str(err) and "sign" in str(err).lower()
    assert _rows(ws) == []


@pytest.mark.asyncio
async def test_the_first_signed_out_agent_in_registry_order_is_named(tmp_path):
    mgr = _manager(str(tmp_path), [
        _status("codex", "Codex", signed_in=False),
        _status("claude-code", "Claude Code", signed_in=False)])
    assert (await _refusal(mgr)).fields["agent"] == "codex"


@pytest.mark.asyncio
async def test_a_signed_out_agent_turned_off_here_is_not_named(tmp_path):
    mgr = _manager(str(tmp_path), [_status("claude-code", "Claude Code", signed_in=False)],
                   disabled=("claude-code",))
    assert (await _refusal(mgr)).code == "no_runner"


@pytest.mark.asyncio
async def test_a_key_only_agent_without_its_key_is_not_called_signed_out(tmp_path):
    """No in-app sign-in exists for it (its credential is a key), so
    "sign in" would be the wrong remedy: today's no_runner stands."""
    status = _status("dsh", "DeepSeek Harness", signed_in=False)
    mgr = _manager(str(tmp_path), [status], manifests={"dsh": _manifest("dsh", login=False)})
    assert (await _refusal(mgr)).code == "no_runner"


@pytest.mark.asyncio
async def test_nothing_installed_is_still_no_runner(tmp_path):
    mgr = _manager(str(tmp_path), [_status("claude-code", "Claude Code",
                                           installed=False, signed_in=False)])
    err = await _refusal(mgr)
    assert err.code == "no_runner" and err.fields == {}


@pytest.mark.asyncio
async def test_a_ready_worker_keeps_missing_api_key(tmp_path):
    mgr = _manager(str(tmp_path), [_status("claude-code", "Claude Code"),
                                   _status("codex", "Codex", signed_in=False)])
    assert (await _refusal(mgr)).code == "missing_api_key"


@pytest.mark.asyncio
async def test_with_a_key_nothing_is_refused(tmp_path):
    ws = str(tmp_path)
    mgr = _manager(ws, [_status("claude-code", "Claude Code", signed_in=False)])
    mgr._build_agent_config_from_project = lambda pid: _config(ws)
    started = []

    async def start(project_id, config, **kwargs):
        started.append(project_id)
    mgr.start_agent = start
    assert await mgr.inject_message("proj", "hello", nonce="n-1",
                                    session_id=SESSION_ID) == "started"
    assert started == ["proj"]


def test_provider_config_error_fields_default_empty():
    assert ProviderConfigError("missing_api_key", "m").fields == {}


def test_route_error_body_carries_the_fields():
    """The start/inject routes answer ``provider_error_detail(e)``."""
    from agent_os.api.routes import agents_v2
    detail = agents_v2.provider_error_detail(ProviderConfigError(
        "agent_signed_out", "Claude Code is signed out.",
        fields={"agent": "claude-code", "agent_name": "Claude Code"}))
    assert detail == {"code": "agent_signed_out", "message": "Claude Code is signed out.",
                      "agent": "claude-code", "agent_name": "Claude Code"}
    # Every existing code keeps its exact two-key body.
    assert agents_v2.provider_error_detail(
        ProviderConfigError("no_runner", "m")) == {"code": "no_runner", "message": "m"}


def test_both_route_sites_use_the_shared_body():
    import inspect
    from agent_os.api.routes import agents_v2
    src = inspect.getsource(agents_v2)
    assert 'detail={"code": e.code, "message": str(e)}' not in src
    assert src.count("detail=provider_error_detail(e)") >= 2
