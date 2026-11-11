# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Agent-CLI probes called from coroutines run off the event loop.

Spec 171 A3: ``SetupEngine.check_agent`` / ``resolve_binary`` /
``get_adapter_config`` end in ``subprocess.run`` with 5–15 s timeouts on a
cache miss (first start, or a CLI that moved after a self-update). Called
synchronously from a route or a worker start, a miss froze every request of
every project. Each call now goes through ``asyncio.to_thread``. The fakes
below record the thread each probe ran on.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException


def _assert_off_loop(seen: list[int]) -> None:
    assert seen, "the probe did not run"
    assert threading.get_ident() not in seen, "a CLI probe ran on the event loop"


def _recorder(seen: list[int], value=None, exc: BaseException | None = None):
    def probe(*_a, **_k):
        seen.append(threading.get_ident())
        if exc is not None:
            raise exc
        return value
    return probe


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_status_route_checks_off_the_loop(monkeypatch):
    from agent_os.api.routes import agents_v2

    seen: list[int] = []
    engine = MagicMock()
    engine.check_agent.side_effect = _recorder(seen, exc=ValueError("nope"))
    monkeypatch.setattr(agents_v2, "_setup_engine", engine)
    with pytest.raises(HTTPException) as err:
        await agents_v2.agent_status("ghost")
    assert err.value.status_code == 404
    _assert_off_loop(seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["trigger_sub_agent_login",
                                   "trigger_sub_agent_update",
                                   "trigger_sub_agent_logout"])
async def test_setup_command_resolves_the_binary_off_the_loop(monkeypatch, route):
    """login / update / logout resolve the CLI path (``resolve_binary``)
    before running the vendor's command."""
    from agent_os.api.routes import settings as settings_routes

    seen: list[int] = []
    monkeypatch.setattr(settings_routes, "_resolve_setup_command",
                        _recorder(seen, value=None))
    monkeypatch.setattr(settings_routes, "_worker_running", lambda _slug: False)
    with pytest.raises(HTTPException):
        await getattr(settings_routes, route)("codex")
    _assert_off_loop(seen)


@pytest.mark.asyncio
async def test_post_login_credential_recheck_resolves_off_the_loop(monkeypatch):
    from agent_os.api.routes import settings as settings_routes

    seen: list[int] = []
    engine = MagicMock()
    engine._registry.get.return_value = SimpleNamespace(slug="codex")
    engine.resolve_binary.side_effect = _recorder(seen, value="/bin/codex")
    engine.check_credentials.return_value = (True, [])
    monkeypatch.setattr(settings_routes, "_setup_engine", engine)
    assert await settings_routes._credentials_now_configured("codex") is True
    _assert_off_loop(seen)


@pytest.mark.asyncio
async def test_api_key_login_resolves_the_binary_off_the_loop(monkeypatch):
    from agent_os.api.routes import settings as settings_routes

    seen: list[int] = []
    engine = MagicMock()
    engine._registry.get.return_value = SimpleNamespace(slug="codex")
    engine.resolve_binary.side_effect = _recorder(seen, value=None)
    monkeypatch.setattr(settings_routes, "_setup_engine", engine)
    with pytest.raises(HTTPException) as err:
        await settings_routes.set_sub_agent_api_key(
            "codex", settings_routes.SetSubAgentApiKeyRequest(api_key="k"))
    assert err.value.status_code == 400   # "not installed"
    _assert_off_loop(seen)


# --------------------------------------------------------------------------
# SubAgentManager
# --------------------------------------------------------------------------

def _sub_agent_manager(engine, manifest):
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

    registry = MagicMock()
    registry.get.return_value = manifest
    project_store = MagicMock()
    project_store.get_project.return_value = {"workspace": "/tmp/ws"}
    return SubAgentManager(
        process_manager=MagicMock(),
        registry=registry,
        setup_engine=engine,
        project_store=project_store,
    )


@pytest.mark.asyncio
async def test_worker_command_probe_builds_the_config_off_the_loop():
    seen: list[int] = []
    engine = MagicMock()
    engine.get_adapter_config.side_effect = _recorder(seen, value={"command": ""})
    manifest = SimpleNamespace(runtime=SimpleNamespace(transport="sdk"))
    mgr = _sub_agent_manager(engine, manifest)
    assert await mgr.probe_worker_commands("claude-code", "/tmp/ws") is None
    _assert_off_loop(seen)


@pytest.mark.asyncio
async def test_worker_start_builds_the_adapter_config_off_the_loop():
    seen: list[int] = []
    engine = MagicMock()
    engine.get_adapter_config.side_effect = _recorder(
        seen, exc=ValueError("Agent 'codex' not installed or binary not found"))
    manifest = SimpleNamespace(runtime=SimpleNamespace(adapter="cli", transport="pipe"))
    mgr = _sub_agent_manager(engine, manifest)
    out = await mgr.start("proj_1", "codex", session_id="sess_1")
    assert out.startswith("Error:") and "not installed" in out
    _assert_off_loop(seen)


@pytest.mark.asyncio
async def test_two_concurrent_starts_of_one_worker_spawn_it_once():
    """The config build is now an await on the start path: a second start
    of the same worker in the same session arriving inside it must not
    spawn a second process. It waits for the first and gets its answer."""
    from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def fake_registry_start(self, project_id, handle, depth=0, **kw):
        calls.append(handle)
        entered.set()
        await release.wait()
        self._adapters.setdefault(("proj_1", "sess_1"), {})[handle] = object()
        return f"Started {handle}"

    engine = MagicMock()
    mgr = _sub_agent_manager(engine, SimpleNamespace())
    mgr._start_from_registry = fake_registry_start.__get__(mgr, SubAgentManager)

    first = asyncio.create_task(mgr.start("proj_1", "codex", session_id="sess_1"))
    await entered.wait()
    second = asyncio.create_task(mgr.start("proj_1", "codex", session_id="sess_1"))
    await asyncio.sleep(0.01)
    release.set()
    results = await asyncio.gather(first, second)

    assert calls == ["codex"], "the worker was spawned twice"
    assert results == ["Started codex", "Started codex"]
    # A later start is a new start (nothing left behind by the single-flight).
    assert not mgr._inflight_starts


# --------------------------------------------------------------------------
# LAN IP lookup
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lan_url_lookup_does_not_block_other_requests(monkeypatch):
    """``getaddrinfo(gethostname())`` blocked a fresh VM's daemon for ~34 s
    per call. The lookup runs in a thread, so a request arriving meanwhile
    is still answered; the answer and its preference order are unchanged."""
    import socket

    from agent_os.api.routes import agents_v2

    seen: list[int] = []

    def slow_getaddrinfo(*_a, **_k):
        seen.append(threading.get_ident())
        time.sleep(0.5)
        return [(socket.AF_INET, None, None, "", ("10.8.0.2", 0)),
                (socket.AF_INET, None, None, "", ("192.168.1.20", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)
    lookup = asyncio.create_task(agents_v2.get_lan_url())
    started = time.monotonic()
    await asyncio.sleep(0.05)          # another request's turn on the loop
    assert time.monotonic() - started < 0.3, "the event loop was blocked"
    assert await lookup == {"ip": "192.168.1.20"}
    _assert_off_loop(seen)
