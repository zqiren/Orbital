# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 172 B against the real ``codex app-server``: which model the transport
puts in ``thread/start``, for a scratch ``$CODEX_HOME`` per case.

No login and no turn: ``config/read``, ``model/list`` and ``thread/start`` are
answered locally, so this never touches the owner's ``~/.codex`` and bills
nothing. The thread/start params are captured by wrapping ``_request``.
Opt-in (spawns codex): ``-m live_daemon``.
"""

from __future__ import annotations

import shutil

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.codex_transport import CodexTransport

pytestmark = pytest.mark.live_daemon

CODEX = shutil.which("codex")
requires_codex = pytest.mark.skipif(CODEX is None, reason="codex binary not on PATH")

_CUSTOM_PROVIDER = """\
model = "gpt-6-luna"
model_provider = "go"

[model_providers.go]
name = "Go"
base_url = "http://127.0.0.1:9/v1"
wire_api = "responses"
"""


async def _thread_start_params(tmp_path, monkeypatch, config_toml: str | None,
                               args: list[str] | None = None):
    home = tmp_path / "codex-home"
    home.mkdir()
    if config_toml is not None:
        (home / "config.toml").write_text(config_toml)
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    t = CodexTransport(autonomy=Autonomy.CHECK_IN)
    sent: list[tuple[str, dict]] = []
    real = t._request

    async def spy(method, params=None, timeout=30.0):
        sent.append((method, dict(params or {})))
        return await real(method, params, timeout)

    t._request = spy
    try:
        await t.start(CODEX, args or [], str(ws))
        [params] = [p for m, p in sent if m == "thread/start"]
        return params, [m for m, _ in sent], t
    finally:
        await t.stop()


@requires_codex
@pytest.mark.asyncio
async def test_config_model_on_a_custom_provider_is_left_to_codex(tmp_path, monkeypatch):
    params, methods, t = await _thread_start_params(tmp_path, monkeypatch, _CUSTOM_PROVIDER)
    assert "config/read" in methods and "model/list" not in methods
    assert "model" not in params
    assert t._effective_model == "gpt-6-luna"     # what Codex opened the thread on


@requires_codex
@pytest.mark.asyncio
async def test_no_config_model_keeps_the_auto_pick(tmp_path, monkeypatch):
    params, methods, _ = await _thread_start_params(tmp_path, monkeypatch, "")
    assert methods.index("config/read") < methods.index("model/list")
    assert params.get("model")


@requires_codex
@pytest.mark.asyncio
async def test_a_broken_config_toml_fails_only_where_it_failed_before(tmp_path, monkeypatch):
    """The failed config/read falls back to the auto-pick; Codex itself then
    refuses thread/start on its broken config, exactly as before 172 (codex
    0.157.1). Orbital's read adds no new failure of its own."""
    from agent_os.agent.transports.codex_transport import CodexRpcError
    with pytest.raises(CodexRpcError) as err:
        await _thread_start_params(tmp_path, monkeypatch, 'model = "broken\n[[[\n')
    assert err.value.method == "thread/start"
    assert "configuration" in str(err.value)


@requires_codex
@pytest.mark.asyncio
async def test_an_orbital_override_wins_over_the_config(tmp_path, monkeypatch):
    params, methods, _ = await _thread_start_params(
        tmp_path, monkeypatch, _CUSTOM_PROVIDER, args=["-m", "gpt-6-sol"])
    assert "config/read" not in methods
    assert params["model"] == "gpt-6-sol"
