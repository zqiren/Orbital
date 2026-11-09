# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 161 §5.1–5.2: only the top of a session gets the dispatch tool.

A spawn whose initiator is ``user_pinned`` (the composer pin) or
``queue_item`` (an automation assigned to the worker) mints a token, records
its ``DispatchContext`` and hands the HTTP MCP config to the transport:
claude-code as ``mcp_servers["orbital"]`` with the bearer header, codex as
``-c`` pairs on the app-server's own argv plus the token in the spawn env.
A worker the manager or another worker dispatched gets nothing — no schema,
no token. The token dies with the adapter.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.daemon_v2.dispatch_tools import TOKEN_ENV, DispatchContext
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_MANIFESTS = os.path.join(_REPO, "agent_os", "agents", "manifests")
URL = "http://127.0.0.1:8541/mcp"


def _mgr(tmp_path, *, url=URL):
    registry = AgentRegistry()
    registry.register(ManifestLoader.load(os.path.join(_MANIFESTS, "claude_code.yaml")))
    registry.register(ManifestLoader.load(os.path.join(_MANIFESTS, "codex.yaml")))
    registry.register(ManifestLoader.load(os.path.join(_MANIFESTS, "minimax-code.yaml")))
    registry.register(ManifestLoader.load(os.path.join(_MANIFESTS, "pi.yaml")))
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    setup_engine = MagicMock()
    setup_engine.get_adapter_config.side_effect = lambda slug, project_workspace: {
        "command": f"/fake/{slug}", "args": [], "workspace": str(ws),
        "approval_patterns": [], "env": {"PATH": "/usr/bin"},
        "network_domains": [], "interactive": False,
    }
    setup_engine.check_all.return_value = []
    project_store = MagicMock()
    project_store.get_project.return_value = {"workspace": str(ws),
                                              "enabled_sub_agents": ["claude-code", "codex"]}
    process_manager = MagicMock()
    process_manager.start = AsyncMock()
    process_manager.stop = AsyncMock()
    mgr = SubAgentManager(process_manager=process_manager, registry=registry,
                          setup_engine=setup_engine, project_store=project_store)
    mgr._dispatch_url = url
    return mgr


async def _spawn(mgr, handle, initiator, *, acp_mounts=True):
    holder = {}

    async def start(config):
        # The real ACP handshake sets this; the mocked adapter never runs it.
        t = holder.get("transport")
        if type(t).__name__ in ("ACPSDKTransport", "PiRPCTransport") and t.dispatch_mcp:
            t.dispatch_mounted = acp_mounts

    with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as adapter_cls, \
         patch("agent_os.daemon_v2.sub_agent_manager.resolve_visible_sub_agent_slugs",
               return_value=["claude-code", "codex"]):
        adapter = MagicMock()
        adapter.start = AsyncMock(side_effect=start)
        adapter.stop = AsyncMock()
        adapter.is_alive = MagicMock(return_value=True)
        def make(*a, **kw):
            holder["transport"] = kw.get("transport")
            return adapter
        adapter_cls.side_effect = make
        result = await mgr.start("p1", handle, session_id="s1", announce=False,
                                 initiator=initiator)
    assert result.startswith("Started"), result
    transport = adapter_cls.call_args.kwargs["transport"]
    config = adapter.start.call_args.args[0]
    return adapter, transport, config


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_top_of_session_claude_code_gets_the_http_config(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, transport, config = await _spawn(mgr, "claude-code", initiator)
    assert type(transport).__name__ == "SDKTransport"
    cfg = transport.dispatch_mcp
    assert cfg["url"] == URL
    token = cfg["token"]
    assert token and len(token) >= 32
    assert mgr.dispatch_context(token) == DispatchContext(
        "p1", "s1", "claude-code", agents=("claude-code", "codex"))
    assert adapter._dispatch_token == token
    # The token is a header for claude-code, never an env var its shell sees.
    assert TOKEN_ENV not in (config.env or {})


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_top_of_session_codex_gets_the_url_and_the_token_env(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, transport, config = await _spawn(mgr, "codex", initiator)
    assert type(transport).__name__ == "CodexTransport"
    assert transport._dispatch_url == URL
    token = config.env[TOKEN_ENV]
    assert mgr.dispatch_context(token).handle == "codex"
    assert adapter._dispatch_token == token


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["management_agent", "worker:claude-code"])
@pytest.mark.parametrize("handle", ["claude-code", "codex"])
async def test_a_dispatched_worker_gets_nothing(tmp_path, initiator, handle):
    mgr = _mgr(tmp_path)
    adapter, transport, config = await _spawn(mgr, handle, initiator)
    if handle == "claude-code":
        assert transport.dispatch_mcp is None
    else:
        assert transport._dispatch_url is None
    assert TOKEN_ENV not in (config.env or {})
    assert mgr._dispatch_tokens == {}
    assert getattr(adapter, "_dispatch_token", None) is None


@pytest.mark.asyncio
async def test_no_server_attached_means_no_token(tmp_path):
    mgr = _mgr(tmp_path, url=None)
    _, transport, _ = await _spawn(mgr, "claude-code", "user_pinned")
    assert transport.dispatch_mcp is None
    assert mgr._dispatch_tokens == {}


@pytest.mark.asyncio
async def test_token_is_revoked_when_the_adapter_stops(tmp_path):
    mgr = _mgr(tmp_path)
    adapter, transport, _ = await _spawn(mgr, "claude-code", "user_pinned")
    token = transport.dispatch_mcp["token"]
    with patch.object(SubAgentManager, "_snapshot_kill_targets", return_value=[]):
        await mgr.stop("p1", "claude-code", session_id="s1")
    assert mgr.dispatch_context(token) is None


@pytest.mark.asyncio
async def test_token_is_revoked_when_the_spawn_fails(tmp_path):
    mgr = _mgr(tmp_path)
    with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as adapter_cls:
        adapter = MagicMock()
        adapter.start = AsyncMock(side_effect=RuntimeError("no binary"))
        adapter.stop = AsyncMock()
        adapter_cls.return_value = adapter
        result = await mgr.start("p1", "claude-code", session_id="s1",
                                 announce=False, initiator="user_pinned")
    assert result.startswith("Error")
    assert mgr._dispatch_tokens == {}


@pytest.mark.asyncio
async def test_send_threads_its_initiator_into_the_spawn(tmp_path):
    mgr = _mgr(tmp_path)
    seen = {}

    async def fake_start(project_id, handle, depth=0, *, session_id=None,
                         announce=True, fresh=False, pinned=False,
                         initiator="management_agent"):
        seen.update(pinned=pinned, initiator=initiator)
        return "Error: stop here"

    mgr.start = fake_start
    await mgr.send("p1", "codex", "task", session_id="s1", initiator="queue_item")
    assert seen == {"pinned": False, "initiator": "queue_item"}


# ---------------------------------------------------------------------------
# The transports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_transport_passes_the_http_server_with_the_bearer_header():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    transport = SDKTransport(dispatch_mcp={"url": URL, "token": "tok-1"})
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client, \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as options:
        client.return_value.connect = AsyncMock()
        await transport.start("claude", [], "/ws")
    assert options.call_args.kwargs["mcp_servers"] == {"orbital": {
        "type": "http", "url": URL, "headers": {"Authorization": "Bearer tok-1"}}}


@pytest.mark.asyncio
async def test_sdk_transport_without_a_context_passes_no_server():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    transport = SDKTransport()
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client, \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as options:
        client.return_value.connect = AsyncMock()
        await transport.start("claude", [], "/ws")
    assert "mcp_servers" not in options.call_args.kwargs


class _StopAfterExec(Exception):
    pass


async def _codex_exec(transport, env):
    seen = {}

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs.get("env")
        raise _StopAfterExec()

    with patch("agent_os.agent.transports.codex_transport.asyncio.create_subprocess_exec",
               side_effect=fake_exec):
        with pytest.raises(_StopAfterExec):
            await transport.start("/fake/codex", ["-m", "gpt-x"], "/ws", env)
    return seen


@pytest.mark.asyncio
async def test_codex_transport_forwards_the_pairs_on_the_app_server_argv():
    from agent_os.agent.transports.codex_transport import CodexTransport
    transport = CodexTransport(dispatch_url=URL)
    seen = await _codex_exec(transport, {TOKEN_ENV: "tok-2", "ORBITAL_SESSION": "s1"})
    assert seen["argv"] == (
        "/fake/codex", "app-server",
        "-c", f"mcp_servers.orbital.url={URL}",
        "-c", f"mcp_servers.orbital.bearer_token_env_var={TOKEN_ENV}",
        "-c", 'mcp_servers.orbital.default_tools_approval_mode="approve"',
    )
    assert seen["env"][TOKEN_ENV] == "tok-2"
    # The model / effort pairs stay per-turn, never on argv.
    assert "-m" not in seen["argv"]
    # The token never rides into the thread's shell env (every command Codex
    # runs would see it); other ORBITAL_* attribution still does.
    assert transport._orbital_env == {"ORBITAL_SESSION": "s1"}


@pytest.mark.asyncio
async def test_codex_transport_without_a_url_execs_bare_app_server():
    from agent_os.agent.transports.codex_transport import CodexTransport
    transport = CodexTransport()
    seen = await _codex_exec(transport, {"ORBITAL_SESSION": "s1"})
    assert seen["argv"] == ("/fake/codex", "app-server")


# Live 2026-11-09 (v0.21.0): Codex 0.157 keeps MCP tools out of the model's
# tool list (code mode: the model must filter ALL_TOOLS to find them), and
# it takes no system prompt. Pinned Codex therefore never learned the
# dispatch tool existed and ran `claude -p` in its shell. The seat-holder
# Codex is told on its first turn, in the note that already carries the
# background-job paragraph.
@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_top_of_session_codex_is_told_about_dispatch_on_its_first_turn(
        tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, _, _ = await _spawn(mgr, "codex", initiator)
    note = adapter._orbital_first_turn_note
    assert "`dispatch` tool of the `orbital` MCP server" in note
    assert "claude-code" in note          # the peers it can hand work to
    assert "shell" in note


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["management_agent", "worker:claude-code"])
async def test_a_dispatched_codex_is_not_told_about_dispatch(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, _, _ = await _spawn(mgr, "codex", initiator)
    note = getattr(adapter, "_orbital_first_turn_note", None)
    assert not note or "dispatch" not in note


# Spec 168 (live 2026-11-09, film rig): pinned Claude Code ran `codex exec`
# from its shell in 2 of 6 runs, so Codex's reply never reached the chat. It
# was the one seat-holder the first-turn line skipped (a transport allowlist
# left out `sdk`). Every seat-holder with dispatch available gets the line on
# its first prompt only; the dispatch context alone decides.
@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_top_of_session_claude_code_is_told_about_dispatch_on_its_first_turn_only(
        tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, transport, _ = await _spawn(mgr, "claude-code", initiator)
    assert type(transport).__name__ == "SDKTransport"
    mgr._build_recap = AsyncMock(return_value="")
    mgr._dispatch_async = AsyncMock()
    prompt = MagicMock(message="fix the sign in bug, then have codex review it",
                       dispatch_id="d1", transcript_path="t", initiator=initiator)
    await mgr._dispatch_prompt_locked(adapter, prompt, "p1", "s1", "claude-code")
    await mgr._dispatch_prompt_locked(adapter, prompt, "p1", "s1", "claude-code")
    first = mgr._dispatch_async.await_args_list[0].args[3]
    second = mgr._dispatch_async.await_args_list[1].args[3]
    assert "`dispatch` tool of the `orbital` MCP server" in first
    assert "codex" in first.split("\n\n")[0]       # the peer it can hand work to
    assert "shell" in first
    assert first.endswith(prompt.message)
    assert second == prompt.message


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["management_agent", "worker:codex"])
async def test_a_dispatched_claude_code_is_not_told_about_dispatch(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, transport, _ = await _spawn(mgr, "claude-code", initiator)
    assert transport.dispatch_mcp is None
    assert not getattr(adapter, "_orbital_first_turn_note", None)


# ACP workers (codebuddy, cursor, dsh, kimi-code, minimax-code) take no
# Orbital system prompt either: the seat-holder gets the endpoint through the
# transport (mounted on session open) and the same first-turn line.
@pytest.mark.asyncio
async def test_top_of_session_acp_worker_gets_the_http_config_and_the_line(tmp_path):
    mgr = _mgr(tmp_path)
    adapter, transport, config = await _spawn(mgr, "minimax-code", "user_pinned")
    assert type(transport).__name__ == "ACPSDKTransport"
    cfg = transport.dispatch_mcp
    assert cfg["url"] == URL
    assert mgr.dispatch_context(cfg["token"]).handle == "minimax-code"
    assert TOKEN_ENV not in (config.env or {})
    assert "dispatch" in adapter._orbital_first_turn_note


@pytest.mark.asyncio
async def test_a_dispatched_acp_worker_gets_nothing(tmp_path):
    mgr = _mgr(tmp_path)
    adapter, transport, _ = await _spawn(mgr, "minimax-code", "management_agent")
    assert transport.dispatch_mcp is None
    assert mgr._dispatch_tokens == {}
    assert not getattr(adapter, "_orbital_first_turn_note", None)


@pytest.mark.asyncio
async def test_acp_agent_without_http_mcp_loses_the_token_and_the_line(tmp_path):
    mgr = _mgr(tmp_path)
    adapter, transport, _ = await _spawn(mgr, "minimax-code", "user_pinned",
                                         acp_mounts=False)
    assert mgr._dispatch_tokens == {}
    assert adapter._dispatch_token is None
    assert not adapter._orbital_first_turn_note


# Pi (pi-rpc) mounts the endpoint through its own MCP client, via Orbital's
# one-file extension passed only to the seat-holder (the transport's half is
# pinned in test_pi_dispatch_extension.py). Pi takes Orbital's system prompt,
# but the seat-holder still gets the same first-turn line as Codex and ACP.
@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_top_of_session_pi_gets_the_http_config_and_the_line(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, transport, config = await _spawn(mgr, "pi", initiator)
    assert type(transport).__name__ == "PiRPCTransport"
    cfg = transport.dispatch_mcp
    assert cfg["url"] == URL
    assert mgr.dispatch_context(cfg["token"]).handle == "pi"
    # The transport puts the token in pi's env itself; the adapter config
    # (the shared spawn env) never carries it.
    assert TOKEN_ENV not in (config.env or {})
    note = adapter._orbital_first_turn_note
    assert "`dispatch` tool of the `orbital` MCP server" in note
    assert "claude-code" in note


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["management_agent", "worker:codex"])
async def test_a_dispatched_pi_gets_nothing(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, transport, _ = await _spawn(mgr, "pi", initiator)
    assert transport.dispatch_mcp is None
    assert mgr._dispatch_tokens == {}
    assert not getattr(adapter, "_orbital_first_turn_note", None)


@pytest.mark.asyncio
async def test_pi_without_the_mount_loses_the_token_and_the_line(tmp_path):
    # pi older than 0.99, its built-in MCP disabled, or the extension failed.
    mgr = _mgr(tmp_path)
    adapter, _, _ = await _spawn(mgr, "pi", "user_pinned", acp_mounts=False)
    assert mgr._dispatch_tokens == {}
    assert adapter._dispatch_token is None
    assert not adapter._orbital_first_turn_note


# 2026-11-09 (keyless smoke D4): the dispatch tool's agent list names only
# workers that can run now — the same test as the pin list — so a Claude seat
# no longer relays "codebuddy, gemini-cli" that are installed but signed out.

@pytest.mark.asyncio
async def test_the_dispatch_list_names_only_ready_workers(tmp_path):
    from types import SimpleNamespace as NS
    mgr = _mgr(tmp_path)
    mgr._project_store.get_project.return_value = {
        "workspace": str(tmp_path / "ws"), "disabled_sub_agents": ["pi"]}

    def status(slug, **kw):
        base = dict(slug=slug, installed=True, dependencies_met=True,
                    credentials_configured=True, update_needed=False, handshake=None)
        base.update(kw)
        return NS(**base)

    mgr._setup_engine.check_all.return_value = [
        status("claude-code"), status("codex"), status("pi"),
        status("codebuddy", credentials_configured=False),
        status("gemini-cli", credentials_configured=False),
        status("built-in"),
    ]
    with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as adapter_cls:
        adapter = MagicMock()
        adapter.start = AsyncMock()
        adapter.stop = AsyncMock()
        adapter.is_alive = MagicMock(return_value=True)
        adapter_cls.return_value = adapter
        result = await mgr.start("p1", "claude-code", session_id="s1", announce=False,
                                 initiator="user_pinned")
    assert result.startswith("Started"), result
    token = adapter_cls.call_args.kwargs["transport"].dispatch_mcp["token"]
    assert mgr.dispatch_context(token).agents == ("claude-code", "codex")
