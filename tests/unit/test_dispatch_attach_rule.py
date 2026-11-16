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
    # Spec 184 §5.3: not the seat note, but the one line saying it has none.
    assert "`dispatch` tool" not in adapter._orbital_first_turn_note
    assert "cannot hand tasks to other agents" in adapter._orbital_first_turn_note


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
    # Spec 184 §5.3: not the seat note, but the one line saying it has none.
    assert "`dispatch` tool" not in adapter._orbital_first_turn_note
    assert "cannot hand tasks to other agents" in adapter._orbital_first_turn_note


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


# ---------------------------------------------------------------------------
# Spec 184 §5.3: a seat-holder that ends up without the tool is told so
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("handle", ["pi", "minimax-code"])
async def test_a_seat_holder_without_the_mount_gets_the_no_tool_line(tmp_path, handle):
    mgr = _mgr(tmp_path)
    adapter, _, _ = await _spawn(mgr, handle, "user_pinned", acp_mounts=False)
    note = adapter._orbital_first_turn_note
    assert "cannot hand tasks to other agents" in note
    assert "`dispatch` tool" not in note          # not the seat note


@pytest.mark.asyncio
async def test_a_seat_holder_with_no_endpoint_gets_the_no_tool_line(tmp_path):
    mgr = _mgr(tmp_path, url=None)
    adapter, _, _ = await _spawn(mgr, "claude-code", "user_pinned")
    assert "cannot hand tasks to other agents" in adapter._orbital_first_turn_note


@pytest.mark.asyncio
async def test_a_seat_holder_without_peers_gets_neither_line(tmp_path):
    mgr = _mgr(tmp_path)
    with patch("agent_os.daemon_v2.sub_agent_manager.resolve_visible_sub_agent_slugs",
               return_value=["pi"]):
        holder = {}

        async def start(config):
            holder["t"].dispatch_mounted = False

        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter") as adapter_cls:
            adapter = MagicMock()
            adapter.start = AsyncMock(side_effect=start)
            adapter.is_alive = MagicMock(return_value=True)

            def make(*a, **kw):
                holder["t"] = kw.get("transport")
                return adapter
            adapter_cls.side_effect = make
            await mgr.start("p1", "pi", session_id="s1", announce=False,
                            initiator="user_pinned")
    assert not adapter._orbital_first_turn_note


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["management_agent", "worker:codex"])
async def test_a_dispatched_worker_never_gets_the_no_tool_line(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    adapter, _, _ = await _spawn(mgr, "pi", initiator)
    assert not adapter._orbital_first_turn_note


# ---------------------------------------------------------------------------
# Spec 184 §5.2: loopback never goes through the user's proxy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("own, inherited, expected", [
    ("a.example", None, "a.example,127.0.0.1,localhost"),
    (None, "corp.example", "corp.example,127.0.0.1,localhost"),
    (None, None, "127.0.0.1,localhost"),
    ("127.0.0.1,b.example", None, "127.0.0.1,b.example,localhost"),
    ("localhost, 127.0.0.1", None, "localhost, 127.0.0.1"),
    ("", "corp.example", ""),   # an explicit empty entry is the worker's value
])
def test_loopback_no_proxy_builds_from_the_value_the_worker_gets(own, inherited, expected):
    from agent_os.daemon_v2.sub_agent_manager import _loopback_no_proxy
    env = {} if own is None else {"NO_PROXY": own, "no_proxy": own}
    inherited_env = {} if inherited is None else {
        "NO_PROXY": inherited, "no_proxy": inherited}
    out = _loopback_no_proxy(env, inherited_env, windows=False)
    if expected == "":
        expected = "127.0.0.1,localhost"
    assert out["NO_PROXY"] == expected
    assert out["no_proxy"] == expected


def test_loopback_no_proxy_on_windows_writes_one_spelling():
    # Windows env names are case-insensitive: both spellings in the child's
    # env block would be a duplicate entry. os.environ there answers either.
    from agent_os.daemon_v2.sub_agent_manager import _loopback_no_proxy
    out = _loopback_no_proxy({}, {"NO_PROXY": "corp.example"}, windows=True)
    assert out == {"NO_PROXY": "corp.example,127.0.0.1,localhost"}


def test_loopback_no_proxy_treats_the_two_spellings_separately():
    from agent_os.daemon_v2.sub_agent_manager import _loopback_no_proxy
    out = _loopback_no_proxy({"NO_PROXY": "a.example"}, {"no_proxy": "b.example"},
                             windows=False)
    assert out["NO_PROXY"] == "a.example,127.0.0.1,localhost"
    assert out["no_proxy"] == "b.example,127.0.0.1,localhost"


@pytest.mark.asyncio
@pytest.mark.parametrize("handle", ["claude-code", "codex", "pi"])
async def test_a_seat_holder_spawn_gets_loopback_in_no_proxy(tmp_path, handle, monkeypatch):
    if os.name == "nt":
        pytest.skip("one case-insensitive spelling on Windows (tested above)")
    monkeypatch.setenv("NO_PROXY", "corp.example")
    monkeypatch.delenv("no_proxy", raising=False)
    mgr = _mgr(tmp_path)
    _, _, config = await _spawn(mgr, handle, "user_pinned")
    assert config.env["NO_PROXY"] == "corp.example,127.0.0.1,localhost"
    assert config.env["no_proxy"] == "127.0.0.1,localhost"


@pytest.mark.asyncio
async def test_a_dispatched_worker_env_is_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "corp.example")
    mgr = _mgr(tmp_path)
    _, _, config = await _spawn(mgr, "codex", "management_agent")
    assert "NO_PROXY" not in config.env and "no_proxy" not in config.env


# ---------------------------------------------------------------------------
# Spec 184 §5.1: a worker Orbital started, taken over by the user, is rebuilt
# once with resume so it gets the tool. Three moments, one test: the pin, the
# user's send, and the turn close.
# ---------------------------------------------------------------------------


class _Rig:
    """A SubAgentManager whose CLIAdapter is a fresh mock per spawn."""

    def __init__(self, mgr):
        self.mgr = mgr
        self.adapters: list = []
        self.transports: list = []
        self.gate: asyncio.Event | None = None     # holds the next start open
        self.dispatched: list[tuple] = []
        mgr._build_recap = AsyncMock(return_value="")

        async def dispatch_async(adapter, project_id, handle, text, *, session_id,
                                 dispatch_id):
            self.dispatched.append((adapter, text, dispatch_id))
        mgr._dispatch_async = dispatch_async
        # Every spawn resumes thread "thr-1".
        mgr._determine_resume = MagicMock(return_value=(
            {"session_id": "thr-1"}, "resumed", None))
        self.ws = MagicMock()
        mgr._ws_manager = self.ws

    def _make(self, *a, **kw):
        adapter = MagicMock()
        transport = kw.get("transport")
        gate = self.gate

        async def start(config):
            if gate is not None:
                await gate.wait()

        adapter.start = AsyncMock(side_effect=start)
        adapter.stop = AsyncMock()
        adapter.is_alive = MagicMock(return_value=True)
        adapter.is_idle = MagicMock(return_value=True)
        adapter._transport = transport
        adapter.display_name = kw.get("display_name")
        self.adapters.append(adapter)
        self.transports.append(transport)
        return adapter

    def __enter__(self):
        self._patches = [
            patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", side_effect=self._make),
            patch("agent_os.daemon_v2.sub_agent_manager.resolve_visible_sub_agent_slugs",
                  return_value=["claude-code", "codex"]),
            patch.object(SubAgentManager, "_snapshot_kill_targets", return_value=[]),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()

    def status_broadcasts(self):
        return [c.args[1] for c in self.ws.broadcast.call_args_list
                if c.args[1].get("type") == "sub_agent.status"]

    def stopped_broadcasts(self):
        return [c.args[1] for c in self.ws.broadcast.call_args_list
                if c.args[1].get("type") == "sub_agent.status"
                and c.args[1].get("status") == "stopped"]


async def _settle(mgr):
    for _ in range(5):
        tasks = [t for t in (list(getattr(mgr, "_rebuild_tasks", ()))
                             + list(getattr(mgr, "_takeover_flights", {}).values()))
                 if not t.done()]
        if not tasks:
            await asyncio.sleep(0)
            return
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_user_send_to_a_worker_orbital_started_rebuilds_it_once(tmp_path, initiator):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        old = rig.adapters[0]
        assert old._dispatch_token is None
        mgr._prompt_active.discard(("p1", "s1", "codex"))   # its turn closed

        await mgr.send("p1", "codex", "hand this to claude-code", session_id="s1",
                       initiator=initiator)
        assert old.stop.await_count == 1
        assert len(rig.adapters) == 2
        new = rig.adapters[1]
        assert new._dispatch_token and mgr.dispatch_context(new._dispatch_token)
        assert rig.transports[1]._dispatch_url == URL
        assert rig.transports[1]._resume_session_id == "thr-1"
        # The user's message reached the rebuilt worker, once.
        assert [a for a, _, _ in rig.dispatched] == [old, new]
        assert rig.dispatched[1][1].endswith("hand this to claude-code")

        mgr._prompt_active.discard(("p1", "s1", "codex"))
        await mgr.send("p1", "codex", "and again", session_id="s1", initiator=initiator)
        assert len(rig.adapters) == 2                 # never rebuilt again
        new.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_daemon_turns_never_rebuild_a_worker_orbital_started(tmp_path):
    # The memory reminder, a worker-result delivery and a job message all
    # carry user_pinned; daemon_turn=True is what tells them apart.
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        for _ in range(3):
            mgr._prompt_active.discard(("p1", "s1", "codex"))
            await mgr.send("p1", "codex", "reminder", session_id="s1",
                           initiator="user_pinned", daemon_turn=True)
        assert len(rig.adapters) == 1
        rig.adapters[0].stop.assert_not_awaited()
        assert mgr._dispatch_tokens == {}


@pytest.mark.asyncio
async def test_a_failed_mount_never_loops_and_a_manager_send_keeps_the_token(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        # A top spawn whose mount failed: no token, but spawned at the top.
        orig = rig._make

        def unmounted(*a, **kw):
            adapter = orig(*a, **kw)
            kw["transport"].dispatch_mounted = False
            return adapter
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", side_effect=unmounted):
            await mgr.send("p1", "codex", "one", session_id="s1", initiator="user_pinned")
        assert rig.adapters[0]._dispatch_token is None
        for _ in range(2):
            mgr._prompt_active.discard(("p1", "s1", "codex"))
            await mgr.send("p1", "codex", "again", session_id="s1", initiator="user_pinned")
        assert len(rig.adapters) == 1

        # A seat-holder the manager later dispatches keeps its tool.
        await mgr.send("p1", "claude-code", "one", session_id="s1", initiator="user_pinned")
        seat = rig.adapters[1]
        token = seat._dispatch_token
        assert token
        mgr._prompt_active.discard(("p1", "s1", "claude-code"))
        await mgr.send("p1", "claude-code", "from the manager", session_id="s1",
                       initiator="management_agent")
        assert len(rig.adapters) == 2
        assert mgr.dispatch_context(token) is not None


@pytest.mark.asyncio
async def test_a_user_message_queued_mid_turn_rebuilds_at_turn_close(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        old = rig.adapters[0]
        r1 = await mgr.send("p1", "codex", "user one", session_id="s1",
                            initiator="user_pinned")
        r2 = await mgr.send("p1", "codex", "user two", session_id="s1",
                            initiator="user_pinned")
        assert "queued" in r1 and "queued" in r2
        old.stop.assert_not_awaited()

        await mgr._on_prompt_turn_closed("p1", "codex", session_id="s1")
        await _settle(mgr)
        assert old.stop.await_count == 1
        new = rig.adapters[1]
        assert new._dispatch_token
        # Order kept: "user one" went out on the rebuilt worker, "user two"
        # queues behind it.
        assert [t for _, t, _ in rig.dispatched][1].endswith("user one")
        await mgr._on_prompt_turn_closed("p1", "codex", session_id="s1")
        await _settle(mgr)
        assert [a for a, _, _ in rig.dispatched] == [old, new, new]
        assert rig.dispatched[2][1].endswith("user two")
        assert len(rig.adapters) == 2


@pytest.mark.asyncio
async def test_a_daemon_turn_at_the_head_waits_for_the_user_message_behind_it(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        old = rig.adapters[0]
        await mgr.send("p1", "codex", "memory reminder", session_id="s1",
                       initiator="user_pinned", daemon_turn=True)
        await mgr.send("p1", "codex", "user message", session_id="s1",
                       initiator="user_pinned")

        await mgr._on_prompt_turn_closed("p1", "codex", session_id="s1")
        await _settle(mgr)
        # The reminder ran on the same process.
        old.stop.assert_not_awaited()
        assert rig.dispatched[-1][0] is old
        assert rig.dispatched[-1][1].endswith("memory reminder")

        await mgr._on_prompt_turn_closed("p1", "codex", session_id="s1")
        await _settle(mgr)
        assert old.stop.await_count == 1
        assert rig.dispatched[-1][0] is rig.adapters[1]
        assert rig.dispatched[-1][1].endswith("user message")


# Owner change to §5.1: the rebuild happens when the chat is pinned, not on
# the first message, so it is over before the user finishes typing.

@pytest.mark.asyncio
async def test_pinning_an_idle_worker_orbital_started_rebuilds_it_quietly(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        mgr._prompt_active.discard(("p1", "s1", "codex"))
        old = rig.adapters[0]
        n_dispatched = len(rig.dispatched)
        rig.ws.broadcast.reset_mock()

        flight = mgr.take_seat("p1", "codex", session_id="s1")
        assert flight is not None
        await flight
        assert old.stop.await_count == 1
        new = rig.adapters[1]
        assert new._dispatch_token
        assert rig.transports[1]._resume_session_id == "thr-1"
        assert "`dispatch` tool" in new._orbital_first_turn_note
        # No model request, nothing for the chat, no chip blink: no status
        # push at all, and the new process (no turn open) reads as idle, not
        # as "running" the way a fresh adapter does before its first turn.
        assert len(rig.dispatched) == n_dispatched
        assert rig.status_broadcasts() == []
        assert new._idle is True

        # A second pin, or the first message, does not rebuild again.
        assert mgr.take_seat("p1", "codex", session_id="s1") is None
        await mgr.send("p1", "codex", "hi", session_id="s1", initiator="user_pinned")
        assert len(rig.adapters) == 2
        assert rig.dispatched[-1][0] is new
        assert "`dispatch` tool" in rig.dispatched[-1][1]   # the seat note rides once


@pytest.mark.asyncio
async def test_pinning_a_worker_mid_turn_does_nothing_until_the_turn_closes(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        assert mgr.take_seat("p1", "codex", session_id="s1") is None
        rig.adapters[0].stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("spawned_by", ["user_pinned", None])
async def test_pinning_a_seat_holder_or_an_empty_slot_does_nothing(tmp_path, spawned_by):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        if spawned_by:
            await mgr.send("p1", "codex", "one", session_id="s1", initiator=spawned_by)
            mgr._prompt_active.discard(("p1", "s1", "codex"))
        assert mgr.take_seat("p1", "codex", session_id="s1") is None
        assert len(rig.adapters) == (1 if spawned_by else 0)


@pytest.mark.asyncio
async def test_pinning_without_a_dispatch_endpoint_does_nothing(tmp_path):
    mgr = _mgr(tmp_path, url=None)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        mgr._prompt_active.discard(("p1", "s1", "codex"))
        assert mgr.take_seat("p1", "codex", session_id="s1") is None
        assert len(rig.adapters) == 1


@pytest.mark.asyncio
async def test_a_message_sent_during_the_pin_rebuild_waits_and_is_answered_once(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        mgr._prompt_active.discard(("p1", "s1", "codex"))
        old = rig.adapters[0]
        rig.gate = asyncio.Event()                     # the rebuilt start hangs

        flight = mgr.take_seat("p1", "codex", session_id="s1")
        for _ in range(500):
            if len(rig.adapters) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(rig.adapters) == 2                  # rebuilt start is open
        send = asyncio.create_task(mgr.send(
            "p1", "codex", "quick message", session_id="s1", initiator="user_pinned"))
        reminder = asyncio.create_task(mgr.send(
            "p1", "codex", "reminder", session_id="s1", initiator="user_pinned",
            daemon_turn=True))
        await asyncio.sleep(0.2)
        assert not send.done()
        assert len(rig.dispatched) == 1                # only the manager task

        rig.gate.set()
        await flight
        result = await send
        await reminder
        assert result.startswith("Message sent to codex"), result
        assert len(rig.adapters) == 2                  # no second rebuild
        assert old.stop.await_count == 1
        new = rig.adapters[1]
        sent = [(a, t) for a, t, _ in rig.dispatched[1:]]
        assert sent[0][0] is new and sent[0][1].endswith("quick message")
        assert "`dispatch` tool" in sent[0][1]
        # The reminder queued behind it on the same rebuilt worker.
        assert len(sent) == 1
        await mgr._on_prompt_turn_closed("p1", "codex", session_id="s1")
        await _settle(mgr)
        assert rig.dispatched[-1][0] is new
        assert rig.dispatched[-1][1].endswith("reminder")
        assert len(rig.adapters) == 2


@pytest.mark.asyncio
async def test_a_fresh_send_during_the_pin_rebuild_waits_then_resets(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        mgr._prompt_active.discard(("p1", "s1", "codex"))
        rig.gate = asyncio.Event()
        flight = mgr.take_seat("p1", "codex", session_id="s1")
        for _ in range(500):
            if len(rig.adapters) == 2:
                break
            await asyncio.sleep(0.01)
        send = asyncio.create_task(mgr.send(
            "p1", "codex", "start over", session_id="s1", initiator="user_pinned",
            fresh=True))
        await asyncio.sleep(0.2)
        # The reset does not tear down the half-built successor.
        rig.adapters[1].stop.assert_not_awaited()
        rig.gate.set()
        await flight
        rig.gate = None
        await send
        # Then it resets as asked: the successor is stopped, a fresh one runs.
        assert rig.adapters[1].stop.await_count == 1
        assert len(rig.adapters) == 3
        assert rig.dispatched[-1][0] is rig.adapters[2]


@pytest.mark.asyncio
async def test_a_failed_pin_rebuild_takes_the_chip_back(tmp_path):
    mgr = _mgr(tmp_path)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "manager task", session_id="s1",
                       initiator="management_agent")
        mgr._prompt_active.discard(("p1", "s1", "codex"))
        rig.ws.broadcast.reset_mock()
        orig = rig._make

        def failing(*a, **kw):
            adapter = orig(*a, **kw)
            adapter.start = AsyncMock(side_effect=RuntimeError("no binary"))
            return adapter
        with patch("agent_os.daemon_v2.sub_agent_manager.CLIAdapter", side_effect=failing):
            await mgr.take_seat("p1", "codex", session_id="s1")
        assert len(rig.stopped_broadcasts()) == 1
