# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Connectors shared with workers: which spawn gets what, and when it rebuilds.

- A pinned (top-of-session) worker and a worker the manager or a worker
  dispatched both get the connectors that are connected, enabled for the
  project and allowed for that agent, on the same ``orbital`` mount. Only the
  top of a session gets ``dispatch`` (spec 161 stands): a dispatched worker's
  token says ``can_dispatch=False``.
- A worker with no effective connectors gets exactly what it gets today.
- A change of the effective set (access, the project toggle, connect /
  disconnect) rebuilds an idle worker on its next message, resuming its
  thread; a transport that cannot mount is never rebuilt for it.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agents.manifest import ManifestLoader
from agent_os.agents.registry import AgentRegistry
from agent_os.connectors import load_catalog
from agent_os.connectors.manager import ConnectorManager
from agent_os.daemon_v2.dispatch_tools import TOKEN_ENV, DispatchContext
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

from tests.unit._mock_mcp_server import build_mock_server, in_memory_opener

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
_MANIFESTS = os.path.join(_REPO, "agent_os", "agents", "manifests")
URL = "http://127.0.0.1:8541/mcp"


async def _cm(tmp_path):
    cm = ConnectorManager(catalog=load_catalog(), credential_store=None,
                          data_dir=str(tmp_path / "data"),
                          session_opener=in_memory_opener(build_mock_server()))
    os.makedirs(tmp_path / "data", exist_ok=True)
    await cm.add_custom("Mock", "https://mock/mcp", "none")
    return cm


def _mgr(tmp_path, cm, *, enabled=("custom-mock",)):
    registry = AgentRegistry()
    for f in ("claude_code.yaml", "codex.yaml", "minimax-code.yaml", "pi.yaml", "aider.yaml"):
        registry.register(ManifestLoader.load(os.path.join(_MANIFESTS, f)))
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    setup_engine = MagicMock()
    setup_engine.get_adapter_config.side_effect = lambda slug, project_workspace: {
        "command": f"/fake/{slug}", "args": [], "workspace": str(ws),
        "approval_patterns": [], "env": {"PATH": "/usr/bin"},
        "network_domains": [], "interactive": False,
    }
    setup_engine.check_all.return_value = []
    project = {"workspace": str(ws), "enabled_sub_agents": ["claude-code", "codex"],
               "enabled_connectors": list(enabled)}
    project_store = MagicMock()
    project_store.get_project.return_value = project
    process_manager = MagicMock()
    process_manager.start = AsyncMock()
    process_manager.stop = AsyncMock()
    mgr = SubAgentManager(process_manager=process_manager, registry=registry,
                          setup_engine=setup_engine, project_store=project_store)
    mgr._dispatch_url = URL
    mgr._connector_manager = cm
    mgr._project = project
    return mgr


async def _spawn(mgr, handle, initiator, *, mounts=True):
    holder = {}

    async def start(config):
        t = holder.get("transport")
        if type(t).__name__ in ("ACPSDKTransport", "PiRPCTransport") and t.dispatch_mcp:
            t.dispatch_mounted = mounts

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


# ---------------------------------------------------------------- who gets what

@pytest.mark.parametrize("initiator", ["user_pinned", "queue_item"])
async def test_a_pinned_worker_gets_dispatch_and_its_connectors(tmp_path, initiator):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    adapter, transport, _ = await _spawn(mgr, "claude-code", initiator)
    ctx = mgr.dispatch_context(transport.dispatch_mcp["token"])
    assert ctx == DispatchContext("p1", "s1", "claude-code",
                                  agents=("claude-code", "codex"),
                                  can_dispatch=True, connectors=("custom-mock",))
    assert adapter._connector_ids == ("custom-mock",)


@pytest.mark.parametrize("initiator", ["management_agent", "worker:codex"])
async def test_a_dispatched_claude_code_gets_its_connectors_but_not_dispatch(
        tmp_path, initiator):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    adapter, transport, config = await _spawn(mgr, "claude-code", initiator)
    cfg = transport.dispatch_mcp
    assert cfg["url"] == URL
    ctx = mgr.dispatch_context(cfg["token"])
    assert ctx.can_dispatch is False and ctx.connectors == ("custom-mock",)
    assert TOKEN_ENV not in (config.env or {})
    note = adapter._orbital_first_turn_note
    assert "dispatch" not in note                 # no seat line, no no-tool line
    assert "Mock" in note                         # the connector line
    # Spec 184 is untouched: it was not spawned at the top.
    assert adapter._spawned_top is False


async def test_a_dispatched_codex_mounts_for_connectors_without_dispatch(tmp_path):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    adapter, transport, config = await _spawn(mgr, "codex", "management_agent")
    assert transport._dispatch_url == URL
    assert transport._serves_dispatch is False
    assert transport._long_tool_calls is True
    ctx = mgr.dispatch_context(config.env[TOKEN_ENV])
    assert ctx.can_dispatch is False and ctx.connectors == ("custom-mock",)


async def test_a_dispatched_acp_worker_mounts_for_connectors(tmp_path):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    adapter, transport, _ = await _spawn(mgr, "minimax-code", "management_agent")
    assert mgr.dispatch_context(transport.dispatch_mcp["token"]).can_dispatch is False


async def test_an_acp_worker_without_http_mcp_loses_the_token(tmp_path):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    adapter, transport, _ = await _spawn(mgr, "minimax-code", "management_agent",
                                         mounts=False)
    assert mgr._dispatch_tokens == {}
    assert adapter._dispatch_token is None
    assert "Mock" not in (adapter._orbital_first_turn_note or "")


@pytest.mark.parametrize("access", [("selected", ["codex"]), ("selected", [])])
async def test_access_not_allowing_the_agent_gives_it_nothing(tmp_path, access):
    cm = await _cm(tmp_path)
    cm.set_access("custom-mock", *access)
    mgr = _mgr(tmp_path, cm)
    adapter, transport, config = await _spawn(mgr, "claude-code", "management_agent")
    assert transport.dispatch_mcp is None
    assert mgr._dispatch_tokens == {}
    assert adapter._connector_ids == ()


async def test_a_project_with_the_connector_off_gives_the_worker_nothing(tmp_path):
    mgr = _mgr(tmp_path, await _cm(tmp_path), enabled=())
    adapter, transport, config = await _spawn(mgr, "claude-code", "management_agent")
    assert transport.dispatch_mcp is None
    assert mgr._dispatch_tokens == {}


async def test_no_endpoint_means_no_token_and_no_record(tmp_path):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    mgr._dispatch_url = None
    adapter, transport, _ = await _spawn(mgr, "claude-code", "management_agent")
    assert transport.dispatch_mcp is None
    assert adapter._connector_ids is None


async def test_a_pty_worker_is_never_recorded_for_connectors(tmp_path):
    mgr = _mgr(tmp_path, await _cm(tmp_path))
    with patch("agent_os.agent.transports.pty_transport.PTYTransport.__init__",
               return_value=None):
        adapter, transport, _ = await _spawn(mgr, "aider", "management_agent")
    assert mgr._dispatch_tokens == {}
    assert adapter._connector_ids is None


# ---------------------------------------------------------------- cost guard

@pytest.mark.parametrize("initiator", ["user_pinned", "management_agent"])
@pytest.mark.parametrize("handle", ["claude-code", "codex"])
async def test_no_connectors_is_byte_identical_to_today(tmp_path, initiator, handle):
    """The same spawn with and without a connector manager: same transport
    config, same env, same first-turn note."""
    seen = []
    for cm in (None, await _cm(tmp_path)):
        mgr = _mgr(tmp_path, cm, enabled=() if cm is not None else ("custom-mock",))
        adapter, transport, config = await _spawn(mgr, handle, initiator)
        token = (transport.dispatch_mcp or {}).get("token") if handle == "claude-code" \
            else (config.env or {}).get(TOKEN_ENV)
        mcp = dict(transport.dispatch_mcp or {}) if handle == "claude-code" else None
        if mcp:
            mcp["token"] = "T"
        env = {k: ("T" if k == TOKEN_ENV else v) for k, v in (config.env or {}).items()}
        seen.append({
            "mcp": mcp,
            "codex": (transport._dispatch_url, transport._dispatch_argv())
            if handle == "codex" else None,
            "env": env,
            "note": adapter._orbital_first_turn_note,
            "ctx": mgr.dispatch_context(token) if token else None,
        })
    assert seen[0] == seen[1]


# ---------------------------------------------------------------- codex argv

async def _codex_argv(transport):
    seen = {}

    class _Stop(Exception):
        pass

    async def fake_exec(*argv, **kwargs):
        seen["argv"] = argv
        raise _Stop()
    with patch("agent_os.agent.transports.codex_transport.asyncio.create_subprocess_exec",
               side_effect=fake_exec):
        with pytest.raises(_Stop):
            await transport.start("/fake/codex", [], "/ws", {TOKEN_ENV: "t"})
    return seen["argv"]


async def test_codex_with_connectors_waits_long_enough_for_the_user():
    from agent_os.agent.transports.codex_transport import (
        CONNECTOR_TOOL_TIMEOUT_S, CodexTransport)
    argv = await _codex_argv(CodexTransport(dispatch_url=URL, long_tool_calls=True))
    assert argv[-2:] == ("-c", f"mcp_servers.orbital.tool_timeout_sec={CONNECTOR_TOOL_TIMEOUT_S}")
    assert CONNECTOR_TOOL_TIMEOUT_S >= 600
    plain = await _codex_argv(CodexTransport(dispatch_url=URL))
    assert not any("tool_timeout_sec" in a for a in plain)


@pytest.mark.parametrize("serves_dispatch,tools,mounted", [
    (True, {"dispatch": {}}, True),
    (True, {"custom-mock__echo_read": {}}, False),
    (False, {"custom-mock__echo_read": {}}, True),
    (False, {}, False),
])
async def test_codex_mount_check_looks_for_what_this_token_serves(serves_dispatch, tools, mounted):
    from agent_os.agent.transports.codex_transport import CodexTransport
    t = CodexTransport(dispatch_url=URL, serves_dispatch=serves_dispatch)
    t._thread_id = "th"
    t._request = AsyncMock(return_value={"data": [{"name": "orbital", "tools": tools}]})
    await t._check_dispatch_mount()
    assert t.dispatch_mounted is mounted


# ---------------------------------------------------------------- rebuild trigger

async def _settle(mgr):
    for _ in range(5):
        tasks = [t for t in list(getattr(mgr, "_rebuild_tasks", ())) if not t.done()]
        if not tasks:
            await asyncio.sleep(0)
            return
        await asyncio.gather(*tasks, return_exceptions=True)


class _Rig:
    def __init__(self, mgr):
        self.mgr = mgr
        self.adapters = []
        self.transports = []
        self.dispatched = []
        mgr._build_recap = AsyncMock(return_value="")

        async def dispatch_async(adapter, project_id, handle, text, *, session_id,
                                 dispatch_id):
            self.dispatched.append((adapter, text))
        mgr._dispatch_async = dispatch_async
        mgr._determine_resume = MagicMock(return_value=(
            {"session_id": "thr-1"}, "resumed", None))
        mgr._ws_manager = MagicMock()

    def _make(self, *a, **kw):
        adapter = MagicMock()
        adapter.start = AsyncMock()
        adapter.stop = AsyncMock()
        adapter.is_alive = MagicMock(return_value=True)
        adapter.is_idle = MagicMock(return_value=True)
        adapter._transport = kw.get("transport")
        self.adapters.append(adapter)
        self.transports.append(kw.get("transport"))
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


def _turn_closed(mgr, handle="claude-code"):
    mgr._prompt_active.discard(("p1", "s1", handle))


@pytest.mark.parametrize("change", ["access", "project_off", "connector_removed"])
async def test_a_changed_set_rebuilds_the_idle_worker_at_its_next_message(tmp_path, change):
    cm = await _cm(tmp_path)
    mgr = _mgr(tmp_path, cm)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "claude-code", "one", session_id="s1", initiator="user_pinned")
        first = rig.adapters[0]
        assert mgr.dispatch_context(first._dispatch_token).connectors == ("custom-mock",)
        _turn_closed(mgr)
        await mgr.send("p1", "claude-code", "two", session_id="s1", initiator="user_pinned")
        assert len(rig.adapters) == 1                  # nothing changed: no rebuild

        if change == "access":
            cm.set_access("custom-mock", "selected", ["codex"])
        elif change == "project_off":
            mgr._project["enabled_connectors"] = []
        else:
            await cm.remove_custom("custom-mock")
        _turn_closed(mgr)
        await mgr.send("p1", "claude-code", "three", session_id="s1", initiator="user_pinned")
        assert first.stop.await_count == 1
        assert len(rig.adapters) == 2
        second = rig.adapters[1]
        # Rebuilt as the same seat-holder, resuming its thread, without the connector.
        ctx = mgr.dispatch_context(second._dispatch_token)
        assert ctx.can_dispatch is True and ctx.connectors == ()
        assert rig.transports[1]._resume_session_id == "thr-1"
        assert rig.dispatched[-1][0] is second
        assert mgr.dispatch_context(first._dispatch_token) is None


async def test_access_granted_later_reaches_the_worker_at_its_next_message(tmp_path):
    cm = await _cm(tmp_path)
    cm.set_access("custom-mock", "selected", [])
    mgr = _mgr(tmp_path, cm)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "codex", "one", session_id="s1", initiator="management_agent")
        assert rig.adapters[0]._dispatch_token is None
        cm.set_access("custom-mock", "all", [])
        _turn_closed(mgr, "codex")
        await mgr.send("p1", "codex", "two", session_id="s1", initiator="management_agent")
        assert len(rig.adapters) == 2
        ctx = mgr.dispatch_context(rig.adapters[1]._dispatch_token)
        assert ctx.can_dispatch is False and ctx.connectors == ("custom-mock",)


async def test_a_change_mid_turn_waits_for_the_turn_to_close(tmp_path):
    cm = await _cm(tmp_path)
    mgr = _mgr(tmp_path, cm)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "claude-code", "one", session_id="s1", initiator="user_pinned")
        cm.set_access("custom-mock", "selected", [])
        # Mid-turn: the message queues, the worker is untouched.
        await mgr.send("p1", "claude-code", "two", session_id="s1", initiator="user_pinned")
        assert len(rig.adapters) == 1
        rig.adapters[0].stop.assert_not_awaited()
        # The turn closes: the queued message reaches a rebuilt worker.
        await mgr._on_prompt_turn_closed("p1", "claude-code", session_id="s1",
                                         cause="turn_complete")
        await _settle(mgr)
        assert len(rig.adapters) == 2
        assert rig.dispatched[-1][0] is rig.adapters[1]
        assert rig.dispatched[-1][1].endswith("two")


async def test_a_worker_that_cannot_mount_is_never_rebuilt_for_connectors(tmp_path):
    cm = await _cm(tmp_path)
    mgr = _mgr(tmp_path, cm)
    with _Rig(mgr) as rig:
        await mgr.send("p1", "claude-code", "one", session_id="s1", initiator="user_pinned")
        rig.adapters[0]._connector_ids = None       # what a pipe/PTY spawn records
        cm.set_access("custom-mock", "selected", [])
        _turn_closed(mgr)
        await mgr.send("p1", "claude-code", "two", session_id="s1", initiator="user_pinned")
        assert len(rig.adapters) == 1


# Live 2026-11-17: Claude Code's own MCP tool timeout ended a write after
# 60 s while the user's approval card was up. A worker that got connectors
# waits as long as Codex does; one without them mounts exactly as before
# (test_dispatch_attach_rule pins that config).
async def test_claude_code_with_connectors_waits_long_enough_for_the_user():
    from agent_os.agent.transports.sdk_transport import (
        CONNECTOR_TOOL_TIMEOUT_MS, SDKTransport)
    transport = SDKTransport(dispatch_mcp={"url": URL, "token": "t", "connectors": True,
                                           "dispatch": False})
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client, \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as options:
        client.return_value.connect = AsyncMock()
        await transport.start("claude", [], "/ws")
    assert options.call_args.kwargs["mcp_servers"] == {"orbital": {
        "type": "http", "url": URL, "headers": {"Authorization": "Bearer t"},
        "timeout": CONNECTOR_TOOL_TIMEOUT_MS}}
    assert CONNECTOR_TOOL_TIMEOUT_MS >= 600_000
