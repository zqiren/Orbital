# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keyless Workbench routing (2026-11-23): ask a signed-in worker ONE
question and leave nothing in its history.

Only the two transports that can do that are used: Claude Code over the SDK
(``--no-session-persistence``, no tools, no MCP servers, one turn) and Codex
over the app-server (an ``ephemeral`` thread: no rollout file, read-only
sandbox, never asks for approval, no MCP servers). Every other transport
(ACP's ``session/new`` writes a session in the user's tool; PTY is a
terminal) is refused without spawning anything. The transports are the ones
workers already run on, in an ``ephemeral`` mode; the CLIs are faked here.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent.transports.codex_transport import CodexTransport
from agent_os.agent.transports.sdk_transport import SDKTransport
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


# ── SDKTransport ephemeral mode ─────────────────────────────────────────

async def _sdk_options(transport) -> dict:
    with patch("agent_os.agent.transports.sdk_transport.ClaudeSDKClient") as client, \
         patch("agent_os.agent.transports.sdk_transport.ClaudeAgentOptions") as options:
        client.return_value.connect = AsyncMock()
        client.return_value.get_server_info = AsyncMock(return_value={})
        await transport.start("claude", [], "/neutral")
        return options.call_args[1]


@pytest.mark.asyncio
async def test_sdk_ephemeral_leaves_no_session_and_has_no_tools():
    opts = await _sdk_options(SDKTransport(ephemeral=True))
    assert opts["extra_args"] == {"no-session-persistence": None}
    assert opts["tools"] == []
    assert opts["strict_mcp_config"] is True
    assert opts["max_turns"] == 1
    assert opts["cwd"] == "/neutral"


@pytest.mark.asyncio
async def test_sdk_worker_mode_is_unchanged():
    opts = await _sdk_options(SDKTransport())
    for key in ("extra_args", "tools", "strict_mcp_config", "max_turns"):
        assert key not in opts


def test_sdk_options_know_every_ephemeral_field():
    """The installed SDK must render each option, or a session would be
    written (``extra_args`` carries the flag the CLI itself documents)."""
    from claude_agent_sdk import ClaudeAgentOptions
    fields = getattr(ClaudeAgentOptions, "__dataclass_fields__", {})
    assert {"extra_args", "tools", "strict_mcp_config", "max_turns"} <= set(fields)


# ── CodexTransport ephemeral mode ───────────────────────────────────────

def test_codex_ephemeral_thread_writes_no_rollout_and_runs_nothing():
    t = CodexTransport(ephemeral=True)
    method, params = t._thread_open_request("/neutral")
    assert method == "thread/start"
    assert params["ephemeral"] is True
    assert params["approvalPolicy"] == "never"
    assert params["sandbox"] == "read-only"
    assert params["config"]["mcp_servers"] == {}
    assert params["cwd"] == "/neutral"


def test_codex_worker_thread_is_unchanged():
    t = CodexTransport()
    _method, params = t._thread_open_request("/ws")
    assert "ephemeral" not in params
    assert "mcp_servers" not in (params.get("config") or {})


def test_codex_ephemeral_skips_the_worker_side_reads():
    t = CodexTransport(ephemeral=True)
    assert t._background_reads is False
    assert CodexTransport()._background_reads is True


# ── SubAgentManager.can_ask_once / ask_once ─────────────────────────────

def _manager(transport_by_slug: dict[str, str], *, statuses=None):
    registry = MagicMock()
    registry.get.side_effect = lambda slug: (
        SimpleNamespace(slug=slug, runtime=SimpleNamespace(transport=transport_by_slug[slug]))
        if slug in transport_by_slug else None)
    setup = MagicMock()
    setup.get_adapter_config.side_effect = lambda slug, project_workspace: {
        "command": f"/bin/{slug}", "args": ["--model", "m-1"], "env": {"X": "1"},
        "workspace": project_workspace}
    setup.check_all_cached.return_value = statuses or []
    return SubAgentManager(process_manager=MagicMock(), registry=registry, setup_engine=setup)


def test_only_sdk_and_codex_app_server_can_ask_once():
    mgr = _manager({"claude-code": "sdk", "codex": "codex-appserver",
                    "kimi-code": "acp-sdk", "aider": "pty", "pi": "pi-rpc"})
    assert mgr.can_ask_once("claude-code") is True
    assert mgr.can_ask_once("codex") is True
    for slug in ("kimi-code", "aider", "pi", "nope"):
        assert mgr.can_ask_once(slug) is False


def test_the_bundled_registry_never_lets_antigravity_route():
    """Merge check (Welcome × Antigravity, 2026-11-24): with the real
    manifests, keyless routing can still only ask Claude Code or Codex.
    Antigravity runs on ACP (session/new writes into agy's own history), so
    it is skipped wherever it sits in the registry order."""
    import os
    from agent_os.agents.registry import AgentRegistry
    registry = AgentRegistry()
    registry.load_directory(os.path.normpath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "agent_os", "agents", "manifests")))
    setup = MagicMock()
    setup.check_all_cached.return_value = []
    mgr = SubAgentManager(process_manager=MagicMock(), registry=registry, setup_engine=setup)
    assert registry.get("antigravity") is not None
    assert registry.get("gemini-cli") is None
    assert mgr.can_ask_once("antigravity") is False
    assert mgr.can_ask_once("claude-code") is True
    assert mgr.can_ask_once("codex") is True


def test_an_agent_whose_login_expired_cannot_ask_once():
    expired = SimpleNamespace(slug="claude-code", auth_expired=True)
    fine = SimpleNamespace(slug="codex", auth_expired=False)
    mgr = _manager({"claude-code": "sdk", "codex": "codex-appserver"},
                   statuses=[expired, fine])
    assert mgr.can_ask_once("claude-code") is False
    assert mgr.can_ask_once("codex") is True


class FakeTransport:
    instances: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = None
        self.sent = []
        self.stopped = False
        self.fail = None
        FakeTransport.instances.append(self)

    async def start(self, command, args, workspace, env=None):
        self.started = (command, list(args), workspace, dict(env or {}))

    async def send(self, message):
        self.sent.append(message)
        if FakeTransport.fail:
            raise FakeTransport.fail
        return '{"project": "p1"}'

    async def stop(self):
        self.stopped = True


@pytest.fixture
def fake_transports(monkeypatch):
    import agent_os.agent.transports.codex_transport as cx
    import agent_os.agent.transports.sdk_transport as sx
    FakeTransport.instances = []
    FakeTransport.fail = None
    monkeypatch.setattr(sx, "SDKTransport", FakeTransport)
    monkeypatch.setattr(cx, "CodexTransport", FakeTransport)
    return FakeTransport


@pytest.mark.asyncio
async def test_ask_once_runs_one_ephemeral_turn_and_stops(fake_transports):
    mgr = _manager({"claude-code": "sdk", "codex": "codex-appserver"})
    assert await mgr.ask_once("claude-code", "route this", cwd="/neutral") == '{"project": "p1"}'
    assert await mgr.ask_once("codex", "route this", cwd="/neutral") == '{"project": "p1"}'
    sdk, codex = fake_transports.instances
    assert sdk.kwargs["ephemeral"] is True and codex.kwargs["ephemeral"] is True
    # The worker's own model setting rides along, as on a dispatch.
    assert sdk.kwargs["model"] == "m-1"
    for t, slug in ((sdk, "claude-code"), (codex, "codex")):
        command, args, workspace, env = t.started
        assert command == f"/bin/{slug}" and workspace == "/neutral"
        assert env["X"] == "1" and "CLAUDECODE" not in env
        assert t.sent == ["route this"] and t.stopped is True
    # The config is resolved for the neutral folder, never a project.
    calls = mgr._setup_engine.get_adapter_config.call_args_list
    assert all(c.kwargs["project_workspace"] == "/neutral" for c in calls)


@pytest.mark.asyncio
async def test_ask_once_stops_the_process_when_the_turn_fails(fake_transports):
    mgr = _manager({"claude-code": "sdk"})
    fake_transports.fail = RuntimeError("signed out")
    with pytest.raises(RuntimeError):
        await mgr.ask_once("claude-code", "route this", cwd="/neutral")
    assert fake_transports.instances[0].stopped is True


@pytest.mark.asyncio
async def test_ask_once_refuses_other_transports_without_spawning(fake_transports):
    mgr = _manager({"kimi-code": "acp-sdk"})
    with pytest.raises(ValueError):
        await mgr.ask_once("kimi-code", "route this", cwd="/neutral")
    assert fake_transports.instances == []
    mgr._setup_engine.get_adapter_config.assert_not_called()


@pytest.mark.asyncio
async def test_an_error_text_from_send_is_a_failure(fake_transports, monkeypatch):
    """``send()`` reports an SDK failure as ``"Error: …"`` text; the route
    must see a failure, not a reply to parse."""
    async def error_text(self, message):
        return "Error: SDK query failed: boom"
    monkeypatch.setattr(FakeTransport, "send", error_text)
    mgr = _manager({"claude-code": "sdk"})
    with pytest.raises(RuntimeError):
        await mgr.ask_once("claude-code", "route this", cwd="/neutral")
    assert fake_transports.instances[0].stopped is True


# ── Nothing written in the neutral folder (live finding 2026-11-23) ─────

def test_sdk_ephemeral_writes_no_usage_ledger(tmp_path):
    """A worker's usage goes to its project's ledger; a routing question has
    no project, so nothing is written (the first live run left
    ``<tmp>/orbital-route/orbital/ledger/usage.jsonl``)."""
    usage = SimpleNamespace(usage={"input_tokens": 10, "output_tokens": 2},
                            total_cost_usd=0.01)
    t = SDKTransport(ephemeral=True)
    t._workspace = str(tmp_path)
    t._capture_usage(usage)
    assert not (tmp_path / "orbital").exists()
    worker = SDKTransport()
    worker._workspace = str(tmp_path)
    worker._capture_usage(usage)
    assert (tmp_path / "orbital" / "ledger").is_dir()   # the worker path is unchanged


def test_codex_ephemeral_writes_no_usage_ledger(tmp_path):
    last = {"totalTokens": 12, "inputTokens": 10, "cachedInputTokens": 0,
            "outputTokens": 2, "reasoningOutputTokens": 0}
    for ephemeral, written in ((True, False), (False, True)):
        ws = tmp_path / f"e{ephemeral}"
        ws.mkdir()
        t = CodexTransport(ephemeral=ephemeral)
        t._workspace = str(ws)
        t._thread_id = "T1"
        t._codex_last_usage[("T1", "U1")] = last
        t._emit_token_usage("U1")
        assert (ws / "orbital" / "ledger").is_dir() is written
