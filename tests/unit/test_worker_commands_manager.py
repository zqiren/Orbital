# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 166: the manager resolves a worker's cold probe (the route only asks)
and hands every new transport the store's publish sink."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2 import worker_commands
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _manager(transport_by_slug: dict[str, str]):
    registry = MagicMock()
    registry.get.side_effect = lambda slug: (
        SimpleNamespace(slug=slug, runtime=SimpleNamespace(transport=transport_by_slug[slug]))
        if slug in transport_by_slug else None)
    setup = MagicMock()
    setup.get_adapter_config.side_effect = lambda slug, project_workspace: {
        "command": f"/bin/{slug}", "workspace": project_workspace, "env": {"X": "1"}}
    return SubAgentManager(process_manager=MagicMock(), registry=registry, setup_engine=setup)


@pytest.mark.asyncio
async def test_probe_dispatches_by_transport(monkeypatch):
    import agent_os.agent.transports.claude_models as cm
    import agent_os.agent.transports.codex_models as xm
    calls = []

    async def fake_claude(binary, workspace, *, env=None, timeout=15.0):
        calls.append(("claude", binary, workspace, env.get("X")))
        return []

    async def fake_codex(binary, workspace, *, env=None, timeout=10.0):
        calls.append(("codex", binary, workspace, env.get("X")))
        return []
    monkeypatch.setattr(cm, "fetch_claude_commands", fake_claude)
    monkeypatch.setattr(xm, "fetch_codex_skills", fake_codex)
    mgr = _manager({"claude-code": "sdk", "codex": "codex-appserver",
                    "kimi-code": "acp-sdk", "aider": "pty"})
    assert await mgr.probe_worker_commands("claude-code", "/ws") == []
    assert await mgr.probe_worker_commands("codex", "/ws") == []
    # No side-effect-free probe for ACP (session/new litters), none for pty,
    # nothing for an unknown slug — and nothing is spawned for them.
    assert await mgr.probe_worker_commands("kimi-code", "/ws") is None
    assert await mgr.probe_worker_commands("aider", "/ws") is None
    assert await mgr.probe_worker_commands("nope", "/ws") is None
    assert [c[0] for c in calls] == ["claude", "codex"]
    assert calls[1] == ("codex", "/bin/codex", "/ws", "1")


def test_sink_publishes_under_slug_and_workspace(tmp_path):
    store = worker_commands.WorkerCommandStore(str(tmp_path / "wc.json"))
    worker_commands.install(store)
    try:
        transport = MagicMock()
        SubAgentManager._attach_commands_sink(transport, "kimi-code", "/ws")
        sink = transport.set_commands_sink.call_args[0][0]
        sink([{"name": "compact", "description": "", "invoke": "/compact ",
               "hint": "", "enabled": True}])
        assert store.get("kimi-code", "/ws")[0]["name"] == "compact"
    finally:
        worker_commands.install(None)
