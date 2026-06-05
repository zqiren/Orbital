# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live spawn-on-demand dispatch (TASK-collapse-dispatch-to-send, TEST RULE 2+5).

Drives the REAL chain — SubAgentManager.send (empty slate) → internal
start() via registry+setup_engine → SDKTransport → real claude subprocess →
dispatch — and asserts the H2 contract on the real path: ONE session
injection for the dispatch (on_message_routed), ZERO standalone "started"
injections, with the piece-2 resume clause riding the dispatch ack.

Requires the real claude binary + claude-agent SDK; skipped otherwise.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from agent_os.agent.transports.sdk_transport import HAS_SDK
from agent_os.agents.registry import AgentRegistry
from agent_os.agents.setup_engine import SetupEngine
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

pytestmark = pytest.mark.live_daemon  # heavy: spawns real claude — opt-in


def _find_claude() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    for c in (Path.home() / ".claude/local/node_modules/.bin/claude",
              Path.home() / ".claude/local/claude"):
        if c.exists():
            return str(c)
    return None


CLAUDE = _find_claude()

requires_claude = pytest.mark.skipif(
    not HAS_SDK or CLAUDE is None, reason="claude SDK/binary not available")

MANIFESTS = os.path.join(
    os.path.dirname(__file__), "..", "..", "agent_os", "agents", "manifests")


@requires_claude
async def test_send_to_not_running_agent_spawns_tasks_and_announces_once(tmp_path):
    registry = AgentRegistry()
    registry.load_directory(MANIFESTS)
    setup_engine = SetupEngine(registry)

    project_store = MagicMock()
    project_store.get_project = MagicMock(
        return_value={"workspace": str(tmp_path)})

    mock_agent_mgr = MagicMock()
    mock_agent_mgr.inject_system_message = AsyncMock(return_value="delivered")
    mock_agent_mgr.record_sub_agent_thread = MagicMock()
    ws = MagicMock()
    observer = LifecycleObserver(agent_manager=mock_agent_mgr, ws_manager=ws)
    pm = ProcessManager(ws_manager=ws, activity_translator=MagicMock(),
                        lifecycle_observer=observer)
    mgr = SubAgentManager(
        process_manager=pm, registry=registry, setup_engine=setup_engine,
        project_store=project_store, lifecycle_observer=observer,
    )

    try:
        # ONE call on an empty slate: spawn + dispatch.
        result = await asyncio.wait_for(
            mgr.send("p1", "claude-code", "Reply with exactly the two letters: OK",
                     session_id="s1"),
            timeout=60.0,
        )
        assert result.startswith("Message sent to claude-code"), result
        # Piece-2 honesty rides the dispatch ack.
        assert "fresh session — first spawn" in result, result

        # H2 on the REAL path: at dispatch time exactly ONE session injection
        # (the on_message_routed line) and NO standalone "started" injection.
        contents = [c.args[1] for c in
                    mock_agent_mgr.inject_system_message.await_args_list]
        started = [c for c in contents if "started (initiated by" in c]
        routed = [c for c in contents if "Message sent to claude-code" in c]
        assert len(started) == 0, (
            f"internal spawn double-announced: {started}")
        assert len(routed) == 1, contents

        # The turn completes honestly (Defect B contract holds end-to-end).
        deadline = time.monotonic() + 150.0
        done: list[str] = []
        while time.monotonic() < deadline:
            done = [c.args[1] for c in
                    mock_agent_mgr.inject_system_message.await_args_list
                    if "completed. Summary:" in c.args[1]]
            if done:
                break
            await asyncio.sleep(0.5)
        assert done, "no honest completion injection arrived"
        assert "OK" in done[0]
    finally:
        # Never leak the claude subprocess.
        await asyncio.wait_for(mgr.stop_all("p1", session_id="s1"), timeout=30.0)
        # Belt and braces: confirm no claude descendants survived.
        me = psutil.Process(os.getpid())
        leftovers = [c.pid for c in me.children(recursive=True)
                     if "claude" in (c.name() or "").lower()]
        for pid in leftovers:
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                pass
        assert leftovers == [], f"claude survived stop_all: {leftovers}"
