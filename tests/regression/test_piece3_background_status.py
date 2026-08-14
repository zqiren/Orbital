# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part A regression: provenance-based background-work classification.

The live-bug class: a sub-agent finishes its *turn* while real background work
(launched via run_in_background) is still alive underneath. The status must
read "background-running" — and an agent holding only warm infrastructure
(MCP servers etc., never registered via provenance) must read "idle".

Classification is by PROVENANCE (tool_use chunks with
metadata.tool_input.run_in_background == true), anchored to the spawned
subtree via a cross-platform psutil direct-child diff — NEVER by
any-live-child, CPU, or creation-time (REPORT-piece3-child-classification.md).
"""

import asyncio
import subprocess
import sys
import time
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.agent.adapters.base import OutputChunk

import psutil

KEY = ("proj_p3", "sess_p3", "claude-code")

# A root process that spawns a direct child AFTER a delay (so the registry's
# baseline excludes it and the capture diff must adopt it) and keeps it alive.
_SPAWNER = (
    "import subprocess, sys, time\n"
    "time.sleep(0.8)\n"
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "p.wait()\n"
)


def _spawn_root(script: str = _SPAWNER) -> psutil.Process:
    proc = subprocess.Popen([sys.executable, "-c", script])
    return psutil.Process(proc.pid)


def _kill_tree(root: psutil.Process):
    try:
        for c in root.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        root.kill()
    except psutil.Error:
        pass


@pytest.fixture
def registry():
    return BackgroundWorkRegistry(capture_window_s=4.0, capture_poll_s=0.1)


# ---------------------------------------------------------------------------
# Registry: anchor capture via cross-platform direct-child diff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_register_anchors_new_direct_child_and_tracks_liveness(registry):
    """A child spawned after registration is adopted as the work anchor;
    liveness follows the subtree. Does NOT depend on the POSIX zsh
    snapshot-shell pattern (cross-platform mechanism, Part A gate)."""
    root = _spawn_root()
    try:
        rec = registry.register(*KEY, command="sleep 60", root_proc=root)
        # capture task adopts the child once it appears (~0.8s)
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and rec.anchor is None:
            await asyncio.sleep(0.1)
        assert rec.anchor is not None, "capture did not adopt the spawned child"
        assert registry.has_live(*KEY) is True
        assert registry.live_commands(*KEY) == ["sleep 60"]

        # killing the anchored subtree flips liveness off
        _kill_tree(rec.anchor)
        await asyncio.sleep(0.2)
        assert registry.has_live(*KEY) is False
        assert registry.live_commands(*KEY) == []
    finally:
        _kill_tree(root)


@pytest.mark.asyncio
async def test_preexisting_infrastructure_is_never_adopted(registry):
    """A child alive BEFORE registration (MCP-server analogue) is baseline
    infrastructure: the record must not anchor to it, and after the capture
    window expires with nothing new the record reads dead — while the
    infrastructure child is still alive."""
    # root spawns its child immediately and keeps it for 60s
    script = (
        "import subprocess, sys, time\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "p.wait()\n"
    )
    root = _spawn_root(script)
    try:
        # wait for the infra child to exist BEFORE registering
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not root.children():
            await asyncio.sleep(0.05)
        assert root.children(), "test setup: infra child did not appear"

        rec = registry.register(*KEY, command="true", root_proc=root)
        await asyncio.sleep(registry.capture_window_s + 0.3)
        assert rec.anchor is None
        assert registry.has_live(*KEY) is False
        assert root.children(), "infrastructure child must remain untouched"
    finally:
        _kill_tree(root)


@pytest.mark.asyncio
async def test_last_background_activity_updates_while_live(registry):
    root = _spawn_root()
    try:
        registry.register(*KEY, command="sleep 60", root_proc=root)
        await asyncio.sleep(1.2)
        assert registry.has_live(*KEY)
        t1 = registry.last_background_activity(*KEY)
        assert t1 is not None
        await asyncio.sleep(0.2)
        assert registry.has_live(*KEY)
        t2 = registry.last_background_activity(*KEY)
        assert t2 >= t1
    finally:
        _kill_tree(root)


# ---------------------------------------------------------------------------
# ProcessManager consume loop: provenance hook
# ---------------------------------------------------------------------------

class _OneShotAdapter:
    """Adapter stub: yields the given chunks, then stops."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._transport = None

    async def read_stream(self):
        for c in self._chunks:
            yield c


@pytest.mark.asyncio
async def test_consume_loop_registers_run_in_background_tool_use():
    pm = ProcessManager(MagicMock(), MagicMock(), lifecycle_observer=None)
    chunks = [
        OutputChunk(text="[Using tool: Bash]", chunk_type="tool_activity",
                    metadata={"tool_name": "Bash",
                              "tool_input": {"command": "sleep 600",
                                             "run_in_background": True}}),
        OutputChunk(text="[Using tool: Bash]", chunk_type="tool_activity",
                    metadata={"tool_name": "Bash",
                              "tool_input": {"command": "ls",
                                             "run_in_background": False}}),
    ]
    await pm.start("proj_p3", "claude-code", _OneShotAdapter(chunks),
                   transcript=None, session_id="sess_p3")
    await asyncio.sleep(0.5)
    recs = pm.background_work.records("proj_p3", "sess_p3", "claude-code")
    assert [r.command for r in recs] == ["sleep 600"]


# ---------------------------------------------------------------------------
# SubAgentManager: three-state status (SDK capability only)
# ---------------------------------------------------------------------------

def _mgr_with_registry(registry):
    pm = MagicMock()
    pm.background_work = registry
    return SubAgentManager(process_manager=pm)


def _stub_adapter(idle=True, supports=True):
    adapter = MagicMock()
    adapter.is_alive.return_value = True
    adapter.is_idle.return_value = idle
    adapter.display_name = "Claude Code"
    adapter._transport = MagicMock()
    adapter._transport.supports_background_status = supports
    return adapter


def _register_live_work(registry, root):
    rec = registry.register(*KEY, command="sleep 60", root_proc=root)
    return rec


@pytest.mark.asyncio
async def test_status_background_running_for_live_provenance_work(registry):
    """Turn done + live registered background work => background-running.
    (A sleeping job at 0% CPU — the exact live-bug shape.)"""
    root = _spawn_root()
    try:
        _register_live_work(registry, root)
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not registry.has_live(*KEY):
            await asyncio.sleep(0.1)
        assert registry.has_live(*KEY)

        mgr = _mgr_with_registry(registry)
        mgr._adapters[("proj_p3", "sess_p3")] = {"claude-code": _stub_adapter(idle=True)}
        active = mgr.list_active("proj_p3", session_id="sess_p3")
        assert active[0]["status"] == "background-running"
        assert mgr.status("proj_p3", "claude-code", session_id="sess_p3") == "background-running"
    finally:
        _kill_tree(root)


@pytest.mark.asyncio
async def test_status_idle_with_only_infrastructure(registry):
    """No provenance records => idle, even if the agent's tree holds warm
    infrastructure (the MCP-server case). NEVER any-live-child."""
    mgr = _mgr_with_registry(registry)
    mgr._adapters[("proj_p3", "sess_p3")] = {"claude-code": _stub_adapter(idle=True)}
    active = mgr.list_active("proj_p3", session_id="sess_p3")
    assert active[0]["status"] == "idle"


@pytest.mark.asyncio
async def test_status_running_while_turn_open(registry):
    mgr = _mgr_with_registry(registry)
    mgr._adapters[("proj_p3", "sess_p3")] = {"claude-code": _stub_adapter(idle=False)}
    active = mgr.list_active("proj_p3", session_id="sess_p3")
    assert active[0]["status"] == "running"


@pytest.mark.asyncio
async def test_degraded_transport_stays_two_state(registry):
    """Transports without the capability (Pipe/ACP/PTY) never read
    background-running, even with live records (scope boundary)."""
    root = _spawn_root()
    try:
        _register_live_work(registry, root)
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not registry.has_live(*KEY):
            await asyncio.sleep(0.1)

        mgr = _mgr_with_registry(registry)
        mgr._adapters[("proj_p3", "sess_p3")] = {
            "claude-code": _stub_adapter(idle=True, supports=False)}
        active = mgr.list_active("proj_p3", session_id="sess_p3")
        assert active[0]["status"] == "idle"
    finally:
        _kill_tree(root)


@pytest.mark.asyncio
async def test_windows_gate_degrades_explicitly(registry):
    """Part A Windows gate (OPEN): when three-state is unsupported on the
    platform, status degrades to two-state — but the registry still tracks
    records (the stop-button warning needs them). Nothing silently lies."""
    registry.three_state_supported = False
    root = _spawn_root()
    try:
        _register_live_work(registry, root)
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not registry.has_live(*KEY):
            await asyncio.sleep(0.1)
        assert registry.has_live(*KEY), "registry must keep tracking on the degraded path"

        mgr = _mgr_with_registry(registry)
        mgr._adapters[("proj_p3", "sess_p3")] = {"claude-code": _stub_adapter(idle=True)}
        active = mgr.list_active("proj_p3", session_id="sess_p3")
        assert active[0]["status"] == "idle"
    finally:
        _kill_tree(root)


@pytest.mark.asyncio
async def test_sdk_transport_declares_capability():
    """The capability flag lives on SDKTransport; degraded transports lack it."""
    from agent_os.agent.transports.pipe_transport import PipeTransport
    from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
    try:
        from agent_os.agent.transports.sdk_transport import SDKTransport
        assert getattr(SDKTransport, "supports_background_status", False) is True
    except ImportError:
        pass  # SDK not installed in this env — capability check moot
    assert getattr(PipeTransport, "supports_background_status", False) is False
    assert getattr(ACPSDKTransport, "supports_background_status", False) is False
    from agent_os.agent.transports.codex_transport import CodexTransport
    # LOCKED (TASK-codex-appserver-transport): Codex is truthfully two-state —
    # no between-turn background work exists (probe A5c). Never set True for
    # UI parity with Claude Code.
    assert getattr(CodexTransport, "supports_background_status", False) is False
