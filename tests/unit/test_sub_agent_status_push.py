# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The worker status pushes itself (``sub_agent.status``).

The bug this protects: a pinned send to an already-running Claude Code
worker left the chip on "Idle" for three minutes. The daemon knew the turn
was open (``_idle=False`` before ``transport.dispatch``) but nothing
broadcast it: the SPA only refetched on unrelated events. Every real
transition now goes through ``SubAgentManager._publish_status``, which
computes the status and broadcasts only on change.

Uses the real ProcessManager consumer, the real CLIAdapter and the real
BackgroundWorkRegistry; the transport is a minimal fake except in the
approval tests, which drive the real SDK and Codex transports.
"""

import asyncio
import subprocess
import sys
from unittest.mock import MagicMock

import psutil
import pytest

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

P, S, H = "proj_status", "proj_status_sess0001", "claude-code"


class _WS:
    def __init__(self):
        self.events: list[dict] = []

    def broadcast(self, project_id, payload):
        self.events.append(dict(payload))

    def statuses(self) -> list[dict]:
        return [e for e in self.events if e.get("type") == "sub_agent.status"]


class _FakeTransport(AgentTransport):
    """Streams whatever the test pushes; non-blocking ``dispatch`` like SDK."""

    supports_background_status = True

    def __init__(self):
        self._q: asyncio.Queue = asyncio.Queue()
        self._alive = True
        self._pending_approvals: dict = {}
        self.dispatched: list[str] = []

    async def start(self, command, args, workspace, env=None):
        pass

    async def send(self, message):
        return None

    async def dispatch(self, message, **_kw):
        self.dispatched.append(message)

    async def read_stream(self):
        while self._alive:
            event = await self._q.get()
            if event is None:
                break
            yield event

    def push(self, event_type, **data):
        self._q.put_nowait(TransportEvent(event_type, data, data.get("text", "")))

    def die(self):
        self._alive = False
        self._q.put_nowait(None)

    async def stop(self):
        self.die()

    def is_alive(self):
        return self._alive


def _rig(transport=None):
    ws = _WS()
    pm = ProcessManager(ws, MagicMock())
    mgr = SubAgentManager(process_manager=pm)
    transport = transport or _FakeTransport()
    adapter = CLIAdapter(handle=H, display_name="Claude Code", transport=transport)
    adapter._idle = True  # an existing worker between turns
    mgr._adapters[(P, S)] = {H: adapter}
    # The registration seam start() calls. getattr only so the red run on
    # the pre-fix code fails on the missing broadcasts, not on this line.
    wire = getattr(mgr, "_wire_status", None)
    if wire is not None:
        wire(adapter, P, S, H)
    return ws, pm, mgr, adapter, transport


async def _until(pred, timeout=3.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def _last(ws):
    statuses = ws.statuses()
    return statuses[-1] if statuses else {}


@pytest.mark.asyncio
async def test_wiring_announces_the_worker_once():
    ws, _pm, _mgr, _adapter, _t = _rig()
    [event] = ws.statuses()
    assert event == {
        "type": "sub_agent.status", "project_id": P, "session_id": S,
        "handle": H, "display_name": "Claude Code", "status": "idle",
        "awaiting_approval": False, "background_commands": [],
    }


@pytest.mark.asyncio
async def test_dispatch_to_an_idle_worker_publishes_running_before_any_output():
    ws, _pm, mgr, adapter, transport = _rig()
    await mgr._dispatch_async(adapter, P, H, "think hard", session_id=S)
    assert transport.dispatched == ["think hard"]
    assert _last(ws)["status"] == "running"
    # Nothing came back from the worker — the push is the dispatch itself.
    assert not [e for e in ws.events if e.get("type") == "chat.sub_agent_message"]


@pytest.mark.asyncio
async def test_turn_complete_publishes_idle_for_dispatched_and_self_turns():
    ws, pm, mgr, adapter, transport = _rig()
    await pm.start(P, H, adapter, session_id=S)
    await mgr._dispatch_async(adapter, P, H, "go", session_id=S)
    assert _last(ws)["status"] == "running"

    transport.push("turn_complete", cause="success")
    await _until(lambda: _last(ws)["status"] == "idle")

    # Spec 129: a turn the CLI starts on its own — its first chunk opens it,
    # its turn_complete(origin=self) closes it. No dispatch involved.
    transport.push("message", text="background shell finished")
    await _until(lambda: _last(ws)["status"] == "running")
    transport.push("turn_complete", cause="success", origin="self")
    await _until(lambda: _last(ws)["status"] == "idle")
    assert [e["status"] for e in ws.statuses()] == [
        "idle", "running", "idle", "running", "idle"]
    await pm.stop(P, H, session_id=S)


@pytest.mark.asyncio
async def test_no_duplicate_broadcast_when_nothing_changed():
    ws, _pm, mgr, adapter, _t = _rig()
    adapter._idle = False
    adapter._idle = False
    adapter._pending_response = False
    await mgr._dispatch_async(adapter, P, H, "again", session_id=S)
    assert [e["status"] for e in ws.statuses()] == ["idle", "running"]


@pytest.mark.asyncio
async def test_background_work_going_live_and_ending_publishes():
    ws, pm, _mgr, adapter, _t = _rig()
    registry = pm.background_work
    registry.capture_window_s = 0.2
    registry.capture_poll_s = 0.02
    registry.watch_poll_s = 0.05
    adapter._idle = True
    # The turn ended with a run_in_background shell still running.
    rec = registry.register(P, S, H, command="sleep 0.6",
                            root_proc=psutil.Process())
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.6)"])
    try:
        assert _last(ws)["status"] == "background-running"
        assert _last(ws)["background_commands"] == ["sleep 0.6"]
        await _until(lambda: rec.anchor is not None)
        await asyncio.to_thread(child.wait)  # reaped: the work is over
        await _until(lambda: _last(ws)["status"] == "idle")
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert [e["status"] for e in ws.statuses()] == [
        "idle", "background-running", "idle"]


@pytest.mark.asyncio
async def test_user_stop_publishes_stopped_once():
    ws, pm, mgr, adapter, _t = _rig()
    await pm.start(P, H, adapter, session_id=S)
    await mgr._dispatch_async(adapter, P, H, "go", session_id=S)
    await mgr.stop(P, H, session_id=S)
    assert [e["status"] for e in ws.statuses()] == ["idle", "running", "stopped"]
    # A later scan or publish for the gone worker says nothing more.
    mgr.list_active(P, session_id=S)
    adapter._idle = True  # a late flag write from the dead adapter
    assert len(ws.statuses()) == 3


@pytest.mark.asyncio
async def test_worker_death_publishes_stopped():
    ws, pm, mgr, adapter, transport = _rig()
    await pm.start(P, H, adapter, session_id=S)
    await mgr._dispatch_async(adapter, P, H, "go", session_id=S)
    transport.die()  # process gone mid-turn, no turn_complete
    await _until(lambda: _last(ws)["status"] == "stopped")
    await pm.stop(P, H, session_id=S)


@pytest.mark.asyncio
async def test_list_active_carries_awaiting_approval_additively():
    _ws, _pm, mgr, adapter, transport = _rig()
    adapter._idle = False
    [entry] = mgr.list_active(P, session_id=S)
    assert entry["status"] == "running" and entry["awaiting_approval"] is False
    transport._pending_approvals["r1"] = asyncio.get_running_loop().create_future()
    [entry] = mgr.list_active(P, session_id=S)
    assert entry["status"] == "running" and entry["awaiting_approval"] is True


# --------------------------------------------------------------- approvals --

@pytest.mark.asyncio
async def test_sdk_permission_prompt_publishes_needs_you_and_its_answer_clears_it():
    from agent_os.agent.transports.sdk_transport import SDKTransport

    transport = SDKTransport()
    transport._alive = True
    ws, _pm, mgr, adapter, _t = _rig(transport)
    adapter._idle = False  # the turn is open
    asking = asyncio.create_task(
        transport._handle_permission("Bash", {"command": "rm x"}, None))
    await _until(lambda: _last(ws)["awaiting_approval"] is True)
    assert _last(ws)["status"] == "running"
    [request_id] = list(transport._pending_approvals)

    assert await mgr.resolve_sub_agent_approval(
        P, request_id, approved=True, session_id=S)
    await asyncio.wait_for(asking, 2.0)
    assert _last(ws)["awaiting_approval"] is False
    assert [(e["status"], e["awaiting_approval"]) for e in ws.statuses()] == [
        ("idle", False), ("running", False), ("running", True), ("running", False)]


@pytest.mark.asyncio
async def test_sdk_cancelled_permission_prompt_clears_needs_you():
    from agent_os.agent.transports.sdk_transport import SDKTransport

    transport = SDKTransport()
    transport._alive = True
    ws, _pm, _mgr, adapter, _t = _rig(transport)
    adapter._idle = False
    asking = asyncio.create_task(
        transport._handle_permission("Bash", {"command": "rm x"}, None))
    await _until(lambda: _last(ws)["awaiting_approval"] is True)
    asking.cancel()  # the CLI withdrew the request (turn interrupted)
    await _until(lambda: _last(ws)["awaiting_approval"] is False)


@pytest.mark.asyncio
async def test_codex_approval_publishes_needs_you_and_decision_clears_it():
    from tests.unit.test_codex_transport import TestApprovals, _wired

    transport, _stdin = _wired(autonomy=Autonomy.CHECK_IN)
    ws, _pm, mgr, adapter, _t = _rig(transport)
    adapter._idle = False
    await transport._route_server_message(dict(TestApprovals.REQUEST))
    assert _last(ws)["awaiting_approval"] is True
    [request_id] = list(transport._pending_approvals)
    assert await mgr.resolve_sub_agent_approval(
        P, request_id, approved=True, session_id=S, decision="accept")
    assert _last(ws)["awaiting_approval"] is False


@pytest.mark.asyncio
async def test_approve_all_bypass_never_flashes_needs_you():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    import time

    transport = SDKTransport()
    transport._alive = True
    ws, _pm, mgr, adapter, _t = _rig(transport)
    adapter._idle = False
    mgr._permission_bypass_until[(P, S, H)] = time.monotonic() + 600
    asking = asyncio.create_task(
        transport._handle_permission("Bash", {"command": "ls"}, None))
    await _until(lambda: bool(transport._pending_approvals))
    [request_id] = list(transport._pending_approvals)
    assert await mgr._on_permission_request(P, H, request_id, session_id=S)
    await asyncio.wait_for(asking, 2.0)
    assert all(e["awaiting_approval"] is False for e in ws.statuses())
