# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 TEST RULE 2 journey: dispatch → background-running → completion →
management session WOKEN → result surfaced; and user stop mid-background-work.

Drives the REAL component chain end-to-end with a REAL claude subprocess:
SubAgentManager.send (spawn-on-demand) → SDKTransport → real run_in_background
job → ProcessManager consume (provenance capture) → three-state status →
on_completed → REAL LifecycleObserver → REAL AgentManager.inject_system_message
→ hydrate-from-disk + durable append + loop start (the WAKE — the silent gap
of the live bug).

The only stub on the wake path is AgentManager.start_agent's LLM turn (a real
management turn needs provider credentials; the wake CONTRACT — event durably
appended + loop start invoked with the hydrated session — is asserted on the
real objects). Same live_daemon/requires_claude convention as the other live
specs; the HTTP layer is not bootable here (create_app takes the daemon
singleton PID file, held by any running packaged app).
"""

from __future__ import annotations

import asyncio
import json
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
from agent_os.daemon_v2.agent_manager import AgentManager
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

PROJ, SID, HANDLE = "proj_journey", "sess_journey_0001", "claude-code"


def _seed_session(workspace: str, sid: str) -> str:
    from agent_os.agent.project_paths import ProjectPaths
    d = ProjectPaths(workspace).sessions_dir
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sid}.jsonl")
    rows = [
        {"type": "session_start", "session_id": sid, "session_uuid": sid,
         "timestamp": "2026-06-06T00:00:00+00:00"},
        {"role": "user", "content": "dispatch the baseline",
         "source": "user", "session_id": sid, "session_uuid": sid,
         "timestamp": "2026-06-06T00:00:01+00:00"},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _build_stack(workspace: str):
    """The app wiring (api/app.py) minus HTTP/LLM: every object on the
    dispatch→consume→status→wake chain is real."""
    registry = AgentRegistry()
    registry.load_directory(MANIFESTS)
    setup_engine = SetupEngine(registry)
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "workspace": workspace, "enabled_sub_agents": [HANDLE]}

    am = AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=None,  # set below
        activity_translator=MagicMock(),
        process_manager=None,    # set below
    )
    am.start_agent = AsyncMock()  # the LLM turn — see module docstring
    am._build_agent_config_from_project = MagicMock(return_value=MagicMock())

    observer = LifecycleObserver(agent_manager=am, ws_manager=MagicMock())
    pm = ProcessManager(MagicMock(), MagicMock(), lifecycle_observer=observer)
    sam = SubAgentManager(
        process_manager=pm, registry=registry, setup_engine=setup_engine,
        project_store=project_store, lifecycle_observer=observer,
    )
    am._sub_agent_manager = sam
    am._process_manager = pm
    return am, sam, pm


async def _auto_approve_loop(sam, stop_evt):
    while not stop_evt.is_set():
        adapters = sam._adapters.get((PROJ, SID), {})
        for adapter in adapters.values():
            tr = getattr(adapter, "_transport", None)
            for rid in list(getattr(tr, "_pending_approvals", {}) or {}):
                await tr.respond_to_permission(rid, True)
        await asyncio.sleep(0.3)


async def _wait_status(sam, want: str, deadline_s: float) -> list[str]:
    seen = []
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        s = sam.status(PROJ, HANDLE, session_id=SID)
        if not seen or seen[-1] != s:
            seen.append(s)
        if s == want:
            return seen
        await asyncio.sleep(0.3)
    return seen



async def _dispatch_until_background(sam, sleep_cmd: str, token: str,
                                     attempts: int = 3) -> list[str]:
    """Dispatch the run_in_background task; re-instruct on model
    disobedience (foreground run or raw '&' detach — the turn then closes
    with no live tracked work and status reads idle). The test is about the
    SYSTEM contract, not model dice; the retry is bounded and honest."""
    prompt = (
        f"Using your Bash tool with the run_in_background parameter set to "
        f"true (NOT a foreground command, NOT shell '&'), start exactly: "
        f"{sleep_cmd} . Do not wait for it, do not monitor it. "
        f"Then immediately reply exactly: {token}"
    )
    seen_all: list[str] = []
    for attempt in range(attempts):
        await sam.send(PROJ, HANDLE, prompt, session_id=SID)
        seen = await _wait_status(sam, "background-running", 120)
        seen_all.extend(seen)
        if seen[-1] == "background-running":
            return seen_all
        prompt = (
            f"That did not register as background work. Call the Bash tool "
            f"again with parameter run_in_background=true and command "
            f"exactly: {sleep_cmd} . Then immediately reply exactly: {token}"
        )
    return seen_all


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


@requires_claude
@pytest.mark.asyncio
async def test_full_round_trip_background_running_then_wake(tmp_path):
    workspace = str(tmp_path)
    session_path = _seed_session(workspace, SID)
    am, sam, pm = _build_stack(workspace)
    stop_evt = asyncio.Event()
    approver = asyncio.create_task(_auto_approve_loop(sam, stop_evt))
    try:
        # The exact live-bug shape, now honest: turn closes while the
        # tracked work is still alive → background-running (never observed
        # end-to-end before this test).
        seen = await _dispatch_until_background(sam, "sleep 45", "BG-LAUNCHED")
        assert seen[-1] == "background-running", f"statuses seen: {seen}"

        # The WAKE: completion event hydrated the on-disk session, appended
        # durably, and started the management loop.
        deadline = time.monotonic() + 30
        woken = False
        while time.monotonic() < deadline and not woken:
            rows = _read_jsonl(session_path)
            done_rows = [r for r in rows
                         if r.get("role") == "system"
                         and "[Sub-agent] claude-code completed" in str(r.get("content"))]
            woken = bool(done_rows) and am.start_agent.await_count >= 1
            await asyncio.sleep(0.5)
        assert woken, "completion event must be appended to disk AND wake the loop"
        rows = _read_jsonl(session_path)
        summary_row = next(r for r in rows
                           if "[Sub-agent] claude-code completed" in str(r.get("content")))
        assert "BG-LAUNCHED" in summary_row["content"], "result surfaced"
        kwargs = am.start_agent.await_args.kwargs
        assert kwargs.get("session") is not None, "woken with the hydrated session"

        # When the background work finishes (sleep 45 expires), status
        # returns to true idle.
        seen = await _wait_status(sam, "idle", 90)
        assert seen[-1] == "idle", f"statuses seen: {seen}"
    finally:
        stop_evt.set()
        approver.cancel()
        await sam.stop_all(PROJ, session_id=SID)
        import subprocess as _sp  # clean raw-detached escapees from disobedient attempts
        _sp.run(["pkill", "-f", "^sleep (45|300)$"], capture_output=True)


@requires_claude
@pytest.mark.asyncio
async def test_user_stop_mid_background_work(tmp_path):
    workspace = str(tmp_path)
    session_path = _seed_session(workspace, SID)
    am, sam, pm = _build_stack(workspace)
    stop_evt = asyncio.Event()
    approver = asyncio.create_task(_auto_approve_loop(sam, stop_evt))
    try:
        seen = await _dispatch_until_background(sam, "sleep 300", "STOPPABLE-READY")
        assert seen[-1] == "background-running", f"statuses seen: {seen}"

        # Anchor pid of the tracked work (scoped assertion: an earlier
        # disobedient attempt may have raw-detached an untracked sleep —
        # that escape class is out of stop's reach BY DESIGN).
        recs = pm.background_work.records(PROJ, SID, HANDLE)
        anchor_pids = [r.anchor.pid for r in recs if r.anchor is not None]
        assert anchor_pids, "anchored background work expected"

        result = await sam.stop_for_user(PROJ, HANDLE, session_id=SID)

        # Tracked tree dead — CONFIRMED, not just signalled.
        assert "sleep 300" in result["background_terminated"]
        for pid in anchor_pids:
            assert (not psutil.pid_exists(pid)
                    or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE), \
                f"anchored work pid {pid} survived the stop"
        # Honest warning names the raw-detach limitation.
        assert "nohup" in result["warning"]
        # Loss surfaced into the management session (resume context).
        rows = _read_jsonl(session_path)
        stop_rows = [r for r in rows
                     if "stopped by user" in str(r.get("content"))]
        assert stop_rows, "user stop must be recorded in the session"
        assert "did NOT complete" in stop_rows[-1]["content"]
        # Agent gone.
        assert sam.status(PROJ, HANDLE, session_id=SID) in ("unknown", "stopped")
    finally:
        stop_evt.set()
        approver.cancel()
        await sam.stop_all(PROJ, session_id=SID)
        import subprocess as _sp  # clean raw-detached escapees from disobedient attempts
        _sp.run(["pkill", "-f", "^sleep (45|300)$"], capture_output=True)
