# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Piece 3 Part D: user stop — cancel turn + kill tracked tree + honest report.

stop_for_user must: terminate tracked background work (confirmed dead),
stop the adapter, report exactly which commands were terminated, carry the
raw-detach honesty warning, and emit a LOUD lifecycle record (never a silent
kill of background work).
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

PROJ, SID, HANDLE = "proj_stop", "sess_stop_0001", "claude-code"

_SPAWNER = (
    "import subprocess, sys, time\n"
    "time.sleep(0.5)\n"
    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    "p.wait()\n"
)


def _spawn_root():
    proc = subprocess.Popen([sys.executable, "-c", _SPAWNER])
    return psutil.Process(proc.pid)


def _kill_tree(root):
    try:
        for c in root.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        root.kill()
    except psutil.Error:
        pass


def _build(registry):
    pm = MagicMock()
    pm.background_work = registry
    pm.stop = AsyncMock()
    observer = MagicMock()
    observer.on_user_stopped = AsyncMock()
    mgr = SubAgentManager(process_manager=pm, lifecycle_observer=observer)
    adapter = MagicMock()
    adapter.stop = AsyncMock()
    mgr._adapters[(PROJ, SID)] = {HANDLE: adapter}
    return mgr, adapter, observer


@pytest.mark.asyncio
async def test_stop_for_user_kills_tracked_work_confirmed_and_reports():
    registry = BackgroundWorkRegistry(capture_window_s=4.0, capture_poll_s=0.1)
    root = _spawn_root()
    try:
        rec = registry.register(PROJ, SID, HANDLE,
                                command="sleep 60", root_proc=root)
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and rec.anchor is None:
            await asyncio.sleep(0.1)
        assert rec.anchor is not None
        anchor_pid = rec.anchor.pid

        mgr, adapter, observer = _build(registry)
        result = await mgr.stop_for_user(PROJ, HANDLE, session_id=SID)

        # Tracked work terminated — and CONFIRMED dead, not just signalled.
        assert result["background_terminated"] == ["sleep 60"]
        assert not psutil.pid_exists(anchor_pid) or \
            psutil.Process(anchor_pid).status() == psutil.STATUS_ZOMBIE
        # Turn/agent stopped.
        adapter.stop.assert_awaited_once()
        assert (PROJ, SID) not in mgr._adapters  # adapter slate emptied
        # Honest warning names the raw-detach limitation.
        assert "nohup" in result["warning"] or "&" in result["warning"]
        # Loud, never silent:
        observer.on_user_stopped.assert_awaited_once()
        kwargs = observer.on_user_stopped.await_args.kwargs
        assert kwargs["terminated"] == ["sleep 60"]
    finally:
        _kill_tree(root)


@pytest.mark.asyncio
async def test_stop_for_user_without_background_work_still_stops():
    registry = BackgroundWorkRegistry()
    mgr, adapter, observer = _build(registry)

    result = await mgr.stop_for_user(PROJ, HANDLE, session_id=SID)

    assert result["status"] == "stopped"
    assert result["background_terminated"] == []
    adapter.stop.assert_awaited_once()
    observer.on_user_stopped.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifecycle_on_user_stopped_injects_and_broadcasts():
    from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
    am = MagicMock()
    am.inject_system_message = AsyncMock()
    ws = MagicMock()
    obs = LifecycleObserver(am, ws)

    await obs.on_user_stopped("proj_x", "claude-code",
                              terminated=["gh api ..."], session_id="s1")

    am.inject_system_message.assert_awaited_once()
    msg = am.inject_system_message.await_args.args[1]
    assert "stopped by user" in msg
    assert "did NOT complete" in msg
    ws.broadcast.assert_called_once()
    payload = ws.broadcast.call_args[0][1]
    assert payload["type"] == "sub_agent.stopped"
    assert payload["background_terminated"] == ["gh api ..."]
