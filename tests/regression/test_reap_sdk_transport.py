# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test for the confirmed sub-agent leak (FINDINGS-reap-confirmation).

Turns the confirmation repro into an assertion: a real ``SDKTransport`` that
connects in task A and is stopped from task B (the daemon's reap/eviction path)
must terminate its claude subprocess. On pre-fix code the SDK's cross-task
``disconnect()`` RuntimeError is swallowed and the process survives — this test
fails. With Direction B (own the kill) it passes.

Requires the real claude binary + claude-agent SDK; skipped otherwise.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import psutil
import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK


def _find_claude() -> str | None:
    found = shutil.which("claude")
    if found:
        return found
    candidates = [
        Path.home() / ".claude/local/node_modules/.bin/claude",
        Path.home() / ".claude/local/claude",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


CLAUDE = _find_claude()


def _alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


@pytest.mark.skipif(not HAS_SDK or CLAUDE is None,
                    reason="claude SDK/binary not available")
async def test_sdk_transport_stop_terminates_subprocess_cross_task(tmp_path):
    transport = SDKTransport(autonomy=Autonomy.HANDS_OFF,
                             system_prompt="disposable regression sub-agent")

    async def task_a_connect_and_run():
        # connect() ENTERS the SDK anyio task group in THIS task — exactly as
        # the daemon's management-loop task does at dispatch.
        await transport.start(command=CLAUDE, args=[], workspace=str(tmp_path), env={})
        await transport.dispatch("Reply with exactly the two letters: OK")
        async for ev in transport.read_stream():
            if ev.event_type == "turn_complete":
                break

    # Task A completes (like the management turn yielding).
    await asyncio.wait_for(asyncio.create_task(task_a_connect_and_run()), timeout=150.0)

    pid = transport._client._transport._process.pid
    assert _alive(pid), "claude subprocess should be alive before stop()"

    # Task B: stop() runs in a DIFFERENT task than connect() — the reap path.
    try:
        await asyncio.wait_for(asyncio.create_task(transport.stop()), timeout=30.0)
    finally:
        # Safety net so a failing assertion never leaves an orphan behind.
        if _alive(pid):
            pass  # asserted below; cleanup in the except path

    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and _alive(pid):
        await asyncio.sleep(0.1)

    still = _alive(pid)
    if still:
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    assert not still, f"claude subprocess {pid} survived cross-task stop()"
    assert transport.is_alive() is False
