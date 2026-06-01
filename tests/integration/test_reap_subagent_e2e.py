# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end reap integration tests (Direction B).

Closes the inferential gap the confirmation left open: the confirmation
reproduced the leak only at the ``SDKTransport`` level. These drive the REAL
daemon reap chain — ``SubAgentManager.stop_all → stop → CLIAdapter.stop →
SDKTransport.stop → kill_process_tree`` — against a REAL claude subprocess,
under the daemon's cross-task condition (connect in task A, reap in task B).

The "daemon" here is the pytest process; the claude subprocess spawns as its
child (same topology Step 0 confirmed for the real daemon). We deliberately do
NOT boot uvicorn or the management LLM: that path only determines *when*
``stop_all`` is called (already established), not whether the kill works. This
test proves the kill works through the real manager/adapter/transport stack.

Requires the real claude binary + claude-agent SDK; skipped otherwise.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

import psutil
import pytest

from agent_os.agent.adapters.base import AdapterConfig
from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
from agent_os.daemon_v2.models import make_session_key
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
SELF = psutil.Process(os.getpid())  # "daemon" analog; claude is its child

requires_claude = pytest.mark.skipif(
    not HAS_SDK or CLAUDE is None, reason="claude SDK/binary not available")


def _alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _claude_descendants() -> list[int]:
    out = []
    for c in SELF.children(recursive=True):
        try:
            if "claude" in c.name().lower():
                out.append(c.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


class _FakeProcMgr:
    async def stop(self, *a, **k):
        return None


async def _dispatch_and_complete(mgr, sk, workspace: str) -> int:
    """Register a real claude SDK sub-agent in ``mgr`` and run one turn.

    connect() runs in task A (like the management-loop task at dispatch); the
    caller reaps from a different task. Returns the claude subprocess PID.
    """
    transport = SDKTransport(autonomy=Autonomy.HANDS_OFF,
                             system_prompt="disposable e2e sub-agent")
    adapter = CLIAdapter(handle="claude-code", display_name="claude-code",
                         transport=transport)
    cfg = AdapterConfig(command=CLAUDE, workspace=workspace,
                        approval_patterns=[], env={}, args=[])

    async def task_a():
        await adapter.start(cfg)  # → transport.start() → connect() in THIS task
        await transport.dispatch("Reply with exactly the two letters: OK")
        async for ev in adapter.read_stream():
            if ev.chunk_type == "turn_complete":
                break

    await asyncio.wait_for(asyncio.create_task(task_a()), timeout=150.0)
    mgr._adapters[sk] = {"claude-code": adapter}
    return transport._client._transport._process.pid


@requires_claude
async def test_daemon_reaps_subagent_end_to_end(tmp_path):
    mgr = SubAgentManager(process_manager=_FakeProcMgr())
    sk = make_session_key("p1", "s1")

    pid = await _dispatch_and_complete(mgr, sk, str(tmp_path))
    assert _alive(pid), "claude should be alive after the turn completes"

    # Reap from a DIFFERENT task — the daemon's reap/eviction path.
    await asyncio.wait_for(
        asyncio.create_task(mgr.stop_all("p1", session_id="s1")), timeout=30.0)

    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline and _alive(pid):
        await asyncio.sleep(0.1)

    still_alive = _alive(pid)
    if still_alive:  # never leave an orphan if the assertion is about to fail
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    assert not still_alive, f"claude {pid} survived the end-to-end reap"
    assert mgr._adapters.get(sk, {}) == {}, "adapter not dropped from _adapters"
    assert _claude_descendants() == [], "claude descendants survived the reap"


@requires_claude
@pytest.mark.skipif(sys.platform == "win32",
                    reason="num_fds is POSIX-only; Windows uses handle accounting")
async def test_fd_count_stable_across_reap_cycles(tmp_path):
    mgr = SubAgentManager(process_manager=_FakeProcMgr())
    cycles = 3

    baseline = SELF.num_fds()
    samples = []
    for i in range(cycles):
        sk = make_session_key("p1", f"s{i}")
        pid = await _dispatch_and_complete(mgr, sk, str(tmp_path))
        await asyncio.wait_for(
            asyncio.create_task(mgr.stop_all("p1", session_id=f"s{i}")),
            timeout=30.0)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline and _alive(pid):
            await asyncio.sleep(0.1)
        assert not _alive(pid), f"cycle {i}: claude {pid} survived"
        await asyncio.sleep(0.3)  # let FD close settle
        samples.append(SELF.num_fds())

    # FD count must not grow monotonically with cycles. Allow a small constant
    # slack for unrelated interpreter/psutil bookkeeping, but reject linear
    # per-cycle growth.
    print(f"\n[fd-leak] baseline={baseline} after-each-cycle={samples}")
    growth = max(samples) - baseline
    assert growth <= 5, (
        f"open-FD count grew by {growth} over {cycles} reap cycles "
        f"(baseline={baseline}, samples={samples}) — descriptor leak")
