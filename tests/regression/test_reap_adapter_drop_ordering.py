# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: SubAgentManager.stop() must not drop the adapter from _adapters
until termination is confirmed (or a bounded timeout force-drops it).

Pre-fix, stop() popped the adapter BEFORE awaiting adapter.stop(), so a teardown
that failed/hung left an untracked live process with no handle to retry the kill
(the root of the leak). Invariant 6: drop only after confirmed-dead-or-bounded
-timeout; on timeout force-drop + log ERROR, never hang.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

SID = "s1"


class _FakeProcMgr:
    async def stop(self, *a, **k):
        return None


class _FakeAdapter:
    def __init__(self, stop_delay: float = 0.0):
        self._stop_delay = stop_delay
        self.stopped = False
        self.stop_started = asyncio.Event()

    async def stop(self):
        self.stop_started.set()
        await asyncio.sleep(self._stop_delay)
        self.stopped = True

    def is_alive(self):
        return not self.stopped


def _mgr_with_adapter(adapter):
    mgr = SubAgentManager(process_manager=_FakeProcMgr())
    sk = make_session_key("p1", SID)
    mgr._adapters[sk] = {"h": adapter}
    return mgr, sk


async def test_adapter_dropped_only_after_confirmed_dead():
    adapter = _FakeAdapter(stop_delay=0.3)
    mgr, sk = _mgr_with_adapter(adapter)

    task = asyncio.create_task(mgr.stop("p1", "h", session_id=SID))
    await adapter.stop_started.wait()

    # MID-stop: adapter must still be registered (not popped pre-emptively).
    assert "h" in mgr._adapters.get(sk, {}), "adapter dropped before stop() finished"

    await task
    # After confirmed-dead: dropped.
    assert "h" not in mgr._adapters.get(sk, {})
    assert adapter.stopped is True


async def test_adapter_force_dropped_on_stop_timeout(caplog):
    adapter = _FakeAdapter(stop_delay=999)  # hangs well past the bound
    mgr, sk = _mgr_with_adapter(adapter)
    mgr.ADAPTER_STOP_TIMEOUT = 0.2  # shrink the bound for a fast test

    with caplog.at_level(logging.ERROR):
        # Must return promptly (force-drop), never hang.
        await asyncio.wait_for(mgr.stop("p1", "h", session_id=SID), timeout=5.0)

    assert "h" not in mgr._adapters.get(sk, {}), "adapter not force-dropped on timeout"
    assert any("did not terminate" in r.getMessage() for r in caplog.records), \
        "expected an ERROR log naming the timeout"
