# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: SubAgentManager.stop() must not drop the adapter from _adapters
until termination is CONFIRMED.

Pre-fix, stop() popped the adapter BEFORE awaiting adapter.stop(), so a teardown
that failed/hung left an untracked live process with no handle to retry the kill
(the root of the leak). Piece 3 Part E tightens the invariant further: the old
"bounded timeout force-drops it" escape hatch is GONE — a hanging adapter.stop()
is escalated to a direct snapshotted tree kill and the slot is released only on
confirmed death (kill-to-confirmation is shielded; callers stop waiting, never
the kill). See tests/regression/test_piece3_teardown_confirms_death.py for the
full Part E matrix.
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


async def test_hanging_adapter_stop_escalates_never_force_drops(caplog):
    """Piece 3 Part E (REWRITTEN from test_adapter_force_dropped_on_stop_timeout):
    a hanging adapter.stop() is escalated to the direct tree kill; with the
    snapshotted tree confirmed dead the slot is released — and the old
    force-drop ('process may leak') line never fires."""
    adapter = _FakeAdapter(stop_delay=999)  # hangs well past the bound
    mgr, sk = _mgr_with_adapter(adapter)
    mgr.ADAPTER_STOP_TIMEOUT = 0.2   # shrink budgets for a fast test
    mgr.STOP_CONFIRM_GRACE = 0.2
    mgr.TEARDOWN_WAIT_BUDGET = 5.0   # ordering preserved: outer > inner

    with caplog.at_level(logging.WARNING):
        # Must return promptly (escalation), never hang.
        await asyncio.wait_for(mgr.stop("p1", "h", session_id=SID), timeout=5.0)

    # No real process tree behind the fake adapter → confirmation succeeds →
    # the slot is released. Crucially: via ESCALATION, not force-drop.
    assert "h" not in mgr._adapters.get(sk, {})
    assert any("escalating" in r.getMessage() for r in caplog.records), \
        "expected a WARNING naming the escalation"
    assert not any("force-dropping" in r.getMessage() for r in caplog.records), \
        "the force-drop leak path must be gone"
