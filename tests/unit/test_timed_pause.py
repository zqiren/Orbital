# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Timed pause: stop(duration_seconds=N) records paused_until; the dispatcher's
_run tick auto-resumes once the deadline passes. A pause without a duration
never auto-resumes (product decision 2026-06-11: pause is a consent boundary;
auto-resume only when the user chose a snooze)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agent_os.queue.dispatcher import QueueDispatcher
from agent_os.queue.models import ItemState, QueueRunState
from agent_os.queue.store import QueueStore
from tests.integration.test_queue_phase2 import _ScriptedAgentManager


async def _wait(predicate, timeout=10.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return False


class _MinimalAgentManager:
    """Minimal fake for stop()-only tests: no sub-agents, no active loop."""

    def get_sub_agent_manager(self):
        return None

    def current_holder_session_id(self, project_id):
        return None

    def get_loop(self, project_id, *, session_id=None):
        return None


def _scripted_with_stop_support(script):
    """Wrap _ScriptedAgentManager with the methods dispatcher.stop() needs."""
    mgr = _ScriptedAgentManager(script)
    mgr.get_sub_agent_manager = lambda: None
    mgr.current_holder_session_id = lambda project_id: None
    return mgr


@pytest.mark.asyncio
async def test_stop_with_duration_records_paused_until(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    mgr = _MinimalAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    result = await dispatcher.stop(duration_seconds=3600)
    state = store.load()
    assert state.state == QueueRunState.PAUSED
    assert state.paused_until is not None
    deadline = datetime.fromisoformat(state.paused_until)
    delta = deadline - datetime.now(timezone.utc)
    assert timedelta(minutes=59) < delta < timedelta(minutes=61)
    assert result["paused_until"] == state.paused_until


@pytest.mark.asyncio
async def test_stop_without_duration_has_no_deadline(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    mgr = _MinimalAgentManager()
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    result = await dispatcher.stop()
    assert store.load().paused_until is None
    assert result.get("paused_until") is None


@pytest.mark.asyncio
async def test_expired_timed_pause_auto_resumes_and_drains(tmp_path):
    """PAUSED with a deadline in the past + one queued item: the running
    dispatcher must flip to RUNNING on its next tick and drain the item."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("snoozed work")
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    store.set_queue_state(QueueRunState.PAUSED, paused_until=past)

    mgr = _scripted_with_stop_support([
        {"reason": "complete", "summary": "done"},
    ])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: (
        store.load().items
        and store.load().items[0].state == ItemState.DONE
    ), timeout=10.0)
    s = store.load()
    await dispatcher.shutdown()
    assert ok, (
        f"timed pause did not auto-resume: queue={s.state.value}, "
        f"paused_until={s.paused_until}, items="
        + ", ".join(f"{it.id}={it.state.value}" for it in s.items)
    )
    assert s.paused_until is None


@pytest.mark.asyncio
async def test_naive_past_deadline_auto_resumes_and_drains(tmp_path):
    """A naive ISO deadline (no UTC offset — what a hand-edited queue.json
    produces) must be interpreted as UTC, not crash the tick with a
    naive-vs-aware comparison TypeError. Past naive deadline + one queued
    item: dispatcher auto-resumes and drains."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("snoozed work")
    store.set_queue_state(
        QueueRunState.PAUSED, paused_until="2000-01-01T00:00:00",
    )

    mgr = _scripted_with_stop_support([
        {"reason": "complete", "summary": "done"},
    ])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()

    ok = await _wait(lambda: (
        store.load().items
        and store.load().items[0].state == ItemState.DONE
    ), timeout=10.0)
    s = store.load()
    await dispatcher.shutdown()
    assert ok, (
        f"naive past deadline did not auto-resume: queue={s.state.value}, "
        f"paused_until={s.paused_until}, items="
        + ", ".join(f"{it.id}={it.state.value}" for it in s.items)
    )
    assert s.paused_until is None


@pytest.mark.asyncio
async def test_untimed_pause_never_auto_resumes(tmp_path):
    """PAUSED without a deadline stays paused across many ticks."""
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("parked work")
    store.set_queue_state(QueueRunState.PAUSED)

    mgr = _scripted_with_stop_support([])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    dispatcher.notify_new_item()
    await asyncio.sleep(1.0)  # several wake/check cycles via notify + tick
    s = store.load()
    await dispatcher.shutdown()
    assert s.state == QueueRunState.PAUSED
    assert s.items[0].state == ItemState.QUEUED


@pytest.mark.asyncio
async def test_malformed_paused_until_is_cleared_not_looping(tmp_path):
    store = QueueStore(tmp_path / "queue.json")
    store.add_item("parked work")  # keep queue non-empty so auto_idle_if_empty is a no-op
    store.set_queue_state(QueueRunState.PAUSED, paused_until="not-a-date")

    mgr = _scripted_with_stop_support([])
    dispatcher = QueueDispatcher(
        project_id="proj_timed", store=store, agent_manager=mgr,
    )
    await dispatcher.start()
    ok = await _wait(lambda: store.load().paused_until is None, timeout=5.0)
    s = store.load()
    await dispatcher.shutdown()
    assert ok
    assert s.state == QueueRunState.PAUSED  # still paused, just deadline dropped
