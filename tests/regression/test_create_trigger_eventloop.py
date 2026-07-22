# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: create_trigger must work when called from a thread pool worker.

Original root cause (fixed pre-tick-loop redesign): TriggerManager._register_timer()
called asyncio.create_task() which requires a running event loop in the current
thread. When CreateTriggerTool.execute() runs in a thread pool worker (via
asyncio.to_thread), there is no event loop in that thread, causing
RuntimeError("no running event loop"). The trigger data was persisted before the
error, so the trigger appeared to exist but no timer was actually scheduled — a
silent success-despite-error. Original fix: TriggerManager captured the event
loop at start() and used asyncio.run_coroutine_threadsafe() in _register_timer().

Post tick-loop redesign: schedule triggers no longer get a per-trigger
asyncio.Task/Future at all (see trigger_manager.py's TICK_INTERVAL_SECONDS tick
loop). register_trigger()/unregister_trigger() for a schedule trigger now just
mutate the plain `_schedule_ids` set/`_trigger_project` dict that the tick loop
reads on its next evaluation — no event loop reference needed, so the original
RuntimeError class of bug cannot recur for this path. These tests are kept (with
their assertions retargeted at `_schedule_ids`) to pin the still-relevant
property: registration from a worker thread must not raise and must not produce
duplicate/stale state.
"""

import asyncio
import tempfile
import pytest
from unittest.mock import MagicMock

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.trigger_manager import TriggerManager


def _make_trigger(trigger_id="trg_test0001", cron="0 7 * * *", enabled=True):
    return {
        "id": trigger_id,
        "name": "Test trigger",
        "enabled": enabled,
        "type": "schedule",
        "schedule": {"cron": cron, "human": "Daily at 7 AM", "timezone": "UTC"},
        "task": "Say hello",
        "autonomy": None,
        "last_triggered": None,
        "trigger_count": 0,
        "created_at": "2020-01-01T00:00:00+00:00",
    }


@pytest.fixture
def project_store():
    """Real ProjectStore (not a bare MagicMock).

    A bare MagicMock used to stand in here, but TriggerManager's tick loop
    now runs concurrently the moment start() is awaited, and its
    orphan-cleanup pass (see _evaluate_due_triggers) unregisters any schedule
    trigger it can't find inside its project's stored trigger list — correct
    behavior for an actually-deleted trigger, but a MagicMock's
    `.get("triggers", [])` iterates as empty, so it made every registered
    trigger look orphaned and the concurrently-running tick loop unregistered
    it mid-test. A real store, seeded the same order CreateTriggerTool
    actually persists a trigger (project_store write, then
    trigger_manager.register_trigger — see agent_os/agent/tools/triggers.py),
    avoids that false-orphan race entirely.
    """
    return ProjectStore(data_dir=tempfile.mkdtemp())


@pytest.fixture
def project_id(project_store):
    return project_store.create_project({
        "name": "Test Project",
        "workspace": tempfile.mkdtemp(),
        "model": "gpt-4",
        "api_key": "sk-test",
    })


@pytest.fixture
def trigger_manager(project_store):
    """TriggerManager wired to the real project_store fixture above."""
    agent_manager = MagicMock()
    return TriggerManager(project_store, agent_manager)


def _persist(project_store, project_id, trigger):
    """Seed project_store with a trigger the way CreateTriggerTool does:
    persist first, so a concurrently-running tick evaluation finds it."""
    project_store.update_project(project_id, {"triggers": [trigger]})


@pytest.mark.asyncio
async def test_register_trigger_from_thread_no_runtime_error(
    project_store, project_id, trigger_manager
):
    """register_trigger() called from a thread pool worker must not raise.

    Historically, asyncio.create_task() inside _register_timer() raised
    'RuntimeError: no running event loop' when called from a non-event-loop
    thread. The tick-loop redesign removed the per-trigger task entirely, but
    this still pins that a worker-thread call succeeds and actually registers.
    """
    await trigger_manager.start()

    trigger = _make_trigger()
    _persist(project_store, project_id, trigger)

    # Simulate what asyncio.to_thread does: run in a thread pool executor
    # This is exactly the code path when CreateTriggerTool.execute() is called
    await asyncio.to_thread(
        trigger_manager.register_trigger, project_id, trigger
    )

    # Trigger should actually be registered in the tick loop's evaluated set
    assert "trg_test0001" in trigger_manager._schedule_ids

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_register_trigger_from_event_loop_thread(
    project_store, project_id, trigger_manager
):
    """register_trigger() called directly from the event loop thread still works."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_test0002")
    _persist(project_store, project_id, trigger)
    trigger_manager.register_trigger(project_id, trigger)

    assert "trg_test0002" in trigger_manager._schedule_ids

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_rapid_toggle_no_duplicate_timers(
    project_store, project_id, trigger_manager
):
    """Rapid register -> unregister -> register must leave exactly one registration."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_toggle")
    _persist(project_store, project_id, trigger)

    # Simulate rapid toggling from a thread (like the UI toggle button)
    await asyncio.to_thread(
        trigger_manager.register_trigger, project_id, trigger
    )
    await asyncio.to_thread(
        trigger_manager.unregister_trigger, "trg_toggle"
    )
    await asyncio.to_thread(
        trigger_manager.register_trigger, project_id, trigger
    )

    assert "trg_toggle" in trigger_manager._schedule_ids
    # No stray/duplicate entries left over from the toggle sequence
    assert trigger_manager._schedule_ids == {"trg_toggle"}

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_unregister_cancels_timer(
    project_store, project_id, trigger_manager
):
    """unregister_trigger() must remove it from the tick loop's evaluated set."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_cancel")
    _persist(project_store, project_id, trigger)
    await asyncio.to_thread(
        trigger_manager.register_trigger, project_id, trigger
    )
    assert "trg_cancel" in trigger_manager._schedule_ids

    await asyncio.to_thread(
        trigger_manager.unregister_trigger, "trg_cancel"
    )
    assert "trg_cancel" not in trigger_manager._schedule_ids

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_register_idempotent_replaces_existing(
    project_store, project_id, trigger_manager
):
    """Calling register_trigger twice for the same trigger_id must not duplicate
    its registration (and must keep the project mapping correct)."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_idem")
    _persist(project_store, project_id, trigger)
    trigger_manager.register_trigger(project_id, trigger)
    assert trigger_manager._schedule_ids == {"trg_idem"}
    assert trigger_manager._trigger_project["trg_idem"] == project_id

    # Register again — should replace, not duplicate
    trigger_manager.register_trigger(project_id, trigger)
    assert trigger_manager._schedule_ids == {"trg_idem"}
    assert trigger_manager._trigger_project["trg_idem"] == project_id

    await trigger_manager.stop()
