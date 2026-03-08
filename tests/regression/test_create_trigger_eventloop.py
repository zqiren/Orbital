# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression test: create_trigger must work when called from a thread pool worker.

Root cause: TriggerManager._register_timer() called asyncio.create_task() which
requires a running event loop in the current thread. When CreateTriggerTool.execute()
runs in a thread pool worker (via asyncio.to_thread), there is no event loop in
that thread, causing RuntimeError("no running event loop"). The trigger data was
persisted before the error, so the trigger appeared to exist but no timer was
actually scheduled — a silent success-despite-error.

Fix: TriggerManager captures the event loop at start() and uses
asyncio.run_coroutine_threadsafe() in _register_timer(), making it safe to call
from any thread.
"""

import asyncio
import pytest
from unittest.mock import MagicMock

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
    }


@pytest.fixture
def trigger_manager():
    """TriggerManager with stubbed dependencies."""
    project_store = MagicMock()
    agent_manager = MagicMock()
    return TriggerManager(project_store, agent_manager)


@pytest.mark.asyncio
async def test_register_trigger_from_thread_no_runtime_error(trigger_manager):
    """register_trigger() called from a thread pool worker must not raise RuntimeError.

    Before the fix, asyncio.create_task() inside _register_timer() raised
    'RuntimeError: no running event loop' when called from a non-event-loop thread.
    """
    await trigger_manager.start()

    trigger = _make_trigger()

    # Simulate what asyncio.to_thread does: run in a thread pool executor
    # This is exactly the code path when CreateTriggerTool.execute() is called
    await asyncio.to_thread(
        trigger_manager.register_trigger, "proj_test", trigger
    )

    # Timer should actually be registered
    assert "trg_test0001" in trigger_manager._timers

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_register_trigger_from_event_loop_thread(trigger_manager):
    """register_trigger() called directly from the event loop thread still works."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_test0002")
    trigger_manager.register_trigger("proj_test", trigger)

    assert "trg_test0002" in trigger_manager._timers

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_rapid_toggle_no_duplicate_timers(trigger_manager):
    """Rapid register -> unregister -> register must leave exactly one timer."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_toggle")

    # Simulate rapid toggling from a thread (like the UI toggle button)
    await asyncio.to_thread(
        trigger_manager.register_trigger, "proj_test", trigger
    )
    await asyncio.to_thread(
        trigger_manager.unregister_trigger, "trg_toggle"
    )
    await asyncio.to_thread(
        trigger_manager.register_trigger, "proj_test", trigger
    )

    assert "trg_toggle" in trigger_manager._timers
    # Only one timer should exist for this trigger_id
    assert len([k for k in trigger_manager._timers if k == "trg_toggle"]) == 1

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_unregister_cancels_timer(trigger_manager):
    """unregister_trigger() must cancel the timer (both asyncio.Task and Future)."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_cancel")
    await asyncio.to_thread(
        trigger_manager.register_trigger, "proj_test", trigger
    )
    assert "trg_cancel" in trigger_manager._timers

    await asyncio.to_thread(
        trigger_manager.unregister_trigger, "trg_cancel"
    )
    assert "trg_cancel" not in trigger_manager._timers

    await trigger_manager.stop()


@pytest.mark.asyncio
async def test_register_idempotent_replaces_existing(trigger_manager):
    """Calling register_trigger twice for same trigger_id replaces the old timer."""
    await trigger_manager.start()

    trigger = _make_trigger(trigger_id="trg_idem")
    trigger_manager.register_trigger("proj_test", trigger)
    first_timer = trigger_manager._timers.get("trg_idem")

    # Register again — should replace, not duplicate
    trigger_manager.register_trigger("proj_test", trigger)
    second_timer = trigger_manager._timers.get("trg_idem")

    assert first_timer is not second_timer
    assert "trg_idem" in trigger_manager._timers

    await trigger_manager.stop()
