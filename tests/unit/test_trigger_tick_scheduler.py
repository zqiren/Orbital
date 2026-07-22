# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Unit tests for TriggerManager's tick-loop scheduler.

Replaces the per-trigger `asyncio.sleep(delay)` timers (which freeze across
macOS system sleep, causing triggers to fire hours late or never — see
trigger_manager.py module docstring) with a single periodic tick loop that
evaluates every registered schedule trigger against wall-clock time on each
tick, catching up on anything missed while the daemon or machine was asleep.

These tests drive `TriggerManager._evaluate_due_triggers()` directly with an
injected `now_fn`, so no real sleeping is involved.
"""

import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

import pytest

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.daemon_v2.trigger_manager import TriggerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# "Now" for every test below: Wednesday 2026-07-22 15:00 UTC.
FIXED_NOW = datetime(2026, 7, 22, 15, 0, 0, tzinfo=timezone.utc)


def _make_project_store(triggers=None):
    """Create a real ProjectStore with a temp directory and one project."""
    tmpdir = tempfile.mkdtemp()
    store = ProjectStore(data_dir=tmpdir)
    pid = store.create_project({
        "name": "Test Project",
        "workspace": tmpdir,
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    if triggers is not None:
        store.update_project(pid, {"triggers": triggers})
    return store, pid


def _make_two_project_store(triggers_a, triggers_b):
    """Create a real ProjectStore with two separate projects."""
    store = ProjectStore(data_dir=tempfile.mkdtemp())
    pid_a = store.create_project({
        "name": "Project A",
        "workspace": tempfile.mkdtemp(),
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    pid_b = store.create_project({
        "name": "Project B",
        "workspace": tempfile.mkdtemp(),
        "model": "gpt-4",
        "api_key": "sk-test",
    })
    store.update_project(pid_a, {"triggers": triggers_a})
    store.update_project(pid_b, {"triggers": triggers_b})
    return store, pid_a, pid_b


def _trigger(trigger_id, cron, *, tz="UTC", last_triggered=None,
             created_at="2020-01-01T00:00:00+00:00", enabled=True,
             name=None, task="Do the thing"):
    return {
        "id": trigger_id,
        "name": name or trigger_id,
        "type": "schedule",
        "enabled": enabled,
        "schedule": {"cron": cron, "timezone": tz, "human": cron},
        "task": task,
        "last_triggered": last_triggered,
        "trigger_count": 0,
        "created_at": created_at,
    }


def _make_agent_mgr(is_running=False):
    mgr = MagicMock()
    mgr.is_running = MagicMock(return_value=is_running)
    mgr.start_agent = AsyncMock()
    mgr._setup_engine = None
    mgr._settings_store = None
    mgr._credential_store = None
    return mgr


def _last_triggered(store, pid, trigger_id):
    project = store.get_project(pid)
    trigger = next(t for t in project["triggers"] if t["id"] == trigger_id)
    return trigger["last_triggered"]


# ===========================================================================
# Due-rule evaluation
# ===========================================================================

class TestDueRuleEvaluation:

    @pytest.mark.asyncio
    async def test_missed_occurrence_fires_once_coalesced(self):
        """last_triggered is 3 days stale (daemon down / machine asleep) —
        the catch-up must coalesce into exactly one fire, not one per missed day."""
        trigger = _trigger(
            "trg_missed", "0 11 * * *",
            last_triggered="2026-07-19T11:05:00+00:00",
        )
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)

        await tm._evaluate_due_triggers()

        assert mgr.start_agent.call_count == 1
        assert _last_triggered(store, pid, "trg_missed") == FIXED_NOW.isoformat()

        # Re-evaluating at the same "now" must not fire again.
        await tm._evaluate_due_triggers()
        assert mgr.start_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_not_yet_due_no_fire(self):
        """Already fired for today's occurrence — must not fire again before
        the next one."""
        trigger = _trigger(
            "trg_notdue", "0 23 * * *",
            last_triggered="2026-07-21T23:00:05+00:00",
        )
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)

        await tm._evaluate_due_triggers()

        mgr.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_created_after_occurrence_no_fire(self):
        """A trigger created after the most recent occurrence must not catch
        up on that pre-creation occurrence."""
        trigger = _trigger(
            "trg_new", "0 11 * * *",
            last_triggered=None,
            created_at="2026-07-22T12:00:00+00:00",  # after today's 11:00 occurrence
        )
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)

        await tm._evaluate_due_triggers()

        mgr.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_weekly_monday_daemon_down_repro(self):
        """Production repro: a weekly Monday-09:00 trigger whose occurrence
        passed while the daemon was down (last_triggered=None, created before
        Monday, now=Wednesday) must fire on the first evaluation."""
        trigger = _trigger(
            "trg_weekly", "0 9 * * 1",
            last_triggered=None,
            created_at="2026-07-17T00:00:00+00:00",  # Friday, before Monday 07-20
        )
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)  # Wednesday
        tm.register_trigger(pid, trigger)

        await tm._evaluate_due_triggers()

        assert mgr.start_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_future_last_triggered_clock_skew_no_crash(self):
        """A last_triggered in the future (clock skew) must be treated as
        not due, without raising."""
        future = (FIXED_NOW.replace(year=2026, month=7, day=23)).isoformat()
        trigger = _trigger(
            "trg_skew", "0 11 * * *",
            last_triggered=future,
        )
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)

        await tm._evaluate_due_triggers()  # must not raise

        mgr.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_timezone_respected(self):
        """cron "0 9 * * 1" with timezone Asia/Shanghai must be evaluated in
        that timezone, not naively as if the cron were UTC.

        now (UTC) = Tue 2026-07-21 02:00Z = Tue 2026-07-21 10:00 Shanghai.
        Correct (Shanghai-aware) prev occurrence: Mon 2026-07-20 09:00+08:00
          = 2026-07-20 01:00Z.
        Naive/buggy (cron treated as UTC) prev occurrence would be:
          2026-07-20 09:00Z (8 hours later).
        last_triggered is set between these two candidates, so the two
        interpretations disagree on due-ness — proving tz handling matters.
        """
        now_shanghai_case = datetime(2026, 7, 21, 2, 0, 0, tzinfo=timezone.utc)
        trigger = _trigger(
            "trg_tz", "0 9 * * 1", tz="Asia/Shanghai",
            last_triggered="2026-07-20T03:00:00+00:00",
        )
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: now_shanghai_case)
        tm.register_trigger(pid, trigger)

        await tm._evaluate_due_triggers()

        # Correct tz-aware due-rule says NOT due; a naive-UTC bug would fire.
        mgr.start_agent.assert_not_called()


# ===========================================================================
# Hold, don't skip (agent busy)
# ===========================================================================

class TestHoldDontSkip:

    @pytest.mark.asyncio
    async def test_agent_busy_held_then_fires_when_freed(self):
        trigger = _trigger("trg_busy", "0 11 * * *", last_triggered=None)
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=True)
        mock_ws = MagicMock()
        tm = TriggerManager(store, mgr, ws_manager=mock_ws, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)
        mock_ws.reset_mock()  # ignore the trigger.created broadcast from register_trigger

        # First held tick: broadcasts trigger.held, does not fire, does not
        # touch last_triggered.
        await tm._evaluate_due_triggers()
        mgr.start_agent.assert_not_called()
        assert _last_triggered(store, pid, "trg_busy") is None
        held_events = [
            c.args[1] for c in mock_ws.broadcast.call_args_list
            if c.args[1].get("type") == "trigger.held"
        ]
        assert len(held_events) == 1
        assert held_events[0]["reason"] == "agent_busy"
        assert held_events[0]["trigger_id"] == "trg_busy"

        # Second consecutive held tick: still no fire, no *additional* broadcast.
        await tm._evaluate_due_triggers()
        mgr.start_agent.assert_not_called()
        held_events = [
            c.args[1] for c in mock_ws.broadcast.call_args_list
            if c.args[1].get("type") == "trigger.held"
        ]
        assert len(held_events) == 1  # still just the one from the first hold

        # Agent frees up: fires on next evaluation.
        mgr.is_running.return_value = False
        await tm._evaluate_due_triggers()
        mgr.start_agent.assert_called_once()
        assert _last_triggered(store, pid, "trg_busy") is not None


# ===========================================================================
# Per-project serialization (one fire per project per tick)
# ===========================================================================

class TestPerProjectSerialization:

    @pytest.mark.asyncio
    async def test_same_project_only_earlier_fires_this_tick(self):
        trigger_a = _trigger("trg_a", "0 7 * * *", last_triggered=None, task="Task A")
        trigger_b = _trigger("trg_b", "0 8 * * *", last_triggered=None, task="Task B")
        store, pid = _make_project_store(triggers=[trigger_a, trigger_b])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger_a)
        tm.register_trigger(pid, trigger_b)

        await tm._evaluate_due_triggers()

        assert mgr.start_agent.call_count == 1
        assert _last_triggered(store, pid, "trg_a") is not None
        assert _last_triggered(store, pid, "trg_b") is None

        # Next tick: the deferred trigger fires.
        await tm._evaluate_due_triggers()
        assert mgr.start_agent.call_count == 2
        assert _last_triggered(store, pid, "trg_b") is not None

    @pytest.mark.asyncio
    async def test_different_projects_both_fire_same_tick(self):
        trigger_a = _trigger("trg_pa", "0 11 * * *", last_triggered=None, task="Task A")
        trigger_b = _trigger("trg_pb", "0 11 * * *", last_triggered=None, task="Task B")
        store, pid_a, pid_b = _make_two_project_store([trigger_a], [trigger_b])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid_a, trigger_a)
        tm.register_trigger(pid_b, trigger_b)

        await tm._evaluate_due_triggers()

        assert mgr.start_agent.call_count == 2
        fired_project_ids = {c.args[0] for c in mgr.start_agent.call_args_list}
        assert fired_project_ids == {pid_a, pid_b}
        assert _last_triggered(store, pid_a, "trg_pa") is not None
        assert _last_triggered(store, pid_b, "trg_pb") is not None
