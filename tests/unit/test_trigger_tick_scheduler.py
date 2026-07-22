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


# ===========================================================================
# Orphan cleanup (review finding 1) — evaluation must self-clean schedule
# triggers whose project or trigger record disappeared out from under it,
# instead of re-checking a dead trigger_id forever.
# ===========================================================================

class TestOrphanCleanup:

    @pytest.mark.asyncio
    async def test_project_deleted_unregisters_trigger(self):
        """delete_project never calls unregister_trigger, so evaluation is
        the only janitor for a schedule trigger whose project is gone."""
        trigger = _trigger("trg_orphan_proj", "0 11 * * *", last_triggered=None)
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)
        assert "trg_orphan_proj" in tm._schedule_ids

        store.delete_project(pid)

        await tm._evaluate_due_triggers()  # must not raise/crash

        assert "trg_orphan_proj" not in tm._schedule_ids
        assert "trg_orphan_proj" not in tm._trigger_project
        mgr.start_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_removed_from_project_unregisters(self):
        """A trigger dropped from its project's triggers list by some path
        other than DeleteTriggerTool (which calls unregister_trigger itself)
        must still be cleaned up by evaluation, without disturbing sibling
        triggers on the same project."""
        trigger = _trigger("trg_removed", "0 11 * * *", last_triggered=None)
        other = _trigger("trg_other", "0 12 * * *", last_triggered=None)
        store, pid = _make_project_store(triggers=[trigger, other])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)
        tm.register_trigger(pid, other)

        # Simulate the trigger vanishing from the project's trigger list via
        # a path that bypasses unregister_trigger.
        store.update_project(pid, {"triggers": [other]})

        await tm._evaluate_due_triggers()  # must not raise/crash

        assert "trg_removed" not in tm._schedule_ids
        assert "trg_removed" not in tm._trigger_project
        # The sibling trigger is unaffected and still fires.
        assert mgr.start_agent.call_count == 1
        assert mgr.start_agent.call_args.args[0] == pid

    @pytest.mark.asyncio
    async def test_disabled_trigger_stays_registered_not_unregistered(self):
        """A trigger toggled to disabled by a path that bypasses
        unregister_trigger must NOT be dropped from the evaluated set —
        re-enabling it later must not require a fresh register_trigger call —
        it just must not fire while disabled."""
        trigger = _trigger("trg_toggle_off", "0 11 * * *", last_triggered=None)
        store, pid = _make_project_store(triggers=[trigger])
        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid, trigger)
        assert "trg_toggle_off" in tm._schedule_ids

        disabled = dict(trigger, enabled=False)
        store.update_project(pid, {"triggers": [disabled]})

        await tm._evaluate_due_triggers()

        assert "trg_toggle_off" in tm._schedule_ids  # still registered
        mgr.start_agent.assert_not_called()


# ===========================================================================
# One bad trigger must not starve others (review finding 2) — a per-item
# exception during firing (e.g. a project_store write failure, which sits
# outside _fire_trigger's own try/except around start_agent) must not abort
# the rest of the tick's due list.
# ===========================================================================

class TestFireExceptionIsolation:

    @pytest.mark.asyncio
    async def test_one_trigger_exception_does_not_starve_others_same_tick(self):
        """Runs against the REAL ProjectStore's live-reference semantics —
        get_project() returns the actual backing dict, not a copy, and
        _fire_trigger mutates trigger["last_triggered"]/["trigger_count"] in
        place *before* calling update_project(). A fix that merely swallows
        the exception (without rolling back that in-place stamp) would still
        leave the trigger looking "already fired" in memory even though
        update_project — and therefore start_agent — never completed. The
        fix under test is _fire_trigger's rollback of that stamp when
        update_project raises (see the try/except around the update_project
        call in _fire_trigger)."""
        trigger_a = _trigger("trg_fail", "0 7 * * *", last_triggered=None, task="Task A")
        trigger_b = _trigger("trg_ok", "0 11 * * *", last_triggered=None, task="Task B")
        store, pid_a, pid_b = _make_two_project_store([trigger_a], [trigger_b])

        # trg_fail has the earlier prev_occ, so it sorts first — without the
        # per-item try/except (fix pass 1), an unhandled exception here would
        # prevent trg_ok (sorted after it) from ever being attempted this
        # tick. The failure is transient: it raises only on trg_fail's first
        # write attempt, then clears — modeling a one-off disk hiccup rather
        # than a permanently broken project.
        real_update_project = store.update_project
        raised_once = {"done": False}

        def flaky_update_project(project_id, updates):
            if project_id == pid_a and not raised_once["done"]:
                raised_once["done"] = True
                raise RuntimeError("simulated transient project_store write failure")
            return real_update_project(project_id, updates)

        store.update_project = flaky_update_project

        mgr = _make_agent_mgr(is_running=False)
        tm = TriggerManager(store, mgr, now_fn=lambda: FIXED_NOW)
        tm.register_trigger(pid_a, trigger_a)
        tm.register_trigger(pid_b, trigger_b)

        await tm._evaluate_due_triggers()  # must not raise/crash

        # B still fires despite A's failure earlier in the sort order.
        assert mgr.start_agent.call_count == 1
        assert mgr.start_agent.call_args.args[0] == pid_b
        assert _last_triggered(store, pid_b, "trg_ok") is not None

        # A's failed fire must be fully rolled back on the LIVE trigger
        # object — not just "no crash," but genuinely still due, not stuck
        # showing "fired" until the next natural cron occurrence.
        assert _last_triggered(store, pid_a, "trg_fail") is None

        # Next tick: the transient failure has cleared, so A actually fires.
        await tm._evaluate_due_triggers()
        assert mgr.start_agent.call_count == 2
        assert mgr.start_agent.call_args.args[0] == pid_a
        assert _last_triggered(store, pid_a, "trg_fail") is not None
