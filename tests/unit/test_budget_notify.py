# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the budget threshold/trip emitter (Budget Piece 2 — Task E).

``agent_os/budget/notify.py`` is the producer side of the relay's Tier-1
``budget.exhausted`` push (plus a sibling ``budget.threshold``). It detects:

  * crossing 80% of the limit within the current window, and
  * the guard tripping (>= 100%),

and emits AT MOST ONCE per (project, window-identity) per threshold. A persisted
last-fired marker keyed by window identity survives a daemon restart and
re-arms on a new window. Codes/numbers only — no display strings.

These tests pin: once-per-window (incl. simulated restart via fresh state +
same marker file), re-arm on new window identity, trip + threshold both in one
window, no-limit no-op, marker corruption tolerance, and payload SHAPE
(codes/numbers only).
"""

import json
import os
from datetime import datetime, timezone

import pytest

from agent_os.budget.notify import (
    THRESHOLD_PCT,
    EVENT_TRIP,
    EVENT_THRESHOLD,
    window_identity,
    maybe_emit_budget_event,
    maybe_emit_trip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spend_returning(amount, *, currency="USD"):
    """A stub spend() returning a fixed converted total (mirrors guard tests)."""
    def _spend(project_dir, window, *, target_currency=None, fx_rates=None,
               now=None, anchor_ts=None):
        return {
            "window": window,
            "by_currency": {},
            "converted_total": {
                "currency": target_currency or currency,
                "amount": amount,
                "estimated": True,
            },
            "breakdown": [],
        }
    return _spend


def _cfg(limit=10.0, *, period="daily", currency="USD", action="pause",
         anchor_ts=None):
    return {
        "budget_limit_usd": limit,
        "budget_period": period,
        "budget_currency": currency,
        "budget_action": action,
        "budget_anchor_ts": anchor_ts,
    }


def _emit_once(workspace, cfg, *, spend, now=None, marker_path=None):
    """Run the emitter once, collecting emitted payloads into a list."""
    emitted = []
    maybe_emit_budget_event(
        workspace,
        cfg,
        emit=emitted.append,
        spend_fn=_spend_returning(spend),
        now=now,
        marker_path=marker_path,
    )
    return emitted


# ---------------------------------------------------------------------------
# window_identity
# ---------------------------------------------------------------------------

class TestWindowIdentity:
    def test_stable_within_same_day(self):
        a = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        b = datetime(2026, 6, 11, 23, 30, tzinfo=timezone.utc)
        assert window_identity("daily", None, a) == window_identity("daily", None, b)

    def test_changes_across_day_rollover(self):
        a = datetime(2026, 6, 11, 23, 30, tzinfo=timezone.utc)
        b = datetime(2026, 6, 12, 0, 30, tzinfo=timezone.utc)
        assert window_identity("daily", None, a) != window_identity("daily", None, b)

    def test_period_is_part_of_identity(self):
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        assert window_identity("daily", None, now) != window_identity("monthly", None, now)

    def test_total_window_keyed_by_anchor(self):
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        a = window_identity("total", "2026-01-01T00:00:00+00:00", now)
        b = window_identity("total", "2026-02-01T00:00:00+00:00", now)
        assert a != b


# ---------------------------------------------------------------------------
# Threshold crossing
# ---------------------------------------------------------------------------

class TestThresholdCrossing:
    def test_crossing_80_pct_fires_once(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        # spend 8.5 / limit 10 = 85% → crosses 80%
        emitted = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now)
        assert len(emitted) == 1
        ev = emitted[0]
        assert ev["type"] == EVENT_THRESHOLD
        assert ev["pct"] == THRESHOLD_PCT == 80

    def test_below_80_does_not_fire(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        emitted = _emit_once(ws, _cfg(limit=10.0), spend=7.9, now=now)
        assert emitted == []

    def test_second_eval_same_window_does_not_refire(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        first = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now)
        assert len(first) == 1
        # Same window, more spend (still < limit) → must NOT re-fire threshold.
        later = datetime(2026, 6, 11, 14, 0, tzinfo=timezone.utc)
        second = _emit_once(ws, _cfg(limit=10.0), spend=9.0, now=later)
        assert second == []

    def test_repeated_eval_5s_cadence_no_spam(self, tmp_path):
        """Simulate the dispatcher's 5s paused cadence: many evals, one fire."""
        ws = str(tmp_path)
        base = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        total = []
        for i in range(20):
            now = base.replace(second=(i % 50))
            total += _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now)
        assert len(total) == 1

    def test_survives_daemon_restart(self, tmp_path):
        """Fresh state objects (new process) + same marker file → no re-fire."""
        ws = str(tmp_path)
        marker = str(tmp_path / "marker.json")
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        first = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now,
                           marker_path=marker)
        assert len(first) == 1
        # "Restart": nothing in memory; only the on-disk marker persists.
        second = _emit_once(ws, _cfg(limit=10.0), spend=8.6, now=now,
                            marker_path=marker)
        assert second == []


# ---------------------------------------------------------------------------
# New window re-arms
# ---------------------------------------------------------------------------

class TestNewWindowRearms:
    def test_day_rollover_refires(self, tmp_path):
        ws = str(tmp_path)
        d1 = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        first = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=d1)
        assert len(first) == 1
        # Next day → new window identity → re-arm.
        d2 = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
        second = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=d2)
        assert len(second) == 1
        assert second[0]["type"] == EVENT_THRESHOLD

    def test_anchor_move_refires_total_window(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        cfg1 = _cfg(limit=10.0, period="total", anchor_ts="2026-01-01T00:00:00+00:00")
        first = _emit_once(ws, cfg1, spend=8.5, now=now)
        assert len(first) == 1
        cfg2 = _cfg(limit=10.0, period="total", anchor_ts="2026-06-01T00:00:00+00:00")
        second = _emit_once(ws, cfg2, spend=8.5, now=now)
        assert len(second) == 1


# ---------------------------------------------------------------------------
# Trip
# ---------------------------------------------------------------------------

class TestTrip:
    def test_trip_fires_trip_event_once(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        first = _emit_once(ws, _cfg(limit=10.0), spend=12.0, now=now)
        # Over limit → trip. Trip event present.
        types = [e["type"] for e in first]
        assert EVENT_TRIP in types
        # Second eval same window → no re-fire of the trip.
        second = _emit_once(ws, _cfg(limit=10.0), spend=13.0, now=now)
        assert EVENT_TRIP not in [e["type"] for e in second]

    def test_threshold_then_trip_both_fire_in_one_window(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        # First eval at 85% → threshold only.
        first = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now)
        assert [e["type"] for e in first] == [EVENT_THRESHOLD]
        # Later eval at 110% → trip (threshold already fired, must not re-fire).
        later = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
        second = _emit_once(ws, _cfg(limit=10.0), spend=11.0, now=later)
        assert [e["type"] for e in second] == [EVENT_TRIP]

    def test_trip_directly_also_emits_threshold_once(self, tmp_path):
        """Jumping straight past 80% to 100%+ in one eval fires BOTH (each once)."""
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        emitted = _emit_once(ws, _cfg(limit=10.0), spend=15.0, now=now)
        types = [e["type"] for e in emitted]
        assert EVENT_THRESHOLD in types
        assert EVENT_TRIP in types
        # Re-eval: nothing re-fires.
        again = _emit_once(ws, _cfg(limit=10.0), spend=16.0, now=now)
        assert again == []


# ---------------------------------------------------------------------------
# No-limit / disconnected / corruption
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_limit_nothing_fires(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        for limit in (None, 0, -5):
            emitted = _emit_once(ws, _cfg(limit=limit), spend=999.0, now=now)
            assert emitted == [], f"limit={limit} should not fire"

    def test_emit_none_is_noop(self, tmp_path):
        """Relay absent → emit callback is None → no exception, no marker spam."""
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        # Should not raise even though it would otherwise cross 80%.
        maybe_emit_budget_event(
            ws, _cfg(limit=10.0), emit=None,
            spend_fn=_spend_returning(8.5), now=now,
        )

    def test_spend_query_failure_is_noop(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)

        def _boom(*a, **k):
            raise RuntimeError("ledger I/O failed")

        emitted = []
        maybe_emit_budget_event(
            ws, _cfg(limit=10.0), emit=emitted.append,
            spend_fn=_boom, now=now,
        )
        assert emitted == []

    def test_corrupt_marker_refires_then_is_stable(self, tmp_path):
        ws = str(tmp_path)
        marker = str(tmp_path / "marker.json")
        with open(marker, "w") as f:
            f.write("{not valid json")
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        # Corrupt marker → re-fire is acceptable.
        first = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now,
                           marker_path=marker)
        assert len(first) == 1
        # After re-write the marker must be valid → steady state, no spam.
        second = _emit_once(ws, _cfg(limit=10.0), spend=8.5, now=now,
                            marker_path=marker)
        assert second == []
        # Marker is now valid JSON.
        with open(marker) as f:
            json.loads(f.read())


# ---------------------------------------------------------------------------
# Payload shape: codes / numbers only
# ---------------------------------------------------------------------------

class TestPayloadShape:
    def test_threshold_payload_codes_only(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        ev = _emit_once(ws, _cfg(limit=10.0, currency="USD", period="daily"),
                        spend=8.5, now=now)[0]
        assert ev["type"] == EVENT_THRESHOLD
        assert ev["pct"] == 80
        assert ev["spend"] == 8.5
        assert ev["limit"] == 10.0
        assert ev["currency"] == "USD"
        assert ev["window"] == "daily"
        # No sentence-like display strings: every string value is a short code
        # with no spaces.
        for k, v in ev.items():
            if isinstance(v, str):
                assert " " not in v, f"field {k}={v!r} looks like a display string"

    def test_trip_payload_codes_only(self, tmp_path):
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        emitted = _emit_once(ws, _cfg(limit=10.0, currency="USD", period="daily"),
                             spend=12.0, now=now)
        trip = [e for e in emitted if e["type"] == EVENT_TRIP][0]
        assert trip["spend"] == 12.0
        assert trip["limit"] == 10.0
        assert trip["currency"] == "USD"
        assert trip["window"] == "daily"
        for k, v in trip.items():
            if isinstance(v, str):
                assert " " not in v, f"field {k}={v!r} looks like a display string"


# ---------------------------------------------------------------------------
# Guard-trip emission site (maybe_emit_trip) — coordinator-ruled dual-site.
# The trip fires at whichever site observes it first; the shared marker key
# guarantees exactly once per (project, window-identity).
# ---------------------------------------------------------------------------

def _trip_once(workspace, *, marker_path=None, now=None, window="daily",
               spend=12.0, limit=10.0, currency="USD", anchor_ts=None,
               emit_list=None):
    emitted = [] if emit_list is None else emit_list
    maybe_emit_trip(
        workspace,
        window=window,
        spend=spend,
        limit=limit,
        currency=currency,
        anchor_ts=anchor_ts,
        emit=emitted.append,
        now=now,
        marker_path=marker_path,
    )
    return emitted


class TestGuardTripSite:
    def test_guard_trip_fires_once(self, tmp_path):
        """No marker, no append — the guard-trip site fires the trip exactly
        once; repeated guard trips in the same window do not re-fire."""
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        first = _trip_once(ws, now=now)
        assert len(first) == 1
        ev = first[0]
        assert ev["type"] == EVENT_TRIP
        assert ev["spend"] == 12.0
        assert ev["limit"] == 10.0
        # Repeated guard evaluations in the same window: no re-fire.
        for _ in range(5):
            assert _trip_once(ws, now=now) == []

    def test_append_site_then_guard_trip_dedups(self, tmp_path):
        """Crossing append fired the trip earlier in the window → a subsequent
        guard trip must NOT re-fire (shared marker key)."""
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        # Append site fires the trip (spend 11 >= limit 10).
        appended = _emit_once(ws, _cfg(limit=10.0), spend=11.0, now=now)
        assert EVENT_TRIP in [e["type"] for e in appended]
        # Guard trip later in the same window: dedup'd.
        later = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)
        assert _trip_once(ws, now=later, spend=11.0) == []

    def test_guard_trip_then_append_site_dedups(self, tmp_path):
        """Reverse order: guard trip fired first → the crossing append must NOT
        re-fire the trip (it may still owe the threshold, which is append-only)."""
        ws = str(tmp_path)
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        assert len(_trip_once(ws, now=now, spend=11.0)) == 1
        # Append-site eval in the same window: trip dedup'd; threshold (also
        # crossed, never fired) still fires once.
        appended = _emit_once(ws, _cfg(limit=10.0), spend=11.0, now=now)
        assert [e["type"] for e in appended] == [EVENT_THRESHOLD]

    def test_new_window_rearms_guard_trip(self, tmp_path):
        ws = str(tmp_path)
        d1 = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        assert len(_trip_once(ws, now=d1)) == 1
        d2 = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
        assert len(_trip_once(ws, now=d2)) == 1

    def test_guard_trip_survives_restart(self, tmp_path):
        """Same marker file, fresh call (new process) → no re-fire."""
        ws = str(tmp_path)
        marker = str(tmp_path / "marker.json")
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        assert len(_trip_once(ws, now=now, marker_path=marker)) == 1
        assert _trip_once(ws, now=now, marker_path=marker) == []

    def test_emit_none_is_noop(self, tmp_path):
        maybe_emit_trip(
            str(tmp_path), window="daily", spend=12.0, limit=10.0,
            currency="USD", emit=None,
            now=datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc),
        )

    def test_missing_decision_fields_noop(self, tmp_path):
        emitted = []
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        for window, spend, limit in (
            (None, 12.0, 10.0), ("daily", None, 10.0), ("daily", 12.0, None),
        ):
            maybe_emit_trip(
                str(tmp_path), window=window, spend=spend, limit=limit,
                currency="USD", emit=emitted.append, now=now,
            )
        assert emitted == []

    def test_trip_payload_codes_only(self, tmp_path):
        now = datetime(2026, 6, 11, 9, 0, tzinfo=timezone.utc)
        ev = _trip_once(str(tmp_path), now=now)[0]
        for k, v in ev.items():
            if isinstance(v, str):
                assert " " not in v, f"field {k}={v!r} looks like a display string"
