# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for the debounced ``budget.spend_updated`` WS broadcast
(Budget Piece 3 — Task P3-C).

The frontend spend displays (header corner + settings meter) are event-driven,
not polled. ``SpendBroadcaster`` emits a lightweight ``budget.spend_updated``
event on each ledger append, DEBOUNCED ≥1s per project: a burst of appends
yields at most a leading + trailing broadcast, and the LAST append's numbers are
always eventually broadcast (the trailing edge never drops the final state).

These pin: leading-edge immediacy, trailing-edge final-state delivery, the
≥1s spacing, the no-limit (limit=None) payload, codes-only shape, and the
compute-failure (guard-style) skip-with-warning contract.
"""

import asyncio
import logging

import pytest

from agent_os.budget.spend_broadcast import (
    SpendBroadcaster,
    SPEND_UPDATED_EVENT,
)


def _drain():
    """A list-backed emit sink + the collected payloads."""
    collected = []
    return collected, collected.append


# ---------------------------------------------------------------------------
# Leading / trailing debounce shape
# ---------------------------------------------------------------------------

class TestDebounceShape:
    @pytest.mark.asyncio
    async def test_single_append_emits_immediately(self):
        """One append in a quiet period → a leading-edge broadcast right away."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=1.0)
        b.submit(window="daily", spend=1.5, limit=10.0, currency="USD")
        assert len(emitted) == 1
        ev = emitted[0]
        assert ev["type"] == SPEND_UPDATED_EVENT
        assert ev["spend"] == 1.5
        assert ev["limit"] == 10.0
        assert ev["window"] == "daily"
        assert ev["currency"] == "USD"
        await b.aclose()

    @pytest.mark.asyncio
    async def test_burst_emits_leading_then_one_trailing_with_last_numbers(self):
        """A burst of N appends within the interval → exactly 2 broadcasts
        (leading + trailing); the trailing carries the LAST append's spend."""
        emitted, sink = _drain()
        # A short interval keeps the test fast; the semantics are identical.
        b = SpendBroadcaster(emit=sink, interval=0.05)
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        b.submit(window="daily", spend=2.0, limit=10.0, currency="USD")
        b.submit(window="daily", spend=3.0, limit=10.0, currency="USD")
        b.submit(window="daily", spend=4.0, limit=10.0, currency="USD")
        # Leading fired synchronously with the first submit.
        assert len(emitted) == 1
        assert emitted[0]["spend"] == 1.0
        # Wait past the interval for the trailing flush.
        await asyncio.sleep(0.12)
        assert len(emitted) == 2, emitted
        # Trailing carries the LAST submitted numbers, not an intermediate one.
        assert emitted[1]["spend"] == 4.0
        await b.aclose()

    @pytest.mark.asyncio
    async def test_appends_spaced_past_interval_emit_each(self):
        """Appends spaced > interval apart each fire their own leading edge."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=0.05)
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        await asyncio.sleep(0.08)
        b.submit(window="daily", spend=2.0, limit=10.0, currency="USD")
        await asyncio.sleep(0.08)
        b.submit(window="daily", spend=3.0, limit=10.0, currency="USD")
        # Each spaced submit is its own leading edge → 3 broadcasts, no pending.
        await asyncio.sleep(0.08)
        spends = [e["spend"] for e in emitted]
        assert spends == [1.0, 2.0, 3.0], spends
        await b.aclose()

    @pytest.mark.asyncio
    async def test_trailing_only_one_per_window_even_for_long_burst(self):
        """Many appends within one interval → still at most leading + ONE
        trailing (the scheduled flush is not re-armed per submit)."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=0.05)
        for i in range(50):
            b.submit(window="daily", spend=float(i), limit=10.0, currency="USD")
        await asyncio.sleep(0.12)
        assert len(emitted) == 2, emitted
        assert emitted[0]["spend"] == 0.0   # leading = first
        assert emitted[1]["spend"] == 49.0  # trailing = last
        await b.aclose()

    @pytest.mark.asyncio
    async def test_no_trailing_when_no_further_submits(self):
        """A lone leading edge with no follow-up submit schedules no trailing
        broadcast (no busy timer firing a duplicate)."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=0.05)
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        assert len(emitted) == 1
        await asyncio.sleep(0.12)
        # Still exactly one — nothing pending, nothing fired.
        assert len(emitted) == 1
        await b.aclose()


# ---------------------------------------------------------------------------
# No-limit / payload shape / failure
# ---------------------------------------------------------------------------

class TestPayload:
    @pytest.mark.asyncio
    async def test_no_limit_still_broadcasts_with_null_limit(self):
        """The corner shows spend even without a limit → broadcast with
        limit=None."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=1.0)
        b.submit(window="daily", spend=2.5, limit=None, currency="USD")
        assert len(emitted) == 1
        assert emitted[0]["spend"] == 2.5
        assert emitted[0]["limit"] is None
        await b.aclose()

    @pytest.mark.asyncio
    async def test_payload_is_codes_and_numbers_only(self):
        """No display strings: every string field is a short code, no spaces."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=1.0)
        b.submit(window="weekly", spend=3.0, limit=20.0, currency="CNY")
        ev = emitted[0]
        assert set(ev.keys()) == {"type", "window", "spend", "limit", "currency"}
        for k, v in ev.items():
            if isinstance(v, str):
                assert " " not in v, f"field {k}={v!r} looks like a display string"
        await b.aclose()

    @pytest.mark.asyncio
    async def test_emit_failure_does_not_propagate(self, caplog):
        """A throwing sink must not crash submit() (append must be unaffected)."""
        def _boom(_payload):
            raise RuntimeError("ws send failed")

        b = SpendBroadcaster(emit=_boom, interval=1.0)
        with caplog.at_level(logging.WARNING):
            # Must not raise.
            b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        await b.aclose()

    @pytest.mark.asyncio
    async def test_emit_none_is_silent_noop(self):
        """No sink wired (relay absent) → submit is a harmless no-op."""
        b = SpendBroadcaster(emit=None, interval=1.0)
        # Must not raise; nothing to assert beyond non-crash.
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        await b.aclose()


# ---------------------------------------------------------------------------
# Idle settle: flush/aclose deliver the pending payload instead of dropping it
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_flushes_pending_trailing_once(self):
        """Closing before the trailing fires DELIVERS it immediately (it used to
        be cancelled), and the disarmed timer never fires a duplicate later."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=0.2)
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        b.submit(window="daily", spend=2.0, limit=10.0, currency="USD")
        assert len(emitted) == 1  # leading only so far
        await b.aclose()
        # The pending final numbers went out on close, not into the bin.
        assert len(emitted) == 2, emitted
        assert emitted[1]["spend"] == 2.0
        # Wait past when the trailing would have fired — no duplicate.
        await asyncio.sleep(0.3)
        assert len(emitted) == 2

    @pytest.mark.asyncio
    async def test_flush_emits_pending_immediately(self):
        """flush() is the idle-settle primitive: the corner lands on the final
        figure the moment work stops, not up to `interval` later."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=10.0)  # trailing far in the future
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        b.submit(window="daily", spend=7.5, limit=10.0, currency="USD")
        assert len(emitted) == 1
        b.flush()
        assert len(emitted) == 2, emitted
        assert emitted[1]["spend"] == 7.5
        await b.aclose()

    @pytest.mark.asyncio
    async def test_flush_with_nothing_pending_is_a_noop(self):
        """No pending payload → no duplicate frame. Safe to call at every idle
        boundary (run-end AND terminate) without spamming the socket."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=1.0)
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        assert len(emitted) == 1  # leading edge already carried the final number
        b.flush()
        b.flush()
        await b.aclose()
        assert len(emitted) == 1

    @pytest.mark.asyncio
    async def test_submit_after_aclose_still_works(self):
        """aclose() must not permanently mute the broadcaster: the SAME loop
        object is what a hot resume runs again, and its spend must keep
        reaching the UI."""
        emitted, sink = _drain()
        b = SpendBroadcaster(emit=sink, interval=0.05)
        b.submit(window="daily", spend=1.0, limit=10.0, currency="USD")
        await b.aclose()
        await asyncio.sleep(0.08)
        b.submit(window="daily", spend=2.0, limit=10.0, currency="USD")
        assert [e["spend"] for e in emitted] == [1.0, 2.0]
        await b.aclose()

    @pytest.mark.asyncio
    async def test_flush_swallows_a_throwing_sink(self):
        """The idle flush runs inside loop teardown — a broken sink must not
        propagate into it."""
        def _boom(_payload):
            raise RuntimeError("ws send failed")

        b = SpendBroadcaster(emit=_boom, interval=10.0)
        b.submit(window="daily", spend=1.0, limit=None, currency="USD")  # leading
        b.submit(window="daily", spend=2.0, limit=None, currency="USD")  # pending
        b.flush()          # must not raise
        await b.aclose()   # must not raise
