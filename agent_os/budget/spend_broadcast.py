# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Debounced ``budget.spend_updated`` WS broadcast (Budget Piece 3 — Task P3-C).

The frontend spend displays (header corner + settings meter) are EVENT-DRIVEN,
not polled. This module emits a lightweight ``budget.spend_updated`` event on
each ledger append carrying the post-append window spend (the same numbers the
meter compares against the limit), DEBOUNCED ≥1s per project.

Why debounce: appends arrive in bursts (a multi-turn run emits one ledger line
per management LLM response, several within a second). Broadcasting one WS frame
per append would be wasteful. The debouncer collapses a burst into at most a
LEADING broadcast (the first append, immediate) plus a TRAILING broadcast (the
LAST append's numbers, after the interval) — so the corner updates promptly AND
never loses the final state.

Async shape (no threads, no busy/polling timers):
  * ``submit`` runs synchronously on the daemon's event loop (called from the
    loop's per-append hook). It either fires a leading broadcast immediately or
    stores the latest payload as pending and arms a SINGLE trailing task via
    ``loop.call_later`` — never a per-second loop. When idle, no timer exists.
  * The trailing task fires once at ``last_emit + interval``, broadcasts the
    pending payload, then disarms. A new burst re-arms it.

Payload carries CODES/NUMBERS ONLY (the binding i18n rule): the client renders
the display from ``window`` / ``spend`` / ``limit`` / ``currency``. ``limit`` is
``None`` when no limit is configured (the corner still shows spend).

Resilience: a throwing emit sink is swallowed (the ledger append must never be
affected by a broadcast failure). ``emit=None`` (relay absent) is a silent
no-op.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# The WS event type the frontend spend displays subscribe to. Routed through the
# SAME on_budget_event → _broadcast → relay forward path as the notify events,
# but it is Tier-3 in the relay client (UI-only, never a push notification).
SPEND_UPDATED_EVENT = "budget.spend_updated"

# Default minimum spacing between broadcasts for one project (the spec's ≥1s).
DEFAULT_INTERVAL = 1.0


class SpendBroadcaster:
    """Per-project leading+trailing debouncer for ``budget.spend_updated``.

    One instance per loop (i.e. per project's active session). ``submit`` is
    cheap and synchronous; the trailing flush is a single ``call_later`` task.

    Leading+trailing (not leading-only) is deliberate: a leading-only debounce
    would drop the final append's numbers when a burst's last submit lands
    inside the cooldown — leaving the corner stuck on a stale mid-burst value.
    The trailing edge guarantees the LAST submitted state is always delivered.
    """

    def __init__(self, *, emit, interval: float = DEFAULT_INTERVAL):
        # ``emit(payload: dict)`` routes the event through the WS broadcast path
        # (the loop wires it to the same sink as the notify events). ``None`` →
        # silent no-op (relay absent / lightweight test loops).
        self._emit = emit
        self._interval = interval
        # Monotonic timestamp of the last actual broadcast (leading or trailing).
        # ``None`` means "never emitted" → the next submit is a leading edge.
        self._last_emit: float | None = None
        # The most-recent payload awaiting a trailing flush, or ``None`` when
        # nothing is pending.
        self._pending: dict | None = None
        # The single scheduled trailing task handle (``loop.call_later``), or
        # ``None`` when disarmed. Exactly one exists at a time.
        self._timer: asyncio.TimerHandle | None = None

    # ------------------------------------------------------------------

    def submit(self, *, window, spend, limit, currency) -> None:
        """Record a post-append spend snapshot and broadcast it, debounced.

        Synchronous and non-blocking. Fires a leading broadcast immediately when
        the project is outside the cooldown; otherwise stores the payload as the
        pending trailing state and arms (once) the trailing flush.
        """
        if self._emit is None:
            return

        payload = {
            "type": SPEND_UPDATED_EVENT,
            "window": window,
            "spend": spend,
            "limit": limit,
            "currency": currency,
        }

        now = self._now()
        if self._last_emit is None or (now - self._last_emit) >= self._interval:
            # Leading edge: outside the cooldown → broadcast immediately.
            self._do_emit(payload, now)
            return

        # Inside the cooldown: stash as the trailing state and arm the flush
        # once. Re-arming on every submit is avoided so the trailing fires at
        # ``last_emit + interval`` (a fixed point), not pushed out indefinitely
        # by a steady stream of appends.
        self._pending = payload
        if self._timer is None:
            delay = max(0.0, (self._last_emit + self._interval) - now)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop (shouldn't happen on the daemon path). Emit
                # the trailing state immediately rather than dropping it.
                self._do_emit(payload, now)
                self._pending = None
                return
            self._timer = loop.call_later(delay, self._flush_trailing)

    # ------------------------------------------------------------------

    def _flush_trailing(self) -> None:
        """Fire the pending trailing broadcast (scheduled by ``call_later``)."""
        self._timer = None
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        self._do_emit(pending, self._now())

    def _do_emit(self, payload: dict, now: float) -> None:
        """Invoke the sink and record the emit time. Never raises."""
        self._last_emit = now
        try:
            self._emit(payload)
        except Exception:  # noqa: BLE001 — a broadcast failure must not affect
            # the ledger append that triggered it, nor crash the trailing timer.
            logger.warning(
                "budget.spend_updated broadcast failed; continuing.",
                exc_info=True,
            )

    @staticmethod
    def _now() -> float:
        # Monotonic clock: debounce spacing must be immune to wall-clock jumps.
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        # ``loop.time()`` IS the monotonic clock the TimerHandle is scheduled
        # against, so ``call_later`` delays line up exactly. Fall back to the
        # module-level monotonic when no loop is running (degenerate path).
        if loop is not None:
            return loop.time()
        import time
        return time.monotonic()

    async def aclose(self) -> None:
        """Cancel any pending trailing task. Idempotent; drops the final state.

        Called when the loop tears down. The trailing broadcast is best-effort
        UI freshness, not durable state, so cancelling a pending flush on close
        is acceptable — the next run's first append re-broadcasts the current
        total as a fresh leading edge.
        """
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending = None
