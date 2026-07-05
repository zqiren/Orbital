# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main agent execution loop.

Owned by Component A. Core while loop: prepare context -> call LLM ->
process response -> execute tools -> repeat.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from collections import deque
from uuid import uuid4

from agent_os.agent.providers.types import (
    StreamAccumulator,
    StreamChunk,
    LLMResponse,
    ContextOverflowError,
    LLMError,
    ErrorCategory,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.tool_result_filters import dispatch_prefilter
from agent_os.agent.tool_result_lifecycle import truncate_consumed_tool_results
from agent_os.budget.ledger import (
    LedgerEvent,
    SOURCE_MANAGEMENT,
    append_event,
)

logger = logging.getLogger(__name__)

# Permanent, always-on per-iteration diagnostics. Kept on a dedicated logger so
# the loop's disposition/exit path is greppable and routable independently of
# the noisy module logger. Records carry a structured `diag` dict on the
# LogRecord (via extra=) plus a greppable "agent.diag.<event>" message. See
# TASK-loop-diagnostic-instrumentation. Low-cost: one log call per iteration.
_diag_logger = logging.getLogger("agent.diag")

# Per-session cache-hit aggregation (Part 4D). Shares the provider's cache
# audit channel so per-call and per-session lines land together in daemon.log.
_cache_logger = logging.getLogger("orbital.cache_audit")

# Number of turns between automatic state checkpoints (turn-count trigger).
# Also used as the global cooldown between ANY two refreshes (except token-pressure).
COOLDOWN_TURNS = 50

# Spec 013: minimum seconds between STARTED background consolidation passes.
# Applies to every scheduler-routed trigger (agent_decided, turn_count,
# agent_decided_coalesced) — NOT to token_pressure or the session boundary,
# which stay synchronous and drain the gate instead.
REFRESH_DEBOUNCE_S = 300


def normalize_tool_call(tc_raw: dict) -> dict:
    """Normalize tool call from nested/flat/string-args formats."""
    if "function" in tc_raw:
        func = tc_raw["function"]
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        return {"id": tc_raw.get("id", ""), "name": func.get("name", ""), "arguments": args}
    args = tc_raw.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    return {"id": tc_raw.get("id", ""), "name": tc_raw.get("name", ""), "arguments": args}


class AgentLoop:
    """Core agent execution loop."""

    def __init__(
        self,
        session,
        provider,
        tool_registry,
        context_manager,
        interceptor=None,
        utility_provider=None,
        fallback_providers=None,
        max_iterations: int = 0,
        token_budget: int = 100_000_000,
        get_budget_config=None,
        on_session_end=None,
        on_session_end_refresh=None,
        project_dir: str | None = None,
        on_budget_event=None,
    ):
        self._session = session
        self._provider = provider
        self._tool_registry = tool_registry
        self._context_manager = context_manager
        self._interceptor = interceptor
        self._utility_provider = utility_provider
        self._fallback_providers = fallback_providers or []
        self._max_iterations = max_iterations
        # PURE token safety-net cap (NOT budget-derived). Budget Piece 2 cut the
        # dollar→token derivation that used to feed this; the gate now only
        # protects against a runaway non-terminating loop. The single
        # dollar-budget enforcement point is the pre-flight guard (below).
        self._token_budget = token_budget
        # LIVE budget config resolver (Budget Piece 2). A zero-arg callable
        # returning the CURRENT project budget config dict (or None). Resolved
        # fresh on EVERY guard evaluation — never snapshotted (the B5 fix). When
        # None, the pre-flight guard is a no-op (e.g. lightweight test loops).
        self._get_budget_config = get_budget_config
        # Budget threshold/trip emitter sink (Budget Piece 2 — Task E). An
        # optional callable ``on_budget_event(payload: dict)`` that routes a
        # codes-only budget event through the relay/WS broadcast path. When None
        # (lightweight test loops, no relay wiring), the emitter is a no-op.
        self._on_budget_event = on_budget_event
        # Per-project token ledger root: the workspace, resolved to
        # {workspace}/orbital/ledger/usage.jsonl via ProjectPaths. When None,
        # ledger emission is skipped (loop runs unchanged). Also the project_dir
        # the pre-flight guard reads the ledger spend from.
        self._project_dir = project_dir
        # Debounced ``budget.spend_updated`` WS broadcaster (Budget Piece 3 —
        # Task P3-C). Fed the post-append window spend after each ledger append
        # (reusing the notify check's already-computed numbers — zero extra
        # spend() query when a limit is set). Routes through the SAME
        # ``on_budget_event`` sink as the notify events. ≥1s leading+trailing
        # debounce per project so a multi-turn burst collapses to at most two
        # frames, the trailing carrying the LAST append's numbers. A None sink
        # makes ``submit`` a silent no-op (lightweight test loops / no relay).
        from agent_os.budget.spend_broadcast import SpendBroadcaster
        self._spend_broadcaster = SpendBroadcaster(emit=on_budget_event)
        self._running = False
        self._llm_failed = False
        self._on_session_end = on_session_end

        # Per-session cache-hit telemetry (Part 4D). Accumulated across every
        # run() of this loop (one loop == one session), logged at each run end.
        # The CONTEXT.md on-judgment-update decision is an empirical bet on the
        # prefix cache; this is the instrument that measures its impact.
        self._cache_read_tokens_total = 0
        self._prompt_input_tokens_total = 0
        self._llm_call_count = 0

        # Periodic state refresh callback (async callable(trigger_name: str))
        # Injected by agent_manager so loop.py stays decoupled from workspace_files.
        self._on_session_end_refresh = on_session_end_refresh

        # Cancellation state (cancel_turn / terminate)
        self._task: asyncio.Task | None = None
        self._inflight_stream: asyncio.Task | None = None
        self._stream_accumulator: StreamAccumulator | None = None
        self._turn_cancelled: bool = False
        self._cancellation_marker_appended_this_turn: bool = False

        # In-flight refresh task (registered with cancel pathway)
        self._refresh_task: asyncio.Task | None = None

        # Spec 013 async checkpoint scheduler state. _refresh_task doubles as
        # the single-flight gate for background passes (the spec's
        # `_refresh_inflight`). Dirty = a trigger arrived while a pass was in
        # flight or debounced; consumed at the turn boundary or session end.
        self._refresh_dirty: bool = False
        self._last_merge_at: float = float("-inf")   # monotonic; -inf → first schedule always passes debounce

        # State refresh tracking
        self._turns_since_last_update: int = 0  # resets after each refresh
        self._last_refresh_iteration: int = 0   # loop iteration# of last refresh
        self._current_iteration: int = 0        # current loop iteration (updated at top of loop)

        # Queue integration (Phase 2 — exit_reason; Phase 3 — queue_state)
        # Reset to defaults at the start of every run().
        self._exit_reason: str = "text"
        self._exit_summary: str | None = None
        self._exit_block_reason: str | None = None
        # "chat" (default) preserves legacy approval-pause; "running" makes
        # the dispatcher treat approval-required tool calls as blocks. The
        # dispatcher sets this before each run via agent_manager helpers.
        self._queue_state: str = "chat"

    @property
    def is_running(self) -> bool:
        return self._running

    def get_completion_state(self) -> "tuple[str, str | None, str | None]":
        """Return how the last run ended: (exit_reason, summary, block_reason).

        ``exit_reason`` is one of ``complete`` | ``blocked`` | ``cancelled`` |
        ``budget_blocked`` | ``text`` (text-only — no completion tool called).
        ``budget_blocked`` (Budget Piece 2) is a first-class exit distinct from
        the others: the pre-flight budget guard tripped, so the loop stopped
        BEFORE spending again. The queue dispatcher routes it explicitly (no
        corrective turn, no contract violation). This accessor formalizes what
        was previously a raw read of the private ``_exit_*`` attributes.
        """
        return (self._exit_reason, self._exit_summary, self._exit_block_reason)

    # ------------------------------------------------------------------
    # Diagnostics (permanent, always-on; see TASK-loop-diagnostic-instrumentation)
    # ------------------------------------------------------------------
    def _diag(self, event: str, **fields) -> None:
        """Emit one structured diagnostic record to the ``agent.diag`` logger.

        Observation only — never mutates session/prompt/cache state. The
        structured payload rides on the LogRecord as ``record.diag`` (via
        ``extra=``) so consumers assert on fields, not free text. project_id is
        not available at the loop layer (it lives in agent_manager); session_id
        / session_uuid identify the run.
        """
        payload = {
            "event": event,
            "session_id": getattr(self._session, "session_id", None),
            "session_uuid": getattr(self._session, "session_uuid", None),
            **fields,
        }
        # Render the payload into the message so it is greppable as a single
        # line in plain daemon.log text (`grep 'agent.diag'`), and ALSO attach
        # it structured on the record (`record.diag`) for programmatic/test use.
        _diag_logger.info(
            "agent.diag %s", json.dumps(payload, default=str, ensure_ascii=False),
            extra={"diag": payload},
        )

    def _note_exit(self, name: str, iteration: int) -> None:
        """Record the named loop-exit/continue path taken this iteration."""
        self._loop_exit_path = name
        self._diag("loop_exit", iteration=iteration, exit=name)

    def _budget_tripped(self):
        """Evaluate the pre-flight budget guard. Returns a tripped
        ``BudgetDecision`` (truthy → trip) or None.

        THE single dollar-budget enforcement point lives in
        ``agent_os/budget/guard.py``; this method only WIRES it into the loop's
        between-LLM-call chokepoint with LIVE config (resolved on every call via
        ``self._get_budget_config`` — never a session-start snapshot). No-op when
        no resolver or no ledger root was plumbed.

        Convertibility: if some spend can't be converted to the limit's currency
        (unknown FX pair), the convertible total is the spend (see the guard
        module docstring). Spend-QUERY failures (ledger I/O) fail OPEN in the
        guard with a logged warning; a malformed spend() return is a contract
        bug and RAISES out of the guard (deliberately loud — never silently
        masked as $0).
        """
        if self._get_budget_config is None or self._project_dir is None:
            return None
        try:
            cfg = self._get_budget_config()
        except Exception:  # noqa: BLE001 — a config read must not crash the loop
            logger.warning("budget config resolver raised; skipping guard",
                           exc_info=True)
            return None
        from agent_os.budget.guard import evaluate_budget
        fx_rates = (cfg or {}).get("fx_rates") if isinstance(cfg, dict) else None
        decision = evaluate_budget(
            self._project_dir, cfg, fx_rates=fx_rates,
        )
        return decision if decision.tripped else None

    def _trip_budget(self, decision, iteration: int) -> None:
        """Record a budget trip: set the first-class ``budget_blocked`` exit
        reason and append a structured timeline event (codes/numbers only).

        Uses the session's typed-event affordance (a structured ``append`` row
        with ``source="budget"`` + a ``payload`` dict) — the same shape the
        queue signals use — NOT an English ``append_system`` sentence, per the
        i18n rule (no display strings; the client renders the message). The
        ``budget_action`` ("pause" | "stop") is normalized by the guard. It is
        carried on ``_exit_block_reason`` so the dispatcher can route pause vs
        stop without re-reading config (``get_completion_state`` surfaces it).

        Also the SECOND trip-emission site (coordinator ruling, Task E): emits
        the ``budget.exhausted`` relay event through the same notify module and
        same marker as the crossing-append site, so the trip push fires at
        whichever happens first — crossing append or guard trip — exactly once
        per (project, window). This covers trips with NO new append (limit
        lowered below current spend; fresh run on an already-over project with
        an absent/stale marker).
        """
        self._exit_reason = "budget_blocked"
        self._exit_block_reason = decision.action
        self._session.append({
            "role": "system",
            "source": "budget",
            "event": "budget_blocked",
            "payload": {
                "action": decision.action,
                "window": decision.window,
                "spend": decision.spend,
                "limit": decision.limit,
                "currency": decision.currency,
            },
        })
        self._emit_trip_notify(decision)
        self._note_exit("budget_blocked", iteration)

    def _emit_trip_notify(self, decision) -> None:
        """Emit the trip relay event from the guard-trip site (dedup'd).

        Delegates to ``budget.notify.maybe_emit_trip`` with the tripped
        ``BudgetDecision``'s own numbers (no spend re-query). The shared marker
        key makes this a no-op when the crossing-append site already fired this
        window. Never raises. No-op when the emitter sink or project_dir was
        not plumbed (lightweight test loops).
        """
        if self._on_budget_event is None or self._project_dir is None:
            return
        try:
            # anchor_ts only matters for the "total" window's identity; read it
            # from the live config (same resolver the guard used).
            anchor_ts = None
            if self._get_budget_config is not None:
                try:
                    cfg = self._get_budget_config()
                    if isinstance(cfg, dict):
                        anchor_ts = cfg.get("budget_anchor_ts")
                except Exception as e:  # noqa: BLE001 — config read must not block emit
                    # Degrades the "total" window identity to epoch-keyed, which
                    # can re-fire one duplicate trip push — observable, not silent.
                    logger.warning(
                        "budget notify: anchor_ts read failed; using None: %s",
                        e, exc_info=True,
                    )
            from agent_os.budget.notify import maybe_emit_trip
            maybe_emit_trip(
                self._project_dir,
                window=decision.window,
                spend=decision.spend,
                limit=decision.limit,
                currency=decision.currency,
                anchor_ts=anchor_ts,
                emit=self._on_budget_event,
            )
        except Exception:  # noqa: BLE001 — never break the loop on notify errors
            logger.warning(
                "budget trip notify failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )

    def _emit_ledger_event(self, active_provider, usage) -> None:
        """Append one token-ledger event for a management LLM response.

        ``active_provider`` is the provider that ACTUALLY served this response
        (post-rotation), so its ``provider`` / ``model`` / ``sdk`` give correct
        attribution. Usage is normalized into the four disjoint token fields via
        the ONLY normalization entry point (``ProviderSemantics.from_sdk`` +
        ``normalize_usage``).

        No-ops when ``project_dir`` was not plumbed. Never raises: this is wrapped
        in its own guard AND ``append_event`` is itself resilient, so a ledger
        failure cannot crash or alter the agent loop. The legacy budget path is
        completely independent of this method.
        """
        if self._project_dir is None:
            return
        try:
            # Imported lazily: budget.normalize -> providers.types pulls in the
            # agent package __init__ (which eagerly imports this module), so a
            # module-level import here would be a circular import.
            from agent_os.budget.normalize import (
                ProviderSemantics,
                normalize_usage,
            )

            sdk = getattr(active_provider, "sdk", None)
            semantics = ProviderSemantics.from_sdk(sdk)
            normalized = normalize_usage(usage, semantics)
            append_event(
                self._project_dir,
                LedgerEvent(
                    session_id=getattr(self._session, "session_id", "") or "",
                    source=SOURCE_MANAGEMENT,
                    provider=getattr(active_provider, "provider", "unknown"),
                    model=getattr(active_provider, "model", "unknown"),
                    usage=normalized,
                ),
            )
        except Exception:  # noqa: BLE001 — never break the loop on ledger errors
            logger.warning(
                "Token-ledger emission failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )
            return
        # Ledger-append is the CAUSAL moment a spend crosses 80%/100% of the
        # window limit (Budget Piece 2 — Task E). Check the threshold/trip
        # crossings here — separately guarded so a notify failure can never
        # affect the ledger append above nor the loop. No-op when no budget
        # config resolver or no emitter sink was plumbed.
        self._check_budget_notify()

    def _check_budget_notify(self) -> None:
        """Run the budget threshold/trip emitter after a ledger append, then
        feed the debounced ``budget.spend_updated`` broadcast (Budget Piece 3).

        Reads the LIVE budget config (same resolver the pre-flight guard uses)
        and delegates to ``budget.notify.maybe_emit_budget_event``, which emits
        AT MOST ONCE per (project, window) for the 80% threshold and the trip.
        It now also RETURNS the post-append management-spend snapshot it computed
        (``window``/``spend``/``limit``/``currency`` in ``budget_currency``) so
        the spend_updated broadcast reuses it — ZERO additional spend() query per
        append when a limit is set. When no limit is configured the notify path
        runs no query (returns None); the corner still shows spend, so we run ONE
        management-only query here for the broadcast. A spend/window-math failure
        skips the broadcast silently (already logged in the notify module).

        Never raises. No-op when ``on_budget_event``/``get_budget_config``/
        ``project_dir`` were not plumbed (e.g. lightweight test loops).
        """
        if (self._on_budget_event is None
                or self._get_budget_config is None
                or self._project_dir is None):
            return
        try:
            cfg = self._get_budget_config()
            fx_rates = (cfg or {}).get("fx_rates") if isinstance(cfg, dict) else None
            from agent_os.budget.notify import maybe_emit_budget_event
            snapshot = maybe_emit_budget_event(
                self._project_dir,
                cfg,
                emit=self._on_budget_event,
                fx_rates=fx_rates,
            )
        except Exception:  # noqa: BLE001 — never break the loop on notify errors
            logger.warning(
                "budget threshold/trip notify failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )
            return
        # Feed the debounced spend_updated broadcast. Wrapped in its own guard so
        # a broadcast failure can never affect the notify check above nor the
        # ledger append nor the loop.
        try:
            if snapshot is None:
                # No limit configured (or a notify-side failure) → run one
                # management-only spend query so the corner still shows spend.
                snapshot = self._compute_spend_snapshot(cfg, fx_rates)
            if snapshot is not None:
                self._spend_broadcaster.submit(
                    window=snapshot["window"],
                    spend=snapshot["spend"],
                    limit=snapshot["limit"],
                    currency=snapshot["currency"],
                )
        except Exception:  # noqa: BLE001 — never break the loop on broadcast errors
            logger.warning(
                "budget spend_updated broadcast failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )

    def _compute_spend_snapshot(self, cfg, fx_rates) -> "dict | None":
        """Compute the post-append management-spend snapshot for the
        ``budget.spend_updated`` broadcast in the NO-LIMIT case only.

        The corner shows spend even without a limit, so when the notify check
        ran no spend query (``maybe_emit_budget_event`` returned None because no
        limit is set) we run exactly ONE management-only query here, converted
        into ``budget_currency`` (consistent with the meter). Returns
        ``{"window", "spend", "limit", "currency"}`` (``limit`` always None on
        this path) or None when spend cannot be computed (guard-style failure →
        skip the broadcast silently with a warning).
        """
        cfg = cfg or {}
        window = cfg.get("budget_period") or "daily"
        currency = cfg.get("budget_currency") or "USD"
        anchor_ts = cfg.get("budget_anchor_ts")
        try:
            from agent_os.budget.ledger import spend as _spend
            result = _spend(
                self._project_dir,
                window,
                target_currency=currency,
                fx_rates=fx_rates or {},
                anchor_ts=anchor_ts,
                # Management-only, consistent with the limit-governed meter
                # (the limit only governs management spend; sub-agent display
                # is a separate block).
                sources=["management"],
            )
            spend_amount = float(result["converted_total"]["amount"])
        except Exception:  # noqa: BLE001 — guard-style: skip the broadcast
            logger.warning(
                "budget spend_updated: spend query failed (session=%s); "
                "skipping broadcast.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )
            return None
        return {
            "window": window,
            "spend": spend_amount,
            "limit": None,
            "currency": currency,
        }

    async def _stream_response(self, context, tool_schemas) -> LLMResponse:
        """Stream LLM response from the primary provider.

        Convenience wrapper preserved for tests that monkey-patch this
        method. Delegates to the unified internal helper.
        """
        return await self._stream_response_internal(
            self._provider, context, tool_schemas,
        )

    async def _stream_response_with(self, provider, context, tool_schemas) -> LLMResponse:
        """Stream LLM response using a specific provider (rotation).

        Routes the primary provider through _stream_response so any test
        monkey-patches on that method continue to fire; otherwise goes
        directly to the unified internal helper. Both paths share the
        same accumulator-publishing semantics so cancel_turn() can rely
        on a consistent contract regardless of which provider is active.
        """
        if provider is self._provider:
            return await self._stream_response(context, tool_schemas)
        return await self._stream_response_internal(
            provider, context, tool_schemas,
        )

    async def _stream_response_internal(self, provider, context, tool_schemas) -> LLMResponse:
        """Single source-of-truth streaming implementation.

        Publishes the accumulator on self._stream_accumulator before any
        await so cancel_turn() can inspect partial state. The async
        iteration is wrapped in a try/finally so CancelledError reaches
        the underlying generator's cleanup path (its httpx context).
        """
        accumulator = StreamAccumulator()
        self._stream_accumulator = accumulator
        try:
            async for chunk in provider.stream(context, tools=tool_schemas):
                accumulator.add(chunk)
                self._session.notify_stream(chunk)
            return accumulator.finalize()
        finally:
            # Do not clear self._stream_accumulator here — cancel_turn()
            # may still need to read it on its way out. The next turn
            # will reassign it at the top of the next call.
            pass

    @staticmethod
    def _rotate_provider(providers, current_idx, cooldowns):
        """Find next available provider not in cooldown. Returns index or None."""
        now = time.monotonic()
        n = len(providers)
        for offset in range(1, n):
            candidate = (current_idx + offset) % n
            if now >= cooldowns.get(candidate, 0):
                return candidate
        return None

    async def run(self, initial_message: str | None = None,
                  initial_nonce: str | None = None) -> None:
        """Main loop entry point."""
        if self._running:
            # Re-entrancy guard: a second concurrent run() on the same loop would
            # write two interleaved turns into one session. Reject rather than
            # silently start a duplicate loop. (Defense-in-depth alongside the
            # single-active-slot discipline.)
            raise RuntimeError(
                f"AgentLoop.run() re-entered while already running "
                f"(session {getattr(self._session, 'session_id', '?')})"
            )
        self._running = True
        # Reset queue exit-reason at the start of every run so the dispatcher
        # always reads a fresh value scoped to this run.
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        # Capture our own task handle so terminate() can cancel us.
        self._task = asyncio.current_task()
        # Every run() invocation begins a fresh turn boundary: clear the
        # sticky cancel-marker flag so a Stop click on this run writes a
        # new marker rather than no-opping on the previous run's flag.
        # Hot-resume paths (_on_loop_done → _start_loop) call run() with
        # no initial_message, so this reset must fire unconditionally —
        # not just when initial_message is provided.
        self._cancellation_marker_appended_this_turn = False
        # Diagnostics: track exit path + appended-row delta for this run.
        self._loop_exit_path = None
        rows_at_run_start = None  # set before the while; guarded in finally
        try:
            # Resolve pending if not resuming from approval pause
            if self._session._paused_for_approval:
                self._session._paused_for_approval = False  # reset here, not in resume()
            else:
                self._session.resolve_pending_tool_calls()

            # Append initial user message
            if initial_message is not None:
                msg: dict = {
                    "role": "user",
                    "content": initial_message,
                    "source": "user",
                }
                if initial_nonce:
                    msg["nonce"] = initial_nonce
                self._session.append(msg)

            # Reset per-run state on all tools (e.g. send counters)
            self._tool_registry.reset_run_state()

            # Reset loop guards
            iteration = 0
            cumulative_tokens = 0
            action_hashes: deque = deque(maxlen=15)
            consecutive_overflows = 0
            llm_retries = 0

            # Fallback provider rotation state
            _all_providers = [self._provider] + list(self._fallback_providers)
            _current_idx = 0
            _cooldowns: dict[int, float] = {}
            _COOLDOWN_SECONDS = 60.0
            _retries_on_current = 0

            # Ping-pong detection state
            _pair_hashes: deque = deque(maxlen=20)
            _prev_action_hash: str | None = None

            # Circuit breaker state (reset per run / new user message)
            error_tracker: dict[str, dict] = {}  # tool_name -> {"error": str, "count": int}
            blocked_tools: set[str] = set()

            # Diagnostics: row count at run start (after the initial user
            # message is appended above) — the run_terminal disposition compares
            # against this to detect a turn that appended nothing.
            rows_at_run_start = len(self._session._messages)

            while True:
                # Drain queued user messages before preparing context
                queued = self._session.pop_queued_messages()
                if queued:
                    # New user input ⇒ a brand new "turn" begins. Reset the
                    # cancellation marker flag so subsequent cancel_turn()
                    # calls write a fresh marker.
                    self._cancellation_marker_appended_this_turn = False
                    for q_item in queued:
                        if isinstance(q_item, tuple):
                            q_content, q_nonce = q_item
                        else:
                            q_content, q_nonce = q_item, None
                        q_msg: dict = {
                            "role": "user",
                            "content": q_content,
                            "source": "user",
                        }
                        if q_nonce:
                            q_msg["nonce"] = q_nonce
                        self._session.append(q_msg)
                    # Reset circuit breaker on new user messages
                    error_tracker.clear()
                    blocked_tools.clear()

                # Check iteration cap (0 = unlimited)
                if self._max_iterations and iteration >= self._max_iterations:
                    self._session.append_system(
                        "Maximum iteration limit reached. Save your current state and stop."
                    )
                    self._note_exit("max_iter", iteration)
                    break

                # Check token budget (PURE non-budget safety-net cap — the
                # dollar→token derivation was cut in Budget Piece 2).
                if cumulative_tokens >= self._token_budget:
                    self._session.append_system(
                        "Token budget exceeded. Save your current state and stop."
                    )
                    self._note_exit("token_budget", iteration)
                    break

                # --- Pre-flight budget guard (Budget Piece 2) ---------------
                # THE single dollar-budget enforcement point. Evaluated BETWEEN
                # LLM calls (here at the top of every tool-loop iteration), with
                # LIVE config. On a trip the loop exits with the first-class
                # ``budget_blocked`` reason; the dispatcher routes it explicitly.
                # Sits before the LLM stream so a tripped budget spends nothing
                # more. Never interrupts an in-flight tool execution.
                _budget_decision = self._budget_tripped()
                if _budget_decision is not None:
                    self._trip_budget(_budget_decision, iteration)
                    break

                iteration += 1

                # --- State refresh: turn-count trigger ---
                # Track current iteration for agent-decided trigger callback.
                self._current_iteration = iteration
                # Increment the per-refresh turn counter and expose it to the
                # context_manager so the agent sees accurate metadata.
                self._turns_since_last_update += 1
                if hasattr(self._context_manager, "_turns_since_last_update"):
                    self._context_manager._turns_since_last_update = self._turns_since_last_update

                # Fire turn-count trigger when cooldown has elapsed — via the
                # spec-013 scheduler (background, single-flight, debounced).
                # Token-pressure trigger is handled separately (before compaction).
                if (
                    self._turns_since_last_update >= COOLDOWN_TURNS
                    and self._on_session_end_refresh is not None
                ):
                    # While debounced/in-flight the counter stays >= COOLDOWN_TURNS
                    # and this re-enters EVERY turn — log only on an actual spawn
                    # (reviewer finding D: avoids per-turn log spam for up to
                    # REFRESH_DEBOUNCE_S). Capture the count before spawn resets it.
                    _turns = self._turns_since_last_update
                    if self.schedule_checkpoint("turn_count").startswith(
                        "Consolidation scheduled"
                    ):
                        logger.info(
                            "State refresh: turn-count trigger at iteration %d "
                            "(%d turns since last update)",
                            iteration, _turns,
                        )
                else:
                    self._maybe_consume_dirty()

                # Reset per-iteration cancellation flags. The
                # cancellation-marker flag is sticky across cancelled
                # iterations: it's only reset when a new user message has
                # been drained (in the queue-drain block above) — so a
                # cancel_turn() called twice with no intervening user
                # message is idempotent.
                self._turn_cancelled = False

                # Diagnostics: row count at the start of this iteration's work,
                # to classify the bottom-of-loop continue as tool_continue
                # (rows grew) vs fell_through (nothing appended).
                rows_at_iter_start = len(self._session._messages)

                # Prepare context
                context = self._context_manager.prepare()

                # Get tool schemas
                tool_schemas = self._tool_registry.schemas()

                # Stream LLM response (using active provider from rotation).
                # Wrap in asyncio.Task so cancel_turn() can interrupt it.
                active_provider = _all_providers[_current_idx]
                self._inflight_stream = asyncio.create_task(
                    self._stream_response_with(
                        active_provider, context, tool_schemas,
                    ),
                    name=f"stream-{self._session.session_uuid}-{iteration}",
                )
                try:
                    response = await self._inflight_stream
                    consecutive_overflows = 0
                    llm_retries = 0
                    _retries_on_current = 0
                except asyncio.CancelledError:
                    # cancel_turn() invoked while streaming. Marker + cost
                    # debit handled by cancel_turn() before propagating.
                    self._inflight_stream = None
                    if self._session.is_stopped():
                        # terminate() also fired — let outer cleanup run.
                        raise
                    # Bare cancel_turn: exit the while loop. The finally
                    # block + _on_loop_done drain any queued messages and
                    # hot-resume a fresh loop task if needed, preserving
                    # sub-agents / browser pages / sandbox state.
                    self._note_exit("cancel", iteration)
                    break
                except ContextOverflowError:
                    self._inflight_stream = None
                    consecutive_overflows += 1
                    if consecutive_overflows >= 3:
                        self._session.append_system(
                            "Context overflow cannot be resolved. Save state and stop."
                        )
                        self._note_exit("context_overflow", iteration)
                        break
                    self._context_manager.reduce_window(0.5)
                    continue
                except LLMError as e:
                    self._inflight_stream = None
                    category = e.category

                    # Abort: non-recoverable errors (401, 403, 400)
                    if category == ErrorCategory.ABORT:
                        self._llm_failed = True
                        self._session.append_system(
                            f"LLM error (non-recoverable): {e.message}. Stopping."
                        )
                        self._note_exit("llm_error", iteration)
                        break

                    # Transient: rotate to next fallback provider
                    if category == ErrorCategory.TRANSIENT:
                        _cooldowns[_current_idx] = time.monotonic() + _COOLDOWN_SECONDS
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            new_provider = _all_providers[rotated]
                            model_name = new_provider.model
                            logger.info("Rotating to fallback model: %s", model_name)
                            self._session.append_meta(
                                "model_swap",
                                provider=getattr(new_provider, "provider", "unknown"),
                                model=model_name,
                                sdk=getattr(new_provider, "sdk", "unknown"),
                                reason=f"transient: {e.message}",
                            )
                            self._session.append_system(
                                f"Primary model unavailable ({e.message}). "
                                f"Switching to {model_name}."
                            )
                            continue

                    # Retry same provider (500, or transient with no fallback)
                    _retries_on_current += 1
                    llm_retries += 1
                    if _retries_on_current >= 3:
                        # Exhausted retries on current — try rotating
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            new_provider = _all_providers[rotated]
                            self._session.append_meta(
                                "model_swap",
                                provider=getattr(new_provider, "provider", "unknown"),
                                model=new_provider.model,
                                sdk=getattr(new_provider, "sdk", "unknown"),
                                reason=f"retries_exhausted: {e.message}",
                            )
                            continue
                        # All providers exhausted
                        self._llm_failed = True
                        self._session.append_system(
                            f"LLM error after {llm_retries} retries: "
                            f"{e.message}. Stopping."
                        )
                        self._note_exit("llm_error", iteration)
                        break
                    await asyncio.sleep(2 ** _retries_on_current)
                    continue

                # Successful stream completion — clear inflight task handle.
                self._inflight_stream = None

                # Diagnostics (Point 1): per-iteration finalized response shape,
                # logged BEFORE any branch decision. `final_usage_chunk_seen`
                # is the explicit named signal for the "no CACHE_AUDIT" anomaly.
                self._diag(
                    "response_shape",
                    iteration=iteration,
                    text_len=len(response.text or ""),
                    tool_call_names=[
                        normalize_tool_call(tc)["name"]
                        for tc in (response.tool_calls or [])
                    ],
                    usage_present=bool(
                        response.usage
                        and (response.usage.input_tokens or response.usage.output_tokens)
                    ),
                    final_usage_chunk_seen=getattr(
                        response, "final_usage_chunk_seen", False
                    ),
                )

                # Track token usage
                if response.usage:
                    cumulative_tokens += response.usage.input_tokens + response.usage.output_tokens
                    # Per-session cache-hit aggregation (Part 4D)
                    self._prompt_input_tokens_total += response.usage.input_tokens
                    self._cache_read_tokens_total += response.usage.cache_read_tokens
                    self._llm_call_count += 1

                # Token ledger (Budget Piece 1): append one disjoint-token event
                # per management LLM response. Attribution uses `active_provider`
                # — the post-rotation provider that ACTUALLY served THIS response
                # (re-read at the top of every iteration from _all_providers[
                # _current_idx]) — never the project's configured model. The
                # ledger is the SINGLE source of recorded spend; the pre-flight
                # guard reads it. Budget Piece 2 deleted the legacy in-loop cost
                # accumulator (_budget_spent_usd / on_cost_update / _estimate_cost_usd)
                # and the post-hoc dollar pause/stop branch that used to live here.
                if response.usage:
                    self._emit_ledger_event(active_provider, response.usage)

                # Text-only response: append and exit
                if not response.has_tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.text,
                        "source": "management",
                    }
                    # Carry reasoning through. finalize() writes
                    # reasoning_content onto raw_message only when reasoning
                    # exists, so mirror the tool-call persist path: include it
                    # when present, never write an empty/None field. Dropping it
                    # here is the silent-vanish bug for reasoned text-only turns.
                    reasoning = (response.raw_message or {}).get("reasoning_content")
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning
                    self._session.append(assistant_msg)
                    truncate_consumed_tool_results(
                        self._session, response.text, iteration,
                    )
                    self._note_exit("text_complete", iteration)
                    break

                # Extract status
                status = response.status_text
                if not status and response.tool_calls:
                    first_tc = normalize_tool_call(response.tool_calls[0])
                    status = f"Using {first_tc['name']}"

                # Append raw assistant message
                raw_msg = dict(response.raw_message)
                raw_msg["source"] = "management"
                if status:
                    raw_msg["_status"] = status
                self._session.append(raw_msg)

                # Truncate tool results consumed by this LLM response
                truncate_consumed_tool_results(
                    self._session, response.text, iteration,
                )

                # ── Queue signal short-circuit ────────────────────────────
                # If the response contains mark_task_complete or
                # mark_task_blocked, exit the loop immediately with the
                # corresponding _exit_reason. Other tool calls in the same
                # response are discarded (the finally block's
                # resolve_pending_tool_calls will write CANCELLED results
                # for them so the JSONL stays valid).
                _queue_signal_exit = False
                for _tc_raw in response.tool_calls:
                    _tc = normalize_tool_call(_tc_raw)
                    if _tc["name"] == "mark_task_complete":
                        summary = (_tc["arguments"] or {}).get("summary", "")
                        if isinstance(summary, str):
                            summary = summary.strip()
                        else:
                            summary = ""
                        self._exit_reason = "complete"
                        self._exit_summary = summary
                        self._session.append({
                            "role": "system",
                            "content": f"Task completed: {summary}" if summary else "Task completed.",
                            "source": "queue_signal",
                            "signal": "complete",
                            "payload": {"summary": summary},
                        })
                        self._note_exit("task_complete", iteration)
                        _queue_signal_exit = True
                        break
                    if _tc["name"] == "mark_task_blocked":
                        reason = (_tc["arguments"] or {}).get("reason", "")
                        if isinstance(reason, str):
                            reason = reason.strip()
                        else:
                            reason = ""
                        self._exit_reason = "blocked"
                        self._exit_block_reason = reason
                        self._session.append({
                            "role": "system",
                            "content": f"Task blocked: {reason}" if reason else "Task blocked.",
                            "source": "queue_signal",
                            "signal": "blocked",
                            "payload": {"reason": reason},
                        })
                        self._note_exit("task_blocked", iteration)
                        _queue_signal_exit = True
                        break
                if _queue_signal_exit:
                    break

                # Normalize and execute tool calls
                tool_calls = [normalize_tool_call(tc) for tc in response.tool_calls]
                exit_outer = False
                intercepted_tc_id = None

                for tc in tool_calls:
                    # Cancellation check: if cancel_turn() fired between or
                    # during tool execution, skip remaining tool calls.
                    if self._turn_cancelled:
                        break

                    # Post-normalization validation
                    if not tc["id"]:
                        tc["id"] = uuid4().hex
                        logger.warning("Tool call missing ID, generated: %s", tc["id"])

                    if not tc["name"]:
                        self._session.append_tool_result(
                            tc["id"],
                            "Error: LLM returned tool call with no function name",
                        )
                        continue

                    # Interceptor check (fail-closed)
                    if self._interceptor is not None:
                        try:
                            should_intercept = self._interceptor.should_intercept(tc)
                        except Exception:
                            should_intercept = True

                        if should_intercept:
                            # Phase 3: under queue-running state, an
                            # approval-required tool call is a block signal,
                            # not a pause. Skip on_intercept (no UI card —
                            # the dispatcher records this as the block reason)
                            # and exit the loop. The dispatcher rotates the
                            # session and proceeds to the next queued item.
                            if self._queue_state == "running":
                                _block_reason = (
                                    f"approval required for tool '{tc['name']}'"
                                )
                                _args_str = ""
                                try:
                                    _args_str = json.dumps(tc.get("arguments") or {})[:200]
                                except (TypeError, ValueError):
                                    _args_str = ""
                                if _args_str and _args_str not in ("{}", '""'):
                                    _block_reason += f" (args: {_args_str})"
                                self._exit_reason = "blocked"
                                self._exit_block_reason = _block_reason
                                self._session.append({
                                    "role": "system",
                                    "content": f"Task blocked: {_block_reason}",
                                    "source": "queue_signal",
                                    "signal": "blocked",
                                    "payload": {
                                        "reason": _block_reason,
                                        "tool": tc["name"],
                                    },
                                })
                                exit_outer = True
                                break

                            reasoning = response.text if response.text and response.text.strip() else None
                            try:
                                self._interceptor.on_intercept(
                                    tc, self._session.recent_activity(),
                                    reasoning=reasoning,
                                )
                            except Exception as e:
                                # on_intercept failed (e.g. WS broadcast hook
                                # crashed). Without this guard the exception
                                # would escape run(), leave the orphan tool
                                # call as CANCELLED, and end the loop task in
                                # error state — leaving the user with no
                                # approval card and no recovery path.
                                # Treat the failure as a denial: append an
                                # error tool result so the agent can react,
                                # and continue the dispatch loop.
                                logger.exception(
                                    "Interceptor.on_intercept raised for "
                                    "tool_call_id=%s tool=%s",
                                    tc.get("id"), tc.get("name"),
                                )
                                self._session.append_tool_result(
                                    tc["id"],
                                    f"Error requesting approval: {e}. "
                                    "The tool was not executed.",
                                )
                                continue
                            intercepted_tc_id = tc["id"]
                            self._session._paused_for_approval = True
                            self._session.pause()
                            self._note_exit("approval_pause", iteration)
                            exit_outer = True
                            break

                    # Circuit breaker: block tool if it has repeated errors
                    if tc["name"] in blocked_tools:
                        blocked_msg = (
                            f"[SYSTEM] The {tc['name']} tool is currently unavailable "
                            f"(repeated error: {error_tracker[tc['name']]['error']}). "
                            f"Work with the tools that are available or ask the user for help."
                        )
                        self._session.append_tool_result(tc["id"], blocked_msg)
                        continue

                    # Execute tool (async-aware)
                    try:
                        if self._tool_registry.is_async(tc["name"]):
                            result: ToolResult = await self._tool_registry.execute_async(
                                tc["name"],
                                tc["arguments"],
                            )
                        else:
                            result: ToolResult = await asyncio.to_thread(
                                self._tool_registry.execute,
                                tc["name"],
                                tc["arguments"],
                            )
                        # Pre-filter tool result before storage
                        filtered_content = dispatch_prefilter(
                            tc["name"], tc["arguments"], result.content,
                        )
                        result = ToolResult(
                            content=filtered_content, meta=result.meta,
                        )

                        self._session.append_tool_result(
                            tc["id"], result.content, meta=result.meta,
                            context_limit=self._context_manager.model_context_limit,
                        )
                        tool_content = result.content
                        # Store result for observation-aware hash
                        _exec_result = result

                        # yield_turn: a dispatch tool that delivered a task to a
                        # sub-agent ends the management turn immediately so the
                        # LLM cannot busy-poll it (which trips the ping-pong
                        # guard). The result is already appended above
                        # (tool_call_id paired), so resolve_pending_tool_calls
                        # will not CANCEL it. The sub-agent's result is pushed
                        # back later via on_completed and the loop restarts.
                        if result.meta and result.meta.get("yield_turn"):
                            logger.info("yield_turn: dispatch tool ended the management turn")
                            self._note_exit("yield_turn", iteration)
                            exit_outer = True
                            break

                        # Pause if credential was requested (wait for user input)
                        if result.meta and result.meta.get("credential_request"):
                            self._session.pause()
                            self._note_exit("credential_request", iteration)
                            exit_outer = True
                            break
                    except Exception as e:
                        tool_content = f"Error executing tool: {e}"
                        self._session.append_tool_result(
                            tc["id"], tool_content
                        )
                        _exec_result = None

                    # Observation-aware repetition detection (post-execution)
                    # Browser tools: skip hash detection (handled by advisory counter in browser tool)
                    if tc["name"] != "browser":
                        result_prefix = str(tool_content)[:500] if tool_content else ""
                        action_hash = hashlib.md5(
                            (tc["name"] + str(tc["arguments"]) + result_prefix).encode()
                        ).hexdigest()

                        action_hashes.append(action_hash)
                        if list(action_hashes).count(action_hash) >= 5:
                            self._session.resolve_pending_tool_calls()
                            self._session.append_system(
                                "Repetitive action detected. Save your state and try a different approach."
                            )
                            self._note_exit("repetition", iteration)
                            exit_outer = True
                            break

                        # Ping-pong / cycle detection: track consecutive pairs
                        if _prev_action_hash is not None:
                            pair = (_prev_action_hash, action_hash)
                            _pair_hashes.append(pair)
                            if list(_pair_hashes).count(pair) >= 3:
                                self._session.resolve_pending_tool_calls()
                                self._session.append_system(
                                    f"Ping-pong pattern detected: you have been "
                                    f"alternating between the same tool calls for "
                                    f"multiple iterations with no progress. "
                                    f"Stop and explain what you are trying to "
                                    f"accomplish, then try a fundamentally "
                                    f"different approach."
                                )
                                self._note_exit("ping_pong", iteration)
                                exit_outer = True
                                break
                        _prev_action_hash = action_hash

                    # Circuit breaker: trip only on genuine tool/infra failures,
                    # never on recoverable input errors the model can correct
                    # (e.g. edit old_text-not-found — the agent should re-read and
                    # retry, not lose the tool). Recoverability is read from the
                    # structured ToolResult marker, not by string-sniffing; a
                    # raised exception (_exec_result is None) is never recoverable.
                    # Identity is compared on the FULL error string so two
                    # genuinely different failures are not collapsed.
                    is_error = isinstance(tool_content, str) and tool_content.startswith("Error")
                    is_recoverable = bool(
                        _exec_result is not None
                        and _exec_result.meta
                        and _exec_result.meta.get("recoverable")
                    )
                    if is_error and not is_recoverable:
                        prev = error_tracker.get(tc["name"])
                        if prev and prev["error"] == tool_content:
                            prev["count"] += 1
                        else:
                            error_tracker[tc["name"]] = {"error": tool_content, "count": 1}
                        if error_tracker[tc["name"]]["count"] >= 2:
                            blocked_tools.add(tc["name"])
                            self._session.append_system(
                                f"[SYSTEM] The {tc['name']} tool is currently unavailable "
                                f"(repeated error: {tool_content}). "
                                f"Work with the tools that are available or ask the user for help."
                            )
                    else:
                        # Successful execution OR a recoverable input error clears
                        # error tracking for this tool.
                        error_tracker.pop(tc["name"], None)

                # Drain deferred messages (lifecycle notifications)
                for msg in self._session.pop_deferred_messages():
                    self._session.append(msg)

                # Cancellation landed mid-tool-loop (cancel_turn fired between
                # tool calls or during a tool's execution). Append marker if
                # not already done, resolve any unstarted tool calls, then
                # exit the loop. The finally block + _on_loop_done drain any
                # queued messages and hot-resume a fresh loop task if needed.
                # terminate() also sets is_stopped(); the cancel-only path
                # exits via break here without needing the is_stopped check
                # below.
                if self._turn_cancelled:
                    if not self._cancellation_marker_appended_this_turn:
                        self._persist_cancellation_marker()
                    # Resolve any tool calls that never got a result — these
                    # are the ones we skipped via the per-tool break above.
                    self._session.resolve_pending_tool_calls()
                    self._note_exit("cancel", iteration)
                    break

                # Resolve unprocessed tool calls from this batch on intercept-break
                if exit_outer and intercepted_tc_id is not None:
                    for tc in tool_calls:
                        tc_id = tc["id"]
                        if tc_id in self._session.pending_tool_calls and tc_id != intercepted_tc_id:
                            self._session.append_tool_result(
                                tc_id,
                                "CANCELLED: Earlier tool call requires approval. "
                                "This tool call will be retried after approval.",
                            )

                # Check if we need to exit the outer loop
                if exit_outer:
                    break

                # Check stopped
                if self._session.is_stopped():
                    self._session.resolve_pending_tool_calls()
                    self._note_exit("stopped", iteration)
                    break

                # Check paused (redundant with exit_outer but kept for safety)
                if self._session.is_paused():
                    self._note_exit("paused", iteration)
                    break

                # Check compaction
                if self._context_manager.should_compact():
                    from agent_os.agent import compaction as compaction_mod

                    # Pre-flight budget guard (Budget Piece 2): the
                    # pre-compaction flush is a real management-LLM call that
                    # spends. Evaluate the SAME single guard before it fires so a
                    # tripped budget stops here rather than buying one more flush
                    # response. On trip, exit with ``budget_blocked`` without
                    # compacting (compaction is a no-spend local op, but the loop
                    # is ending anyway — the next resume re-evaluates).
                    _flush_budget = self._budget_tripped()
                    if _flush_budget is not None:
                        self._trip_budget(_flush_budget, iteration)
                        break

                    # Token-pressure trigger: fire refresh BEFORE compaction.
                    # Exempt from cooldown — data preservation trumps redundancy.
                    if self._on_session_end_refresh is not None:
                        # Spec 013: never two consolidations at once — finish
                        # any background pass before the blocking pre-compaction one.
                        await self.drain_refresh()
                        logger.info(
                            "State refresh: token-pressure trigger at iteration %d "
                            "(context at %.0f%%)",
                            iteration,
                            self._context_manager.usage_percentage * 100,
                        )
                        self._refresh_task = asyncio.create_task(
                            self._run_refresh("token_pressure", iteration),
                            name=f"refresh-tp-{self._session.session_uuid}-{iteration}",
                        )
                        try:
                            await self._refresh_task
                        except asyncio.CancelledError:
                            logger.info(
                                "Token-pressure refresh cancelled at iteration %d",
                                iteration,
                            )
                            if self._session.is_stopped():
                                raise
                        finally:
                            self._refresh_task = None

                    # Pre-compaction memory flush: give agent one turn to save state
                    try:
                        self._session.append_system(compaction_mod.MEMORY_FLUSH_PROMPT)
                        flush_llm = self._utility_provider or self._provider
                        flush_context = self._context_manager.prepare()
                        flush_response = await flush_llm.complete(flush_context)
                        # Token ledger (Budget Piece 1): the pre-compaction
                        # flush is a real management-LLM response with real
                        # spend. It is served by flush_llm (the utility
                        # provider when configured, else the primary) — NOT
                        # the rotated active_provider — so attribute to
                        # flush_llm. Purely additive: the legacy budget path
                        # deliberately does not count the flush, and that
                        # stays unchanged.
                        if getattr(flush_response, "usage", None):
                            self._emit_ledger_event(
                                flush_llm, flush_response.usage,
                            )
                        flush_text = flush_response.text or ""
                        if not compaction_mod.is_silent_response(flush_text):
                            # Agent wants to save state — execute any tool calls.
                            # FO-1 invariant (audit 2026-06-03): this flush executor is
                            # intentionally NOT interceptor-gated, and is scoped to project-state
                            # persistence only (MEMORY_FLUSH_PROMPT directs PROJECT_STATE.md writes;
                            # the flush completion is given no tool schemas). Do not repurpose it to
                            # run arbitrary approval-required tools without routing through the
                            # interceptor.
                            if flush_response.tool_calls:
                                raw_msg = dict(flush_response.raw_message)
                                raw_msg["source"] = "management"
                                self._session.append(raw_msg)
                                for tc_raw in flush_response.tool_calls:
                                    try:
                                        tc = normalize_tool_call(tc_raw)
                                        if not tc["id"]:
                                            tc["id"] = uuid4().hex
                                        if not tc["name"]:
                                            self._session.append_tool_result(
                                                tc["id"],
                                                "Error: tool call has no function name",
                                            )
                                            continue
                                        if self._tool_registry.is_async(tc["name"]):
                                            result = await self._tool_registry.execute_async(
                                                tc["name"], tc["arguments"],
                                            )
                                        else:
                                            result = await asyncio.to_thread(
                                                self._tool_registry.execute,
                                                tc["name"], tc["arguments"],
                                            )
                                        self._session.append_tool_result(tc["id"], result.content)
                                    except Exception as e:
                                        tc_id = tc.get("id") if isinstance(tc, dict) else tc_raw.get("id", uuid4().hex)
                                        self._session.append_tool_result(
                                            tc_id, f"Error: {e}",
                                        )
                            else:
                                flush_msg = {
                                    "role": "assistant",
                                    "content": flush_text,
                                    "source": "management",
                                }
                                # Carry reasoning through (same omission as the
                                # main text-only persist). Reasoning lives on the
                                # flush response's raw_message when present.
                                _reasoning = (
                                    flush_response.raw_message or {}
                                ).get("reasoning_content")
                                if _reasoning:
                                    flush_msg["reasoning_content"] = _reasoning
                                self._session.append(flush_msg)
                    except (ContextOverflowError, LLMError):
                        # Context too full for flush — skip and proceed to compaction
                        pass
                    except Exception:
                        # Any other error during flush — don't crash, just compact
                        logger.warning("Pre-compaction flush failed, proceeding to compaction")

                    await compaction_mod.run(
                        self._session,
                        self._provider,
                        utility_provider=self._utility_provider,
                    )

                    # Post-compaction reorientation
                    workspace = getattr(
                        getattr(self._context_manager, "_base_ctx", None),
                        "workspace", None,
                    )
                    if workspace:
                        compaction_mod.inject_reorientation(workspace, self._session)

                # Bottom of the loop body: no named branch broke out, so this
                # iteration is continuing to the next. Classify as tool_continue
                # when rows grew this iteration, else fell_through — the
                # catch-all for a path that produced no appended row (the
                # original-bug shape), captured even though no named branch fired.
                _appended_this_iter = len(self._session._messages) - rows_at_iter_start
                self._note_exit(
                    "tool_continue" if _appended_this_iter > 0 else "fell_through",
                    iteration,
                )

        finally:
            if not self._session._paused_for_approval:
                self._session.resolve_pending_tool_calls()
            # Drain any remaining deferred messages
            for msg in self._session.pop_deferred_messages():
                self._session.append(msg)
            # Per-session cache-hit summary (Part 4D). Logged at every run-end
            # boundary; the totals are cumulative across the session, so the
            # last line for a session is its session total. Lets us compare
            # cache hit rate before/after the CONTEXT.md on-judgment change.
            if self._llm_call_count > 0 and self._prompt_input_tokens_total > 0:
                _rate = self._cache_read_tokens_total / self._prompt_input_tokens_total
                _cache_logger.info(
                    "[CACHE_SESSION] session=%s calls=%d prompt_tokens=%d "
                    "cached_tokens=%d cache_rate=%.1f%%",
                    getattr(self._session, "session_uuid", "?"),
                    self._llm_call_count,
                    self._prompt_input_tokens_total,
                    self._cache_read_tokens_total,
                    _rate * 100,
                )
            # NOTE: fire-and-forget asyncio.create_task(self._on_session_end())
            # was previously here. Removed in TASK-cancel-arch-04: the
            # synchronous session-end inside agent_manager.new_session() is
            # now the sole authoritative summarization path. The fire-and-
            # forget had no strong reference and would emit
            # "Task was destroyed but it is pending" on cancel.

            # Diagnostics (Point 3): run-level terminal disposition. This is the
            # authoritative capture of the original-bug fingerprint —
            # `none_appended` means the loop exited without persisting any new
            # row this turn. Runs even while an exception propagates through the
            # finally (detected via sys.exc_info), which is logged as `error`.
            _base = (
                rows_at_run_start
                if rows_at_run_start is not None
                else len(self._session._messages)
            )
            _rows_appended = len(self._session._messages) - _base
            if sys.exc_info()[0] is not None:
                _disposition = "error"
            elif _rows_appended == 0:
                _disposition = "none_appended"
            elif self._cancellation_marker_appended_this_turn or self._turn_cancelled:
                _disposition = "cancel"
            else:
                _disposition = "normal"
            self._diag(
                "run_terminal",
                disposition=_disposition,
                rows_appended=_rows_appended,
                exit_path=self._loop_exit_path,
            )

            self._running = False

    # ------------------------------------------------------------------
    # State refresh helpers
    # ------------------------------------------------------------------

    async def _run_refresh(self, trigger_name: str, iteration: int) -> None:
        """Execute a state refresh and update checkpoint metadata.

        Runs the session-end routine with bypass_idempotency=True so it can
        fire mid-session. Updates context_manager checkpoint fields after a
        successful run so the agent sees accurate last-update metadata on
        the next turn.

        Broadcasts state_refresh.lifecycle WS events (in_progress / done /
        failed) via the on_session_end_refresh callback, which is wired by
        agent_manager to include ws.broadcast calls.

        Always resets _turns_since_last_update and _last_refresh_iteration
        regardless of success/failure so cooldown still engages (prevents
        rapid retry storms on persistent LLM failures).
        """
        if self._on_session_end_refresh is None:
            return

        try:
            await self._on_session_end_refresh(trigger_name)
        except Exception:
            logger.exception("State refresh failed (trigger=%s)", trigger_name)
        finally:
            # Always record the refresh so cooldown engages
            self._turns_since_last_update = 0
            self._last_refresh_iteration = iteration
            # Push metadata into context_manager so the agent sees it
            ts = datetime.now(timezone.utc).isoformat()
            if hasattr(self._context_manager, "_last_checkpoint_turn"):
                self._context_manager._last_checkpoint_turn = iteration
                self._context_manager._last_checkpoint_ts = ts
                self._context_manager._turns_since_last_update = 0

    def _spawn_refresh(self, trigger: str) -> None:
        """Start a background consolidation pass. SYNCHRONOUS — no awaits
        between the caller's in-flight check and create_task (cooperative
        event loop ⇒ the single-flight gate is TOCTOU-free; spec 013 race #3).
        """
        iteration = self._current_iteration
        self._last_merge_at = time.monotonic()
        self._refresh_dirty = False           # fresh pass reads files fresh — covers all prior triggers
        # Reset at SPAWN (and again in _run_refresh's finally): turns counted
        # while a slow pass runs are deliberately discarded — benign, and it
        # prevents turn_count refire churn while the pass is in flight.
        self._turns_since_last_update = 0
        task = asyncio.create_task(
            self._run_refresh(trigger, iteration),
            name=f"refresh-{trigger}-{self._session.session_uuid}-{iteration}",
        )
        self._refresh_task = task

        def _on_refresh_done(t: asyncio.Task) -> None:
            if self._refresh_task is t:
                self._refresh_task = None
            if not t.cancelled() and t.exception() is not None:
                # _run_refresh swallows callback exceptions; this catches
                # crashes in _run_refresh itself.
                logger.error(
                    "Background state refresh crashed (trigger=%s)",
                    trigger, exc_info=t.exception(),
                )

        task.add_done_callback(_on_refresh_done)

    def schedule_checkpoint(self, trigger: str = "agent_decided") -> str:
        """Single-flight, debounced, coalescing scheduler (spec 013).

        Synchronous by design: no await between the in-flight check and the
        spawn, so two callers can never both observe 'no pass in flight'.
        """
        if self._on_session_end_refresh is None:
            return "State refresh unavailable in this session."
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_dirty = True
            return ("Consolidation already in flight — your request was "
                    "coalesced into it.")
        if time.monotonic() - self._last_merge_at < REFRESH_DEBOUNCE_S:
            self._refresh_dirty = True
            return ("Consolidation ran recently — deferred until the debounce "
                    "window elapses. Your edits are already persisted.")
        self._spawn_refresh(trigger)
        return "Consolidation scheduled in background."

    def _maybe_consume_dirty(self) -> None:
        """Turn-boundary pickup of a coalesced/deferred trigger (spec 013)."""
        if (
            self._refresh_dirty
            and self._on_session_end_refresh is not None
            and (self._refresh_task is None or self._refresh_task.done())
            and time.monotonic() - self._last_merge_at >= REFRESH_DEBOUNCE_S
        ):
            self._spawn_refresh("agent_decided_coalesced")

    async def drain_refresh(self) -> None:
        """Await any in-flight background pass. Called before the two passes
        that stay synchronous (token_pressure, session boundary) so two
        consolidations never overlap (spec 013 race #3)."""
        task = self._refresh_task
        if task is not None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # _run_refresh already logged it

    async def trigger_checkpoint(self) -> str:
        """Public entry point for the agent-decided refresh trigger.

        Spec 013: schedules a BACKGROUND pass and returns the scheduler's
        status string immediately — never blocks the loop.
        """
        return self.schedule_checkpoint("agent_decided")

    # ------------------------------------------------------------------
    # Cancellation API (cancel_turn / terminate)
    # ------------------------------------------------------------------

    async def cancel_turn(self) -> None:
        """Interrupt the current turn. Loop exits, agent stays alive.

        Backs the POST /api/v2/agents/{pid}/cancel HTTP verb (the UI
        Stop button). Writes a single cancellation marker to the session
        JSONL, cancels any in-flight LLM stream, and breaks the loop's
        while body so the task completes; _on_loop_done then drains any
        queued messages and either broadcasts idle or hot-resumes for
        queued input.

        Idempotent: safe to call when no turn is in flight, or twice in
        rapid succession. Exactly one cancellation marker is written per
        cancellation event regardless of when the cancel landed (during
        stream, between stream and tools, or mid-tool-loop).

        terminate() — used by stop_agent / new_session — sets the stop
        flag in addition to calling cancel_turn(), which causes the
        loop's CancelledError handler to re-raise instead of break (the
        terminate path also tears down sub-agents, browser pages, etc.).
        cancel_turn() alone preserves all of that ancillary state.
        """
        # Idempotency: if we have already initiated a cancel for the
        # current turn (marker pending or appended), do nothing. The flag
        # is reset unconditionally at run() entry so the hot-resume path
        # (run() with no initial_message after a cancelled turn) does
        # not inherit the prior turn's True flag.
        if self._cancellation_marker_appended_this_turn:
            return

        if self._inflight_stream is None or self._inflight_stream.done():
            # No turn in flight; possibly idle, possibly between LLM and
            # tool exec, possibly mid-tool-loop. Set the flag so the loop
            # body skips remaining tool calls and the post-tool block
            # appends the marker if appropriate.
            self._turn_cancelled = True
            return

        # Claim the inflight task BEFORE any await.
        inflight = self._inflight_stream
        self._inflight_stream = None

        # Mark cancelled FIRST so any concurrent loop work observes it,
        # then mark "marker pending for this turn" to make further
        # cancel_turn() calls no-ops until a new turn starts.
        self._turn_cancelled = True
        self._cancellation_marker_appended_this_turn = True

        # Persist the marker SYNCHRONOUSLY before the await so its JSONL
        # position is adjacent to the cancelled turn. The loop's break-
        # on-cancel exit means nothing else writes to the session after
        # this synchronous append before the task completes, so marker
        # position is trivially correct under the current design.
        self._persist_cancellation_marker_inner()

        inflight.cancel()
        try:
            await asyncio.wait_for(inflight, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            logger.exception("inflight stream raised during cancel_turn")
        # Budget Piece 2: the legacy partial-output cost debit
        # (_debit_cancelled_turn_cost → _budget_spent_usd / on_cost_update) was
        # deleted. The token ledger is the single source of recorded spend and
        # records whole responses; partial cancelled-stream tokens are not
        # ledgered (the provider's final usage chunk never arrived), which is
        # acceptable — the next guard evaluation reads the ledger total.

    async def terminate(self) -> None:
        """End the loop. Cancels current turn and exits the while True.

        Session JSONL preserved; agent task done; handle cleanup is the
        caller's responsibility (caller awaits self._task or shields it).
        Do NOT await self._task here — this would deadlock if terminate
        was invoked from within the loop's own task.
        """
        # Cancel any in-flight refresh task before cancelling the turn.
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
        # Cancel any pending debounced spend_updated trailing broadcast so it
        # does not fire after teardown (best-effort UI freshness, not durable
        # state — dropping a pending trailing on terminate is acceptable).
        await self._spend_broadcaster.aclose()
        await self.cancel_turn()
        # Cooperative flag: loop's between-iterations check picks this up.
        self._session.stop()
        if self._task is not None and not self._task.done():
            # If we're called from a different task (typical for /stop),
            # cancel the loop task so it exits the inflight stream wait
            # via the CancelledError -> is_stopped() raise branch.
            cur = None
            try:
                cur = asyncio.current_task()
            except RuntimeError:
                pass
            if cur is not self._task:
                self._task.cancel()

    def _persist_cancellation_marker(self) -> None:
        """Append a system message indicating user cancellation.

        Called exactly once per cancellation event. Sets the per-turn
        flag so cancel_turn() and the post-tool-loop block do not double-
        append. The flag is reset at the top of each loop iteration.
        """
        if self._cancellation_marker_appended_this_turn:
            return
        self._cancellation_marker_appended_this_turn = True
        self._persist_cancellation_marker_inner()

    def _persist_cancellation_marker_inner(self) -> None:
        """Write the cancellation marker without re-checking the flag.

        Internal helper used by cancel_turn() once it has already claimed
        the per-turn marker slot via _cancellation_marker_appended_this_turn.

        Sets _exit_reason = "cancelled" synchronously so the queue
        dispatcher's _await_and_handle observes the cancellation as soon as
        the loop task terminates. Setting this here (rather than from the
        dispatcher) makes the signal correct for any cancel path, including
        the /cancel HTTP verb that doesn't go through QueueDispatcher.pause.
        """
        self._exit_reason = "cancelled"
        self._session.append({
            "role": "system",
            "content": "[cancelled by user]",
            "source": "management",
            "cancelled_by_user": True,
        })
