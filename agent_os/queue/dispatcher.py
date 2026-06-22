# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""QueueDispatcher — per-project queue-draining task.

The dispatcher owns one asyncio.Task per project. It pulls queued items,
runs the agent loop on the project's session, and routes on three exit
signals: complete → advance, blocked → bypass, text → contract violation
(Concern 4: agent must declare an outcome on queue items).

The dispatcher does NOT own the agent loop — AgentManager does. The
dispatcher reads loop._exit_reason after each run via AgentManager.get_loop.

Pause/resume: pause() cancels the live attempt's turn (via cancel_turn) so
the agent comes to rest on the parked attempt's session — without rotating
or swapping sessions. The user's chat messages while paused land in that
same parked attempt session, so on resume the agent sees the clarifying
turns in conversation history. resume() flips queue state back to RUNNING
and re-spawns a loop run on the parked attempt. The dispatcher task stays
alive across pause/resume cycles. After an advance drains the last
queueable item, the store auto-transitions the queue to IDLE.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemRecord,
    ItemState,
    QueueRunState,
    QueueState,
)
from agent_os.queue.store import QueueStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueDispatcher:
    """Owns the queue-draining asyncio.Task for one project."""

    IDLE_WAIT_TIMEOUT_SEC = 5.0
    LOOP_WAIT_POLL_SEC = 2.0
    # How often the held slot re-checks for the session's next turn and for
    # stop / pause / shutdown while a continuation is pending (the session
    # ended a turn without a verdict but has more turns to go). Small enough to
    # stay responsive; the total wait is bounded by the backstop deadline.
    HOLD_POLL_SEC = 0.5
    SUB_AGENT_STOP_TIMEOUT_SEC = 10.0
    # After auto-starting an agent, poll run-status this often until it settles
    # to 'idle' (so the queue item hot-resumes a dedicated run instead of being
    # folded into the freshly-launched loop's in-flight turn). Bounded by the
    # timeout, after which we inject anyway rather than hang.
    AGENT_READY_POLL_SEC = 0.1
    AGENT_READY_TIMEOUT_SEC = 30.0

    def __init__(
        self,
        project_id: str,
        store: QueueStore,
        agent_manager,
        ws_manager=None,
        max_runtime_seconds: int = 1800,
        workspace: str = "",
    ):
        self._project_id = project_id
        self._store = store
        self._agent_manager = agent_manager
        self._ws = ws_manager
        self._max_runtime_seconds = max_runtime_seconds
        # Project workspace — used to resolve queue-item file_refs into the
        # <attached_files> prefix (attachment symmetry with the chat composer).
        self._workspace = workspace
        self._task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._idle_event = asyncio.Event()
        # Legacy per-item stall flag. Concern 4 removed the queue-item code
        # path that set this — text-only exits on queue items are now
        # contract violations and rotate immediately. The field is retained
        # (and still cleared on resume) because the _run loop's idle guard
        # reads it; nothing in the queue dispatch path sets it anymore.
        self._stalled_item_id: Optional[str] = None
        # Increments on every stop(). _await_and_handle snapshots this at
        # the start of an attempt; if the generation differs at the end, a
        # stop fired mid-attempt and we MUST NOT close/advance/stall — the
        # state belongs to the next resume(). A bare boolean flag would
        # race with resume() resetting it before the handler resumed.
        self._stop_generation = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the dispatcher task. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._shutting_down = False
        self._ensure_run_task()
        logger.info("dispatcher(%s): started", self._project_id)

    def _ensure_run_task(self) -> None:
        """Idempotently (re)create the main drain task.

        Safety net for B3: if the ``_run`` loop ever dies (e.g. some
        CancelledError path the layer-1 guard didn't anticipate), the queue
        would stay RUNNING with no task pulling items. Any transition back to
        RUNNING (start / resume) calls this to self-heal a dead loop. A no-op
        when the task is already live so we never double-spawn — two ``_run``
        loops would race on the single agent slot.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def reclaim_on_startup(self) -> dict:
        """Reconcile queue state with on-disk records at daemon startup.

        Contract per D6:
        - If queue state is PAUSED: leave everything as is. The user
          previously paused the queue; the dispatcher will idle until
          /queue/resume (or /queue/start) is invoked.
        - Otherwise (RUNNING or IDLE) and items are in RUNNING state: those
          attempts were interrupted by daemon death. Close each open
          attempt with outcome=INTERRUPTED, increment interrupted_count.
          interrupted_count >= 2 → mark BLOCKED (poison pill protection).
          Otherwise → requeue at head with priority=1.

        Called by AgentManager during agent (re)start, before start().
        Returns a summary dict for logging.
        """
        state = self._store.load()
        summary: dict = {
            "queue_state": state.state.value,
            "reclaimed_items": [],
            "blocked_items": [],
        }
        if state.state == QueueRunState.PAUSED:
            logger.info(
                "dispatcher(%s): startup with PAUSED queue; no reclaim",
                self._project_id,
            )
            return summary

        # RUNNING / IDLE: walk items in RUNNING state.
        for item in list(state.items):
            if item.state != ItemState.RUNNING:
                continue
            # Close any open attempt
            if item.attempts and item.attempts[-1].outcome is None:
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.INTERRUPTED,
                    block_reason="interrupted by daemon restart",
                )
            new_count = self._store.increment_interrupted(item.id)
            if new_count >= 2:
                self._store.set_item_state(item.id, ItemState.BLOCKED)
                summary["blocked_items"].append(item.id)
                logger.warning(
                    "dispatcher(%s): item %s blocked after %d interruptions",
                    self._project_id, item.id, new_count,
                )
            else:
                self._store.set_item_state(item.id, ItemState.QUEUED)
                self._store.move_to_head(item.id)
                summary["reclaimed_items"].append(item.id)
                logger.info(
                    "dispatcher(%s): item %s requeued at head (interruptions=%d)",
                    self._project_id, item.id, new_count,
                )
        return summary

    async def shutdown(self) -> None:
        """Full teardown — used by AgentManager.stop_agent. NOT the same as
        Phase 4 stop(): this kills the dispatcher task entirely."""
        self._shutting_down = True
        self._idle_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        logger.info("dispatcher(%s): shut down", self._project_id)

    def notify_new_item(self) -> None:
        """Wake the dispatcher if it is sleeping. Safe from sync code."""
        self._idle_event.set()

    # ------------------------------------------------------------------
    # Phase 4: stop / resume
    # ------------------------------------------------------------------

    async def stop(self, duration_seconds: Optional[float] = None) -> dict:
        """Pause queue draining. Cancel the live attempt's turn so the agent
        comes to rest on the parked attempt's session — without rotating or
        swapping sessions. User chat messages while paused land in that same
        parked session; on resume the agent sees them in conversation history.

        The in-flight attempt is NOT closed and the queue is NOT advanced —
        both will happen on resume via _await_and_handle's cancelled-branch
        early-return preserving the attempt for resume to pick up.

        Body {"duration_seconds": N} = timed pause (auto-resumes after N
        seconds); no body = pause until resumed.
        """
        self._stop_generation += 1
        paused_until: Optional[str] = None
        # The route already enforces gt=0 (422 on zero/negative); this guard
        # covers direct callers — the redundancy is intentional.
        if duration_seconds is not None and duration_seconds > 0:
            paused_until = (
                datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
            ).isoformat()
        # Set queue state first so the main loop sees PAUSED on its next tick.
        # A stop()/timed pause is always a user-initiated pause (reason "user");
        # the store's precedence rule keeps it sticky over a later budget pause.
        self._store.set_queue_state(
            QueueRunState.PAUSED, paused_until=paused_until, reason="user",
        )

        # Terminate any sub-agents under a budget so they don't keep churning
        # against the user while the dispatcher is paused.
        sub_mgr = self._agent_manager.get_sub_agent_manager()
        if sub_mgr is not None:
            try:
                await asyncio.wait_for(
                    sub_mgr.stop_all(self._project_id),
                    timeout=self.SUB_AGENT_STOP_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "dispatcher(%s): stop_all timed out; sub-agents may leak",
                    self._project_id,
                )
            except Exception:
                logger.exception(
                    "dispatcher(%s): stop_all raised; continuing",
                    self._project_id,
                )

        # Halt the active loop via terminate(). Internally this calls
        # cancel_turn() (which writes the cancellation marker synchronously
        # and sets loop._exit_reason = "cancelled"), then session.stop() +
        # task.cancel() so the loop task actually exits. The dispatcher's
        # _await_and_handle then reads _exit_reason and takes the
        # cancelled-branch early return. The session JSONL is preserved
        # in place — no rotation, no swap — so chat messages sent during
        # pause land in the parked attempt's session, and resume picks up
        # exactly where pause left off.
        holder_sid = self._agent_manager.current_holder_session_id(self._project_id)
        loop_obj = self._agent_manager.get_loop(self._project_id, session_id=holder_sid)
        if loop_obj is not None:
            try:
                await loop_obj.terminate()
            except Exception:
                logger.exception(
                    "dispatcher(%s): terminate during pause raised; continuing",
                    self._project_id,
                )

        self._broadcast_state_changed(QueueRunState.PAUSED.value)
        logger.info(
            "dispatcher(%s): paused; parked attempt session preserved",
            self._project_id,
        )
        return {"status": "paused", "paused_until": paused_until}

    async def resume(self) -> dict:
        """Re-activate queue draining. If a parked attempt exists, start a
        new loop run on its already-active session (no swap needed — pause
        left the session in place). Otherwise, just set the queue back to
        running and let the main loop pick up the next queued item.
        """
        self._stalled_item_id = None
        self._store.set_queue_state(QueueRunState.RUNNING)

        # B3 safety net: ensure the main drain loop is alive before we resume.
        # If a prior pause cancellation killed _run (the layer-1 guard is the
        # primary defense, but belt-and-suspenders), resume alone must bring
        # the queue back to life — re-create the task here so that after the
        # parked item finishes, the loop is there to pull the next queued item.
        self._ensure_run_task()

        # Find a parked attempt: head item in RUNNING with an attempts list.
        head = self._store.head()
        parked_item = head if (head and head.state == ItemState.RUNNING) else None

        if parked_item is not None and parked_item.attempts:
            session_id = parked_item.attempts[-1].session_id
            logger.info(
                "dispatcher(%s): resuming item %s on parked session %s",
                self._project_id, parked_item.id, session_id,
            )
            # No switch_session needed — pause did not swap, the parked
            # session is still active. Set the loop's queue_state so the
            # approval branch picks it up before _start_loop fires.
            loop_obj = self._agent_manager.get_loop(self._project_id, session_id=session_id)
            if loop_obj is not None:
                try:
                    loop_obj._queue_state = "running"
                except Exception:
                    pass
            # Spawn a handler task: it starts the loop and waits for exit.
            # Pass session_id so _start_loop locates the handle correctly
            # under the F7 tuple-keyed handles map (switch_session above
            # re-keyed it to the parked session_id).
            asyncio.create_task(self._resume_attempt(parked_item, session_id))
        else:
            # No parked attempt — kick the main loop in case items are queued.
            self._idle_event.set()

        self._broadcast_state_changed(QueueRunState.RUNNING.value)
        return {
            "status": "running",
            "resumed_item_id": parked_item.id if parked_item else None,
        }

    async def _resume_attempt(self, item: ItemRecord, session_id: str) -> None:
        """Re-start the agent loop on an already-parked attempt session,
        then route the eventual exit through the same handler used by
        first-time dispatch.

        ``session_id`` is the parked attempt's session_id; required so
        ``_start_loop`` locates the handle under its current F7 tuple key
        (``switch_session`` re-keyed it during the caller's swap).
        """
        try:
            await self._agent_manager._start_loop(
                self._project_id, session_id=session_id,
            )
        except Exception:
            logger.exception(
                "dispatcher(%s): _start_loop on resume failed for item %s",
                self._project_id, item.id,
            )
            return
        await self._await_and_handle(item, session_id)

    # ------------------------------------------------------------------
    # Concern 5: retry a BLOCKED item on its preserved session
    # ------------------------------------------------------------------

    async def retry_blocked_item(
        self, item_id: str, new_input: str, *, mode: str,
    ) -> dict:
        """Re-dispatch a BLOCKED item on the same session as its prior attempt.

        mode="answer": inject `new_input` raw (used by question-card answers).
        mode="edit":   wrap with `[QUEUE ITEM | id=... | attempt=N+1]` header.

        Hot-resume semantics: pulls the most recent attempt's session_id and
        calls AgentManager.switch_session to reload that JSONL as the active
        session. switch_session also resets `_cold_resume_injected` and
        related ContextManager flags so memory files re-read fresh on the
        next loop run. A new AttemptRecord is appended sharing the prior
        session_id (we are continuing it, not opening a new one).
        """
        if mode not in ("answer", "edit"):
            raise ValueError(f"mode must be 'answer' or 'edit', got {mode!r}")

        # 1. Look up the item; assert BLOCKED + has a prior attempt
        qstate = self._store.load()
        item = next((it for it in qstate.items if it.id == item_id), None)
        if item is None:
            raise KeyError(f"item {item_id} not found")
        if item.state != ItemState.BLOCKED:
            raise ValueError(
                f"item {item_id} is not BLOCKED (state={item.state.value})"
            )
        if not item.attempts:
            raise ValueError(
                f"item {item_id} has no attempts to retry from"
            )

        # 2. Pull session_id from the most recent attempt
        prior_session_id = item.attempts[-1].session_id

        # 3 + 4. Swap the active session back to the parked attempt's JSONL.
        # switch_session resets _cold_resume_injected / _recovery_injected /
        # _window_factor / _last_usage_pct on the ContextManager for us.
        try:
            await self._agent_manager.switch_session(
                self._project_id, prior_session_id, start_loop=False,
            )
        except Exception:
            logger.exception(
                "dispatcher(%s): switch to prior session %s failed for retry "
                "of item %s",
                self._project_id, prior_session_id, item_id,
            )
            raise

        # 5. Mint a new attempt record. attempt_number is len+1; the new
        # attempt reuses prior_session_id because retries continue the
        # same JSONL, they don't open a new one.
        attempt_number = len(item.attempts) + 1
        self._store.append_attempt(
            item_id, AttemptRecord(session_id=prior_session_id),
        )

        # 7. State → RUNNING. Also set loop._queue_state so Phase-3
        # approval-as-block behaviour fires on the retry.
        self._store.set_item_state(item_id, ItemState.RUNNING)
        loop_obj = self._agent_manager.get_loop(self._project_id, session_id=prior_session_id)
        if loop_obj is not None:
            try:
                loop_obj._queue_state = "running"
            except Exception:
                pass

        # 6. Inject — headered re-wrap for edit, raw user message for answer.
        # Edit-mode reuses the same HEADER_CONTRACT block as a first dispatch
        # so the agent sees the same contract on every queue-item user
        # message regardless of attempt number.
        if mode == "edit":
            injected = (
                f"[QUEUE ITEM | id={item_id} | attempt={attempt_number}]\n"
                + self.HEADER_CONTRACT
                + new_input
            )
        else:  # mode == "answer"
            injected = new_input

        try:
            # F7: pass session_id so inject_message locates the handle re-
            # keyed by switch_session above (not the default-session handle
            # which doesn't exist) — otherwise it would auto-start a NEW
            # agent and violate single-slot discipline.
            await self._agent_manager.inject_message(
                self._project_id, injected,
                session_id=prior_session_id,
            )
        except Exception:
            logger.exception(
                "dispatcher(%s): inject failed during retry of item %s",
                self._project_id, item_id,
            )
            self._store.close_latest_attempt(
                item_id,
                outcome=AttemptOutcome.INTERRUPTED,
                block_reason="inject failed during retry",
            )
            self._store.set_item_state(item_id, ItemState.BLOCKED)
            raise

        logger.info(
            "dispatcher(%s): retrying item %s attempt=%d on session=%s "
            "(mode=%s)",
            self._project_id, item_id, attempt_number, prior_session_id, mode,
        )

        # 8. Spawn a handler task that awaits the loop and routes the exit
        # through the standard handler — same shape as _resume_attempt.
        # Note: inject_message (Case 2: idle session + alive handle) already
        # calls _start_loop internally, so we do NOT call _start_loop here.
        asyncio.create_task(self._retry_attempt_handler(item, prior_session_id))

        self._broadcast_state_changed(QueueRunState.RUNNING.value)
        return {
            "status": "retry_started",
            "item_id": item_id,
            "attempt_number": attempt_number,
            "session_id": prior_session_id,
            "mode": mode,
        }

    async def _retry_attempt_handler(self, item: ItemRecord, session_id: str) -> None:
        """Wait for the retry's loop run to finish and route the exit.

        Mirrors the shape of _resume_attempt — _await_and_handle only reads
        `item.id`, so the stale `state` / `attempts` on the captured item
        object don't cause incorrect routing. ``session_id`` is the retry's
        (reused prior) session, threaded so loop reads target it.
        """
        await self._await_and_handle(item, session_id)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            while not self._shutting_down:
                try:
                    # Catch deletions that occurred while idle/paused so an
                    # emptied queue advances to IDLE without needing an
                    # advance event to trigger the helper.
                    self._store.auto_idle_if_empty()
                    qstate = self._store.load()
                    if qstate.state == QueueRunState.PAUSED:
                        if self._timed_pause_expired(qstate):
                            logger.info(
                                "dispatcher(%s): timed pause expired; auto-resuming",
                                self._project_id,
                            )
                            await self.resume()
                        elif self._budget_auto_resume_ready(qstate):
                            logger.info(
                                "dispatcher(%s): budget pause cleared (now under "
                                "limit, action!=stop); auto-resuming",
                                self._project_id,
                            )
                            await self.resume()
                        else:
                            await self._wait_idle()
                        continue
                    if qstate.state == QueueRunState.IDLE:
                        await self._wait_idle()
                        continue

                    if self._stalled_item_id is not None:
                        await self._wait_idle()
                        continue

                    # If an attempt is already in-flight — either via the
                    # main path's await on _dispatch_one (this branch can't
                    # be re-entered) OR via a parallel _resume_attempt task —
                    # head() will be RUNNING. Idle until rotation drops the
                    # head's state out of RUNNING.
                    head = self._store.head()
                    if head is not None and head.state == ItemState.RUNNING:
                        await self._wait_idle()
                        continue

                    item = self._store.next_queued()
                    if item is None:
                        await self._wait_idle()
                        continue

                    await self._dispatch_one(item)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "dispatcher(%s): unhandled error, backing off",
                        self._project_id,
                    )
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    async def _wait_idle(self) -> None:
        self._idle_event.clear()
        try:
            await asyncio.wait_for(
                self._idle_event.wait(),
                timeout=self.IDLE_WAIT_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            self._idle_event.clear()

    def _timed_pause_expired(self, qstate: QueueState) -> bool:
        """True when a snooze deadline exists and has passed. Malformed
        deadlines are cleared (stay paused, drop the timer) so a corrupt
        value can't trip a resume loop or spam the log every tick."""
        if qstate.paused_until is None:
            return False
        try:
            deadline = datetime.fromisoformat(qstate.paused_until)
            if deadline.tzinfo is None:
                # Hand-edited / legacy value without an offset: our own writes
                # are always UTC-aware, so interpret naive as UTC.
                deadline = deadline.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.warning(
                "dispatcher(%s): malformed paused_until %r; clearing",
                self._project_id, qstate.paused_until,
            )
            # Drop the corrupt deadline but stay PAUSED — preserve the existing
            # pause_reason (don't downgrade a budget pause to user here).
            self._store.set_queue_state(
                QueueRunState.PAUSED,
                paused_until=None,
                reason=qstate.pause_reason,
            )
            return False
        return datetime.now(timezone.utc) >= deadline

    def _budget_auto_resume_ready(self, qstate: QueueState) -> bool:
        """Decide whether a budget-paused queue may LAZILY auto-resume (P2-D).

        Lazy, query-driven — NO timers, NO background tasks. Called only from
        the PAUSED branch of ``_run`` on the dispatcher's EXISTING evaluation
        cadence (the ``_wait_idle`` IDLE_WAIT_TIMEOUT_SEC tick, plus any
        ``notify_new_item`` nudge — e.g. a limit raised via the PUT route). The
        window boundary (period roll-over) is discovered by the same guard query
        the loop uses; we never schedule a midnight wake-up.

        Returns True iff ALL hold:
          * ``pause_reason == "budget"`` — a USER pause never auto-resumes, and
            we only ever touch budget-paused queues (precedence is enforced in
            the store; here we just read the reason).
          * the project's CURRENT ``budget_action`` is NOT ``stop`` — the
            binding coordinator ruling: "nothing ever auto-resumes a stop". Only
            ``pause``-mode projects auto-resume.
          * the SAME guard query the loop runs (``evaluate_budget``) reports the
            project is now UNDER limit (``decision.tripped is False``) — limit
            raised, period rolled over, anchor moved, or pricing corrected.

        Fail-safe: any error resolving config or evaluating the guard → False
        (stay paused). Auto-resume must never fire on a transient read error.
        """
        if qstate.pause_reason != "budget":
            return False
        # Resolve the LIVE budget config via the shared reader (same source the
        # loop's pre-flight guard uses). Tolerate a manager test-double without
        # the accessor: no config → cannot prove under-limit → stay paused.
        cfg_fn = getattr(self._agent_manager, "build_budget_config", None)
        if not callable(cfg_fn):
            return False
        try:
            cfg = cfg_fn(self._project_id)
        except Exception:  # noqa: BLE001 — a bad read must not auto-resume
            logger.warning(
                "dispatcher(%s): budget config read failed; staying paused",
                self._project_id, exc_info=True,
            )
            return False

        from agent_os.budget.guard import (
            evaluate_budget,
            normalize_budget_action,
            ACTION_STOP,
        )

        # Coordinator ruling: a project now in stop-mode NEVER auto-resumes,
        # even if it happens to be under limit. Re-checked on the CURRENT config.
        action = normalize_budget_action((cfg or {}).get("budget_action"))
        if action == ACTION_STOP:
            return False

        fx_rates = (cfg or {}).get("fx_rates") if isinstance(cfg, dict) else None
        try:
            decision = evaluate_budget(
                self._workspace or self._project_id, cfg, fx_rates=fx_rates,
            )
        except Exception:  # noqa: BLE001 — guard failure must not auto-resume
            logger.warning(
                "dispatcher(%s): budget guard eval failed; staying paused",
                self._project_id, exc_info=True,
            )
            return False
        # Under limit now → clear to resume. Still tripped → stay paused (no
        # flap: we return False and the loop idles, leaving reason="budget").
        return not decision.tripped

    # ------------------------------------------------------------------
    # Per-item dispatch
    # ------------------------------------------------------------------

    async def _dispatch_one(self, item: ItemRecord) -> None:
        # The dispatcher is the project's session-lifecycle manager. Gate on
        # onboarding: a project with no captured state must not be
        # auto-started (same gate /queue/start applied before).
        if not self._agent_manager.is_onboarding_complete(self._project_id):
            logger.info(
                "dispatcher(%s): onboarding incomplete; not auto-starting agent",
                self._project_id,
            )
            await self._wait_idle()
            return

        # Busy-slot guard: a project has exactly one active-loop slot. If
        # another session already holds it (e.g. a long-running onboarding /
        # research turn the user kicked off, or a still-draining attempt),
        # minting a fresh session and injecting would race that turn and
        # inject_message would reject — burning an attempt and (pre-fix)
        # wedging the head item as a zombie RUNNING. Back off WITHOUT minting
        # a session/attempt; the main loop re-polls every IDLE_WAIT_TIMEOUT_SEC
        # and will dispatch once the slot is free. (Reuses the existing
        # current_holder_session_id accessor — no new machinery.)
        #
        # getattr-guarded: the real AgentManager always defines this accessor;
        # only lightweight test doubles that model a single ready session omit
        # it. For those, "no holder" (slot free) is the correct semantic, so we
        # default to a lambda returning None rather than crashing the dispatch.
        _holder_fn = getattr(
            self._agent_manager, "current_holder_session_id", lambda _pid: None,
        )
        holder = _holder_fn(self._project_id)
        if holder is not None:
            logger.info(
                "dispatcher(%s): loop slot held by session %s; deferring "
                "dispatch of item %s (no attempt minted)",
                self._project_id, holder, item.id,
            )
            await self._wait_idle()
            return

        # Each queue item runs in its OWN fresh session — the same new_session
        # the user gets from "+ new session", then an inject. No rotation, no
        # reuse of a prior session, no action on whatever finished before.
        # new_session mints the id; the session materializes (handle, loop,
        # JSONL) on the inject below via inject_message Case 3.
        try:
            minted = await self._agent_manager.new_session(self._project_id)
        except Exception:
            logger.exception(
                "dispatcher(%s): new_session failed for item %s",
                self._project_id, item.id,
            )
            await self._wait_idle()
            return
        session_id = minted["session_id"]

        # Compute attempt_number BEFORE append_attempt so the header reflects
        # the new attempt's 1-based index. First dispatch: len(attempts)==0
        # → attempt_number=1. Concern-5 retries arrive via _resume_attempt /
        # the retry path (which reuse prior_session_id), not through here.
        attempt_number = len(item.attempts) + 1

        self._store.set_item_state(item.id, ItemState.RUNNING)
        self._store.append_attempt(item.id, AttemptRecord(session_id=session_id))

        logger.info(
            "dispatcher(%s): dispatching item %s attempt=%d (session=%s)",
            self._project_id, item.id, attempt_number, session_id,
        )

        # Prepend the <attached_files> block for any staged file_refs, so a
        # dispatched item carries attachments the same way a user's Send does.
        attach_prefix = ""
        if getattr(item, "file_refs", None):
            try:
                from agent_os.api.routes._attachment_formatter import (
                    format_prefix_from_refs,
                )
                attach_prefix = format_prefix_from_refs(
                    self._workspace, item.file_refs,
                )
            except Exception:
                logger.warning(
                    "dispatcher(%s): attachment prefix failed for item %s",
                    self._project_id, item.id, exc_info=True,
                )

        header = (
            f"[QUEUE ITEM | id={item.id} | attempt={attempt_number}]\n"
            + self.HEADER_CONTRACT
        )
        wrapped_content = attach_prefix + header + item.content

        try:
            # Inject into the fresh session. inject_message Case 3 auto-starts
            # a loop for this new session_id in queue mode (queue_state=
            # "running"), so the completion contract is enforced from the
            # first turn. This is the same inject the user's Send uses, plus
            # the queue flag — there is no separate start, so no auto-start
            # race to wait out.
            await self._agent_manager.inject_message(
                self._project_id, wrapped_content,
                session_id=session_id, queue_state="running",
            )
        except Exception:
            logger.exception(
                "dispatcher(%s): inject failed for item %s",
                self._project_id, item.id,
            )
            # Do NOT leave the item RUNNING — that strands the queue head as a
            # zombie (RUNNING with no live attempt), wedging the dispatcher
            # until daemon restart (observed live on item_cb75800b6b18 during
            # the 2026-06-08 queue stress run). Close the attempt INTERRUPTED
            # and reclaim it with the SAME poison-pill convention
            # reclaim_on_startup uses: bump interrupted_count, re-queue at head
            # under the cap, BLOCKED at the cap.
            self._store.close_latest_attempt(
                item.id,
                outcome=AttemptOutcome.INTERRUPTED,
                block_reason="inject failed",
            )
            self._reclaim_interrupted_item(item.id)
            self._idle_event.set()
            return

        await self._await_and_handle(item, session_id)

    # Poison-pill threshold for reclaiming an interrupted item. interrupted
    # attempts >= this → BLOCKED instead of re-queued. Mirrors the constant
    # baked into reclaim_on_startup so a runtime inject failure and a
    # restart-time interruption are treated identically.
    INTERRUPTED_REQUEUE_CAP = 2

    def _reclaim_interrupted_item(self, item_id: str) -> None:
        """Re-queue or BLOCK an item whose latest attempt was INTERRUPTED.

        Shared by the inject-failure path in _dispatch_one and (in spirit) the
        startup reconciler: increment interrupted_count, re-queue at head with
        priority while under INTERRUPTED_REQUEUE_CAP, mark BLOCKED at the cap.
        The attempt records are left intact for diagnosis.
        """
        new_count = self._store.increment_interrupted(item_id)
        if new_count >= self.INTERRUPTED_REQUEUE_CAP:
            self._store.set_item_state(item_id, ItemState.BLOCKED)
            logger.warning(
                "dispatcher(%s): item %s blocked after %d interruptions",
                self._project_id, item_id, new_count,
            )
        else:
            self._store.set_item_state(item_id, ItemState.QUEUED)
            self._store.move_to_head(item_id)
            logger.info(
                "dispatcher(%s): item %s requeued at head after inject failure "
                "(interruptions=%d)",
                self._project_id, item_id, new_count,
            )

    # ------------------------------------------------------------------
    # Contract delivery: header + corrective turn
    # ------------------------------------------------------------------

    # The contract reminder embedded in every queue-item user message,
    # between the [QUEUE ITEM | id | attempt] metadata line and the item
    # content. H1 verification (12 LLM calls, 2 models × 2 placements ×
    # 3 samples) showed header-only delivery yields strictly better final
    # outcomes than system-prompt delivery — same for Kimi, dramatically
    # better for deepseek (0/3 → 3/3 signal rate after corrective turn).
    # First-turn signal rate is 0% under either placement on ambiguous
    # tasks, so the corrective turn does the load-bearing work; the header
    # makes the corrective injection legible to the model when it fires.
    HEADER_CONTRACT = (
        "You are working on a queue item. When you finish, call "
        "mark_task_complete(summary). If you cannot proceed — stuck, "
        "missing info, need to ask the user — call mark_task_blocked(reason); "
        "put any question for the user directly in reason. Do not end "
        "with plain text.\n"
    )

    # The stern message the dispatcher injects when the agent exits
    # text-only on its first turn of an attempt. The agent gets ONE more
    # chance to signal correctly before the dispatcher records a contract
    # violation. Keep the two tool names and the "no plain text" rule
    # in sync with HEADER_CONTRACT — drift between the two would confuse
    # weaker models that lean on consistency in nearby context.
    CORRECTIVE_TURN_PROMPT = (
        "[SYSTEM: You exited without declaring outcome. Call "
        "mark_task_complete if you finished the task, or mark_task_blocked "
        "if you cannot proceed. Do not respond with text. This is your "
        "final turn — the next text-only exit will be recorded as a "
        "contract violation and the queue will advance past this item.]"
    )

    @staticmethod
    def _read_completion(loop_obj) -> "tuple[str, str | None, str | None]":
        """Read (exit_reason, summary, block_reason) from the loop.

        Prefers the formal AgentLoop.get_completion_state() accessor; falls back
        to the raw _exit_* attributes for lightweight test doubles. Returns a
        text-only exit when there is no loop.
        """
        if loop_obj is None:
            return ("text", None, None)
        getter = getattr(loop_obj, "get_completion_state", None)
        if callable(getter):
            return getter()
        return (
            getattr(loop_obj, "_exit_reason", "text"),
            getattr(loop_obj, "_exit_summary", None),
            getattr(loop_obj, "_exit_block_reason", None),
        )

    async def _await_and_handle(self, item: ItemRecord, session_id: str) -> None:
        """Wait for the in-flight loop task to finish then route the outcome
        based on AgentLoop._exit_reason. Honors watchdog, stop, and (CHANGE
        2 of the architecture amendments) gives the agent one corrective
        turn if it exits text-only on a queue item.

        ``session_id`` is the session this item is running in (a fresh one per
        item from _dispatch_one, or the parked/prior session on resume/retry).
        All loop reads target it explicitly — a project may now hold several
        sessions, so the first-handle ``_find_project_handle`` is ambiguous.

        When an item finishes, the dispatcher takes NO action on the
        now-idle session — it simply stays on disk, navigable, and the next
        item gets its own fresh session. (There is no rotation step.)

        The corrective_turn_used flag is a local variable, so a retry of a
        blocked item (which calls _await_and_handle afresh from
        _retry_attempt_handler) gets its own corrective turn.
        """
        corrective_turn_used = False

        while True:
            loop_obj = self._agent_manager.get_loop(self._project_id, session_id=session_id)
            gen_at_start = self._stop_generation

            try:
                await asyncio.wait_for(
                    self._wait_for_loop_done(session_id),
                    timeout=self._max_runtime_seconds,
                )
            except asyncio.CancelledError:
                # B3: pause() halts the live attempt by calling terminate(),
                # which does task.cancel() on the agent-loop task. That
                # cancellation surfaces in _wait_for_loop_done (it awaits
                # asyncio.shield(task)) as CancelledError and is re-raised; it
                # then lands HERE. This is NOT a cancellation of the dispatcher
                # task itself — _run / the dispatcher task is only ever
                # cancelled via self._task.cancel() under self._shutting_down
                # (see shutdown()). So if we are not shutting down, treat this
                # exactly like the pause guard below: clean return, attempt
                # preserved (no close, no advance, no rotation), letting _run
                # continue and idle on PAUSED. Swallowing CancelledError is
                # safe here precisely because the cancel target was the child
                # loop task, not the _run task.
                #
                # Only a genuine dispatcher shutdown (self._shutting_down) — or
                # a cancellation that arrived without a corresponding pause
                # (no _stop_generation bump) — should propagate to kill _run.
                if self._shutting_down:
                    raise
                if self._stop_generation != gen_at_start:
                    logger.info(
                        "dispatcher(%s): item %s loop cancelled by pause; "
                        "attempt preserved (no close, no advance, no rotation)",
                        self._project_id, item.id,
                    )
                    return
                # A cancellation with no pause and no shutdown is unexpected —
                # re-raise so it isn't silently swallowed (preserves asyncio
                # cancellation semantics for any path we didn't anticipate).
                raise
            except asyncio.TimeoutError:
                logger.warning(
                    "dispatcher(%s): item %s exceeded runtime cap %ds; terminating",
                    self._project_id, item.id, self._max_runtime_seconds,
                )
                try:
                    if loop_obj is not None:
                        await loop_obj.terminate()
                except Exception:
                    logger.exception(
                        "dispatcher(%s): terminate after watchdog failed",
                        self._project_id,
                    )
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.INTERRUPTED,
                    block_reason="exceeded runtime cap",
                )
                self._log_attempt_close(
                    item.id, "interrupted",
                    first_turn_signaled=False,
                    corrective_used=corrective_turn_used,
                )
                # No rotation: the timed-out session stays idle on disk and
                # remains navigable; the next item gets its own fresh session.
                self._store.set_item_state(item.id, ItemState.BLOCKED)
                self._broadcast_advance(item.id, "interrupted")
                if self._store.auto_idle_if_empty():
                    self._broadcast_state_changed(QueueRunState.IDLE.value)
                self._idle_event.set()
                return

            # If pause() was called mid-attempt, the loop was terminated by
            # terminate(). We MUST NOT close the attempt — preserve it
            # exactly as it was so resume can pick up cleanly on the same
            # session, with any chat messages the user sent in the meantime
            # already appended to the JSONL.
            if self._stop_generation != gen_at_start:
                logger.info(
                    "dispatcher(%s): item %s loop terminated by pause; "
                    "attempt preserved (no close, no advance, no rotation)",
                    self._project_id, item.id,
                )
                return

            loop_obj = self._agent_manager.get_loop(self._project_id, session_id=session_id)
            exit_reason, exit_summary, exit_block_reason = self._read_completion(loop_obj)

            # Cancelled out-of-band (e.g. via the /cancel HTTP endpoint while
            # a queue item is dispatching) — distinct from pause(), which
            # bumps _stop_generation and is handled by the guard above. The
            # attempt is closed as INTERRUPTED so it can be reclaimed on the
            # next dispatch, but the queue is NOT advanced and the session
            # is NOT rotated — same early-return shape as the pause guard.
            if exit_reason == "cancelled":
                logger.info(
                    "dispatcher(%s): item %s loop exited cancelled; "
                    "attempt closed INTERRUPTED, no advance, no rotation",
                    self._project_id, item.id,
                )
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.INTERRUPTED,
                    block_reason="cancelled",
                )
                self._log_attempt_close(
                    item.id, "interrupted",
                    first_turn_signaled=False,
                    corrective_used=corrective_turn_used,
                    reason="cancelled",
                )
                return

            # Budget trip (Budget Piece 2). The loop's pre-flight guard tripped
            # BEFORE spending again and exited with the first-class
            # ``budget_blocked`` reason. Route it EXPLICITLY — this is the branch
            # that kills the A2/A3 paid-corrective-turn bug:
            #   * NO corrective turn (no inject_system_message)
            #   * NO contract-violation outcome
            # ``exit_block_reason`` carries the normalized action ("pause" |
            # "stop"); the loop set it in _trip_budget.
            #
            # In BOTH modes the queue goes PAUSED with reason="budget" so the
            # remaining items don't dispatch and keep burning spend. The
            # difference is the in-flight item's disposition:
            #   * pause → item returns to the FRONT of the queue in ``queued``
            #     state, so work resumes where it stopped after unblock. The
            #     attempt is closed INTERRUPTED (JSONL stays valid) but
            #     interrupted_count is NOT bumped — a budget block is not a
            #     poison-pill interruption, so it must never trip the cap.
            #   * stop → the item is marked terminal (BLOCKED), never requeued;
            #     nothing auto-resumes a stop. The session ends via the loop's
            #     normal graceful exit (the loop already broke out cleanly).
            if exit_reason == "budget_blocked":
                action = exit_block_reason or "pause"
                logger.info(
                    "dispatcher(%s): item %s budget_blocked (action=%s); "
                    "no corrective turn, no contract violation; queue→PAUSED"
                    " reason=budget",
                    self._project_id, item.id, action,
                )
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.INTERRUPTED,
                    block_reason="budget_blocked",
                )
                if action == "stop":
                    # Terminal: mark BLOCKED, do NOT requeue.
                    self._store.set_item_state(item.id, ItemState.BLOCKED)
                    self._broadcast_advance(item.id, "blocked")
                else:
                    # Pause: return the item to the FRONT of the queue, ready to
                    # resume after unblock. No interrupted_count bump.
                    self._store.set_item_state(item.id, ItemState.QUEUED)
                    self._store.move_to_head(item.id)
                self._log_attempt_close(
                    item.id, "budget_blocked",
                    first_turn_signaled=False,
                    corrective_used=corrective_turn_used,
                    reason=action,
                )
                # ALWAYS pass reason="budget" explicitly. The store's default is
                # "user"; an omitted reason on an already-budget-paused queue
                # would silently flip it to "user" and kill auto-resume.
                self._store.set_queue_state(
                    QueueRunState.PAUSED, reason="budget",
                )
                self._broadcast_state_changed(QueueRunState.PAUSED.value)
                return

            if exit_reason == "complete":
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.COMPLETED,
                    summary=exit_summary or "",
                )
                self._log_attempt_close(
                    item.id, "completed",
                    first_turn_signaled=not corrective_turn_used,
                    corrective_used=corrective_turn_used,
                )
                self._store.set_item_state(item.id, ItemState.DONE)
                self._broadcast_advance(item.id, "completed")
                if self._store.auto_idle_if_empty():
                    self._broadcast_state_changed(QueueRunState.IDLE.value)
                self._idle_event.set()
                return

            if exit_reason == "blocked":
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.BLOCKED,
                    block_reason=exit_block_reason or "",
                )
                self._log_attempt_close(
                    item.id, "blocked",
                    first_turn_signaled=not corrective_turn_used,
                    corrective_used=corrective_turn_used,
                    reason=exit_block_reason,
                )
                self._store.set_item_state(item.id, ItemState.BLOCKED)
                self._broadcast_advance(item.id, "blocked")
                if self._store.auto_idle_if_empty():
                    self._broadcast_state_changed(QueueRunState.IDLE.value)
                self._idle_event.set()
                return

            # Session-level verdict (not turn-level): a verdict-less turn-end
            # is terminal ONLY when the session has no pending continuation. A
            # turn that ended to await a dispatched sub-agent ("yield_turn") is
            # NOT a verdict — the session has more turns to go; its next turn is
            # auto-triggered by the sub-agent completion push-back
            # (on_completed -> inject_system_message -> _start_loop). Hold the
            # dispatch slot (session stays active, queue does not advance) and
            # re-await the next turn, while staying responsive to
            # stop / pause / shutdown. Everything else still in the "text"
            # bucket (genuine stalls, terminal-error paths) falls through to the
            # corrective-turn -> contract-violation path below, unchanged.
            exit_path = getattr(loop_obj, "_loop_exit_path", None)
            if self._continuation_pending(exit_path, session_id):
                hold = await self._hold_slot_for_continuation(
                    item, session_id, gen_at_start,
                )
                if hold == "resume":
                    # The session's next turn started; re-await + classify it.
                    continue
                if hold in ("paused", "shutdown"):
                    # Preserve the attempt exactly (mirror the pause guard
                    # above): no close, no advance, no rotation. resume() /
                    # teardown owns the parked session.
                    logger.info(
                        "dispatcher(%s): item %s slot-hold released (%s); "
                        "attempt preserved, no advance",
                        self._project_id, item.id, hold,
                    )
                    return
                if hold == "cancelled":
                    # Out-of-band cancel during the hold — close INTERRUPTED so
                    # the attempt can be reclaimed; no advance, no rotation
                    # (mirror the cancelled branch above).
                    self._store.close_latest_attempt(
                        item.id,
                        outcome=AttemptOutcome.INTERRUPTED,
                        block_reason="cancelled",
                    )
                    self._log_attempt_close(
                        item.id, "interrupted",
                        first_turn_signaled=False,
                        corrective_used=corrective_turn_used,
                        reason="cancelled",
                    )
                    return
                # hold == "timeout": the continuation never arrived within the
                # backstop deadline (e.g. a hung sub-agent whose push-back never
                # fired). Self-recover via the runtime-cap disposition — close
                # the attempt and BLOCK so the queue advances rather than
                # pinning the head forever.
                logger.warning(
                    "dispatcher(%s): item %s slot-hold exceeded %ss awaiting "
                    "continuation; closing BLOCKED so the queue self-recovers",
                    self._project_id, item.id, self._max_runtime_seconds,
                )
                self._store.close_latest_attempt(
                    item.id,
                    outcome=AttemptOutcome.INTERRUPTED,
                    block_reason="exceeded hold deadline awaiting continuation",
                )
                self._log_attempt_close(
                    item.id, "interrupted",
                    first_turn_signaled=False,
                    corrective_used=corrective_turn_used,
                )
                self._store.set_item_state(item.id, ItemState.BLOCKED)
                self._broadcast_advance(item.id, "interrupted")
                if self._store.auto_idle_if_empty():
                    self._broadcast_state_changed(QueueRunState.IDLE.value)
                self._idle_event.set()
                return

            # text-only on a queue item.
            # CHANGE 2: give the agent ONE corrective turn before recording
            # a contract violation. Inject a stern system reminder telling
            # the agent it must call a signal next, then restart the loop
            # on the same session and wait for the next exit.
            if not corrective_turn_used:
                logger.info(
                    "dispatcher(%s): item %s exited text-only; injecting "
                    "corrective system message and restarting loop",
                    self._project_id, item.id,
                )
                try:
                    # Seam 3 / D1 (Root D): forward the item's session id.
                    # inject_system_message is passthrough-None, so omitting it
                    # keys (pid, None) → handle-miss → "no_session" and the
                    # corrective message is silently dropped. The session this
                    # item runs in is in scope; route the message to it.
                    await self._agent_manager.inject_system_message(
                        self._project_id, self.CORRECTIVE_TURN_PROMPT,
                        session_id=session_id,
                    )
                except Exception:
                    # If inject fails we cannot ask the agent again — fall
                    # through to the violation branch. Leave the flag False
                    # so the reason text reflects the actual situation
                    # (the agent didn't get a corrective turn).
                    logger.exception(
                        "dispatcher(%s): corrective-turn injection failed; "
                        "falling through to contract-violation close",
                        self._project_id,
                    )
                else:
                    # Inject succeeded; loop is restarting. Mark the flag
                    # so a SECOND text-only exit gets the "ignored" message.
                    corrective_turn_used = True
                    continue

            # Either:
            # (a) this is the second text-only exit (corrective turn was
            #     used and ignored), or
            # (b) the corrective injection itself failed and we're now
            #     forced to record a violation.
            # Both are contract violations from the queue's perspective.
            contract_reason = (
                "agent exited without declaring outcome — contract "
                "violation (corrective turn ignored)"
                if corrective_turn_used
                else "agent exited without declaring outcome — contract violation"
            )
            self._store.close_latest_attempt(
                item.id,
                outcome=AttemptOutcome.BLOCKED,
                block_reason=contract_reason,
            )
            self._log_attempt_close(
                item.id, "violation",
                first_turn_signaled=False,
                corrective_used=corrective_turn_used,
                reason=contract_reason,
            )
            self._store.set_item_state(item.id, ItemState.BLOCKED)
            self._broadcast_advance(item.id, "blocked")
            if self._store.auto_idle_if_empty():
                self._broadcast_state_changed(QueueRunState.IDLE.value)
            self._idle_event.set()
            return

    def _log_attempt_close(
        self,
        item_id: str,
        outcome: str,
        *,
        first_turn_signaled: bool,
        corrective_used: bool,
        reason: str | None = None,
    ) -> None:
        """Structured single-line log for an attempt close.

        Pin format: every attempt produces exactly one of these lines, with
        a consistent set of fields. Lets us tail `dispatcher(.*): close `
        and extract first-turn signal rate over a session without parsing
        free-form messages.
        """
        reason_snip = (reason or "").replace("\n", " ")[:80]
        logger.info(
            "dispatcher(%s): close item=%s outcome=%s "
            "first_turn_signaled=%s corrective_used=%s reason=%r",
            self._project_id, item_id, outcome,
            first_turn_signaled, corrective_used, reason_snip,
        )

    async def _wait_for_loop_done(self, session_id: str) -> None:
        while not self._shutting_down:
            task = self._agent_manager.get_loop_task(
                self._project_id, session_id=session_id,
            )
            if task is None or task.done():
                return
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=self.LOOP_WAIT_POLL_SEC,
                )
                return
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                return

    # ------------------------------------------------------------------
    # Session-level verdict: continuation-pending slot-hold
    # ------------------------------------------------------------------

    def _continuation_pending(self, exit_path, session_id: str) -> bool:
        """Whether a verdict-less turn-end has a pending continuation — i.e.
        the session has more turns to go before a real complete/blocked
        verdict, so it must stay active in its slot rather than terminate.

        ``_exit_reason`` reads ``"text"`` for all of these, so the diagnostic
        ``_loop_exit_path`` identifies the pattern and the pattern's own
        pending signal confirms it. Currently recognizes a sub-agent dispatch
        (``yield_turn``) whose sub-agent is still live for this session — the
        same "busy" signal ``_on_loop_done`` computes (agent_manager
        ``list_active`` status ``"running"``). Its next turn arrives via the
        existing completion push-back. (``credential_request`` is a planned
        fast-follow on this same slot-hold primitive: its resume trigger needs
        its own verification before it can be carved out here.)
        """
        if exit_path == "yield_turn":
            return self._sub_agent_running(session_id)
        return False

    def _sub_agent_running(self, session_id: str) -> bool:
        """True iff a sub-agent for this session reports status ``"running"``."""
        sub_mgr = self._agent_manager.get_sub_agent_manager()
        if sub_mgr is None:
            return False
        try:
            active = sub_mgr.list_active(self._project_id, session_id=session_id)
        except Exception:
            logger.exception(
                "dispatcher(%s): sub-agent liveness probe failed; treating as "
                "no pending continuation", self._project_id,
            )
            return False
        return any((a or {}).get("status") == "running" for a in (active or []))

    async def _hold_slot_for_continuation(
        self, item: ItemRecord, session_id: str, gen_at_start: int,
    ) -> str:
        """Hold the dispatch slot while the session awaits its next turn.

        The session reached a verdict-less turn-end with a pending continuation,
        so its loop task is already ``done()`` — ``_wait_for_loop_done`` would
        return immediately and cannot be the wait. Poll for the session's NEXT
        turn (a fresh loop task installed by the push-back: on_completed ->
        inject_system_message -> _start_loop) while remaining responsive to
        stop / pause / shutdown. This runs INSIDE the ``_run`` task's call stack
        (it is awaited directly, never detached), so shutdown's
        ``self._task.cancel()`` reaches it.

        Returns one of:
          * ``"resume"``    — the next turn started; re-await + classify it.
          * ``"paused"``    — user pause/stop (``_stop_generation`` bump) or
                              stored ``QueueRunState.PAUSED``; preserve attempt.
          * ``"shutdown"``  — dispatcher teardown (``_shutting_down``).
          * ``"cancelled"`` — the loop/session was cancelled out-of-band.
          * ``"timeout"``   — the backstop deadline elapsed (continuation never
                              arrived).
        """
        parked_task = self._agent_manager.get_loop_task(
            self._project_id, session_id=session_id,
        )
        # Backstop (owner decision: included, reusing the runtime cap). If the
        # continuation never arrives — e.g. a hung sub-agent whose push-back
        # never fires — the hold expires so the queue self-recovers instead of
        # pinning the head forever. Set ``max_runtime_seconds`` falsy to rely on
        # manual stop instead.
        deadline = None
        if self._max_runtime_seconds:
            deadline = time.monotonic() + self._max_runtime_seconds

        while True:
            # --- interruptibility (mandatory) ---
            if self._shutting_down:
                return "shutdown"
            if self._stop_generation != gen_at_start:
                return "paused"
            try:
                if self._store.load().state == QueueRunState.PAUSED:
                    return "paused"
            except Exception:
                logger.exception(
                    "dispatcher(%s): queue-state read during slot-hold failed; "
                    "continuing to hold", self._project_id,
                )
            loop_obj = self._agent_manager.get_loop(
                self._project_id, session_id=session_id,
            )
            if loop_obj is not None:
                if getattr(loop_obj, "_exit_reason", None) == "cancelled":
                    return "cancelled"
                sess = getattr(loop_obj, "_session", None)
                is_stopped = getattr(sess, "is_stopped", None)
                if callable(is_stopped) and is_stopped():
                    return "cancelled"

            # --- next-turn detection ---
            # A fresh loop task (different identity from the one we parked on)
            # means the push-back installed the session's next turn.
            cur = self._agent_manager.get_loop_task(
                self._project_id, session_id=session_id,
            )
            if cur is not None and cur is not parked_task:
                return "resume"

            # --- backstop ---
            if deadline is not None and time.monotonic() >= deadline:
                return "timeout"

            await asyncio.sleep(self.HOLD_POLL_SEC)

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------

    def _broadcast_advance(self, item_id: str, outcome: str) -> None:
        if self._ws is None:
            return
        try:
            self._ws.broadcast(self._project_id, {
                "type": "queue.item_advanced",
                "project_id": self._project_id,
                "item_id": item_id,
                "outcome": outcome,
            })
        except Exception:
            pass

    def _broadcast_state_changed(self, state: str) -> None:
        if self._ws is None:
            return
        # Carry pause_reason so the frontend can distinguish a budget block from
        # a manual pause (Budget Piece 3 renders it). Read it from the store so
        # every call site stays in sync with the just-written state — it is None
        # for any non-PAUSED state by construction (set_queue_state clears it).
        try:
            pause_reason = self._store.load().pause_reason
        except Exception:
            pause_reason = None
        try:
            self._ws.broadcast(self._project_id, {
                "type": "queue.state_changed",
                "project_id": self._project_id,
                "state": state,
                "pause_reason": pause_reason,
            })
        except Exception:
            pass
