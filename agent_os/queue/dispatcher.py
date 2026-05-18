# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""QueueDispatcher — per-project queue-draining task.

The dispatcher owns one asyncio.Task per project. It pulls queued items,
runs the agent loop on the project's session, and routes on three exit
signals: complete → advance, blocked → bypass, text → stall for chat.

The dispatcher does NOT own the agent loop — AgentManager does. The
dispatcher reads loop._exit_reason after each run via AgentManager.get_loop.

Stop/resume (Phase 4): stop() terminates the live attempt's loop without
closing the attempt and swaps the active session to a dedicated per-project
chat session. resume() swaps back to the parked attempt session and starts
a fresh loop run on it. The dispatcher task stays alive across stop/resume
cycles; it just transitions between RUNNING (drain queue) and PAUSED
(idle while user chats) states. After an advance drains the last queueable
item, the store auto-transitions the queue to IDLE.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemRecord,
    ItemState,
    QueueRunState,
)
from agent_os.queue.store import QueueStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueDispatcher:
    """Owns the queue-draining asyncio.Task for one project."""

    IDLE_WAIT_TIMEOUT_SEC = 5.0
    LOOP_WAIT_POLL_SEC = 2.0
    SUB_AGENT_STOP_TIMEOUT_SEC = 10.0

    def __init__(
        self,
        project_id: str,
        store: QueueStore,
        agent_manager,
        ws_manager=None,
        max_runtime_seconds: int = 1800,
    ):
        self._project_id = project_id
        self._store = store
        self._agent_manager = agent_manager
        self._ws = ws_manager
        self._max_runtime_seconds = max_runtime_seconds
        self._task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._idle_event = asyncio.Event()
        # Per-item stall flag: a text-only exit means "pause for question",
        # so we sit on the item until the user removes it, resumes via
        # injection, or the queue is stopped.
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
        self._task = asyncio.create_task(self._run())
        logger.info("dispatcher(%s): started", self._project_id)

    def reclaim_on_startup(self) -> dict:
        """Reconcile queue state with on-disk records at daemon startup.

        Contract per D6:
        - If queue state is PAUSED: leave everything as is. The user
          previously paused the queue; the same chat-mode applies after
          restart. The dispatcher will idle until /queue/resume.
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

    async def stop(self) -> dict:
        """Pause queue draining. Switch the active session to a dedicated
        per-project chat session, terminate any live attempt loop, but do
        NOT close the in-flight attempt or rotate. The session JSONL is
        preserved so resume can pick up exactly where things left off."""
        self._stop_generation += 1
        # Set queue state first so the main loop sees PAUSED on its next tick.
        self._store.set_queue_state(QueueRunState.PAUSED)

        # Terminate the live attempt's loop, if any. AgentManager.switch_session
        # does the terminate; we additionally clear sub-agents under a budget.
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

        # Mint or reuse the chat session id, persist it, then swap.
        qstate = self._store.load()
        chat_sid = qstate.chat_session_id
        if not chat_sid:
            chat_sid = f"chat_{uuid4().hex[:12]}"
            self._store.set_chat_session_id(chat_sid)

        try:
            await self._agent_manager.switch_session(
                self._project_id, chat_sid, start_loop=False,
            )
        except Exception:
            logger.exception(
                "dispatcher(%s): switch to chat session failed",
                self._project_id,
            )

        self._broadcast_state_changed(QueueRunState.PAUSED.value)
        logger.info(
            "dispatcher(%s): paused; active session is chat_session=%s",
            self._project_id, chat_sid,
        )
        return {"status": "paused", "chat_session_id": chat_sid}

    async def resume(self) -> dict:
        """Re-activate queue draining. If a parked attempt exists, swap the
        active session back to it and start a new loop run on the same
        session. Otherwise, just set the queue back to running and let the
        main loop pick up the next queued item."""
        self._stalled_item_id = None
        self._store.set_queue_state(QueueRunState.RUNNING)

        # Find a parked attempt: head item in RUNNING with an attempts list.
        head = self._store.head()
        parked_item = head if (head and head.state == ItemState.RUNNING) else None

        if parked_item is not None and parked_item.attempts:
            session_id = parked_item.attempts[-1].session_id
            logger.info(
                "dispatcher(%s): resuming item %s on parked session %s",
                self._project_id, parked_item.id, session_id,
            )
            try:
                await self._agent_manager.switch_session(
                    self._project_id, session_id, start_loop=False,
                )
            except Exception:
                logger.exception(
                    "dispatcher(%s): switch to parked session failed",
                    self._project_id,
                )
            # Set the loop's queue_state before starting it so the approval
            # branch picks it up.
            loop_obj = self._agent_manager.get_loop(self._project_id)
            if loop_obj is not None:
                try:
                    loop_obj._queue_state = "running"
                except Exception:
                    pass
            # Spawn a handler task: it starts the loop and waits for exit.
            asyncio.create_task(self._resume_attempt(parked_item))
        else:
            # No parked attempt — kick the main loop in case items are queued.
            self._idle_event.set()

        self._broadcast_state_changed(QueueRunState.RUNNING.value)
        return {
            "status": "running",
            "resumed_item_id": parked_item.id if parked_item else None,
        }

    async def _resume_attempt(self, item: ItemRecord) -> None:
        """Re-start the agent loop on an already-parked attempt session,
        then route the eventual exit through the same handler used by
        first-time dispatch."""
        try:
            await self._agent_manager._start_loop(self._project_id)
        except Exception:
            logger.exception(
                "dispatcher(%s): _start_loop on resume failed for item %s",
                self._project_id, item.id,
            )
            return
        await self._await_and_handle(item)

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
                    if qstate.state in (
                        QueueRunState.PAUSED,
                        QueueRunState.IDLE,
                    ):
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

    # ------------------------------------------------------------------
    # Per-item dispatch
    # ------------------------------------------------------------------

    async def _dispatch_one(self, item: ItemRecord) -> None:
        session = self._agent_manager.get_session(self._project_id)
        if session is None:
            logger.warning(
                "dispatcher(%s): no agent session, waiting for one",
                self._project_id,
            )
            await self._wait_idle()
            return

        self._store.set_item_state(item.id, ItemState.RUNNING)
        attempt = AttemptRecord(session_id=session.session_id)
        self._store.append_attempt(item.id, attempt)

        logger.info(
            "dispatcher(%s): dispatching item %s (session=%s)",
            self._project_id, item.id, session.session_id,
        )

        loop_obj = self._agent_manager.get_loop(self._project_id)
        if loop_obj is not None:
            try:
                loop_obj._queue_state = "running"
            except Exception:
                pass

        try:
            await self._agent_manager.inject_message(
                self._project_id, item.content,
            )
        except Exception:
            logger.exception(
                "dispatcher(%s): inject failed for item %s",
                self._project_id, item.id,
            )
            self._store.close_latest_attempt(
                item.id,
                outcome=AttemptOutcome.INTERRUPTED,
                block_reason="inject failed",
            )
            return

        await self._await_and_handle(item)

    async def _await_and_handle(self, item: ItemRecord) -> None:
        """Wait for the in-flight loop task to finish then route the outcome
        based on AgentLoop._exit_reason. Honors the watchdog and stop signals."""
        loop_obj = self._agent_manager.get_loop(self._project_id)
        gen_at_start = self._stop_generation

        try:
            await asyncio.wait_for(
                self._wait_for_loop_done(),
                timeout=self._max_runtime_seconds,
            )
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
            # Rotate BEFORE setting state to BLOCKED so a parallel main-loop
            # tick can't pick up the next queued item while the rotation is
            # still in flight. Belt-and-suspenders with the head-RUNNING
            # guard in _run.
            await self._rotate_session_for_advance()
            self._store.set_item_state(item.id, ItemState.BLOCKED)
            self._broadcast_advance(item.id, "interrupted")
            if self._store.auto_idle_if_empty():
                self._broadcast_state_changed(QueueRunState.IDLE.value)
            self._idle_event.set()
            return

        # If stop() was called mid-attempt, the loop was terminated by
        # switch_session. We MUST NOT close the attempt — preserve it
        # exactly as it was so resume can pick up cleanly.
        if self._stop_generation != gen_at_start:
            logger.info(
                "dispatcher(%s): item %s loop terminated by stop; "
                "attempt preserved (no close, no advance, no rotation)",
                self._project_id, item.id,
            )
            return

        loop_obj = self._agent_manager.get_loop(self._project_id)
        exit_reason = getattr(loop_obj, "_exit_reason", "text") if loop_obj else "text"
        exit_summary = getattr(loop_obj, "_exit_summary", None) if loop_obj else None
        exit_block_reason = (
            getattr(loop_obj, "_exit_block_reason", None) if loop_obj else None
        )

        if exit_reason == "complete":
            self._store.close_latest_attempt(
                item.id,
                outcome=AttemptOutcome.COMPLETED,
                summary=exit_summary or "",
            )
            await self._rotate_session_for_advance()
            self._store.set_item_state(item.id, ItemState.DONE)
            logger.info(
                "dispatcher(%s): item %s completed", self._project_id, item.id,
            )
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
            await self._rotate_session_for_advance()
            self._store.set_item_state(item.id, ItemState.BLOCKED)
            logger.info(
                "dispatcher(%s): item %s blocked: %s",
                self._project_id, item.id, exit_block_reason or "(no reason)",
            )
            self._broadcast_advance(item.id, "blocked")
            if self._store.auto_idle_if_empty():
                self._broadcast_state_changed(QueueRunState.IDLE.value)
            self._idle_event.set()
            return

        # text-only → stall
        logger.info(
            "dispatcher(%s): item %s exited text-only; stalling for user input",
            self._project_id, item.id,
        )
        self._stalled_item_id = item.id

    async def _rotate_session_for_advance(self) -> None:
        try:
            await self._agent_manager.new_session(self._project_id)
        except Exception:
            logger.exception(
                "dispatcher(%s): new_session failed during advance",
                self._project_id,
            )

    async def _wait_for_loop_done(self) -> None:
        while not self._shutting_down:
            task = self._agent_manager.get_loop_task(self._project_id)
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
        try:
            self._ws.broadcast(self._project_id, {
                "type": "queue.state_changed",
                "project_id": self._project_id,
                "state": state,
            })
        except Exception:
            pass
