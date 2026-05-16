# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""QueueDispatcher — per-project queue-draining task.

Phase 1 scope: pull queued items, mark them running, dispatch one attempt
through the existing agent loop via inject_message, wait for the loop to
finish. Phase 1 has no signal detection, so after the loop returns the
dispatcher STALLS (item stays `running`, no rotation, no advance). Phase 2
wires the loop._exit_reason check that drives advance/block/stall.

The dispatcher does not own the agent loop — AgentManager does. The
dispatcher is a producer of work items into AgentManager.inject_message
and a consumer of "loop task done" signals from AgentManager._handles.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

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

    # Poll interval when no items are ready
    IDLE_WAIT_TIMEOUT_SEC = 5.0
    # Poll interval while waiting for the inner loop task to finish
    LOOP_WAIT_POLL_SEC = 2.0

    def __init__(
        self,
        project_id: str,
        store: QueueStore,
        agent_manager,
        ws_manager=None,
    ):
        self._project_id = project_id
        self._store = store
        self._agent_manager = agent_manager
        self._ws = ws_manager
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._idle_event = asyncio.Event()
        # Phase-1 stall flag: once the loop returns text-only, the dispatcher
        # parks until told otherwise. Phase 2 replaces with exit_reason check.
        self._stalled_item_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the dispatcher loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run())
        logger.info("dispatcher(%s): started", self._project_id)

    async def stop(self) -> None:
        """Stop the dispatcher task. Does NOT terminate the inner agent loop
        (AgentManager owns that). Phase 4 adds richer stop-as-pause."""
        self._stopping = True
        self._idle_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        self._task = None
        logger.info("dispatcher(%s): stopped", self._project_id)

    def notify_new_item(self) -> None:
        """Wake the dispatcher if it is sleeping. Safe to call from sync code."""
        self._idle_event.set()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            while not self._stopping:
                try:
                    qstate = self._store.load()
                    if qstate.state == QueueRunState.STOPPED:
                        await self._wait_idle()
                        continue

                    # Phase-1 stall: once an item has been dispatched and the
                    # loop returned, we don't rotate. Sit on this item until
                    # the user removes it or restarts.
                    if self._stalled_item_id is not None:
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

        # Move item into running and open an attempt
        self._store.set_item_state(item.id, ItemState.RUNNING)
        attempt = AttemptRecord(session_id=session.session_id)
        self._store.append_attempt(item.id, attempt)

        logger.info(
            "dispatcher(%s): dispatching item %s (session=%s)",
            self._project_id, item.id, session.session_id,
        )

        # Tell the loop it's draining a queue so Phase 3 approval-as-block
        # semantics activate. Default ("chat") is preserved for non-queue work.
        loop_obj = self._agent_manager.get_loop(self._project_id)
        if loop_obj is not None:
            try:
                loop_obj._queue_state = "draining"
            except Exception:
                pass

        # Inject the item content as a user message. inject_message handles
        # all three cases (idle session → hot resume, running → queue,
        # no handle → auto-start).
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

        # Wait for the loop to finish this run.
        await self._wait_for_loop_done()

        # Read exit_reason and route the outcome.
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
            self._store.set_item_state(item.id, ItemState.DONE)
            logger.info(
                "dispatcher(%s): item %s completed", self._project_id, item.id,
            )
            self._broadcast_advance(item.id, "completed")
            await self._rotate_session_for_advance()
            return

        if exit_reason == "blocked":
            self._store.close_latest_attempt(
                item.id,
                outcome=AttemptOutcome.BLOCKED,
                block_reason=exit_block_reason or "",
            )
            self._store.set_item_state(item.id, ItemState.BLOCKED)
            logger.info(
                "dispatcher(%s): item %s blocked: %s",
                self._project_id, item.id, exit_block_reason or "(no reason)",
            )
            self._broadcast_advance(item.id, "blocked")
            await self._rotate_session_for_advance()
            return

        # exit_reason == "text" (or unknown) → pause for question. Item stays
        # running, dispatcher stalls until the user takes action (resume,
        # remove, inject chat message, etc.).
        logger.info(
            "dispatcher(%s): item %s exited text-only; stalling for user input",
            self._project_id, item.id,
        )
        self._stalled_item_id = item.id

    async def _rotate_session_for_advance(self) -> None:
        """Rotate the session so the next attempt starts clean.

        Called after a completed or blocked outcome. Wraps AgentManager.
        new_session() in a guard so dispatcher failures here log and continue
        rather than killing the whole drain loop.
        """
        try:
            await self._agent_manager.new_session(self._project_id)
        except Exception:
            logger.exception(
                "dispatcher(%s): new_session failed during advance",
                self._project_id,
            )

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

    async def _wait_for_loop_done(self) -> None:
        """Poll the AgentManager handle for the current loop task to finish."""
        # Access handle through a public-ish getter to keep the coupling
        # visible. AgentManager exposes _handles directly today; we use it
        # via a small accessor on AgentManager added in Phase 1 patches.
        while not self._stopping:
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
