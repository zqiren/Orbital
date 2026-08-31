# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pinned-chat consolidation triggers (spec 074 §3.5).

While a chat session is pinned to a sub-agent the management LLM takes zero
turns, so the manager-loop consolidation triggers (turn count, token
pressure, session end) never run. This module owns the two replacement
triggers that keep Layer-1 tidy for pinned exchanges:

1. **Unpin / retarget** — the sessions PATCH route calls ``trigger()``
   whenever the dropdown moves away from the current worker. Always fires;
   nothing user-facing awaits it (the incoming worker is briefed by the
   transcript recap, which does not depend on this pass).
2. **Quiescence** — every pinned terminal event (completed, errored,
   stopped, …) starts/resets a timer (default 600s) via
   ``note_pinned_terminal`` (wired to ``LifecycleObserver``'s
   ``pinned_terminal_hook``); a new pinned dispatch cancels it via
   ``note_pinned_dispatch``. On expiry with no new dispatch the pass fires
   once per quiet period. Covers long pins at natural pauses and abandoned
   sessions — deliberately NO trigger on session close/delete.

Per-completion passes were rejected by design (every pinned message is a
dispatch — that would be one LLM pass per exchange). The pass itself is
``run_session_end_routine`` — already daemon-invokable: it needs only a
session, a provider, and a ``WorkspaceFileManager``, none of which are
loop-bound — run as a single-flight detached task with a dirty flag: a
trigger landing during a running pass coalesces into exactly one re-run.

The input window is "messages since the last pass" (``since_index``),
tracked in memory per session key; a daemon restart resets it to the whole
tail, mirroring the in-memory idempotency guard the routine already uses.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Default quiet period before a consolidation fires (spec 074: 10 minutes).
DEFAULT_QUIESCENCE_S = 600.0


class PinnedConsolidationCoordinator:
    """Owns the quiescence timers, single-flight passes, and dirty flags.

    One instance per daemon, constructed and wired in
    ``agents_v2.configure``. All entry points are synchronous, cheap, and
    never raise — a consolidation trigger must never break a dispatch or a
    PATCH.
    """

    def __init__(self, agent_manager, project_store, *,
                 quiescence_s: float = DEFAULT_QUIESCENCE_S):
        self._agent_manager = agent_manager
        self._project_store = project_store
        self._quiescence_s = quiescence_s
        # (project_id, session_id) -> pending quiescence timer task.
        self._timers: dict[tuple[str, str | None], asyncio.Task] = {}
        # Keys with a pass currently running (single-flight).
        self._running: set[tuple[str, str | None]] = set()
        # Keys re-triggered while running — exactly one re-run after.
        self._dirty: set[tuple[str, str | None]] = set()
        # (project_id, session_id) -> message count at the START of the last
        # pass; the next pass distills only messages after it.
        self._last_pass_index: dict[tuple[str, str | None], int] = {}
        # Strong refs to detached pass tasks (asyncio keeps only weak ones).
        self._pass_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Trigger entry points
    # ------------------------------------------------------------------

    def note_pinned_dispatch(self, project_id: str,
                             session_id: str | None) -> None:
        """A new pinned dispatch — the exchange is active; cancel the timer."""
        self._cancel_timer((project_id, session_id))

    def note_pinned_terminal(self, project_id: str,
                             session_id: str | None) -> None:
        """A pinned dispatch reached a terminal state — start/reset the timer."""
        key = (project_id, session_id)
        self._cancel_timer(key)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop (sync test harness) — quiescence simply disarmed.
            return
        task = loop.create_task(self._quiescence_wait(key))
        self._timers[key] = task

    def trigger(self, project_id: str, session_id: str | None,
                reason: str) -> None:
        """Fire a consolidation pass now, as a detached single-flight task.

        Used by the unpin/retarget PATCH (always) and by the quiescence
        timer on expiry. A trigger during a running pass sets the dirty flag
        for one re-run instead of stacking passes.
        """
        key = (project_id, session_id)
        self._cancel_timer(key)
        if key in self._running:
            self._dirty.add(key)
            logger.info(
                "pinned consolidation for %s/%s already running "
                "(trigger=%s) — marked dirty for one re-run",
                project_id, session_id, reason,
            )
            return
        self._running.add(key)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._running.discard(key)
            logger.warning(
                "pinned consolidation trigger (%s) for %s/%s outside an "
                "event loop — skipped", reason, project_id, session_id,
            )
            return
        task = loop.create_task(self._run_guarded(key, reason))
        self._pass_tasks.add(task)
        task.add_done_callback(self._pass_tasks.discard)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cancel_timer(self, key: tuple[str, str | None]) -> None:
        task = self._timers.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    async def _quiescence_wait(self, key: tuple[str, str | None]) -> None:
        try:
            await asyncio.sleep(self._quiescence_s)
        except asyncio.CancelledError:
            return
        self._timers.pop(key, None)
        self.trigger(key[0], key[1], reason="quiescence")

    async def _run_guarded(self, key: tuple[str, str | None],
                           reason: str) -> None:
        """Run one pass; on completion honor the dirty flag with one re-run.

        The caller has already claimed the ``_running`` slot for ``key``.
        """
        try:
            await self._run_pass(key, reason)
        except Exception:
            logger.exception(
                "pinned consolidation pass failed for %s/%s (trigger=%s)",
                key[0], key[1], reason,
            )
        finally:
            self._running.discard(key)
        if key in self._dirty:
            self._dirty.discard(key)
            self.trigger(key[0], key[1], reason=f"{reason}+dirty")

    def _resolve_session(self, project_id: str, session_id: str | None):
        """Live handle session if hydrated, else the on-disk JSONL."""
        session = self._agent_manager.get_session(
            project_id, session_id=session_id)
        if session is not None:
            return session
        if session_id is None:
            return None
        return self._agent_manager._load_session_from_disk(
            project_id, session_id)

    async def _run_pass(self, key: tuple[str, str | None],
                        reason: str) -> None:
        from agent_os.agent.workspace_files import (
            WorkspaceFileManager,
            run_session_end_routine,
        )

        project_id, session_id = key
        session = self._resolve_session(project_id, session_id)
        if session is None:
            logger.info(
                "pinned consolidation (%s): no session for %s/%s — nothing "
                "to consolidate", reason, project_id, session_id,
            )
            return

        config = self._agent_manager._build_agent_config_from_project(
            project_id)
        provider, _fallbacks, utility_provider, _info = (
            self._agent_manager._build_llm_providers(config)
        )
        workspace_files = WorkspaceFileManager(config.workspace)

        # Window start captured BEFORE the (long) LLM pass: rows appended
        # while it runs belong to the NEXT window.
        start_count = len(session.get_messages())
        since = self._last_pass_index.get(key)
        logger.info(
            "pinned consolidation pass starting for %s/%s (trigger=%s, "
            "window=%s..%s)", project_id, session_id, reason,
            since or 0, start_count,
        )
        outcome = await run_session_end_routine(
            session=session,
            provider=provider,
            workspace_files=workspace_files,
            utility_provider=utility_provider,
            session_uuid=session.session_uuid,
            bypass_idempotency=True,
            project_id=project_id,
            since_index=since,
            pinned_exchange=True,
        )
        self._last_pass_index[key] = start_count
        logger.info(
            "pinned consolidation pass done for %s/%s: %s",
            project_id, session_id, outcome,
        )
