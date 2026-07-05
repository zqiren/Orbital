# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fanout join/gather core (spec 009, W3).

``FanoutRegistry`` owns the state a batch of parallel native-worker tasks
needs to be woken exactly once, together, regardless of how they individually
terminate (complete / error / stall / user-stop / project teardown).
``SubAgentManager.dispatch_fanout`` (Task 2) mints the group at dispatch time;
``LifecycleObserver`` (also Task 2) routes each worker's terminal event
through ``absorb_terminal`` instead of the normal per-sub-agent injection.

Concurrency note: all group-mutation code paths here are synchronous between
the "check" and the "mutate" — a single-threaded asyncio event loop makes a
plain dict/attribute mutation atomic as long as no ``await`` sits between the
check and the write. ``absorb_terminal`` is a plain (non-async) function for
exactly this reason: it must be safely callable from a sync call site
mid-observer-event without the caller needing to await it, and the
"is this the last task" check-and-flip must not yield to another task in
between.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# kind values absorb_terminal accepts from the lifecycle observer, and the
# FanoutTask.status they map to.
_TERMINAL_STATUS = {
    "completed": "completed",
    "error": "error",
    "failed": "error",
    "interrupted": "interrupted",
    "stopped": "interrupted",
}

_NON_TERMINAL_STATUSES = {"running"}


@dataclass
class FanoutTask:
    handle: str
    label: str
    brief: str
    allowed_paths: list[str] | None = None
    forbidden_paths: list[str] | None = None
    status: str = "running"        # running|completed|error|stalled|interrupted
    summary: str = ""
    transcript_path: str = ""
    last_activity: float = 0.0     # event-loop time.monotonic()
    # The worker's real chat session id (issues 2+3, round 2) — lets the
    # frontend drill-in fetch the worker's own /chat transcript and render it
    # chat-shaped instead of the flat entries view. None only for a task
    # whose adapter somehow lacks a session_uuid (defensive; every
    # NativeWorkerAdapter sets one at construction).
    session_uuid: str | None = None
    # Wall-clock ms timestamp of the terminal transition (issue 1, round 2):
    # lets the frontend freeze each row's own duration instead of sharing one
    # never-freezing batch countdown. Stamped in BOTH terminal-transition
    # sites — absorb_terminal's terminal flip and resolve_group's reap flip
    # (running -> stalled) — since a task can go terminal via either path.
    completed_at_ms: int | None = None


@dataclass
class FanoutGroup:
    fanout_id: str
    project_id: str
    session_id: str
    tasks: dict[str, FanoutTask]   # keyed by handle
    max_runtime_s: int = 3600
    stall_after_s: int = 600
    resolved: bool = False
    created_at: float = field(default_factory=time.monotonic)
    _watchdog_task: "asyncio.Task | None" = field(default=None, repr=False)


class FanoutRegistry:
    """Owns every live fanout group, keyed by ``(project_id, handle)`` for
    O(1) terminal-event routing plus ``(project_id, session_id, fanout_id)``
    for lookup/creation."""

    def __init__(self, *, inject, broadcast, stop_worker) -> None:
        """``inject(project_id, content, session_id) -> awaitable`` — the
        ``inject_system_message`` funnel. ``broadcast(project_id, payload)`` —
        WS (synchronous, matches ``WebSocketManager.broadcast``).
        ``stop_worker(project_id, handle, session_id=...) -> awaitable`` —
        stops one worker. ``session_id`` is ALWAYS passed by keyword here
        (this class calls it that way internally) so
        ``stop_worker=SubAgentManager.stop`` can be wired directly — that
        real method declares ``session_id`` keyword-only
        (``async def stop(self, project_id, handle, *, session_id=None)``);
        a positional-only stub would TypeError."""
        self._inject = inject
        self._broadcast = broadcast
        self._stop_worker = stop_worker
        # handle -> group, for O(1) terminal-event / activity routing.
        self._by_handle: dict[str, FanoutGroup] = {}
        # (project_id, session_id, fanout_id) -> group.
        self._groups: dict[tuple[str, str, str], FanoutGroup] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def create_group(self, project_id: str, session_id: str,
                     tasks: list[FanoutTask], *,
                     max_runtime_s: int) -> FanoutGroup:
        """Register a new group. ``tasks`` are expected fresh (status
        "running", already carrying handle/label/brief/transcript_path).
        The group's ``fanout_id`` is recovered from the first task's handle
        (``worker:<fanout_id>-<index>``) — every task in one dispatch shares
        the same fanout id by construction."""
        fanout_id = _fanout_id_from_handle(tasks[0].handle) if tasks else ""
        group = FanoutGroup(
            fanout_id=fanout_id,
            project_id=project_id,
            session_id=session_id,
            tasks={t.handle: t for t in tasks},
            max_runtime_s=max_runtime_s,
        )
        self._groups[(project_id, session_id, fanout_id)] = group
        for t in tasks:
            self._by_handle[t.handle] = group
        return group

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def owner_group(self, project_id: str, handle: str,
                    session_id: str | None) -> FanoutGroup | None:
        """The group owning ``handle``, or None if it isn't a fanout member
        (or belongs to a different project/session than claimed)."""
        group = self._by_handle.get(handle)
        if group is None:
            return None
        if group.project_id != project_id:
            return None
        if session_id is not None and group.session_id != session_id:
            return None
        return group

    # ------------------------------------------------------------------
    # Activity / terminal events
    # ------------------------------------------------------------------

    def note_activity(self, project_id: str, handle: str,
                      session_id: str | None) -> None:
        """Bump ``last_activity`` for ``handle``'s task. No-op for a
        non-fanout handle, an already-resolved group, or an already-terminal
        task (a stale activity ping racing resolution must not resurrect a
        task the group already gave up on)."""
        group = self.owner_group(project_id, handle, session_id)
        if group is None or group.resolved:
            return
        task = group.tasks.get(handle)
        if task is None or task.status not in _NON_TERMINAL_STATUSES:
            return
        task.last_activity = time.monotonic()

    def absorb_terminal(self, project_id: str, handle: str,
                        session_id: str | None, *, kind: str,
                        summary: str, transcript_path: str) -> bool:
        """Record a worker terminal event. Returns True when ``handle``
        belongs to a fanout group (caller must then SKIP the per-worker
        session injection but still WS-broadcast). Triggers group resolution
        when the last task lands."""
        group = self.owner_group(project_id, handle, session_id)
        if group is None:
            return False
        task = group.tasks.get(handle)
        if task is None:
            return False

        status = _TERMINAL_STATUS.get(kind, "error")
        # Idempotency: a task already terminal (e.g. a duplicate/late event
        # racing the watchdog's own stop-then-mark) must not flip status or
        # double-count toward "last task lands".
        already_terminal = task.status not in _NON_TERMINAL_STATUSES
        if not already_terminal:
            task.status = status
            task.summary = summary
            task.transcript_path = transcript_path or task.transcript_path
            task.completed_at_ms = int(time.time() * 1000)

        self._broadcast(project_id, {
            "type": "fanout.task_update",
            "project_id": project_id,
            "session_id": group.session_id,
            "fanout_id": group.fanout_id,
            "handle": handle,
            "status": task.status,
            "completed_at_ms": task.completed_at_ms,
        })

        if not already_terminal and not group.resolved:
            all_terminal = all(
                t.status not in _NON_TERMINAL_STATUSES
                for t in group.tasks.values()
            )
            if all_terminal:
                asyncio.create_task(
                    self.resolve_group(group, reason="all_completed"),
                    name=f"fanout-resolve-{group.fanout_id}",
                )
        return True

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    async def start_watchdog(self, group: FanoutGroup) -> None:
        """Spawn the background timer task that stalls silent workers and
        enforces the hard ``max_runtime_s`` ceiling. Idempotent: calling
        twice on the same group replaces the prior watchdog task (defensive
        — callers are expected to call this exactly once per group)."""
        if group._watchdog_task is not None and not group._watchdog_task.done():
            return
        group._watchdog_task = asyncio.create_task(
            self._watchdog_loop(group),
            name=f"fanout-watchdog-{group.fanout_id}",
        )

    async def _watchdog_loop(self, group: FanoutGroup) -> None:
        # Poll interval scales with the group's own stall threshold (1/20th
        # of it, clamped to [5ms, 5s]) so a test override of stall_after_s
        # gets fast detection without production runs (stall_after_s=600)
        # busy-polling every few milliseconds for up to an hour.
        interval_s = max(0.005, min(5.0, group.stall_after_s / 20))
        try:
            while not group.resolved:
                await asyncio.sleep(interval_s)
                if group.resolved:
                    return
                now = time.monotonic()
                stalled: list[FanoutTask] = []
                for task in group.tasks.values():
                    if task.status not in _NON_TERMINAL_STATUSES:
                        continue
                    if now - task.last_activity >= group.stall_after_s:
                        stalled.append(task)
                if stalled:
                    # Status flip + stop_worker for every non-terminal task
                    # (not just the ones that literally tripped the stall
                    # check — a stall resolves the WHOLE group) now happens
                    # once, inside resolve_group's own reap loop below —
                    # see that loop's docstring for why this consolidation
                    # matters (avoids double stop_worker calls).
                    await self.resolve_group(group, reason="stalled")
                    return
                if now - group.created_at >= group.max_runtime_s:
                    await self.resolve_group(group, reason="max_runtime")
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "fanout watchdog loop crashed for group %s — resolving with "
                "whatever partial state exists rather than orphaning the "
                "management session (R4)", group.fanout_id,
            )
            try:
                await self.resolve_group(group, reason="watchdog_error")
            except Exception:
                logger.exception(
                    "fanout watchdog: resolve_group also failed for %s",
                    group.fanout_id,
                )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve_group(self, group: FanoutGroup, *, reason: str) -> None:
        """Idempotent. Stops stragglers, composes ONE summary message and
        delivers it via ``inject``, broadcasts ``fanout.completed``.

        ``reason`` (all_completed|stalled|max_runtime|watchdog_error) is NOT
        part of the injected content — the join-summary format is FROZEN
        (frontend-parsed; see the format block above ``_clean_summary``) and
        has no slot for it. It's logged here for operator observability and
        still carried on the ``fanout.completed`` broadcast payload.

        CRITICAL (round-3 review): all three watchdog-internal call sites in
        ``_watchdog_loop`` call this method directly (``await
        self.resolve_group(...)``) from WITHIN ``group._watchdog_task``
        itself — i.e. this coroutine can be executing AS that very task.
        Cancelling a task from inside its own execution is a SELF-cancel:
        ``Task.cancel()`` has no ``_fut_waiter`` to cancel immediately (the
        task isn't suspended — it IS the caller), so it only sets
        ``_must_cancel`` and takes effect at the task's NEXT genuine
        suspension. Every ``await`` below this point that actually yields to
        the event loop (real ``stop_worker``/``inject`` implementations both
        do) would then have ``CancelledError`` thrown into it instead of
        resuming normally — a ``BaseException`` that blows past every
        ``except Exception`` guard here, aborting resolution mid-flight
        (join injection and/or ``fanout.completed`` never fire) and leaking
        the group (``resolved`` is already ``True``, but the cleanup at the
        bottom never runs, and nothing re-triggers it). The
        ``absorb_terminal``-triggered path (the "last task lands" case)
        does NOT have this problem — it spawns resolve_group via
        ``asyncio.create_task``, a genuinely different task than the
        watchdog's, so cancelling the watchdog there is a normal cross-task
        cancel. Guard against the self-cancel case specifically rather than
        skipping the cancel unconditionally (a live watchdog from a
        DIFFERENT task, e.g. the absorb_terminal path resolving while the
        watchdog is still polling, must still be stopped).
        """
        if group.resolved:
            return
        group.resolved = True
        logger.info(
            "fanout group %s resolving (reason=%s)", group.fanout_id, reason,
        )

        if (group._watchdog_task is not None
                and not group._watchdog_task.done()
                and group._watchdog_task is not asyncio.current_task()):
            group._watchdog_task.cancel()

        # Reap EVERY task's adapter registration here — not just stragglers.
        # A task still "running" at this point (timeout/interrupted paths;
        # the all-complete path never has one) is force-flipped to
        # "stalled" before stopping, same as before. A task that is already
        # terminal (completed/error/interrupted, recorded by
        # absorb_terminal) is stopped too, WITHOUT touching its status —
        # workers are one-shot (native_worker.py) and never reused once
        # terminal, so leaving them registered in
        # ``SubAgentManager._adapters`` forever only wastes a
        # MAX_CONCURRENT_SUBAGENTS slot (the cap check in dispatch_fanout
        # counts every registered adapter regardless of status) — this is
        # exactly the completed-fanout cap-exhaustion bug this reap closes.
        # ``stop_worker`` (``SubAgentManager.stop``) is safe to call on an
        # already-idle/terminal ``NativeWorkerAdapter``: it is the silent
        # reap path (no lifecycle/observer event fires from it), and
        # ``AgentLoop.cancel_turn()`` is an idempotent no-op when no turn is
        # in flight, so calling it here after ``_background_send`` has
        # already completed is harmless.
        for task in group.tasks.values():
            if task.status in _NON_TERMINAL_STATUSES:
                task.status = "stalled"
                task.completed_at_ms = int(time.time() * 1000)
            try:
                await self._stop_worker(
                    group.project_id, task.handle,
                    session_id=group.session_id,
                )
            except Exception:
                logger.exception(
                    "resolve_group: stop_worker failed for %s", task.handle,
                )

        # FROZEN format (frontend parses this — see fanout.py module
        # docstring reference / task-2-report.md): original task order
        # (dict insertion order from create_group, NOT re-sorted), one
        # header line + one line per task, optional trailing guidance
        # paragraph the frontend ignores.
        ordered = list(group.tasks.values())
        succeeded = sum(1 for t in ordered if t.status == "completed")
        total = len(ordered)
        lines = [f"[Fanout {group.fanout_id}] {succeeded}/{total} succeeded."]
        for t in ordered:
            text = _clean_summary(t.summary or "(no output)")
            lines.append(
                f"- [{t.status}] {t.label} ({t.handle}): {text} "
                f"| transcript: {t.transcript_path or 'unknown'}"
            )
        # Display split (UI leak fix): the header + per-task lines are what
        # the user should see; the trailing guidance paragraph is agent-facing
        # and rides only in the LLM content. The frontend renders
        # _meta.display_content when present and falls back to content for
        # legacy messages.
        display_content = "\n".join(lines)
        lines.append("")
        lines.append(
            "Synthesize these results into a single reply for the user — "
            "unlike a single sub-agent's completion, fanout results are not "
            "auto-echoed as chat bubbles, so the user has not seen this "
            "content yet."
        )
        content = "\n".join(lines)

        try:
            # session_id MUST be passed by keyword: the real collaborator
            # (AgentManager.inject_system_message) declares it keyword-only.
            # The brief's own test stub accepts it either way, but a
            # positional call here would TypeError against production wiring
            # the moment a real project used fanout end-to-end.
            await self._inject(group.project_id, content,
                               session_id=group.session_id,
                               meta={"display_content": display_content})
        except Exception:
            # Guard failures (Task-2 brief teardown requirement): a dead
            # session must not raise out of resolution — the group is
            # already marked resolved above, so this never retries into a
            # duplicate injection.
            logger.warning(
                "resolve_group: inject failed for fanout %s (project=%s "
                "session=%s) — the group is still marked resolved",
                group.fanout_id, group.project_id, group.session_id,
                exc_info=True,
            )

        self._broadcast(group.project_id, {
            "type": "fanout.completed",
            "project_id": group.project_id,
            "session_id": group.session_id,
            "fanout_id": group.fanout_id,
            "succeeded": succeeded,
            "total": total,
            "reason": reason,
            "tasks": [
                {
                    "handle": t.handle,
                    "label": t.label,
                    "status": t.status,
                    "transcript_path": t.transcript_path,
                    "completed_at_ms": t.completed_at_ms,
                    "session_uuid": t.session_uuid,
                }
                for t in ordered
            ],
        })

        # Free the routing entries — a resolved group's handles are dead
        # weight for absorb_terminal/note_activity lookups.
        for handle in group.tasks:
            self._by_handle.pop(handle, None)
        self._groups.pop(
            (group.project_id, group.session_id, group.fanout_id), None,
        )


def _fanout_id_from_handle(handle: str) -> str:
    """Recover the fanout id from a ``worker:<fanout_id>-<index>`` handle.
    Mirrors ``native_worker._fanout_id_from_handle`` (kept local/duplicated
    rather than imported — this module must not depend on native_worker's
    internals, only on the handle string shape both sides agree on)."""
    body = handle.split(":", 1)[-1]
    fanout_id, _, _ = body.rpartition("-")
    return fanout_id or body


_SUMMARY_LIMIT = 200


def _clean_summary(text: str) -> str:
    """Per the FROZEN join-summary format (team-lead spec): first 200 chars,
    newlines replaced with spaces, so a multi-line worker summary/error
    can't break the one-line-per-task contract the frontend parses."""
    return text.replace("\n", " ").replace("\r", " ")[:_SUMMARY_LIMIT]
