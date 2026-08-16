# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""TriggerManager — schedule and file-watch trigger execution for Agent OS.

Manages trigger lifecycle: loads triggers from project configs on startup,
registers cron-based schedules and watchdog file observers, fires triggers
by calling agent_manager.start_agent(), and updates trigger state
(last_triggered, trigger_count) in project config.
"""

import asyncio
import fnmatch
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4

from croniter import croniter
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


def generate_trigger_id() -> str:
    """Generate a unique trigger ID."""
    return "trg_" + uuid4().hex[:8]


def validate_watch_path(watch_path: str, workspace: str) -> str | None:
    """Validate that watch_path resolves within workspace. Returns error or None."""
    resolved = os.path.realpath(os.path.join(workspace, watch_path))
    workspace_real = os.path.realpath(workspace)
    if not resolved.startswith(workspace_real + os.sep) and resolved != workspace_real:
        return f"watch_path '{watch_path}' resolves outside workspace"
    return None


def validate_trigger(trigger: dict, workspace: str | None = None) -> str | None:
    """Validate a trigger dict. Returns error string or None if valid."""
    if not trigger.get("name"):
        return "Trigger name is required"
    ttype = trigger.get("type")
    if ttype not in ("schedule", "file_watch"):
        return f"Invalid trigger type: {ttype}. Must be 'schedule' or 'file_watch'"
    if ttype == "schedule":
        schedule = trigger.get("schedule")
        if not schedule or not schedule.get("cron"):
            return "Schedule trigger requires schedule.cron"
        cron = schedule["cron"]
        if not croniter.is_valid(cron):
            return f"Invalid cron expression: {cron}"
    if ttype == "file_watch":
        if not trigger.get("watch_path"):
            return "file_watch trigger requires watch_path"
        if workspace:
            path_error = validate_watch_path(trigger["watch_path"], workspace)
            if path_error:
                return path_error
    if not trigger.get("task"):
        return "Trigger task is required"
    return None


class TriggerManager:
    """Manages scheduled triggers for all projects.

    On startup, loads all triggers from all projects, adds enabled schedule
    triggers to a single periodic tick loop, and starts watchdog observers
    for file_watch triggers. When a trigger fires, calls
    agent_manager.start_agent() with the task as initial_message.

    Schedule triggers are evaluated by one tick loop (`TICK_INTERVAL_SECONDS`)
    rather than a per-trigger `asyncio.sleep(delay_to_next_occurrence)` timer.
    Per-trigger timers run on the monotonic clock, which freezes across macOS
    system sleep — a daily trigger armed before a sleep fires late by the
    accumulated sleep duration, and a multi-day timer (e.g. weekly) can miss
    its occurrence entirely across a sleep+restart. The tick loop instead
    re-evaluates wall-clock due-ness from scratch every tick, so a missed
    occurrence is caught up (and coalesced into a single fire) the next time
    the loop runs — see `_evaluate_due_triggers`.
    """

    TICK_INTERVAL_SECONDS = 60

    def __init__(self, project_store, agent_manager, ws_manager=None, now_fn=None):
        self._project_store = project_store
        self._agent_manager = agent_manager
        self._ws = ws_manager
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        # Instance attribute (not the class constant) so tests can shrink it.
        self.tick_interval_seconds = self.TICK_INTERVAL_SECONDS
        self._schedule_ids: set[str] = set()  # trigger_ids evaluated each tick
        self._held: set[str] = set()  # trigger_ids currently holding on agent_busy
        self._file_observers: dict[str, Observer] = {}
        self._debounce_timers: dict[str, asyncio.TimerHandle] = {}
        self._debounce_buffers: dict[str, list[str]] = {}
        self._trigger_project: dict[str, str] = {}  # trigger_id → project_id
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tick_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Load all triggers from all projects, register the tick loop's
        schedule set, start file-watch observers, and start the tick loop."""
        self._running = True
        self._loop = asyncio.get_running_loop()
        projects = self._project_store.list_projects()
        schedule_count = 0
        file_watch_count = 0
        for project in projects:
            triggers = project.get("triggers", [])
            project_id = project["project_id"]
            for trigger in triggers:
                if not trigger.get("enabled", True):
                    continue
                ttype = trigger.get("type")
                if ttype == "schedule":
                    if self._register_schedule(project_id, trigger):
                        schedule_count += 1
                elif ttype == "file_watch":
                    self._start_file_watch(project_id, trigger)
                    file_watch_count += 1
        logger.info(
            "TriggerManager started: %d schedule + %d file_watch triggers registered",
            schedule_count, file_watch_count,
        )
        # The tick loop runs an immediate catch-up evaluation before its
        # first sleep (see _tick_loop), so daemon-restart catch-up needs no
        # special-casing here.
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        """Cancel the tick loop, stop all file observers, and shut down."""
        self._running = False
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        self._schedule_ids.clear()
        self._held.clear()
        for trigger_id in list(self._file_observers):
            self._stop_file_watch(trigger_id)
        # Cancel any pending debounce timers
        for handle in self._debounce_timers.values():
            handle.cancel()
        self._debounce_timers.clear()
        self._debounce_buffers.clear()
        self._trigger_project.clear()
        logger.info("TriggerManager stopped")

    def register_trigger(
        self, project_id: str, trigger: dict, *, broadcast: bool = True
    ) -> None:
        """Register a single trigger (called after create/update).

        ``broadcast=False`` arms the scheduler/observer silently — for callers
        that announce the change themselves (an edit is a ``trigger.updated``,
        not a ``trigger.created``; see ``apply_trigger_update``).
        """
        trigger_id = trigger["id"]
        # Drop any existing registration/observer first (idempotent re-register).
        # Silent: a re-register is not a deletion, and announcing one made every
        # UI list drop the row.
        self.unregister_trigger(trigger_id, broadcast=False)
        self._trigger_project[trigger_id] = project_id
        if trigger.get("enabled", True):
            ttype = trigger.get("type")
            if ttype == "schedule":
                self._register_schedule(project_id, trigger)
            elif ttype == "file_watch":
                self._start_file_watch(project_id, trigger)
        # Broadcast creation event for real-time UI updates
        if broadcast:
            self._broadcast(project_id, {
                "type": "trigger.created",
                "project_id": project_id,
                "trigger": trigger,
            })

    def unregister_trigger(self, trigger_id: str, *, broadcast: bool = True) -> None:
        """Unregister a trigger (called after delete or disable).

        Removes it from the tick loop's evaluated set and clears any
        held-streak tracking. Also stops file-watch observers.

        ``broadcast=False`` disarms without announcing a delete — used when the
        record still exists (disable / re-register).
        """
        self._schedule_ids.discard(trigger_id)
        self._held.discard(trigger_id)
        self._stop_file_watch(trigger_id)
        # Broadcast deletion event
        project_id = self._trigger_project.pop(trigger_id, None)
        if project_id and broadcast:
            self._broadcast(project_id, {
                "type": "trigger.deleted",
                "project_id": project_id,
                "trigger_id": trigger_id,
            })

    def apply_trigger_update(self, project_id: str, trigger: dict) -> None:
        """Re-arm an EDITED trigger and announce it as ``trigger.updated``.

        ``trigger.created``/``trigger.deleted`` mean exactly what they say —
        a record appeared or went away. Enable/disable and field edits are
        neither: they used to travel as created/deleted (because the toggle
        route called register/unregister), which made a disabled automation
        vanish from every live list until the next refetch. Callers that
        mutate an existing record use this instead.

        ``register_trigger`` is idempotent and only arms when ``enabled`` — so
        it correctly disarms a trigger that was just switched off.
        """
        self.register_trigger(project_id, trigger, broadcast=False)
        self._broadcast(project_id, {
            "type": "trigger.updated",
            "project_id": project_id,
            "trigger": trigger,
        })

    # ---- File-watch observer lifecycle ----

    def _start_file_watch(self, project_id: str, trigger: dict) -> None:
        """Create and start a watchdog Observer for a file_watch trigger."""
        trigger_id = trigger["id"]
        watch_path = trigger.get("watch_path", "")
        patterns = trigger.get("patterns", [])
        recursive = trigger.get("recursive", False)
        debounce_seconds = trigger.get("debounce_seconds", 5)

        # Resolve watch_path relative to workspace
        project = self._project_store.get_project(project_id)
        if project is None:
            logger.warning("Cannot start file_watch %s: project %s not found", trigger_id, project_id)
            return
        workspace = project.get("workspace", "")
        abs_path = os.path.realpath(os.path.join(workspace, watch_path))

        # Security: verify within workspace
        workspace_real = os.path.realpath(workspace)
        if not abs_path.startswith(workspace_real + os.sep) and abs_path != workspace_real:
            logger.warning("file_watch %s: path '%s' outside workspace, skipping", trigger_id, watch_path)
            return

        # Create directory if it doesn't exist
        os.makedirs(abs_path, exist_ok=True)

        handler = _DebouncedHandler(
            trigger_id=trigger_id,
            project_id=project_id,
            patterns=patterns,
            debounce_seconds=debounce_seconds,
            trigger_manager=self,
        )
        observer = Observer()
        observer.schedule(handler, abs_path, recursive=recursive)
        observer.start()
        self._file_observers[trigger_id] = observer
        self._trigger_project[trigger_id] = project_id
        logger.info(
            "file_watch %s started: watching '%s' (patterns=%s, recursive=%s, debounce=%ds)",
            trigger_id, abs_path, patterns or ["*"], recursive, debounce_seconds,
        )

    def _stop_file_watch(self, trigger_id: str) -> None:
        """Stop and clean up a file-watch observer."""
        observer = self._file_observers.pop(trigger_id, None)
        if observer is not None:
            observer.stop()
            observer.join(timeout=5)
            logger.info("file_watch %s stopped", trigger_id)
        # Clean up debounce state
        handle = self._debounce_timers.pop(trigger_id, None)
        if handle is not None:
            handle.cancel()
        self._debounce_buffers.pop(trigger_id, None)

    def _on_file_event(self, trigger_id: str, project_id: str, file_path: str,
                       debounce_seconds: int) -> None:
        """Called from watchdog handler thread. Buffers events and schedules debounce."""
        if not self._running or self._loop is None:
            return
        # Buffer the changed file
        buf = self._debounce_buffers.setdefault(trigger_id, [])
        buf.append(file_path)
        # Reset debounce timer (must schedule on event loop thread)
        self._loop.call_soon_threadsafe(
            self._reset_debounce, trigger_id, project_id, debounce_seconds,
        )

    def _reset_debounce(self, trigger_id: str, project_id: str, debounce_seconds: int) -> None:
        """Reset the debounce timer for a trigger (runs on event loop thread)."""
        existing = self._debounce_timers.pop(trigger_id, None)
        if existing is not None:
            existing.cancel()
        handle = self._loop.call_later(
            debounce_seconds,
            self._debounce_flush, trigger_id, project_id,
        )
        self._debounce_timers[trigger_id] = handle

    def _debounce_flush(self, trigger_id: str, project_id: str) -> None:
        """Flush debounce buffer and fire the trigger (runs on event loop thread)."""
        self._debounce_timers.pop(trigger_id, None)
        changed_files = self._debounce_buffers.pop(trigger_id, [])
        if changed_files:
            # Deduplicate
            changed_files = list(dict.fromkeys(changed_files))
            asyncio.ensure_future(
                self._fire_trigger(project_id, trigger_id, changed_files=changed_files)
            )

    # ---- Schedule tick-loop lifecycle ----

    def _register_schedule(self, project_id: str, trigger: dict) -> bool:
        """Add a schedule trigger to the tick loop's evaluated set.

        Idempotent: registering the same trigger_id again is a no-op on set
        membership. The tick loop re-reads schedule/cron/last_triggered fresh
        from project_store on every evaluation (see _evaluate_due_triggers),
        so there is no cached per-trigger state here to go stale or duplicate.
        Returns False (and logs+skips, same as before) for an invalid cron.
        """
        trigger_id = trigger["id"]
        cron = trigger.get("schedule", {}).get("cron")
        if not cron or not croniter.is_valid(cron):
            logger.warning("Skipping trigger %s: invalid cron '%s'", trigger_id, cron)
            return False
        self._trigger_project[trigger_id] = project_id
        self._schedule_ids.add(trigger_id)
        return True

    async def _tick_loop(self) -> None:
        """Periodic tick: evaluate all due schedule triggers, then sleep.

        The first evaluation runs immediately (before the first sleep) so a
        trigger missed during machine sleep or a daemon restart is caught up
        promptly instead of waiting for its next natural occurrence.
        """
        while self._running:
            try:
                await self._evaluate_due_triggers()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error evaluating due triggers")
            try:
                await asyncio.sleep(self.tick_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _evaluate_due_triggers(self) -> None:
        """Evaluate every registered schedule trigger and fire the due ones.

        Due rule (per trigger, in the trigger's own timezone): prev_occ = the
        most recent cron occurrence strictly before now. The trigger is due
        iff prev_occ > max(last_triggered, created_at) — a missing/unparsable
        last_triggered counts as never-fired, and a trigger created after an
        occurrence must not catch up on that pre-creation occurrence. Firing
        sets last_triggered = now (in _fire_trigger), which subsumes any
        older missed occurrences into this one catch-up fire.

        Firing is serialized per project (at most one fire per project per
        tick, in prev_occ order, tie-broken by created_at then id) and
        awaited strictly sequentially — no create_task fan-out — so there is
        no check-then-start race between simultaneous triggers. A due
        trigger whose project agent is already running is held (not fired,
        last_triggered untouched) and re-checked next tick; "trigger.held" is
        broadcast only on the first held tick of a streak.

        A schedule trigger whose project (or whose trigger record within
        that project) no longer exists is unregistered on the spot — neither
        the delete-project API path nor a direct project_store edit that
        drops a trigger ever calls unregister_trigger, so this evaluation is
        the only janitor for that state. A disabled trigger is left
        registered (just skipped) so re-enabling it later needs no fresh
        register_trigger call.
        """
        import pytz

        now = self._now_fn()
        due: list[tuple[datetime, str, str, str, dict]] = []

        for trigger_id in list(self._schedule_ids):
            project_id = self._trigger_project.get(trigger_id)
            if project_id is None:
                continue
            project = self._project_store.get_project(project_id)
            if project is None:
                logger.info(
                    "Trigger %s: project %s no longer exists, unregistering",
                    trigger_id, project_id,
                )
                self.unregister_trigger(trigger_id)
                continue
            trigger = next(
                (t for t in project.get("triggers", []) if t.get("id") == trigger_id),
                None,
            )
            if trigger is None:
                logger.info(
                    "Trigger %s: no longer present in project %s, unregistering",
                    trigger_id, project_id,
                )
                self.unregister_trigger(trigger_id)
                continue
            if not trigger.get("enabled", True):
                continue
            schedule = trigger.get("schedule", {})
            cron_expr = schedule.get("cron")
            if not cron_expr or not croniter.is_valid(cron_expr):
                continue

            tz_name = schedule.get("timezone", "UTC")
            try:
                tz = pytz.timezone(tz_name)
            except pytz.UnknownTimeZoneError:
                logger.warning(
                    "Trigger %s: unknown timezone '%s', falling back to UTC",
                    trigger_id, tz_name,
                )
                tz = pytz.UTC

            try:
                prev_occ = croniter(cron_expr, now.astimezone(tz)).get_prev(datetime)
            except Exception:
                logger.exception(
                    "Trigger %s: failed computing previous occurrence", trigger_id
                )
                continue
            if prev_occ.tzinfo is None:
                prev_occ = tz.localize(prev_occ)

            baseline = self._trigger_baseline(trigger)
            if prev_occ <= baseline:
                # Not due (or no longer due) — clear any held-streak tracking.
                self._held.discard(trigger_id)
                continue

            due.append((prev_occ, trigger.get("created_at") or "", trigger_id, project_id, trigger))

        # Deterministic order: earliest missed occurrence first, tie-broken
        # by creation time then id.
        due.sort(key=lambda item: (item[0], item[1], item[2]))

        fired_projects: set[str] = set()
        for prev_occ, _created_at, trigger_id, project_id, trigger in due:
            if project_id in fired_projects:
                # Another trigger already used this project's one-fire-per-tick
                # slot; this one stays due and is picked up next tick.
                logger.debug(
                    "Trigger %s: deferring to next tick (project %s already fired this tick)",
                    trigger_id, project_id,
                )
                continue
            if self._agent_manager.is_running(project_id):
                if trigger_id not in self._held:
                    self._held.add(trigger_id)
                    self._broadcast(project_id, {
                        "type": "trigger.held",
                        "project_id": project_id,
                        "trigger_id": trigger_id,
                        "trigger_name": trigger.get("name", trigger_id),
                        "reason": "agent_busy",
                        "timestamp": now.isoformat(),
                    })
                else:
                    logger.debug("Trigger %s: still held (agent busy)", trigger_id)
                continue
            self._held.discard(trigger_id)
            try:
                await self._fire_trigger(project_id, trigger_id)
            except Exception:
                # _fire_trigger's own try/except only wraps start_agent —
                # the project_store.update_project stamping above it can
                # still raise, and _fire_trigger rolls that stamp back on
                # its own live trigger object before re-raising. Don't let
                # one trigger's failure starve every trigger sorted after it
                # this tick; it stays genuinely due and is retried next tick.
                logger.exception(
                    "Trigger %s: unhandled error firing for project %s",
                    trigger_id, project_id,
                )
            fired_projects.add(project_id)

    @staticmethod
    def _parse_iso(value) -> datetime | None:
        """Parse an ISO-8601 timestamp string; return None if missing/unparsable."""
        if not value or not isinstance(value, str):
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _trigger_baseline(self, trigger: dict) -> datetime:
        """The due-rule baseline: max(last_triggered, created_at).

        Missing/unparsable last_triggered is treated as never-fired (falls
        back to created_at). If both are missing/unparsable, the baseline is
        the epoch, so the trigger is due as soon as it has any occurrence.
        A last_triggered in the future (clock skew) is honored as-is — it
        simply raises the bar so nothing looks due.
        """
        candidates = [
            parsed for parsed in (
                self._parse_iso(trigger.get("last_triggered")),
                self._parse_iso(trigger.get("created_at")),
            )
            if parsed is not None
        ]
        if not candidates:
            return datetime.min.replace(tzinfo=timezone.utc)
        return max(candidates)

    async def _fire_trigger(self, project_id: str, trigger_id: str,
                             changed_files: list[str] | None = None) -> None:
        """Execute a trigger: start the agent with the trigger's task."""
        project = self._project_store.get_project(project_id)
        if project is None:
            logger.warning("Trigger %s: project %s not found, unregistering", trigger_id, project_id)
            self.unregister_trigger(trigger_id)
            return

        triggers = project.get("triggers", [])
        trigger = next((t for t in triggers if t.get("id") == trigger_id), None)
        if trigger is None:
            logger.warning("Trigger %s not found in project %s", trigger_id, project_id)
            self.unregister_trigger(trigger_id)
            return

        if not trigger.get("enabled", True):
            logger.debug("Trigger %s is disabled, skipping", trigger_id)
            return

        trigger_name = trigger.get("name", trigger_id)
        human_schedule = trigger.get("schedule", {}).get("human", "")

        # Check if agent is already running BEFORE updating state
        if self._agent_manager.is_running(project_id):
            logger.info("Trigger %s: agent already running for project %s, skipping", trigger_id, project_id)
            self._broadcast(project_id, {
                "type": "trigger.skipped",
                "project_id": project_id,
                "trigger_id": trigger_id,
                "trigger_name": trigger_name,
                "reason": "agent_busy",
                "timestamp": self._now_fn().isoformat(),
            })
            return

        task_content = trigger.get("task", "")
        trigger_type = trigger.get("type", "schedule")

        # Build initial message with trigger context
        if trigger_type == "file_watch" and changed_files:
            files_str = ", ".join(changed_files[:20])  # Cap at 20 filenames
            if len(changed_files) > 20:
                files_str += f" (and {len(changed_files) - 20} more)"
            initial_message = (
                f"[Triggered by file_watch '{trigger_name}']\n\n"
                f"Changed files: {files_str}\n\n{task_content}"
            )
        else:
            initial_message = (
                f"[Triggered by schedule '{trigger_name}'"
                + (f" ({human_schedule})" if human_schedule else "")
                + f"]\n\n{task_content}"
            )

        # Update trigger state only when we will actually fire. trigger is a
        # live reference into project_store's backing dict (get_project()
        # does not return a copy), so this stamp is visible immediately —
        # including to the next tick's evaluation — regardless of whether
        # update_project() below actually persists it. If it raises (e.g. a
        # disk write failure), roll the stamp back on this same live object
        # before re-raising: otherwise the trigger would look "already
        # fired" in memory forever, even though nothing was persisted and
        # the agent below was never started, starving it until its next
        # natural cron occurrence instead of being retried next tick.
        old_last_triggered = trigger.get("last_triggered")
        old_trigger_count = trigger.get("trigger_count", 0)
        now_iso = self._now_fn().isoformat()
        trigger["last_triggered"] = now_iso
        trigger["trigger_count"] = old_trigger_count + 1
        try:
            self._project_store.update_project(project_id, {"triggers": triggers})
        except Exception:
            trigger["last_triggered"] = old_last_triggered
            trigger["trigger_count"] = old_trigger_count
            raise

        # Start the agent
        try:
            from agent_os.daemon_v2.models import AgentConfig
            from agent_os.agent.prompt_builder import Autonomy

            # Always use project-level autonomy (triggers inherit from project)
            autonomy_str = project.get("autonomy", "hands_off")
            try:
                autonomy = Autonomy(autonomy_str)
            except ValueError:
                autonomy = Autonomy.HANDS_OFF

            # Resolve API key / model / base_url through the same fallback
            # chain as inject_message: project → credential store → global settings
            settings_store = getattr(self._agent_manager, '_settings_store', None)
            credential_store = getattr(self._agent_manager, '_credential_store', None)
            global_settings = settings_store.get() if settings_store else None
            cred_key = credential_store.get_api_key() if credential_store else None
            api_key = (
                project.get("api_key")
                or cred_key
                or (global_settings.llm.api_key if global_settings else None)
                or ""
            )
            base_url = project.get("base_url") or (
                global_settings.llm.base_url if global_settings else None
            )
            model = (
                project.get("model")
                or (global_settings.llm.model if global_settings else None)
                or ""
            )

            # Compute available sub-agents from setup_engine, minus the
            # project's disabled_sub_agents denylist. Legacy ``enabled_sub_agents``
            # is informational-only in v1.
            disabled = set(project.get("disabled_sub_agents", []) or [])
            setup_engine = getattr(self._agent_manager, '_setup_engine', None)
            if setup_engine is not None:
                available = setup_engine.check_all()
                trigger_enabled_sub_agents = [
                    a.slug for a in available
                    if a.installed and a.slug != "built-in"
                    and a.slug not in disabled
                ]
            else:
                trigger_enabled_sub_agents = [
                    s for s in (project.get("enabled_sub_agents") or [])
                    if s not in disabled
                ]

            config = AgentConfig(
                workspace=project["workspace"],
                model=model,
                api_key=api_key,
                base_url=base_url,
                autonomy=autonomy,
                sdk=project.get("sdk", "openai"),
                provider=project.get("provider", "custom"),
                project_name=project.get("name", ""),
                project_instructions=project.get("instructions", ""),
                sub_agent_deployment_instructions=(project.get(
                    "sub_agent_deployment_instructions", ""
                ) or ""),
                is_scratch=project.get("is_scratch", False),
                agent_name=project.get("agent_name", project.get("name", "")),
                enabled_sub_agents=trigger_enabled_sub_agents,
                disabled_sub_agents=list(disabled),
                budget_limit_usd=project.get("budget_limit_usd"),
                budget_action=project.get("budget_action", "pause"),
            )
            await self._agent_manager.start_agent(
                project_id, config,
                initial_message=initial_message,
                trigger_source=trigger_type,
                trigger_name=trigger_name,
            )
            logger.info("Trigger %s fired: started agent for project %s", trigger_id, project_id)

            self._broadcast(project_id, {
                "type": "trigger.fired",
                "project_id": project_id,
                "trigger_id": trigger_id,
                "trigger_name": trigger_name,
                "timestamp": now_iso,
            })
        except ValueError as e:
            # Single-slot guard (start_agent): another session holds the
            # project's active-loop slot, so this trigger is blocked this cycle.
            # Not an error — skip quietly without broadcasting trigger.fired.
            logger.info("Trigger %s: not started — %s", trigger_id, e)
        except Exception as e:
            logger.exception("Trigger %s: failed to start agent for project %s", trigger_id, project_id)
            # Surface the failure to the UI — this path was log-only, so a
            # trigger with a missing/invalid API key failed silently forever.
            from agent_os.daemon_v2.provider_errors import classify_llm_error
            code, message = classify_llm_error(e)
            self._broadcast(project_id, {
                "type": "agent.status",
                "project_id": project_id,
                "status": "error",
                "reason": message,
                "error_code": code,
                "source": "trigger",
                "trigger_id": trigger_id,
                "trigger_name": trigger_name,
                "timestamp": self._now_fn().isoformat(),
            })

    def _broadcast(self, project_id: str, event: dict) -> None:
        """Broadcast a WebSocket event if ws_manager is available."""
        if self._ws is not None:
            self._ws.broadcast(project_id, event)


class _DebouncedHandler(FileSystemEventHandler):
    """Watchdog event handler that debounces file events through TriggerManager."""

    def __init__(self, trigger_id: str, project_id: str, patterns: list[str],
                 debounce_seconds: int, trigger_manager: TriggerManager):
        super().__init__()
        self._trigger_id = trigger_id
        self._project_id = project_id
        self._patterns = patterns
        self._debounce_seconds = debounce_seconds
        self._tm = trigger_manager

    def _matches_patterns(self, path: str) -> bool:
        """Check if a file path matches any of the configured glob patterns."""
        if not self._patterns:
            return True  # No patterns = match all files
        basename = os.path.basename(path)
        return any(fnmatch.fnmatch(basename, pat) for pat in self._patterns)

    def on_any_event(self, event):
        """Called by watchdog on any filesystem event."""
        # Only react to file events (not directory events)
        if event.is_directory:
            return
        src = getattr(event, "src_path", "")
        if not src or not self._matches_patterns(src):
            return
        self._tm._on_file_event(
            self._trigger_id, self._project_id, src, self._debounce_seconds,
        )
