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
import concurrent.futures
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

    On startup, loads all triggers from all projects and registers
    asyncio timers for enabled schedule triggers. When a trigger fires,
    calls agent_manager.start_agent() with the task as initial_message.
    """

    def __init__(self, project_store, agent_manager, ws_manager=None):
        self._project_store = project_store
        self._agent_manager = agent_manager
        self._ws = ws_manager
        self._timers: dict[str, asyncio.Task | concurrent.futures.Future] = {}
        self._file_observers: dict[str, Observer] = {}
        self._debounce_timers: dict[str, asyncio.TimerHandle] = {}
        self._debounce_buffers: dict[str, list[str]] = {}
        self._trigger_project: dict[str, str] = {}  # trigger_id → project_id
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        """Load all triggers from all projects and register timers/observers."""
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
                    self._register_timer(project_id, trigger)
                    schedule_count += 1
                elif ttype == "file_watch":
                    self._start_file_watch(project_id, trigger)
                    file_watch_count += 1
        logger.info(
            "TriggerManager started: %d schedule + %d file_watch triggers registered",
            schedule_count, file_watch_count,
        )

    async def stop(self) -> None:
        """Cancel all timers, stop all file observers, and shut down."""
        self._running = False
        for trigger_id, handle in self._timers.items():
            handle.cancel()
        self._timers.clear()
        for trigger_id in list(self._file_observers):
            self._stop_file_watch(trigger_id)
        # Cancel any pending debounce timers
        for handle in self._debounce_timers.values():
            handle.cancel()
        self._debounce_timers.clear()
        self._debounce_buffers.clear()
        self._trigger_project.clear()
        logger.info("TriggerManager stopped")

    def register_trigger(self, project_id: str, trigger: dict) -> None:
        """Register a single trigger (called after create/update)."""
        trigger_id = trigger["id"]
        # Cancel existing timer/observer if any
        self.unregister_trigger(trigger_id)
        self._trigger_project[trigger_id] = project_id
        if trigger.get("enabled", True):
            ttype = trigger.get("type")
            if ttype == "schedule":
                self._register_timer(project_id, trigger)
            elif ttype == "file_watch":
                self._start_file_watch(project_id, trigger)
        # Broadcast creation event for real-time UI updates
        self._broadcast(project_id, {
            "type": "trigger.created",
            "project_id": project_id,
            "trigger": trigger,
        })

    def unregister_trigger(self, trigger_id: str) -> None:
        """Unregister a trigger (called after delete or disable).

        Handles both asyncio.Task (from event loop thread) and
        concurrent.futures.Future (from run_coroutine_threadsafe).
        Also stops file-watch observers.
        """
        handle = self._timers.pop(trigger_id, None)
        if handle is not None:
            handle.cancel()
        self._stop_file_watch(trigger_id)
        # Broadcast deletion event
        project_id = self._trigger_project.pop(trigger_id, None)
        if project_id:
            self._broadcast(project_id, {
                "type": "trigger.deleted",
                "project_id": project_id,
                "trigger_id": trigger_id,
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

    # ---- Schedule timer lifecycle ----

    def _register_timer(self, project_id: str, trigger: dict) -> None:
        """Create an asyncio task that sleeps until next cron time, then fires.

        Thread-safe: uses run_coroutine_threadsafe when called from a non-event-loop
        thread (e.g. tool execute via asyncio.to_thread), or asyncio.create_task when
        called from the event loop thread directly.
        """
        trigger_id = trigger["id"]
        schedule = trigger.get("schedule", {})
        cron = schedule.get("cron")
        tz_name = schedule.get("timezone", "UTC")
        if not cron or not croniter.is_valid(cron):
            logger.warning("Skipping trigger %s: invalid cron '%s'", trigger_id, cron)
            return

        # Cancel existing timer before creating new one (idempotent)
        existing = self._timers.pop(trigger_id, None)
        if existing is not None:
            existing.cancel()

        coro = self._timer_loop(project_id, trigger_id, cron, tz_name)

        # Detect whether we're in the event loop thread or a worker thread
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            # Called from event loop thread (e.g. during start() or direct call)
            self._timers[trigger_id] = asyncio.create_task(coro)
        elif self._loop is not None:
            # Called from a worker thread (e.g. via asyncio.to_thread)
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            self._timers[trigger_id] = future
        else:
            logger.warning(
                "Trigger %s: no event loop captured, cannot schedule timer", trigger_id
            )

    async def _timer_loop(self, project_id: str, trigger_id: str,
                          cron_expr: str, tz_name: str = "UTC") -> None:
        """Loop: compute next fire time in the trigger's timezone, sleep, fire, repeat."""
        import pytz
        try:
            tz = pytz.timezone(tz_name)
        except pytz.UnknownTimeZoneError:
            logger.warning("Trigger %s: unknown timezone '%s', falling back to UTC", trigger_id, tz_name)
            tz = pytz.UTC

        while self._running:
            try:
                now = datetime.now(tz)
                cron = croniter(cron_expr, now)
                next_fire = cron.get_next(datetime)
                # Convert to UTC for delay calculation
                if next_fire.tzinfo is None:
                    next_fire = tz.localize(next_fire)
                now_utc = datetime.now(timezone.utc)
                delay = (next_fire.astimezone(timezone.utc) - now_utc).total_seconds()
                if delay < 0:
                    delay = 0
                logger.debug(
                    "Trigger %s: next fire in %.0f seconds at %s (%s)",
                    trigger_id, delay, next_fire.isoformat(), tz_name
                )
                await asyncio.sleep(delay)
                await self._fire_trigger(project_id, trigger_id)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in timer loop for trigger %s", trigger_id)
                await asyncio.sleep(60)  # Back off on error

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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

        # Update trigger state only when we will actually fire
        now_iso = datetime.now(timezone.utc).isoformat()
        trigger["last_triggered"] = now_iso
        trigger["trigger_count"] = trigger.get("trigger_count", 0) + 1
        self._project_store.update_project(project_id, {"triggers": triggers})

        # Start the agent
        try:
            from agent_os.daemon_v2.models import AgentConfig
            from agent_os.agent.prompt_builder import Autonomy

            # Resolve autonomy: trigger override or project default
            autonomy_str = trigger.get("autonomy") or project.get("autonomy", "hands_off")
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
                is_scratch=project.get("is_scratch", False),
                agent_name=project.get("agent_name", project.get("name", "")),
                enabled_sub_agents=project.get("enabled_sub_agents", []),
                budget_limit_usd=project.get("budget_limit_usd"),
                budget_action=project.get("budget_action", "ask"),
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
        except Exception:
            logger.exception("Trigger %s: failed to start agent for project %s", trigger_id, project_id)

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
