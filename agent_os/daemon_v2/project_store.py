# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""JSON-file-based project store.

Runtime fields (``runtime.budget_spent_usd`` in particular) are written on
every LLM call, which made the per-turn ``_save()`` call the dominant disk
write of the daemon. Track C3 (perf): batch runtime writes in memory and
flush them to ``projects.json`` on a fixed interval, on graceful shutdown,
or when an accumulated dollar threshold is crossed.

Semantics are unchanged: ``get_project`` always returns the freshest in-memory
value, so reads after writes are consistent even before a flush hits disk.
Only the disk-write cadence changes.
"""

import asyncio
import json
import logging
import os
import re
import threading
from uuid import uuid4



logger = logging.getLogger(__name__)


DEFAULT_NOTIFICATION_PREFS = {
    "task_completed": True,
    "errors": True,
    "agent_messages": True,
    "trigger_started": False,
}

# Default value for fields that are merged into every ``get_project`` response
# so legacy records missing the key behave as if it were explicitly ``False``.
# Keep this flat (top-level) — it is not a runtime-managed field; it persists
# across daemon restarts and records an architectural decision about whether
# the project has ever been reconciled with the bundled default skills.
DEFAULT_PROJECT_FIELDS: dict = {
    "default_skills_reconciled": False,
    "sub_agent_deployment_instructions": "",
    # Spec 082 — the credential card this project runs on. None means "follow
    # the global default card"; ``migration_note`` carries a one-line reason
    # the UI shows above the card picker (card_incomplete / needs_card /
    # card_deleted). Both default here so a legacy record reads as "on the
    # default, nothing to explain" without a migration pass of its own.
    "card_id": None,
    "migration_note": None,
}

# Pre-cards per-project LLM fields (spec 082 §3.7 step 5). They still LOAD —
# the migration reads them — but never survive a save: a project row must not
# be able to carry a plaintext key again, and a stale model/base_url snapshot
# is exactly what the card lookup exists to delete.
_LEGACY_LLM_FIELDS = ("api_key", "model", "provider", "base_url", "sdk")

# Queue runtime cap defaults. Applied lazily when reading
# get_max_runtime_seconds so legacy projects use the default without a
# migration.
DEFAULT_MAX_RUNTIME_SECONDS = 1800  # 30 min
MIN_MAX_RUNTIME_SECONDS = 60
MAX_MAX_RUNTIME_SECONDS = 86400  # 24h


# Default cadence: flush dirty runtime state to disk every 10 seconds.
# Worst-case data-loss window on hard crash is bounded by this.
DEFAULT_FLUSH_INTERVAL_SECONDS = 10.0

# Default threshold: if accumulated unflushed spend for any single project
# crosses this dollar value, flush synchronously. Bounds data loss in
# dollar terms in addition to time terms — useful for high-cost models
# where 10 seconds of activity could be several dollars.
DEFAULT_BUDGET_FLUSH_THRESHOLD_USD = 0.50


class ProjectStore:
    """CRUD for projects using JSON files on disk.

    Runtime fields (e.g. ``budget_spent_usd``) are write-batched: kept in
    memory and flushed to disk on a timer, on shutdown, or on a threshold
    crossing. See module docstring.
    """

    def __init__(
        self,
        data_dir: str,
        flush_interval_seconds: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        budget_flush_threshold_usd: float = DEFAULT_BUDGET_FLUSH_THRESHOLD_USD,
    ):
        self._data_dir = data_dir
        self._projects_file = os.path.join(data_dir, "projects.json")
        self._projects: dict[str, dict] = {}
        self._load()

        # Batching configuration
        self._flush_interval = flush_interval_seconds
        self._budget_flush_threshold = budget_flush_threshold_usd

        # In-memory pending runtime updates per project. Keyed by project_id,
        # value is the merged "runtime" dict awaiting a flush. We mirror the
        # update into self._projects immediately so reads are consistent;
        # this dict tracks WHAT needs to land on disk.
        self._pending_runtime: dict[str, dict] = {}

        # Telemetry: counts since last successful flush (per project) and
        # cumulative count of timer ticks that found nothing to flush.
        self._runtime_updates_since_flush: dict[str, int] = {}
        self._skipped_flushes: int = 0

        # Baseline of budget_spent_usd that's been confirmed-on-disk.
        # Used to compute the unflushed delta for the threshold check.
        # Populated lazily on first update_runtime for that project.
        self._disk_budget_baseline: dict[str, float] = {}

        # Sync lock guards mutation of the in-memory dicts above. update_runtime
        # is called from inside the agent loop's async run() (synchronously,
        # from the same event-loop thread), but flush_runtime may also be
        # invoked from a background asyncio task or a test thread, so we
        # protect the dict mutations with a threading lock. Disk I/O happens
        # outside the lock to avoid holding it across a syscall.
        self._lock = threading.Lock()

        # Background flush task handle (set by start_flush_task).
        self._flush_task: asyncio.Task | None = None
        self._flush_stop_event: asyncio.Event | None = None

    def _agent_name_taken(self, agent_name: str, exclude_pid: str | None = None) -> bool:
        """Check if agent_name is already used by another project."""
        for pid, proj in self._projects.items():
            if pid == exclude_pid:
                continue
            if proj.get("agent_name", proj.get("name", "")) == agent_name:
                return True
        return False

    def _load(self) -> None:
        if os.path.exists(self._projects_file):
            with open(self._projects_file, "r", encoding="utf-8") as f:
                self._projects = json.load(f)
        else:
            self._projects = {}

    def _save(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        with open(self._projects_file, "w", encoding="utf-8") as f:
            json.dump(self._projects, f, indent=2, ensure_ascii=False)

    def list_projects(self) -> list[dict]:
        return list(self._projects.values())

    def get_project(self, project_id: str) -> dict | None:
        project = self._projects.get(project_id)
        if project is not None:
            prefs = project.get("notification_prefs", {})
            project["notification_prefs"] = {**DEFAULT_NOTIFICATION_PREFS, **prefs}
            # Merge defaults for top-level project fields so legacy records
            # surface the expected False/zero/empty default without requiring
            # an explicit migration step.
            for key, default_value in DEFAULT_PROJECT_FIELDS.items():
                project.setdefault(key, default_value)
            # Per-project connector enablement (Spec 011 §2). Kept out of the
            # DEFAULT_PROJECT_FIELDS loop because that loop's setdefault would
            # alias one shared list across every legacy project; the literal
            # here creates a fresh list on first access instead.
            project.setdefault("enabled_connectors", [])
        return project

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Strip characters that are problematic in filenames."""
        s = re.sub(r'[^\w\s-]', '', name)
        return s.strip()

    def create_project(self, config: dict) -> str:
        pid = "proj_" + uuid4().hex[:12]
        # Telemetry counter (spec 046 §4). Scratch is auto-created on first
        # run — counting it would make every fresh install look like a user
        # action, so only real projects count.
        if not config.get("is_scratch"):
            from agent_os import telemetry

            telemetry.emit("project_created")
            telemetry.latch("first_project")
        # Sanitize project name to avoid filesystem issues
        raw_name = config.get("name", "")
        if raw_name:
            config["name"] = self._sanitize_name(raw_name)
        # Default agent_name to project name if not provided
        config.setdefault("agent_name", config.get("name", ""))
        # Default is_scratch to False
        config.setdefault("is_scratch", False)
        # Validate agent_name uniqueness
        if self._agent_name_taken(config["agent_name"]):
            raise ValueError(f"agent_name '{config['agent_name']}' already in use")
        project = {
            "project_id": pid,
            **{k: v for k, v in config.items() if k not in _LEGACY_LLM_FIELDS},
        }
        self._projects[pid] = project
        self._save()
        return pid

    def update_project(self, project_id: str, updates: dict) -> None:
        project = self._projects.get(project_id)
        if project is None:
            return
        # Validate agent_name uniqueness if being changed
        if "agent_name" in updates and self._agent_name_taken(updates["agent_name"], exclude_pid=project_id):
            raise ValueError(f"agent_name '{updates['agent_name']}' already in use")
        # Partial merge for notification_prefs
        if "notification_prefs" in updates and isinstance(updates["notification_prefs"], dict):
            existing = project.get("notification_prefs", {})
            updates["notification_prefs"] = {**DEFAULT_NOTIFICATION_PREFS, **existing, **updates["notification_prefs"]}
        project.update(updates)
        # Drop the legacy dead config on save (TASK-network-config-cleanup):
        # old records load fine, but the key does not survive a save round-trip.
        project.pop("network_extra_domains", None)
        for field in _LEGACY_LLM_FIELDS:
            project.pop(field, None)
        self._save()

    def reorder_projects(self, ordered_ids: list[str]) -> list[str]:
        """Stamp a manual display order onto the listed projects (spec 056).

        Writes ``sort_key = <position>`` for every id in ``ordered_ids`` in a
        single pass followed by ONE ``_save()``, so the whole order lands
        atomically — a partial write can never strand two projects sharing a
        key. Idempotent, and there is nothing to janitor when a project is
        deleted: the keys of the survivors stay valid (merely non-contiguous)
        and the next reorder re-densifies them.

        Ids that no longer exist are skipped. Projects NOT named in
        ``ordered_ids`` are left untouched: they keep whatever key they had,
        or stay keyless — and a keyless project sorts to the bottom in
        creation order (see ``_project_sort_key`` in the list route). Callers
        are expected to send the complete list they are ordering; sending a
        subset can collide positions with untouched projects, which the list
        route's stable sort then breaks by creation order.

        Returns the ids that were actually stamped.
        """
        applied: list[str] = []
        for position, project_id in enumerate(ordered_ids):
            project = self._projects.get(project_id)
            if project is None:
                continue
            project["sort_key"] = position
            applied.append(project_id)
        self._save()
        return applied

    def find_scratch_project(self) -> dict | None:
        """Return the scratch project if one exists."""
        for proj in self._projects.values():
            if proj.get("is_scratch"):
                return proj
        return None

    def update_runtime(self, project_id: str, updates: dict) -> None:
        """Update runtime-managed fields (separate from user-editable config).

        Stored under a "runtime" key in the project dict. Unlike user-facing
        updates, this does NOT immediately rewrite ``projects.json`` — the
        change lands in memory and is flushed asynchronously. ``get_project``
        sees the new value immediately; disk consistency is restored on the
        next flush tick, on graceful shutdown, or when an accumulated
        ``budget_spent_usd`` delta crosses the configured threshold.

        Callers that need synchronous on-disk persistence should call
        ``flush_runtime()`` after the update.
        """
        must_flush_now = False
        with self._lock:
            project = self._projects.get(project_id)
            if project is None:
                return
            # Mirror into the live project dict so reads are consistent.
            runtime = project.setdefault("runtime", {})
            runtime.update(updates)
            # Track in pending state for next flush.
            self._pending_runtime.setdefault(project_id, {}).update(updates)
            # Increment telemetry counter.
            self._runtime_updates_since_flush[project_id] = (
                self._runtime_updates_since_flush.get(project_id, 0) + 1
            )
            # Threshold check: only triggered by budget_spent_usd updates.
            if "budget_spent_usd" in updates:
                # Baseline = whatever is currently on disk. Lazily seed from
                # the loaded value the first time we see this project.
                if project_id not in self._disk_budget_baseline:
                    # On first update for this project, seed the baseline from
                    # whatever is currently on disk (0.0 if no prior runtime
                    # block). This bounds the threshold check against actual
                    # disk state, not against the in-memory in-flight state.
                    self._disk_budget_baseline[project_id] = (
                        self._load_disk_baseline(project_id)
                    )
                delta = abs(
                    float(updates["budget_spent_usd"])
                    - self._disk_budget_baseline[project_id]
                )
                if delta >= self._budget_flush_threshold:
                    must_flush_now = True

        if must_flush_now:
            self.flush_runtime()

    def _load_disk_baseline(self, project_id: str) -> float:
        """Read the on-disk budget value for this project at load time.

        Used to seed the threshold-check baseline. If the file is fresh
        (no runtime block) the baseline is 0.0.
        """
        # Re-read from a snapshot of what _load saw — that's exactly what
        # the constructor put into self._projects, so we don't need to hit
        # disk again here. But if the project was just created in-process
        # and immediately had update_runtime called, no flush has happened
        # yet, so the on-disk view is also 0.0.
        # We approximate "on-disk budget" by reading projects.json.
        try:
            if os.path.exists(self._projects_file):
                with open(self._projects_file, "r", encoding="utf-8") as f:
                    on_disk = json.load(f)
                return float(
                    on_disk.get(project_id, {})
                    .get("runtime", {})
                    .get("budget_spent_usd", 0.0)
                )
        except (OSError, ValueError, KeyError):
            pass
        return 0.0

    def flush_runtime(self) -> None:
        """Persist any pending runtime updates to ``projects.json``.

        Idempotent: if no updates are pending, increments the
        ``_skipped_flushes`` counter and returns without touching disk.
        Safe to call from any thread.
        """
        # Snapshot pending state under the lock, then release it before
        # doing disk I/O. The lock protects dict mutation only.
        with self._lock:
            if not self._pending_runtime:
                self._skipped_flushes += 1
                return
            # Compute the new baseline for each project being flushed.
            new_baselines: dict[str, float] = {}
            for pid, pending in self._pending_runtime.items():
                if "budget_spent_usd" in pending:
                    new_baselines[pid] = float(pending["budget_spent_usd"])
            project_count = len(self._pending_runtime)
            update_count = sum(self._runtime_updates_since_flush.values())
            # Clear pending + counters BEFORE the write. The in-memory
            # self._projects already holds the merged state from
            # update_runtime, so _save() will write it.
            self._pending_runtime.clear()
            self._runtime_updates_since_flush.clear()
            for pid, baseline in new_baselines.items():
                self._disk_budget_baseline[pid] = baseline

        # Disk I/O outside the lock.
        try:
            self._save()
            logger.info(
                "project_store: flushed runtime updates "
                "(projects=%d, updates=%d, skipped_total=%d)",
                project_count, update_count, self._skipped_flushes,
            )
        except Exception:
            logger.exception("project_store: failed to flush runtime updates")
            # Best-effort: do not re-queue, since self._projects has the
            # merged state and the next update will trigger another flush.
            # A persistent disk failure should surface in the logs.

    async def start_flush_task(self) -> None:
        """Spawn the periodic flush task on the running event loop.

        Idempotent: a second call is a no-op if the task is already running.
        Must be called from inside an asyncio event loop (e.g. a FastAPI
        startup handler).
        """
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_stop_event = asyncio.Event()
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info(
            "project_store: started flush task (interval=%.1fs, threshold=$%.2f)",
            self._flush_interval, self._budget_flush_threshold,
        )

    async def stop_flush_task(self) -> None:
        """Stop the periodic flush task and drain pending state.

        Graceful-shutdown contract: any in-memory updates land on disk
        before this returns (modulo disk errors, which are logged).
        Idempotent.
        """
        if self._flush_stop_event is not None:
            self._flush_stop_event.set()
        task = self._flush_task
        self._flush_task = None
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("project_store: flush task did not stop in time; cancelling")
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        # Final drain even if the task was already dead.
        self.flush_runtime()

    async def _flush_loop(self) -> None:
        """Background task body: flush on every interval until stopped."""
        assert self._flush_stop_event is not None
        try:
            while not self._flush_stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._flush_stop_event.wait(),
                        timeout=self._flush_interval,
                    )
                    # If we get here, stop was signalled — exit loop.
                    break
                except asyncio.TimeoutError:
                    # Normal path: interval elapsed without a stop signal.
                    pass
                try:
                    self.flush_runtime()
                except Exception:
                    logger.exception("project_store: flush_loop iteration failed")
        except asyncio.CancelledError:
            # Outer cancel — re-raise so awaiters see it.
            raise

    def delete_project(self, project_id: str) -> None:
        if self._projects.get(project_id, {}).get("is_scratch"):
            raise ValueError("cannot delete scratch project")
        # Flush any pending runtime updates for this project before deleting
        # so we don't strand updates in memory pointing at a removed key.
        with self._lock:
            self._pending_runtime.pop(project_id, None)
            self._runtime_updates_since_flush.pop(project_id, None)
            self._disk_budget_baseline.pop(project_id, None)
        self._projects.pop(project_id, None)
        self._save()

    def get_max_runtime_seconds(self, project_id: str) -> int:
        """Return the per-attempt runtime cap. Falls back to default."""
        project = self._projects.get(project_id) or {}
        value = project.get("max_runtime_seconds")
        if not isinstance(value, int):
            return DEFAULT_MAX_RUNTIME_SECONDS
        return max(MIN_MAX_RUNTIME_SECONDS, min(MAX_MAX_RUNTIME_SECONDS, value))
