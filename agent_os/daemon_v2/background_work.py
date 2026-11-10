# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Background-work registry — provenance-based classification (Piece 3 Part A).

Classifies a sub-agent's child processes into REAL background work vs warm
infrastructure. The source of truth is PROVENANCE: the consume loop registers
a record when it sees a ``tool_use`` chunk whose
``metadata.tool_input.run_in_background`` is true. Liveness is anchored to the
spawned subtree via a cross-platform psutil direct-child diff captured around
the registration moment.

Explicitly NOT used for classification (all three tested and FAILED —
``docs/investigations/REPORT-piece3-child-classification.md``):
- "any live child" — warm infrastructure (MCP servers, browsers) is present
  from the agent's first second and must read idle;
- CPU activity — a sleeping background job shows 0%, identical to idle infra;
- creation-time-vs-turn — infrastructure is also born mid-turn (lazy browser
  launch) and mutates on its own (updater spawn/zombie).

Anchor mechanism (cross-platform by construction, Part A Windows gate):
at registration we snapshot the agent root's direct children (the
infrastructure baseline); a capture task then polls for a NEW direct child —
that subtree is the work anchor, tracked as a ``psutil.Process`` (identity-
checked by create-time, so PID reuse reads dead, not alive). A child matching
claude-code's ``~/.claude/shell-snapshots`` launcher pattern is adopted at
once; failing that, a new child that outlasts the capture window is. The
pattern is a REFINEMENT, never the sole mechanism.

WINDOWS GATE — OPENED 2026-11-02 (spec 147 Phase 2): ``three_state_supported``
is True on every platform, win32 included. The gate was opened on the strength
of nine of nine Windows run_in_background shells tracked correctly in field
feedback bundles (2026-10-28 / 2026-11-01). If the Windows process shape turns
out wrong somewhere, the failure mode is display only: the status dot stays
green after the shell ends or never lights for it — not a crash, and eviction
stays descendant-gated so no work is killed. The instance attribute remains so
tests (and a future re-close) can set it False for the two-state path.

Known accepted limitation (do not chase): work detached via raw foreground
``&``/``nohup`` reparents away from the agent tree and is invisible to both
provenance and any tree walk. It equally escapes the reap kill.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

# claude-code's Bash tool launches background jobs under a shell sourcing a
# snapshot file at this path fragment (observed live, Git Bash included). The
# bootstrap that WRITES the snapshot quotes a native path, so on Windows only
# the launcher matches.
_LAUNCHER_FRAGMENT = ".claude/shell-snapshots/"

Key = tuple[str, str, str]  # (project_id, session_id, handle)


@dataclass
class BackgroundWorkRecord:
    command: str
    registered_at: float
    anchor: "psutil.Process | None" = None
    capture_task: "asyncio.Task | None" = field(default=None, repr=False)

    def capture_pending(self) -> bool:
        return self.anchor is None and self.capture_task is not None \
            and not self.capture_task.done()

    def is_live(self) -> bool:
        """Live = anchored subtree root still running (identity-checked), or
        anchor capture still in flight (bounded by the capture window)."""
        if self.anchor is not None:
            try:
                return self.anchor.is_running()
            except psutil.Error:
                return False
        return self.capture_pending()


class BackgroundWorkRegistry:
    """Tracks registered background work per (project_id, session_id, handle)."""

    def __init__(self, capture_window_s: float = 6.0,
                 capture_poll_s: float = 0.25,
                 watch_poll_s: float = 1.0):
        self.capture_window_s = capture_window_s
        self.capture_poll_s = capture_poll_s
        # How often a key with live work is re-checked so its END is pushed
        # (liveness is otherwise only read on demand).
        self.watch_poll_s = watch_poll_s
        # listener(project_id, session_id, handle): the key's work went live
        # or ended. SubAgentManager recomputes the worker status from it.
        self._listener = None
        self._watchers: dict[Key, asyncio.Task] = {}
        # Windows gate, opened 2026-11-02 (see module docstring). Instance
        # attr so tests can exercise the degraded two-state path.
        self.three_state_supported = True
        self._records: dict[Key, list[BackgroundWorkRecord]] = {}
        self._last_activity: dict[Key, float] = {}

    # ------------------------------------------------------------- write ----
    def register(self, project_id: str, session_id: str, handle: str, *,
                 command: str, root_proc: "psutil.Process | None") -> BackgroundWorkRecord:
        """Register provenance for a run_in_background launch and start the
        anchor capture against ``root_proc``'s direct children."""
        key = (project_id, session_id, handle)
        rec = BackgroundWorkRecord(command=command, registered_at=time.monotonic())
        self._records.setdefault(key, []).append(rec)
        self._last_activity[key] = time.monotonic()

        if root_proc is not None:
            try:
                baseline = {c.pid for c in root_proc.children()}
            except psutil.Error:
                baseline = set()
            baseline |= self._claimed_pids(key)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                rec.capture_task = loop.create_task(
                    self._capture_anchor(key, rec, root_proc, baseline),
                    name=f"bgwork-capture-{handle}",
                )
        logger.info(
            "background work registered for %s/%s/%s: %r (anchor capture %s)",
            project_id, session_id, handle, command[:80],
            "started" if rec.capture_task else "unavailable",
        )
        self._notify(key)
        self._ensure_watcher(key)
        return rec

    # ------------------------------------------------------------ push ----
    def set_listener(self, listener) -> None:
        """``listener(project_id, session_id, handle)`` is called when a
        key's background work goes live or ends."""
        self._listener = listener

    def _notify(self, key: Key) -> None:
        if self._listener is None:
            return
        try:
            self._listener(*key)
        except Exception:
            logger.exception("background work listener raised for %s", key)

    def _ensure_watcher(self, key: Key) -> None:
        task = self._watchers.get(key)
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._watchers[key] = loop.create_task(
            self._watch(key), name=f"bgwork-watch-{key[2]}")

    async def _watch(self, key: Key) -> None:
        """Re-check a key's work until none is live, then tell the
        listener once: the end of background work is a status change that
        nothing else observes."""
        try:
            while True:
                await asyncio.sleep(self.watch_poll_s)
                if not self._records.get(key):
                    break  # cleared / terminated — that path notified
                if not self._prune(key):
                    self._notify(key)
                    break
        finally:
            if self._watchers.get(key) is asyncio.current_task():
                self._watchers.pop(key, None)

    async def _capture_anchor(self, key: Key, rec: BackgroundWorkRecord,
                              root_proc: "psutil.Process", baseline: set[int]) -> None:
        """Adopt the NEW direct child of the agent root that is the work: the
        shell-snapshots launcher as soon as it appears, else whichever new
        child outlasts the window. A first-come pick would take the CLI's own
        snapshot bootstrap (``bash -lc echo "$PATH"``), which exits within a
        second, and the worker would read idle while its shell runs."""
        deadline = time.monotonic() + self.capture_window_s
        while True:
            expired = time.monotonic() >= deadline
            try:
                candidates = [
                    c for c in root_proc.children()
                    if c.pid not in baseline and c.pid not in self._claimed_pids(key)
                ]
            except psutil.Error:
                return  # root gone — nothing to anchor
            chosen = next((c for c in candidates if self._is_launcher(c)), None)
            if chosen is None and expired and candidates:
                chosen = candidates[0]
            if chosen is not None:
                rec.anchor = chosen
                self._last_activity[key] = time.monotonic()
                logger.info(
                    "background work anchored for %s: pid=%s cmd=%r",
                    key, chosen.pid, rec.command[:80],
                )
                return
            if expired:
                break
            await asyncio.sleep(self.capture_poll_s)
        logger.info(
            "background work anchor capture expired for %s cmd=%r "
            "(work likely finished within the window)", key, rec.command[:80],
        )

    @staticmethod
    def _is_launcher(proc: "psutil.Process") -> bool:
        try:
            return any(_LAUNCHER_FRAGMENT in part for part in proc.cmdline())
        except psutil.Error:
            return False

    def _claimed_pids(self, key: Key) -> set[int]:
        return {r.anchor.pid for r in self._records.get(key, [])
                if r.anchor is not None}

    # -------------------------------------------------------------- read ----
    def _prune(self, key: Key) -> list[BackgroundWorkRecord]:
        recs = self._records.get(key, [])
        live = [r for r in recs if r.is_live()]
        if len(live) != len(recs):
            self._records[key] = live
            self._last_activity[key] = time.monotonic()
        return live

    def has_live(self, project_id: str, session_id: str, handle: str) -> bool:
        key = (project_id, session_id, handle)
        live = self._prune(key)
        if live:
            self._last_activity[key] = time.monotonic()
        return bool(live)

    def live_commands(self, project_id: str, session_id: str, handle: str) -> list[str]:
        return [r.command for r in self._prune((project_id, session_id, handle))]

    def records(self, project_id: str, session_id: str, handle: str) -> list[BackgroundWorkRecord]:
        """Raw records (no pruning) — provenance log, incl. finished work."""
        return list(self._records.get((project_id, session_id, handle), []))

    def last_background_activity(self, project_id: str, session_id: str,
                                 handle: str) -> float | None:
        """Monotonic timestamp of the most recent observed background-work
        activity for this key (registration, anchoring, observed-live,
        transition-to-dead). The eviction timer resets on this (Part B)."""
        return self._last_activity.get((project_id, session_id, handle))

    # ----------------------------------------------------------- teardown ----
    async def terminate_all(self, project_id: str, session_id: str, handle: str,
                            *, grace_s: float = 3.0) -> list[str]:
        """Kill all live anchored subtrees for this key and CONFIRM death.
        Returns the commands whose work was terminated (for loud loss
        reporting — Parts D/E). Raw ``&``-detached work is out of reach by
        design (accepted limitation)."""
        key = (project_id, session_id, handle)
        terminated: list[str] = []
        for rec in self._prune(key):
            if rec.capture_task is not None and not rec.capture_task.done():
                rec.capture_task.cancel()
            if rec.anchor is None:
                continue
            try:
                victims = [rec.anchor] + rec.anchor.children(recursive=True)
            except psutil.Error:
                continue
            for p in victims:
                try:
                    p.terminate()
                except psutil.Error:
                    pass
            _, alive = psutil.wait_procs(victims, timeout=grace_s)
            for p in alive:
                try:
                    p.kill()
                except psutil.Error:
                    pass
            _, still_alive = psutil.wait_procs(alive, timeout=grace_s)
            if still_alive:
                logger.error(
                    "background work for %s cmd=%r NOT confirmed dead "
                    "(pids=%s)", key, rec.command[:80],
                    [p.pid for p in still_alive],
                )
            else:
                terminated.append(rec.command)
        self._records.pop(key, None)
        self._last_activity[key] = time.monotonic()
        self._notify(key)
        return terminated

    def clear(self, project_id: str, session_id: str, handle: str) -> None:
        key = (project_id, session_id, handle)
        for rec in self._records.get(key, []):
            if rec.capture_task is not None and not rec.capture_task.done():
                rec.capture_task.cancel()
        had = self._records.pop(key, None)
        if had:
            self._notify(key)
