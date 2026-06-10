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
checked by create-time, so PID reuse reads dead, not alive). On POSIX a child
matching claude-code's ``~/.claude/shell-snapshots`` launcher pattern is
preferred when several candidates appear; the pattern is a REFINEMENT, never
the sole mechanism.

WINDOWS GATE — OPEN: the Windows process shape of a run_in_background shell
has NOT been observed (REPORT-piece3-child-classification.md scoped it; this
machine cannot run the observation). ``three_state_supported`` is therefore
False on win32: status stays two-state there (explicit, logged degradation —
eviction remains descendant-gated so no work can be killed; nothing silently
reads idle while liveness is unverifiable). Do not flip this without the
observed Windows trace.

Known accepted limitation (do not chase): work detached via raw foreground
``&``/``nohup`` reparents away from the agent tree and is invisible to both
provenance and any tree walk. It equally escapes the reap kill.
"""

import asyncio
import logging
import sys
import time
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)

# POSIX refinement: claude-code's Bash tool launches background jobs under a
# shell sourcing a snapshot file at this path fragment (observed live).
_POSIX_LAUNCHER_FRAGMENT = ".claude/shell-snapshots/"

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
                 capture_poll_s: float = 0.25):
        self.capture_window_s = capture_window_s
        self.capture_poll_s = capture_poll_s
        # Part A Windows gate (see module docstring). Instance attr so tests
        # can exercise the degraded path without platform patching.
        self.three_state_supported = sys.platform != "win32"
        if not self.three_state_supported:
            logger.warning(
                "BackgroundWorkRegistry: three-state status DISABLED on this "
                "platform (Windows liveness shape unverified — Part A gate "
                "open). Status degrades to two-state; eviction stays "
                "descendant-gated."
            )
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
        return rec

    async def _capture_anchor(self, key: Key, rec: BackgroundWorkRecord,
                              root_proc: "psutil.Process", baseline: set[int]) -> None:
        """Adopt the first NEW direct child of the agent root as the work
        anchor. POSIX: prefer the shell-snapshots launcher when several
        candidates appear in the window."""
        deadline = time.monotonic() + self.capture_window_s
        while time.monotonic() < deadline:
            try:
                candidates = [
                    c for c in root_proc.children()
                    if c.pid not in baseline and c.pid not in self._claimed_pids(key)
                ]
            except psutil.Error:
                return  # root gone — nothing to anchor
            if candidates:
                chosen = candidates[0]
                if sys.platform != "win32" and len(candidates) > 1:
                    for c in candidates:
                        try:
                            if any(_POSIX_LAUNCHER_FRAGMENT in part
                                   for part in c.cmdline()):
                                chosen = c
                                break
                        except psutil.Error:
                            continue
                rec.anchor = chosen
                self._last_activity[key] = time.monotonic()
                logger.info(
                    "background work anchored for %s: pid=%s cmd=%r",
                    key, chosen.pid, rec.command[:80],
                )
                return
            await asyncio.sleep(self.capture_poll_s)
        logger.info(
            "background work anchor capture expired for %s cmd=%r "
            "(work likely finished within the window)", key, rec.command[:80],
        )

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
        return terminated

    def clear(self, project_id: str, session_id: str, handle: str) -> None:
        key = (project_id, session_id, handle)
        for rec in self._records.get(key, []):
            if rec.capture_task is not None and not rec.capture_task.done():
                rec.capture_task.cancel()
        self._records.pop(key, None)
