# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Background jobs (spec 134 Part B): fanout with process workers.

``shell(background=true)`` (and the ``orbital/bin/bg`` launcher Codex and Pi
workers run) starts a command that outlives the turn. ``JobRegistry`` is the
process-worker twin of ``FanoutRegistry``:

* **Groups.** Every background call in one assistant message joins one
  ``JobGroup`` (keyed by the loop's tool-batch id), like one fanout.
* **Poller.** One task ticks every ``POLL_INTERVAL_S``: our own children are
  ``poll()``-ed (which also reaps them); launcher jobs, which Orbital did not
  spawn, end when their ``exit`` file appears or their pid + create_time
  identity is gone. The tick also rotates ``output.log`` and picks up new
  launcher folders under each watched workspace's ``orbital/jobs/``.
* **Join.** A group wakes its session once every job has ended, or at once
  when any job ends without success. Jobs still running after an early wake
  get their own wake when they end. One ``inject_system_message`` row per
  wake, ``meta.event = "job_terminal"``, lists each job's state, runtime,
  exit code and last ``TAIL_LINES`` lines. A worker's group (a launcher job
  from Codex/Pi) is delivered to that worker as a normal dispatch instead.
* **Check-ins.** While a group still has running jobs, it checks in every
  ``CHECKIN_INTERVAL_S`` (runtime, whether output grew, last lines). Ending
  the turn means keep waiting. After ``SILENT_CHECKIN_LIMIT`` check-ins in a
  row with no new output and no user message, check-ins stop for that group
  (a note tells the user); the final exit still wakes.
* **Guards.** The same command failing ``SAME_FAILURE_LIMIT`` times in a row,
  or ``HOURLY_WAKE_CAP`` job wakes in an hour, turns further wakes into
  appended, non-waking rows until the user writes again.

Jobs do not survive quitting Orbital (decision (a), 2026-10-26): ``stop_all``
ends every job tree at quit, Windows jobs sit in a kill-on-close Job Object
held for the app's lifetime, and after a crash the next ``start`` ends any
leftover job found by pid + create_time in ``job.json`` and appends a
non-waking note to the owning session. There is no reattach and no ``lost``
state. ``job.json`` holds just what that cleanup and the note need.

Threading: ``ShellTool`` runs on the tool pool, so ``open_group`` /
``spawn_job`` / ``adopt`` / ``cancel`` are called from worker threads. Shared
state is guarded by one ``RLock``; WS broadcasts and deliveries are handed to
the event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import psutil

from agent_os.agent.project_paths import ProjectPaths

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform == "win32"

#: A check-in wake every 15 minutes while a group still has running jobs
#: (decision (b)). ``ORBITAL_JOB_CHECKIN_S`` overrides it for tests and the
#: daemon smoke only.
CHECKIN_INTERVAL_S = 15 * 60
CHECKIN_ENV = "ORBITAL_JOB_CHECKIN_S"
POLL_INTERVAL_S = 2.0
MAX_RUNNING_PER_PROJECT = 8
ROTATE_AT_BYTES = 64 * 1024 * 1024
ROTATE_KEEP_BYTES = 4 * 1024 * 1024
JOBS_DIR_CEILING_BYTES = 1024 * 1024 * 1024
RETENTION_S = 7 * 24 * 3600
RETENTION_GROUPS = 50
RETENTION_EVERY_S = 3600
TAIL_LINES = 20
TAIL_LINE_CHARS = 400
SILENT_CHECKIN_LIMIT = 3
SAME_FAILURE_LIMIT = 3
HOURLY_WAKE_CAP = 20
ANNOUNCE_DELAY_S = 0.5

#: ``meta.event`` of the rows this module writes. ``job_terminal`` and
#: ``job_checkin`` wake the session (``agent_manager._WAKE_EVENTS``);
#: ``job_started`` / ``job_note`` never do.
EVENT_TERMINAL = "job_terminal"
EVENT_CHECKIN = "job_checkin"
EVENT_STARTED = "job_started"
EVENT_NOTE = "job_note"

RUNNING = "running"
EXITED = "exited"
CANCELLED = "cancelled"
STOPPED = "stopped"          # ended because Orbital quit or crashed

# job state -> the FanoutCard status vocabulary the job rows reuse.
_CARD_STATUS = {RUNNING: "running", CANCELLED: "interrupted", STOPPED: "interrupted"}


def checkin_interval_s() -> float:
    raw = os.environ.get(CHECKIN_ENV)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return float(CHECKIN_INTERVAL_S)


def _fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m}m" if m else f"{h}h"


def _fmt_interval(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(round(seconds / 60))
        return f"{minutes} minute" + ("" if minutes == 1 else "s")
    secs = int(round(seconds))
    return f"{secs} second" + ("" if secs == 1 else "s")


def kill_command(pid: int) -> str:
    """The command the agent runs to stop job *pid* (and everything it started)."""
    if _IS_WINDOWS:
        return f"taskkill /T /F /PID {pid}"
    return f"kill -TERM -- -{pid}"


def _write_json_atomic(path: str, data: dict) -> None:
    tmp = f"{path}.{uuid.uuid4().hex[:6]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


def _read_json(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _read_exit_file(job_dir: str) -> int | None | bool:
    """The launcher's recorded exit code; False when no ``exit`` file yet."""
    try:
        with open(os.path.join(job_dir, "exit"), "r", encoding="utf-8",
                  errors="replace") as f:
            first = (f.read().split() or [""])[0]
    except OSError:
        return False
    try:
        return int(first)
    except ValueError:
        return None


def _identity_alive(pid: int | None, create_time: float | None, *,
                    marker: str | None = None) -> bool:
    """True when *pid* is still the process we started (pid + create_time).

    With no recorded create_time (a launcher job Orbital never adopted), the
    process counts only if its command line names the job (*marker*)."""
    if not pid:
        return False
    try:
        proc = psutil.Process(int(pid))
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None:
            return abs(proc.create_time() - float(create_time)) < 1.0
        if marker:
            return marker in " ".join(proc.cmdline())
        return True
    except (psutil.Error, OSError, ValueError):
        return False


def _create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def _kill_job_tree(pid: int | None, *, proc=None) -> None:
    """End a job and everything it started. Never raises."""
    if not pid:
        return
    from agent_os.platform.proc_runner import kill_process_tree

    try:
        if proc is not None:
            proc.kill_tree()
        else:
            kill_process_tree(int(pid), pgid=None if _IS_WINDOWS else int(pid))
    except Exception:
        logger.warning("could not end background job pid %s", pid, exc_info=True)


def tail_lines(path: str, n: int = TAIL_LINES, *, max_bytes: int = 65536) -> list[str]:
    """The last *n* lines of *path*, read by seeking (never the whole file)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # the first line is probably cut
    out = []
    for line in lines[-n:]:
        if len(line) > TAIL_LINE_CHARS:
            line = line[:TAIL_LINE_CHARS] + " …"
        out.append(line)
    return out


class JobRefused(Exception):
    """A background job was not started; the message is for the agent."""


@dataclass
class Job:
    id: str
    group_id: str
    project_id: str
    session_id: str
    workspace: str
    command: str
    agent: str | None = None        # None: the management agent; else the worker handle
    source: str = "shell"           # "shell" (Orbital spawned it) | "launcher"
    pid: int | None = None
    create_time: float | None = None
    started_at: float = field(default_factory=time.time)
    state: str = RUNNING
    exit_code: int | None = None
    ended_at: float | None = None
    noted: bool = True              # False: a stop note is still owed to the session
    reported: bool = False          # its end has been delivered in a join
    rotated_bytes: int = 0
    checkin_bytes: int = 0
    scratch: list[str] = field(default_factory=list, repr=False)
    proc: object | None = field(default=None, repr=False)

    @property
    def dir(self) -> str:
        return ProjectPaths(self.workspace).job_dir(self.id)

    @property
    def log_path(self) -> str:
        return os.path.join(self.dir, "output.log")

    @property
    def rel_log_path(self) -> str:
        return f"orbital/jobs/{self.id}/output.log"

    @property
    def succeeded(self) -> bool:
        return self.state == EXITED and self.exit_code == 0

    @property
    def card_status(self) -> str:
        if self.state == EXITED:
            return "completed" if self.exit_code == 0 else "error"
        return _CARD_STATUS.get(self.state, "error")

    def runtime_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    def log_size(self) -> int:
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def bytes_written(self) -> int:
        return self.rotated_bytes + self.log_size()

    def record(self) -> dict:
        """``job.json``: what orphan cleanup and the stop note need."""
        return {
            "id": self.id, "group_id": self.group_id,
            "project_id": self.project_id, "session_id": self.session_id,
            "agent": self.agent or "", "source": self.source,
            "command": self.command, "pid": self.pid,
            "create_time": self.create_time, "started_at": self.started_at,
            "state": self.state, "exit_code": self.exit_code,
            "ended_at": self.ended_at, "noted": self.noted,
        }

    def describe(self) -> dict:
        """The API / WS shape of one job."""
        return {
            "job_id": self.id, "group_id": self.group_id,
            "project_id": self.project_id, "session_id": self.session_id,
            "agent": self.agent, "command": self.command, "pid": self.pid,
            "state": self.state, "status": self.card_status,
            "exit_code": self.exit_code,
            "started_at_ms": int(self.started_at * 1000),
            "ended_at_ms": int(self.ended_at * 1000) if self.ended_at else None,
            "output": self.rel_log_path,
        }

    def end_phrase(self) -> str:
        runtime = _fmt_duration(self.runtime_s())
        if self.state == RUNNING:
            return f"still running ({runtime})"
        if self.state == CANCELLED:
            return f"cancelled after {runtime}"
        if self.state == STOPPED:
            return f"stopped after {runtime}"
        if self.exit_code is None:
            return f"ended after {runtime}, exit code unknown"
        if self.exit_code == 0:
            return f"exited 0 after {runtime}"
        if self.exit_code < 0:
            return f"failed: killed by signal {-self.exit_code} after {runtime}"
        return f"failed: exit {self.exit_code} after {runtime}"


@dataclass
class JobGroup:
    id: str
    project_id: str
    session_id: str
    workspace: str
    agent: str | None = None
    jobs: dict[str, Job] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    last_checkin: float = field(default_factory=time.monotonic)
    silent_checkins: int = 0
    checkins_stopped: bool = False
    user_spoke: bool = False
    done: bool = False

    def running(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.state == RUNNING]

    def describe(self) -> dict:
        return {"group_id": self.id, "jobs": [j.describe() for j in self.jobs.values()]}


Inject = Callable[..., Awaitable[str]]


class JobRegistry:
    """Owns every live background job; see the module docstring."""

    def __init__(self, *, inject: Inject, broadcast: Callable[[str, dict], None],
                 dispatch_worker: Callable[..., Awaitable[object]] | None = None,
                 workspaces: Callable[[], list[tuple[str, str]]] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 poll_interval_s: float = POLL_INTERVAL_S) -> None:
        """``inject(project_id, content, *, session_id, meta)`` — the
        ``inject_system_message`` funnel. ``dispatch_worker(project_id,
        session_id, handle, content)`` sends a worker its job message as a
        normal dispatch. ``workspaces()`` lists ``(project_id, workspace)`` for
        the boot cleanup and retention."""
        self._inject = inject
        self._broadcast_fn = broadcast
        self._dispatch_worker = dispatch_worker
        self._workspaces = workspaces or (lambda: [])
        self._clock = clock
        self._poll_interval_s = poll_interval_s
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._groups: dict[str, JobGroup] = {}
        self._batch_groups: dict[tuple, str] = {}
        # workspace -> project_id, scanned each tick for launcher folders.
        self._watched: dict[str, str] = {}
        self._seen_dirs: dict[str, set[str]] = {}
        self._failures: dict[tuple[str, str], tuple[str, int]] = {}
        self._wakes: dict[tuple[str, str], deque] = {}
        self._announce: dict[str, asyncio.TimerHandle] = {}
        self._size_cache: dict[str, tuple[float, int]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._last_retention = 0.0
        self._stopped = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Boot: end jobs a crashed Orbital left behind, owe their notes,
        apply retention, then start the poller."""
        global _ACTIVE
        _ACTIVE = self
        self._loop = asyncio.get_running_loop()
        try:
            notes = await asyncio.to_thread(self._boot_cleanup)
        except Exception:
            logger.exception("background-job boot cleanup failed")
            notes = []
        for project_id, session_id, content, meta in notes:
            await self._safe_inject(project_id, content, session_id, meta)
        self._last_retention = self._clock()
        self._task = asyncio.create_task(self._poll_loop(), name="job-registry-poll")

    async def shutdown(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await asyncio.to_thread(self.stop_all)

    def stop_all(self) -> int:
        """End every running job tree (quit / daemon shutdown). Thread-safe,
        synchronous, never raises. Returns how many were running."""
        with self._lock:
            if self._stopped:
                return 0
            self._stopped = True
            running = [j for j in self._jobs.values() if j.state == RUNNING]
            procs = [j.proc for j in self._jobs.values() if j.proc is not None]
        for job in running:
            _kill_job_tree(job.pid, proc=job.proc)
            job.state = STOPPED
            job.ended_at = time.time()
            job.noted = False          # the next launch appends the note
            self._persist(job)
        for proc in procs:
            # Closing a kill-on-close Job Object (Windows) also ends whatever
            # a finished job left behind: jobs die with Orbital.
            try:
                proc.close()
            except Exception:
                pass
        if running:
            logger.info("stopped %d background job(s) on shutdown", len(running))
        return len(running)

    def running_count(self, project_id: str | None = None) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state == RUNNING
                       and (project_id is None or j.project_id == project_id))

    def watch(self, workspace: str, project_id: str) -> None:
        """Scan *workspace*'s ``orbital/jobs/`` for launcher jobs each tick."""
        if workspace:
            with self._lock:
                self._watched[workspace] = project_id

    # ------------------------------------------------------------------
    # Starting jobs (called from the tool pool)
    # ------------------------------------------------------------------

    def admission(self, project_id: str, workspace: str) -> str | None:
        """Why a new job may not start right now, or None."""
        if self.running_count(project_id) >= MAX_RUNNING_PER_PROJECT:
            return (f"Error: not started — this project already has "
                    f"{MAX_RUNNING_PER_PROJECT} background jobs running, the "
                    f"limit. Wait for one to end, or stop one, first.")
        if self._jobs_dir_bytes(workspace) > JOBS_DIR_CEILING_BYTES:
            return ("Error: not started — orbital/jobs/ holds more than 1 GB "
                    "of job output. Delete old job folders under orbital/jobs/ "
                    "first.")
        return None

    def open_group(self, project_id: str, session_id: str, workspace: str, *,
                   batch_key: str | None = None, agent: str | None = None) -> JobGroup:
        """The group this call joins: the one already opened by the same tool
        batch (one assistant message), else a new one."""
        with self._lock:
            key = (project_id, session_id, agent, batch_key)
            if batch_key is not None:
                gid = self._batch_groups.get(key)
                group = self._groups.get(gid) if gid else None
                if group is not None and not group.done:
                    return group
            group = JobGroup(id=f"grp_{uuid.uuid4().hex[:8]}", project_id=project_id,
                             session_id=session_id, workspace=workspace, agent=agent,
                             created_at=self._clock(), last_checkin=self._clock())
            self._groups[group.id] = group
            if batch_key is not None:
                self._batch_groups[key] = group.id
            return group

    def spawn_job(self, group: JobGroup, command: str, argv: list[str], *,
                  env: dict | None = None) -> Job:
        """Start *argv* detached as a job of *group* (Orbital is its parent)."""
        from agent_os.platform.proc_runner import spawn

        job = Job(id=f"job_{uuid.uuid4().hex[:8]}", group_id=group.id,
                  project_id=group.project_id, session_id=group.session_id,
                  workspace=group.workspace, command=command, agent=group.agent)
        os.makedirs(job.dir, exist_ok=True)
        # append=True opens the log O_APPEND, so rotation can truncate it
        # under the running writer.
        proc = spawn(argv, cwd=group.workspace, log_path=job.log_path, env=env,
                     append=True, kill_on_close=True)
        job.proc = proc
        job.pid = proc.pid
        job.create_time = _create_time(proc.pid)
        self._register(group, job)
        return job

    def adopt(self, job_dir: str, *, group: JobGroup | None = None,
              project_id: str | None = None,
              command: str | None = None) -> Job | None:
        """Track a job the launcher started (its folder has ``job.json``).

        ``command`` is what the job card and wakes show, when the launcher
        ran a wrapper around it (the shell tool's Windows ``.ps1``); by
        default it is the launcher's own record of the command."""
        data = _read_json(os.path.join(job_dir, "job.json"))
        if not data:
            return None
        job_id = str(data.get("id") or os.path.basename(job_dir))
        with self._lock:
            known = self._jobs.get(job_id)
            if known is not None:
                return known
        if data.get("state", RUNNING) != RUNNING:
            return None
        workspace = os.path.dirname(os.path.dirname(os.path.dirname(job_dir)))
        if not command:
            try:
                with open(os.path.join(job_dir, "cmd"), "r", encoding="utf-8",
                          errors="replace") as f:
                    command = f.read().strip()
            except OSError:
                command = str(data.get("command") or "")
        pid = data.get("pid")
        pid = int(pid) if isinstance(pid, (int, float)) or str(pid).isdigit() else None
        session_id = str(data.get("session_id") or "")
        agent = str(data.get("agent") or "") or None
        pid_project = str(data.get("project_id") or project_id or "")
        if group is None:
            gid = str(data.get("group_id") or "")
            with self._lock:
                group = self._groups.get(gid)
            if group is None or group.done:
                group = self.open_group(pid_project, session_id, workspace, agent=agent)
        job = Job(id=job_id, group_id=group.id, project_id=group.project_id,
                  session_id=group.session_id, workspace=workspace, command=command,
                  agent=group.agent, source="launcher", pid=pid,
                  started_at=float(data.get("started_at") or time.time()))
        if job.pid and _identity_alive(job.pid, None, marker=job_id):
            job.create_time = _create_time(job.pid)
        self._register(group, job)
        return job

    def _register(self, group: JobGroup, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            group.jobs[job.id] = job
            self._seen_dirs.setdefault(job.workspace, set()).add(job.id)
            self._size_cache.pop(job.workspace, None)
        self._persist(job)
        self._on_loop(self._schedule_announce, group.id)

    def start_text(self, job: Job) -> str:
        """The tool result the agent acts on."""
        return (
            f"Started background job {job.id} (pid {job.pid}) in group {job.group_id}.\n"
            f"Output: {job.rel_log_path}\n"
            f"Orbital will send you a message when it finishes, and check in every "
            f"{_fmt_interval(checkin_interval_s())} while it runs. Your turn ends "
            f"after this message's tool calls; do not wait, sleep or poll for it. "
            f"Orbital tracks only the process this call started: anything it "
            f"detaches gets no message, so don't promise the user one.\n"
            f"To check early: read the output file. To stop it: {kill_command(job.pid or 0)}"
        )

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, project_id: str, job_id: str, *, notify: bool = True) -> Job | None:
        """End a running job's tree; final state ``cancelled``. Blocking (the
        tree kill waits a short grace) — call it off the event loop."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.project_id != project_id:
                return None
            if job.state != RUNNING:
                return job
            job.state = "cancelling"
        self._persist(job)
        _kill_job_tree(job.pid, proc=job.proc)
        code = None
        if job.proc is not None:
            code = job.proc.wait(timeout=2.0)
        self._finish(job, CANCELLED, code)
        group = self._groups.get(job.group_id)
        if group is not None:
            if notify:
                self._on_loop(self._after_end, group.id)
            else:
                job.reported = True
                self._on_loop(self._after_end, group.id)
        return job

    def cancel_scope(self, project_id: str, session_id: str | None = None) -> list[str]:
        """Cancel every running job of a project (or one of its sessions),
        without waking anyone — the session or project is being deleted."""
        with self._lock:
            targets = [j for j in self._jobs.values()
                       if j.project_id == project_id and j.state == RUNNING
                       and (session_id is None or j.session_id == session_id)]
        for job in targets:
            self.cancel(project_id, job.id, notify=False)
        return [j.id for j in targets]

    def is_running_dir(self, path: str) -> bool:
        """True when *path* is (or is inside) a running job's folder."""
        norm = os.path.normcase(os.path.realpath(path))
        with self._lock:
            for job in self._jobs.values():
                if job.state != RUNNING:
                    continue
                d = os.path.normcase(os.path.realpath(job.dir))
                if norm == d or d.startswith(norm.rstrip(os.sep) + os.sep):
                    return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_jobs(self, project_id: str, session_id: str | None = None) -> list[dict]:
        with self._lock:
            return [j.describe() for j in self._jobs.values()
                    if j.project_id == project_id
                    and (session_id is None or j.session_id == session_id)]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def note_user_message(self, project_id: str, session_id: str) -> None:
        """A user message in this session: it counts as attention for the
        silent check-in guard and resets the wake-loop guards."""
        with self._lock:
            for group in self._groups.values():
                if group.project_id == project_id and group.session_id == session_id:
                    group.user_spoke = True
            self._failures.pop((project_id, session_id), None)
            self._wakes.pop((project_id, session_id), None)

    # ------------------------------------------------------------------
    # Poller
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_s)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("background-job poll tick failed")

    async def tick(self) -> None:
        """One poll: endings, rotation, new launcher jobs, check-ins, retention."""
        await asyncio.to_thread(self._scan_watched)
        ended: list[Job] = []
        with self._lock:
            running = [j for j in self._jobs.values() if j.state == RUNNING]
        for job in running:
            finished, code = self._probe(job)
            if finished:
                self._finish(job, EXITED, code)
                ended.append(job)
            else:
                self._maybe_rotate(job)
        for gid in {j.group_id for j in ended}:
            await self._after_end_async(gid)
        await self._checkins()
        if self._clock() - self._last_retention >= RETENTION_EVERY_S:
            self._last_retention = self._clock()
            await asyncio.to_thread(self._retention_all)

    def _probe(self, job: Job) -> tuple[bool, int | None]:
        if job.proc is not None:
            code = job.proc.poll()
            return (code is not None, code)
        recorded = _read_exit_file(job.dir)
        if recorded is not False:
            return True, recorded
        if _identity_alive(job.pid, job.create_time, marker=job.id):
            return False, None
        # Gone without an exit file: the runner itself was killed. Look once
        # more — the exit file may have landed between the two checks.
        recorded = _read_exit_file(job.dir)
        return True, (None if recorded is False else recorded)

    def _finish(self, job: Job, state: str, code: int | None) -> None:
        with self._lock:
            if job.state not in (RUNNING, "cancelling"):
                return
            job.state = state
            job.exit_code = code
            job.ended_at = time.time()
        logger.info("job %s ended: state=%s code=%s session=%s",
                    job.id, state, code, job.session_id)
        for path in job.scratch:
            try:
                os.remove(path)
            except OSError:
                pass
        self._persist(job)
        self._broadcast(job.project_id, {
            "type": "fanout.task_update", "project_id": job.project_id,
            "session_id": job.session_id, "fanout_id": job.group_id,
            "handle": job.id, "status": job.card_status,
            "completed_at_ms": int(job.ended_at * 1000),
            "job": job.describe(),
        })

    def _maybe_rotate(self, job: Job) -> None:
        size = job.log_size()
        if size <= ROTATE_AT_BYTES:
            return
        try:
            with open(job.log_path, "rb") as src:
                src.seek(max(0, size - ROTATE_KEEP_BYTES))
                keep = src.read(ROTATE_KEEP_BYTES)
            with open(job.log_path + ".1", "wb") as dst:
                dst.write(keep)
            # The writer holds the file O_APPEND, so its next write lands at
            # the new end instead of leaving a hole.
            with open(job.log_path, "r+b") as f:
                f.truncate(0)
            job.rotated_bytes += size
            logger.info("rotated %s at %d bytes", job.log_path, size)
        except OSError:
            logger.warning("could not rotate %s", job.log_path, exc_info=True)

    def _scan_watched(self) -> None:
        with self._lock:
            watched = dict(self._watched)
        for workspace, project_id in watched.items():
            jobs_dir = ProjectPaths(workspace).jobs_dir
            try:
                names = os.listdir(jobs_dir)
            except OSError:
                continue
            seen = self._seen_dirs.setdefault(workspace, set())
            for name in names:
                if name in seen or not name.startswith("job_"):
                    continue
                path = os.path.join(jobs_dir, name)
                if not os.path.exists(os.path.join(path, "job.json")):
                    continue  # the launcher writes job.json last
                seen.add(name)
                try:
                    self.adopt(path, project_id=project_id)
                except Exception:
                    logger.warning("could not adopt job folder %s", path, exc_info=True)

    # ------------------------------------------------------------------
    # Join
    # ------------------------------------------------------------------

    def _after_end(self, gid: str) -> None:
        """Loop-side: schedule the join check for group *gid*."""
        asyncio.ensure_future(self._after_end_async(gid))

    async def _after_end_async(self, gid: str) -> None:
        with self._lock:
            group = self._groups.get(gid)
            if group is None:
                return
            ended = [j for j in group.jobs.values()
                     if j.state not in (RUNNING, "cancelling") and not j.reported]
            if not ended:
                if not group.running() and not group.done:
                    group.done = True
                    self._broadcast_completed(group)
                return
            failed = [j for j in ended if not j.succeeded]
            all_ended = not group.running()
            if not failed and not all_ended:
                return
            for j in ended:
                j.reported = True
            if all_ended:
                group.done = True
        content, display = self._compose_join(group, failed)
        meta = {"event": EVENT_TERMINAL, "kind": "job_join",
                "display_content": display, "job_group": group.describe()}
        await self._deliver(group, content, meta,
                            failed_command=failed[0].command if failed else None)
        if group.done:
            self._broadcast_completed(group)

    def _compose_join(self, group: JobGroup, failed: list[Job]) -> tuple[str, str]:
        jobs = list(group.jobs.values())
        running = group.running()
        n = len(jobs)
        noun = "job" if n == 1 else "jobs"
        # Count every ended failure, not just this wake's: a later wake of a
        # group that failed early must not read as all-good.
        failures = [j for j in jobs if j.state not in (RUNNING, "cancelling")
                    and not j.succeeded]
        if failures:
            head = f"{len(failures)} of {n} background {noun} did not succeed"
        else:
            head = f"{n} background {noun} finished"
        if running:
            head += f"; {len(running)} still running"
        header = f"[Background jobs {group.id}] {head}."
        display = [header]
        body = [header]
        for j in jobs:
            line = f"- {j.id} `{j.command}`: {j.end_phrase()}"
            display.append(line)
            body.append(line)
            if j.state != RUNNING:
                tail = tail_lines(j.log_path)
                if tail:
                    body.append(f"  last {len(tail)} lines of {j.rel_log_path}:")
                    body.extend(f"    {t}" for t in tail)
                else:
                    body.append(f"  no output ({j.rel_log_path} is empty)")
        body.append("")
        if running:
            body.append("You will get another message when the jobs still running end. "
                        "To stop one: " + "; ".join(
                            f"{j.id}: {kill_command(j.pid or 0)}" for j in running) + ".")
        else:
            body.append("These jobs have ended. Orbital isn't tracking anything they "
                        "left running, so don't tell the user you'll be notified "
                        "about it.")
        body.append("Read the full output in the files above if you need more than "
                    "these lines. The user has not seen this message; tell them the "
                    "outcome.")
        return "\n".join(body), "\n".join(display)

    def _broadcast_completed(self, group: JobGroup) -> None:
        self._broadcast(group.project_id, {
            "type": "fanout.completed", "project_id": group.project_id,
            "session_id": group.session_id, "fanout_id": group.id,
            "succeeded": sum(1 for j in group.jobs.values() if j.succeeded),
            "total": len(group.jobs), "reason": "jobs_ended",
            "tasks": [{"handle": j.id, "label": j.command, "status": j.card_status,
                       "completed_at_ms": int(j.ended_at * 1000) if j.ended_at else None}
                      for j in group.jobs.values()],
        })

    # ------------------------------------------------------------------
    # Check-ins
    # ------------------------------------------------------------------

    async def _checkins(self) -> None:
        interval = checkin_interval_s()
        now = self._clock()
        due: list[tuple[JobGroup, bool]] = []
        with self._lock:
            for group in self._groups.values():
                if group.done or group.checkins_stopped or not group.running():
                    continue
                if now - group.last_checkin < interval:
                    continue
                grew = False
                for j in group.running():
                    written = j.bytes_written()
                    if written > j.checkin_bytes:
                        grew = True
                    j.checkin_bytes = written
                if grew or group.user_spoke:
                    group.silent_checkins = 0
                else:
                    group.silent_checkins += 1
                group.user_spoke = False
                group.last_checkin = now
                due.append((group, grew))
        for group, grew in due:
            content, display = self._compose_checkin(group, grew)
            meta = {"event": EVENT_CHECKIN, "kind": "job_checkin",
                    "display_content": display, "job_group": group.describe()}
            await self._deliver(group, content, meta)
            if group.silent_checkins >= SILENT_CHECKIN_LIMIT:
                group.checkins_stopped = True
                ids = ", ".join(j.id for j in group.running())
                note = (f"[Background jobs {group.id}] Orbital stopped checking in on "
                        f"{ids}: no new output for {SILENT_CHECKIN_LIMIT} check-ins "
                        f"in a row. The agent will still be told when "
                        f"{'it ends' if len(group.running()) == 1 else 'they end'}.")
                await self._safe_inject(group.project_id, note, group.session_id, {
                    "event": EVENT_NOTE, "kind": "job_checkins_stopped",
                    "suppress_wake": True, "display_content": note,
                    "job_group": group.describe()})

    def _compose_checkin(self, group: JobGroup, grew: bool) -> tuple[str, str]:
        running = group.running()
        n = len(running)
        header = (f"[Background jobs {group.id}] Check-in: {n} job"
                  f"{'' if n == 1 else 's'} still running.")
        display = [header]
        body = [header]
        for j in running:
            growth = ("output grew since the last check-in" if grew
                      else "no new output since the last check-in")
            line = f"- {j.id} `{j.command}`: running {_fmt_duration(j.runtime_s())}, {growth}"
            display.append(line)
            body.append(line)
            tail = tail_lines(j.log_path)
            if tail:
                body.append(f"  last {len(tail)} lines of {j.rel_log_path}:")
                body.extend(f"    {t}" for t in tail)
        body.append("")
        body.append("To keep waiting, just end your turn: you will get a message when it "
                    "ends. If it looks stuck or wrong, stop it: " + "; ".join(
                        f"{j.id}: {kill_command(j.pid or 0)}" for j in running) + ".")
        return "\n".join(body), "\n".join(display)

    # ------------------------------------------------------------------
    # Delivery + guards
    # ------------------------------------------------------------------

    async def _deliver(self, group: JobGroup, content: str, meta: dict, *,
                       failed_command: str | None = None) -> None:
        key = (group.project_id, group.session_id)
        stop_line = None
        with self._lock:
            if meta.get("event") == EVENT_TERMINAL:
                if failed_command is not None:
                    last = self._failures.get(key)
                    count = last[1] + 1 if last and last[0] == failed_command else 1
                    self._failures[key] = (failed_command, count)
                    if count > SAME_FAILURE_LIMIT:
                        stop_line = ("Stopped auto-waking: this command failed "
                                     f"{SAME_FAILURE_LIMIT} times in a row. Ask the user.")
                else:
                    self._failures.pop(key, None)
            wakes = self._wakes.setdefault(key, deque())
            now = self._clock()
            while wakes and now - wakes[0] > 3600:
                wakes.popleft()
            if stop_line is None and len(wakes) >= HOURLY_WAKE_CAP:
                stop_line = (f"Stopped auto-waking: {HOURLY_WAKE_CAP} background-job "
                             "wakes in the last hour. Ask the user.")
            if stop_line is None:
                wakes.append(now)
        if stop_line is not None:
            content = f"{content}\n{stop_line}"
            meta = {**meta, "suppress_wake": True,
                    "display_content": f"{meta.get('display_content', '')}\n{stop_line}"}
            await self._safe_inject(group.project_id, content, group.session_id, meta)
            return
        if group.agent and self._dispatch_worker is not None:
            try:
                sent = await self._dispatch_worker(group.project_id, group.session_id,
                                                   group.agent, content)
                # SubAgentManager.send reports a failure as "Error: …".
                if not (isinstance(sent, str) and sent.startswith("Error")):
                    return
                logger.warning("job %s: dispatch to %s refused: %s", group.id,
                               group.agent, sent)
            except Exception:
                logger.warning("job %s: dispatch to %s failed; appending to the "
                               "session instead", group.id, group.agent, exc_info=True)
        await self._safe_inject(group.project_id, content, group.session_id, meta)

    async def _safe_inject(self, project_id: str, content: str, session_id: str,
                           meta: dict) -> str | None:
        if not project_id or not session_id:
            logger.info("job message with no session to deliver to: %s", content[:200])
            return None
        try:
            result = await self._inject(project_id, content, session_id=session_id, meta=meta)
            logger.info("job message delivered: event=%s session=%s result=%s",
                        meta.get("event"), session_id, result)
            return result
        except Exception:
            logger.warning("job message delivery failed (project=%s session=%s)",
                           project_id, session_id, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Live card
    # ------------------------------------------------------------------

    def _schedule_announce(self, gid: str) -> None:
        """Loop-side: one ``fanout.started`` per group, after the batch's last
        job joined it (the calls of one message start within milliseconds)."""
        if self._loop is None:
            return
        old = self._announce.pop(gid, None)
        if old is not None:
            old.cancel()
        self._announce[gid] = self._loop.call_later(ANNOUNCE_DELAY_S, self._announce_now, gid)

    def _announce_now(self, gid: str) -> None:
        self._announce.pop(gid, None)
        with self._lock:
            group = self._groups.get(gid)
            if group is None:
                return
            desc = group.describe()
            tasks = [{"handle": j.id, "label": j.command,
                      "job": {"project_id": j.project_id, "job_id": j.id,
                              "agent": j.agent}}
                     for j in group.jobs.values()]
        self._broadcast(group.project_id, {
            "type": "fanout.started", "project_id": group.project_id,
            "session_id": group.session_id, "fanout_id": gid, "tasks": tasks,
            "kind": "jobs",
        })
        if group.agent:
            # A worker's job has no tool row in the manager's chat; this
            # non-waking row is what the card is rebuilt from on reload.
            who = group.agent
            text = f"{who} started a background job: " + "; ".join(
                f"{t['handle']} `{t['label']}`" for t in tasks)
            asyncio.ensure_future(self._safe_inject(
                group.project_id, text, group.session_id,
                {"event": EVENT_STARTED, "kind": "job_started", "suppress_wake": True,
                 "display_content": text, "job_group": desc}))

    def _broadcast(self, project_id: str, payload: dict) -> None:
        self._on_loop(self._broadcast_fn, project_id, payload)

    def _on_loop(self, fn, *args) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            try:
                fn(*args)
            except Exception:
                logger.debug("job registry callback failed", exc_info=True)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            fn(*args)
        else:
            loop.call_soon_threadsafe(fn, *args)

    # ------------------------------------------------------------------
    # Disk: job.json, ceiling, retention, boot cleanup
    # ------------------------------------------------------------------

    def _persist(self, job: Job) -> None:
        try:
            os.makedirs(job.dir, exist_ok=True)
            _write_json_atomic(os.path.join(job.dir, "job.json"), job.record())
        except OSError:
            logger.warning("could not write %s/job.json", job.dir, exc_info=True)

    def _jobs_dir_bytes(self, workspace: str) -> int:
        cached = self._size_cache.get(workspace)
        now = self._clock()
        if cached and now - cached[0] < 10:
            return cached[1]
        total = 0
        for root, _dirs, files in os.walk(ProjectPaths(workspace).jobs_dir):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        self._size_cache[workspace] = (now, total)
        return total

    def _retention_all(self) -> None:
        workspaces = {w for _p, w in self._safe_workspaces()}
        with self._lock:
            workspaces.update(self._watched)
            workspaces.update(j.workspace for j in self._jobs.values())
        for workspace in workspaces:
            self.apply_retention(workspace)

    def apply_retention(self, workspace: str, *, now: float | None = None) -> list[str]:
        """Delete ended job folders older than ``RETENTION_S`` or outside the
        newest ``RETENTION_GROUPS`` groups. Never a running job's folder."""
        now = time.time() if now is None else now
        jobs_dir = ProjectPaths(workspace).jobs_dir
        try:
            names = os.listdir(jobs_dir)
        except OSError:
            return []
        entries = []
        for name in names:
            path = os.path.join(jobs_dir, name)
            data = _read_json(os.path.join(path, "job.json"))
            if data is None or data.get("state") in (RUNNING, "cancelling"):
                continue
            with self._lock:
                live = self._jobs.get(name)
            if live is not None and live.state == RUNNING:
                continue
            entries.append((name, path, data))
        groups: dict[str, float] = {}
        for _name, _path, data in entries:
            gid = str(data.get("group_id") or data.get("id"))
            started = float(data.get("started_at") or 0)
            groups[gid] = max(groups.get(gid, 0.0), started)
        keep_groups = set(sorted(groups, key=groups.get, reverse=True)[:RETENTION_GROUPS])
        removed = []
        for name, path, data in entries:
            gid = str(data.get("group_id") or data.get("id"))
            ended = float(data.get("ended_at") or data.get("started_at") or 0)
            if gid in keep_groups and now - ended < RETENTION_S:
                continue
            if self.is_running_dir(path):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(name)
            with self._lock:
                job = self._jobs.pop(name, None)
                if job is not None:
                    grp = self._groups.get(job.group_id)
                    if grp is not None:
                        grp.jobs.pop(name, None)
                        if not grp.jobs:
                            self._groups.pop(grp.id, None)
        return removed

    def _safe_workspaces(self) -> list[tuple[str, str]]:
        try:
            return [(p, w) for p, w in self._workspaces() if w]
        except Exception:
            logger.warning("job registry: could not list workspaces", exc_info=True)
            return []

    def _boot_cleanup(self) -> list[tuple[str, str, str, dict]]:
        """End jobs a crashed or force-quit Orbital left behind; return the
        notes owed to their sessions. Runs before the poller starts."""
        notes = []
        for project_id, workspace in self._safe_workspaces():
            jobs_dir = ProjectPaths(workspace).jobs_dir
            try:
                names = os.listdir(jobs_dir)
            except OSError:
                continue
            for name in names:
                path = os.path.join(jobs_dir, name)
                data = _read_json(os.path.join(path, "job.json"))
                if data is None:
                    continue
                state = data.get("state")
                owed = state in (RUNNING, "cancelling") or (
                    state == STOPPED and data.get("noted") is False)
                if not owed:
                    continue
                pid = data.get("pid")
                job_id = str(data.get("id") or name)
                exit_code = _read_exit_file(path)
                ended_text = "was stopped because Orbital closed"
                new_state, code = STOPPED, None
                if state in (RUNNING, "cancelling"):
                    if _identity_alive(pid, data.get("create_time"), marker=job_id):
                        _kill_job_tree(pid)
                    elif exit_code is not False:
                        new_state, code = EXITED, exit_code
                        ended_text = ("ended while Orbital was closed (exit code "
                                      f"{'unknown' if exit_code is None else exit_code})")
                data.update(state=new_state, exit_code=code, noted=True,
                            ended_at=data.get("ended_at") or time.time())
                try:
                    _write_json_atomic(os.path.join(path, "job.json"), data)
                except OSError:
                    pass
                session_id = str(data.get("session_id") or "")
                pid_owner = str(data.get("project_id") or project_id)
                command = str(data.get("command") or "")
                if not command:
                    try:
                        with open(os.path.join(path, "cmd"), encoding="utf-8",
                                  errors="replace") as f:
                            command = f.read().strip()
                    except OSError:
                        pass
                text = f"Background job {job_id} (`{command}`) {ended_text}."
                ended_ms = int(float(data["ended_at"]) * 1000)
                card = {"group_id": str(data.get("group_id") or job_id), "jobs": [{
                    "job_id": job_id, "command": command, "state": new_state,
                    "ended_at_ms": ended_ms,
                    "status": ("completed" if new_state == EXITED and code == 0
                               else "error" if new_state == EXITED else "interrupted"),
                    "exit_code": code}]}
                if session_id:
                    notes.append((pid_owner, session_id, text, {
                        "event": EVENT_NOTE, "kind": "job_stopped",
                        "suppress_wake": True, "display_content": text,
                        "job_group": card}))
            self.apply_retention(workspace)
        return notes


#: The registry of this process, for the desktop quit path (which exits with
#: ``os._exit`` and never reaches the app's shutdown hooks).
_ACTIVE: JobRegistry | None = None


def running_job_count() -> int:
    registry = _ACTIVE
    return registry.running_count() if registry is not None else 0


def stop_all_jobs() -> int:
    registry = _ACTIVE
    return registry.stop_all() if registry is not None else 0
