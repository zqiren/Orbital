# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The file-history service (spec 065 §5.1, §5.4, §5.7).

One watchdog ``Observer`` with one recursive watch per project root. The
handler is a doorbell: on watchdog's thread it drops denied paths (a string
test) and hops to the event loop; all per-path state lives on the loop.
Each path settles after ``QUIET_SECONDS`` without events (or
``MAX_WAIT_SECONDS`` after its first event) and is flushed on a worker
thread under the project's store lock.

Attribution (§5.4): Orbital's own writers wrap their write in
``recording(path, attrib)`` so the version is exact and recorded while the
lock is held (the watcher echo then finds the hash equal). Everything the
watcher sees is credited to the sessions running at event time
(``running``, two-plus = ambiguous) or ``outside``; changes found by the
start reconcile are ``closed``.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from watchdog.events import FileSystemEventHandler

from agent_os.history import rules
from agent_os.history.store import HistoryStore, epoch_to_iso, now_iso, sha256

logger = logging.getLogger(__name__)

QUIET_SECONDS = 2.0
# A file appended without pause (a log) still gets a version a minute.
MAX_WAIT_SECONDS = 60.0
# Repairs dropped events and records offline edits; observer liveness check.
RECONCILE_SECONDS = 600.0
_MAX_FLUSH_RETRIES = 5
_SKIP_EVENTS_KEEP = 2000

USER_ATTRIB = {"kind": "user"}
ORBITAL_ATTRIB = {"kind": "orbital"}


def tool_attrib(session_id: str | None, agent: str = "orbital", *, worker: str | None = None) -> dict:
    """Attribution for Orbital's own write/edit tools (exact)."""
    if not session_id:
        return dict(ORBITAL_ATTRIB)
    out = {"kind": "tool", "sessions": [{"session_id": session_id, "agents": [agent]}]}
    if worker:
        out["worker"] = worker
    return out


def candidates_attrib(cands: dict[str, set[str]]) -> dict:
    """Watcher attribution from the candidate set of a debounce window."""
    if not cands:
        return {"kind": "outside"}
    sessions = [{"session_id": sid, "agents": sorted(agents)} for sid, agents in sorted(cands.items())]
    out = {"kind": "running", "sessions": sessions}
    if len(sessions) > 1:
        out["ambiguous"] = True
    return out


_ACTIVE: "FileHistory | None" = None
_tls = threading.local()


def active() -> "FileHistory | None":
    return _ACTIVE


class HistoryUnavailable(RuntimeError):
    pass


class _Retry(Exception):
    pass


@dataclass
class _Pending:
    first_ts: float
    first_mono: float
    cands: dict[str, set[str]] = field(default_factory=dict)
    timer: object | None = None
    due: bool = False
    retries: int = 0

    def add(self, cands) -> None:
        for sid, agents in cands:
            self.cands.setdefault(sid, set()).update(agents)


class _Project:
    def __init__(self, pid: str, workspace: str):
        self.pid = pid
        self.workspace = workspace
        self.ws_real = os.path.realpath(workspace)
        self.store: HistoryStore | None = None
        self.status = "building"
        self.reason: str | None = None
        self.watch = None
        self.ready = False
        self.closed = False
        self.pending: dict[str, _Pending] = {}
        self.flush_task: asyncio.Task | None = None
        self.start_task: asyncio.Task | None = None
        self.reconcile_timer = None
        self.nested: tuple[str, ...] = ()
        self.enclosing: tuple[str, ...] = ()
        self.listed: set[str] = set()
        self.not_listed: set[str] = set()
        self.listing_stale = True
        self.skipped: dict[str, str] = {}
        self.skip_cache: dict[str, tuple[int, int, str]] = {}
        self.skip_events: deque = deque(maxlen=_SKIP_EVENTS_KEEP)

    def rel_of(self, path) -> str | None:
        p = os.fsdecode(path)
        for base in (self.ws_real, self.workspace):
            if p.startswith(base + os.sep):
                return rules.to_rel(p[len(base) + 1:])
        return None

    def set_unavailable(self, reason: str) -> None:
        self.status, self.reason, self.ready = "unavailable", reason, False
        logger.warning("file history unavailable for %s (%s): %s", self.pid, self.workspace, reason)

    def note_skip(self, rel: str, reason: str, ts: str, attrib: dict) -> None:
        self.skipped[rel] = reason
        self.skip_events.append({"ts": ts, "path": rel, "reason": reason, "attrib": attrib})


class FileHistory:
    """App-level owner of every project's history (like ``TriggerManager``)."""

    def __init__(self, project_store, candidates_fn=None, *, cap_bytes: int | None = None,
                 quiet: float | None = None, max_wait: float | None = None,
                 reconcile_every: float | None = None, observer_factory=None):
        self._project_store = project_store
        self._candidates_fn = candidates_fn
        self._cap = cap_bytes
        self.quiet = QUIET_SECONDS if quiet is None else quiet
        self.max_wait = MAX_WAIT_SECONDS if max_wait is None else max_wait
        self.reconcile_every = RECONCILE_SECONDS if reconcile_every is None else reconcile_every
        self._observer_factory = observer_factory
        self._observer = None
        self._observer_error: str | None = None
        self._projects: dict[str, _Project] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tick_task: asyncio.Task | None = None
        self._running = False

    # ── lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        global _ACTIVE
        self._loop = asyncio.get_running_loop()
        self._running = True
        _ACTIVE = self
        try:
            factory = self._observer_factory
            if factory is None:
                from watchdog.observers import Observer as factory  # noqa: N813
            self._observer = factory()
            self._observer.start()
        except Exception as e:  # noqa: BLE001
            self._observer = None
            self._observer_error = f"observer_failed: {e}"
            logger.warning("file history: observer failed to start: %s", e)
        for project in list(self._project_store.list_projects() or []):
            pid = project.get("project_id")
            if pid:
                await self.add_project(pid)
        if self.reconcile_every > 0:
            self._tick_task = asyncio.ensure_future(self._tick())

    async def stop(self) -> None:
        global _ACTIVE
        self._running = False
        if self._tick_task is not None:
            self._tick_task.cancel()
        for proj in list(self._projects.values()):
            await self._close_project(proj, final_flush=True)
        self._projects.clear()
        if self._observer is not None:
            try:
                self._observer.stop()
                await asyncio.to_thread(self._observer.join, 5)
            except Exception:  # noqa: BLE001
                logger.debug("file history: observer stop failed", exc_info=True)
            self._observer = None
        if _ACTIVE is self:
            _ACTIVE = None

    async def wait_ready(self, pid: str, timeout: float = 30.0) -> None:
        """Test/smoke helper: wait for the project's first pass."""
        proj = self._projects.get(pid)
        if proj is not None and proj.start_task is not None:
            await asyncio.wait_for(asyncio.shield(proj.start_task), timeout)

    async def add_project(self, pid: str) -> None:
        """Schedule the project's watch and start its baseline / reconcile.
        Idempotent; a changed workspace is rescheduled."""
        project = self._project_store.get_project(pid)
        workspace = (project or {}).get("workspace") or ""
        old = self._projects.get(pid)
        if old is not None:
            if old.workspace == workspace:
                return
            await self._close_project(old, final_flush=True)
            self._projects.pop(pid, None)
        if not workspace:
            return
        proj = _Project(pid, workspace)
        self._projects[pid] = proj
        self._recompute_nesting()
        if not os.path.isdir(workspace):
            proj.set_unavailable("workspace_missing")
            return
        if self._observer is None:
            proj.set_unavailable(self._observer_error or "observer_failed")
            return
        store = HistoryStore(workspace, cap_bytes=self._cap)
        first = not store.exists()
        try:
            await asyncio.to_thread(store.load)
        except Exception as e:  # noqa: BLE001
            proj.set_unavailable(f"store_unreadable: {e}")
            return
        proj.store = store
        proj.status = "building" if first else "watching"
        try:
            proj.watch = self._observer.schedule(_Doorbell(self, proj), workspace, recursive=True)
        except Exception as e:  # noqa: BLE001
            proj.set_unavailable(f"observer_failed: {e}")
            return
        proj.start_task = asyncio.ensure_future(self._initial_pass(proj, "baseline" if first else "closed"))

    async def remove_project(self, pid: str) -> None:
        """Unschedule before the project's ``orbital/`` is removed, so no
        late flush recreates the store."""
        proj = self._projects.pop(pid, None)
        if proj is not None:
            await self._close_project(proj, final_flush=False)
        self._recompute_nesting()

    async def sync_projects(self) -> None:
        """Bring the watched set in line with the project store."""
        known = {p.get("project_id"): p for p in (self._project_store.list_projects() or [])}
        for pid in list(self._projects):
            if pid not in known:
                await self.remove_project(pid)
        for pid in known:
            if pid:
                await self.add_project(pid)

    async def _close_project(self, proj: _Project, *, final_flush: bool) -> None:
        if proj.reconcile_timer is not None:
            proj.reconcile_timer.cancel()
        for p in proj.pending.values():
            if p.timer is not None:
                p.timer.cancel()
        if proj.start_task is not None and not proj.start_task.done():
            proj.start_task.cancel()
        if final_flush and proj.ready and proj.pending and proj.store is not None:
            items = list(proj.pending.items())
            proj.pending.clear()
            try:
                await asyncio.wait_for(asyncio.to_thread(self._flush_sync, proj, items), 10)
            except Exception:  # noqa: BLE001
                logger.debug("file history: final flush failed for %s", proj.pid, exc_info=True)
        proj.pending.clear()
        if proj.watch is not None and self._observer is not None:
            try:
                self._observer.unschedule(proj.watch)
            except Exception:  # noqa: BLE001
                pass
            proj.watch = None
        if proj.store is not None:
            def _mark_closed():
                with proj.store.lock:
                    proj.closed = True
            await asyncio.to_thread(_mark_closed)
        proj.closed = True
        proj.ready = False

    def _recompute_nesting(self) -> None:
        projs = list(self._projects.values())
        for p in projs:
            nested, enclosing = [], []
            for q in projs:
                if q is p:
                    continue
                if q.ws_real.startswith(p.ws_real + os.sep):
                    nested.append(rules.to_rel(q.ws_real[len(p.ws_real) + 1:]))
                elif p.ws_real.startswith(q.ws_real + os.sep):
                    enclosing.append(q.pid)
            p.nested, p.enclosing = tuple(sorted(nested)), tuple(enclosing)

    async def _initial_pass(self, proj: _Project, mode: str) -> None:
        try:
            await asyncio.to_thread(self._reconcile_sync, proj, mode)
        except rules.ListingUnavailable as e:
            proj.set_unavailable(f"listing_failed: {e}")
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("file history: first pass failed for %s", proj.pid, exc_info=True)
            proj.set_unavailable(f"baseline_failed: {e}")
            return
        if proj.closed:
            return
        proj.status, proj.ready = "watching", True
        if any(p.due for p in proj.pending.values()):
            self._schedule_flush(proj)

    async def _tick(self) -> None:
        while self._running:
            await asyncio.sleep(self.reconcile_every)
            try:
                await self.reconcile_all()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.warning("file history: reconcile tick failed", exc_info=True)

    async def reconcile_all(self) -> None:
        await self.sync_projects()
        alive = self._observer is not None and self._observer.is_alive()
        for proj in list(self._projects.values()):
            if not alive and proj.status != "unavailable":
                proj.set_unavailable("observer_dead")
                continue
            if proj.ready:
                await self._reconcile(proj, "outside")

    async def _reconcile(self, proj: _Project, mode: str) -> None:
        try:
            await asyncio.to_thread(self._reconcile_sync, proj, mode)
        except rules.ListingUnavailable as e:
            proj.set_unavailable(f"listing_failed: {e}")

    # ── doorbell → loop ───────────────────────────────────────────────

    def _ring(self, proj: _Project, rel: str | None) -> None:
        """Watchdog thread → loop. No state is touched here."""
        loop = self._loop
        if loop is None or not self._running:
            return
        try:
            loop.call_soon_threadsafe(self._on_event, proj, rel)
        except RuntimeError:
            pass

    def _candidates(self, proj: _Project) -> list[tuple[str, list[str]]]:
        if self._candidates_fn is None:
            return []
        out: list[tuple[str, list[str]]] = []
        for pid in (proj.pid,) + proj.enclosing:
            try:
                out.extend(self._candidates_fn(pid) or [])
            except Exception:  # noqa: BLE001
                logger.debug("file history: candidates failed for %s", pid, exc_info=True)
        return out

    def _on_event(self, proj: _Project, rel: str | None) -> None:
        """On the loop: start/extend the path's debounce window."""
        if proj.closed or self._projects.get(proj.pid) is not proj:
            return
        if rel is None:
            # A directory moved or deleted: its files get no events of their
            # own on every platform. Reconcile soon instead.
            if proj.reconcile_timer is not None:
                proj.reconcile_timer.cancel()
            proj.reconcile_timer = self._loop.call_later(
                self.quiet, lambda: asyncio.ensure_future(self._reconcile_if_ready(proj)))
            return
        if rel == ".gitignore" or rel.endswith("/.gitignore"):
            proj.listing_stale = True
        now, mono = time.time(), time.monotonic()
        p = proj.pending.get(rel)
        if p is None:
            p = _Pending(first_ts=now, first_mono=mono)
            proj.pending[rel] = p
        p.add(self._candidates(proj))
        if p.timer is not None:
            p.timer.cancel()
        delay = max(0.0, min(self.quiet, p.first_mono + self.max_wait - mono))
        p.timer = self._loop.call_later(delay, self._due, proj, rel, p)

    async def _reconcile_if_ready(self, proj: _Project) -> None:
        proj.reconcile_timer = None
        if proj.ready and not proj.closed:
            await self._reconcile(proj, "outside")

    def _due(self, proj: _Project, rel: str, p: _Pending) -> None:
        p.timer, p.due = None, True
        if proj.pending.get(rel) is not p or not proj.ready:
            return  # held until the first pass finishes
        self._schedule_flush(proj)

    def _schedule_flush(self, proj: _Project) -> None:
        if proj.flush_task is not None and not proj.flush_task.done():
            return  # the running loop picks up newly due paths
        proj.flush_task = asyncio.ensure_future(self._flush_loop(proj))

    async def _flush_loop(self, proj: _Project) -> None:
        while not proj.closed:
            items = [(rel, p) for rel, p in proj.pending.items() if p.due]
            if not items:
                return
            for rel, _p in items:
                del proj.pending[rel]
            try:
                retry = await asyncio.to_thread(self._flush_sync, proj, items)
            except Exception:  # noqa: BLE001
                logger.warning("file history: flush failed for %s", proj.pid, exc_info=True)
                retry = [(rel, p) for rel, p in items if p.retries < _MAX_FLUSH_RETRIES]
            for rel, p in retry:
                p.retries += 1
                if p.retries > _MAX_FLUSH_RETRIES:
                    continue
                newer = proj.pending.get(rel)
                if newer is not None:
                    newer.first_ts, newer.first_mono = p.first_ts, p.first_mono
                    for sid, agents in p.cands.items():
                        newer.cands.setdefault(sid, set()).update(agents)
                    continue
                p.due = False
                proj.pending[rel] = p
                p.timer = self._loop.call_later(self.quiet, self._due, proj, rel, p)

    # ── flush (worker thread) ─────────────────────────────────────────

    def _flush_sync(self, proj: _Project, items) -> list:
        store = proj.store
        retry = []
        if store is None:
            return retry
        relisted = [False]
        for rel, p in items:
            try:
                with store.lock:
                    if proj.closed:
                        return []
                    self._flush_one(proj, rel, candidates_attrib(p.cands),
                                    epoch_to_iso(p.first_ts), relisted)
            except _Retry:
                retry.append((rel, p))
            except Exception:  # noqa: BLE001
                logger.warning("file history: flush of %s failed", rel, exc_info=True)
        with store.lock:
            if not proj.closed:
                store.save_index()
                store.prune_if_needed()
        return retry

    def _is_member(self, proj: _Project, rel: str, relisted: list | None = None) -> bool:
        if rules.is_layer1(rel):
            return True
        if rules.is_denied(rel, proj.nested):
            return False
        if not proj.listing_stale:
            if rel in proj.listed:
                return True
            if rel in proj.not_listed:
                return False
        if relisted is None or not relisted[0]:
            try:
                self._relist(proj)
            except rules.ListingUnavailable:
                logger.warning("file history: listing failed for %s", proj.pid, exc_info=True)
                return False
            if relisted is not None:
                relisted[0] = True
        if rel in proj.listed:
            return True
        proj.not_listed.add(rel)
        return False

    def _relist(self, proj: _Project) -> None:
        listing = rules.list_tracked(proj.workspace, proj.nested)
        proj.listed = set(listing)
        proj.not_listed = set()
        proj.listing_stale = False

    def _flush_one(self, proj: _Project, rel: str, attrib: dict, ts: str, relisted) -> None:
        store = proj.store
        abs_path = os.path.join(proj.workspace, rel)
        if not os.path.isfile(abs_path):
            if not os.path.lexists(abs_path) and store.current(rel) is not None:
                store.record(rel, None, attrib=attrib, ts=ts, save_index=False)
            return
        if not self._is_member(proj, rel, relisted):
            return
        data, st = self._read_settled(abs_path)
        reason = rules.classify_bytes(rel, data)
        if reason:
            proj.note_skip(rel, reason, ts, attrib)
            return
        store.record(rel, data, attrib=attrib, ts=ts, mtime_ns=st.st_mtime_ns, save_index=False)

    @staticmethod
    def _read_settled(abs_path: str):
        """Read with a stat before and after; a file still being written
        (or share-locked on Windows) is retried at the next quiet window."""
        try:
            st1 = os.stat(abs_path)
            if rules.is_office(abs_path):
                limit = rules.OFFICE_MAX_BYTES
            else:
                limit = rules.TEXT_MAX_BYTES
            with open(abs_path, "rb") as f:
                data = f.read(limit + 1)
            st2 = os.stat(abs_path)
        except OSError as e:
            raise _Retry() from e
        if (st1.st_size, st1.st_mtime_ns) != (st2.st_size, st2.st_mtime_ns):
            raise _Retry()
        if len(data) <= limit and len(data) != st2.st_size:
            raise _Retry()
        return data, st2

    # ── reconcile (worker thread) ─────────────────────────────────────

    def _reconcile_sync(self, proj: _Project, mode: str) -> None:
        """Listing + stat, no events. ``baseline`` on a project's first
        start; ``closed`` at a later start (changes made while Orbital was
        not running); ``outside`` at the periodic tick (missed events)."""
        store = proj.store
        if store is None or proj.closed:
            return
        listing = rules.list_tracked(proj.workspace, proj.nested)
        layer1 = [rel for rel in rules.LAYER1_FILES
                  if os.path.isfile(os.path.join(proj.workspace, rel))]
        everything = list(dict.fromkeys(listing + layer1))
        with store.lock:
            proj.listed, proj.not_listed, proj.listing_stale = set(listing), set(), False
        seen = set(everything)
        skipped: dict[str, str] = {}
        n = 0
        for rel in everything:
            if proj.closed:
                return
            if mode == "outside" and rel in proj.pending:
                # periodic repair: a live change, its flush attributes it.
                # (At start, FSEvents may replay the offline edit itself as
                # an event; the start reconcile records it as ``closed``.)
                continue
            abs_path = os.path.join(proj.workspace, rel)
            try:
                st = os.stat(abs_path)
            except OSError:
                continue
            cached = proj.skip_cache.get(rel)
            if cached and cached[:2] == (st.st_size, st.st_mtime_ns):
                skipped[rel] = cached[2]
                continue
            with store.lock:
                entry = store.current(rel)
                if entry and entry.get("size") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns:
                    continue
                try:
                    data, st = self._read_settled(abs_path)
                except _Retry:
                    continue
                reason = rules.classify_bytes(rel, data)
                if reason:
                    skipped[rel] = reason
                    proj.skip_cache[rel] = (st.st_size, st.st_mtime_ns, reason)
                    continue
                ts = epoch_to_iso(st.st_mtime)
                if entry is None and mode == "baseline":
                    store.record(rel, data, attrib={"kind": "baseline"}, kind="baseline",
                                 ts=ts, mtime_ns=st.st_mtime_ns, save_index=False)
                else:
                    store.record(rel, data, attrib={"kind": mode}, ts=ts,
                                 mtime_ns=st.st_mtime_ns, save_index=False)
                n += 1
                if n % 200 == 0:
                    store.save_index()
        with store.lock:
            if proj.closed:
                return
            for rel in list(store.index):
                if rel in seen or (mode == "outside" and rel in proj.pending):
                    continue
                if not os.path.lexists(os.path.join(proj.workspace, rel)):
                    store.record(rel, None, attrib={"kind": "outside" if mode == "baseline" else mode},
                                 save_index=False)
            proj.skipped = skipped
            store.save_index()
            store.prune_if_needed()

    # ── Orbital's own writes ──────────────────────────────────────────

    def project_for_path(self, path: str) -> tuple[_Project, str] | None:
        """The registered project owning ``path`` (the deepest workspace
        containing it) and the workspace-relative path."""
        parent, name = os.path.split(os.path.abspath(path))
        real = os.path.join(os.path.realpath(parent), name)
        best: _Project | None = None
        for proj in self._projects.values():
            if real.startswith(proj.ws_real + os.sep):
                if best is None or len(proj.ws_real) > len(best.ws_real):
                    best = proj
        if best is None:
            return None
        return best, rules.to_rel(real[len(best.ws_real) + 1:])

    def _record_after_write(self, proj: _Project, rel: str, attrib: dict) -> None:
        store = proj.store
        if store is None or proj.closed or not proj.ready:
            return
        if rules.is_denied(rel, proj.nested) or not self._is_member(proj, rel):
            return
        abs_path = os.path.join(proj.workspace, rel)
        if not os.path.isfile(abs_path):
            store.record(rel, None, attrib=attrib)
            return
        try:
            data, st = self._read_settled(abs_path)
        except _Retry:
            return  # the watcher's flush records it
        reason = rules.classify_bytes(rel, data)
        if reason:
            proj.note_skip(rel, reason, now_iso(), attrib)
            return
        store.record(rel, data, attrib=attrib, mtime_ns=st.st_mtime_ns)

    # ── restore (worker thread) ───────────────────────────────────────

    def get_project(self, pid: str) -> _Project | None:
        return self._projects.get(pid)

    def status(self, pid: str) -> dict:
        proj = self._projects.get(pid)
        if proj is None:
            return {"status": "unavailable", "reason": "not_watched"}
        out = {"status": proj.status}
        if proj.reason:
            out["reason"] = proj.reason
        return out

    def current_attrib(self, pid: str) -> dict:
        """Attribution 'at this moment' (on the loop): used by restore for
        the content it captures before replacing it."""
        proj = self._projects.get(pid)
        if proj is None:
            return {"kind": "outside"}
        cands: dict[str, set[str]] = {}
        for sid, agents in self._candidates(proj):
            cands.setdefault(sid, set()).update(agents)
        return candidates_attrib(cands)

    def restore(self, pid: str, items: list[dict], *, capture_attrib: dict | None = None,
                by: str = "user") -> list[dict]:
        proj = self._projects.get(pid)
        if proj is None or proj.store is None or proj.status == "unavailable" or proj.closed:
            raise HistoryUnavailable(self.status(pid).get("reason") or "unavailable")
        capture_attrib = capture_attrib or {"kind": "outside"}
        return [self._restore_one(proj, it, capture_attrib, by) for it in items]

    def _restore_one(self, proj: _Project, item: dict, capture_attrib: dict, by: str) -> dict:
        store = proj.store
        rel = rules.to_rel(os.path.normpath(str(item.get("path") or ""))).lstrip("/")
        out = {"path": rel}
        target_sha = item.get("sha")
        expect_sha = item.get("expect_sha")
        layer1 = rules.is_layer1(rel)
        abs_path = os.path.join(proj.workspace, rel)
        real = os.path.join(os.path.realpath(os.path.dirname(abs_path)), os.path.basename(abs_path))
        if not real.startswith(proj.ws_real + os.sep):
            return {**out, "status": "refused", "reason": "outside_workspace"}
        if rules.is_denied(rel, proj.nested):
            return {**out, "status": "refused", "reason": "not_tracked"}
        with store.lock:
            try:
                cur = self._read_current(abs_path)
            except PermissionError:
                return {**out, "status": "busy"}
            cur_sha = sha256(cur) if cur is not None else None
            if cur_sha != expect_sha:
                return {**out, "status": "conflict", "current_sha": cur_sha}
            entry = store.current(rel)
            if cur_sha != (entry["sha"] if entry else None):
                # what is about to be replaced is always in the store first
                store.record(rel, cur, attrib=capture_attrib)
            if target_sha == cur_sha:
                return {**out, "status": "ok", "current_sha": cur_sha, "unchanged": True}
            src = self._find_version(store, rel, target_sha, item.get("version"))
            attrib = {"kind": "restore", "by": by}
            if src is not None:
                attrib.update({"from_v": src.get("v"), "from_ts": src.get("ts")})
            if target_sha is None:
                if layer1:
                    return {**out, "status": "refused", "reason": "layer1_delete"}
                try:
                    os.remove(abs_path)
                except FileNotFoundError:
                    pass
                except PermissionError:
                    return {**out, "status": "busy"}
                rec = store.record(rel, None, attrib=attrib, kind="delete")
                return {**out, "status": "ok", "current_sha": None, "version": rec}
            if not store.has_blob(target_sha):
                return {**out, "status": "refused", "reason": "unknown_version"}
            data = store.get_blob(target_sha)
            try:
                if layer1:
                    post = self._write_layer1(proj, rel, data)
                else:
                    self._atomic_write_workspace(abs_path, data)
                    post = data
            except PermissionError:
                return {**out, "status": "busy"}
            try:
                mtime_ns = os.stat(abs_path).st_mtime_ns
            except OSError:
                mtime_ns = None
            rec = store.record(rel, post, attrib=attrib, kind="restore", mtime_ns=mtime_ns)
            return {**out, "status": "ok", "current_sha": rec["sha"] if rec else sha256(post),
                    "version": rec}

    @staticmethod
    def _read_current(abs_path: str) -> bytes | None:
        if not os.path.isfile(abs_path):
            return None
        with open(abs_path, "rb") as f:
            return f.read()

    @staticmethod
    def _find_version(store: HistoryStore, rel: str, sha, v) -> dict | None:
        if v is not None:
            rec = store.version_by_id(int(v))
            if rec is not None and rec["path"] == rel:
                return rec
        for rec in reversed(store.versions_for(rel)):
            if rec.get("sha") == sha:
                return rec
        return None

    @staticmethod
    def _atomic_write_workspace(abs_path: str, data: bytes) -> None:
        from agent_os.agent.workspace_files import _atomic_replace
        parent = os.path.dirname(abs_path)
        os.makedirs(parent, exist_ok=True)
        tmp = os.path.join(parent, f"{rules.RESTORE_TMP_PREFIX}{secrets.token_hex(6)}.tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        try:
            _atomic_replace(tmp, abs_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.remove(tmp)
            raise

    @staticmethod
    def _write_layer1(proj: _Project, rel: str, data: bytes) -> bytes:
        """D11: a Layer-1 restore goes through WorkspaceFileManager.write
        (prepare + its atomic replace); the version records what is on disk
        after, which may be restamped."""
        from agent_os.agent.workspace_files import WorkspaceFileManager
        key = rules.LAYER1_FILES[rel]
        _tls.suppress = getattr(_tls, "suppress", 0) + 1
        try:
            WorkspaceFileManager(proj.workspace).write(key, data.decode("utf-8", errors="replace"))
        finally:
            _tls.suppress -= 1
        with open(os.path.join(proj.workspace, rel), "rb") as f:
            return f.read()

    def set_name(self, pid: str, rel: str, sha: str, name: str) -> dict:
        proj = self._projects.get(pid)
        if proj is None or proj.store is None:
            raise HistoryUnavailable("not_watched")
        return proj.store.set_name(rel, sha, name)


class _Doorbell(FileSystemEventHandler):
    """watchdog handler: string tests and a hop to the loop, nothing else."""

    def __init__(self, svc: FileHistory, proj: _Project):
        super().__init__()
        self._svc, self._proj = svc, proj

    def dispatch(self, event) -> None:
        et = getattr(event, "event_type", "")
        if et in ("opened", "closed_no_write"):
            return
        proj = self._proj
        if getattr(event, "is_directory", False):
            if et in ("moved", "deleted"):
                rel = proj.rel_of(event.src_path)
                if rel is not None and not rules.is_denied(rel + "/x", proj.nested):
                    self._svc._ring(proj, None)
            return
        paths = [event.src_path]
        if et == "moved":
            paths.append(getattr(event, "dest_path", ""))
        for path in paths:
            if not path:
                continue
            rel = proj.rel_of(path)
            if rel is None or rules.is_denied(rel, proj.nested):
                continue
            self._svc._ring(proj, rel)


@contextlib.contextmanager
def recording(path: str, attrib: dict):
    """Wrap one of Orbital's own writes so its version is exact.

    Holds the owning project's store lock across the write and the record,
    so a watcher flush for the path waits and then finds the hash equal.
    No active history, an untracked path, or a project still building →
    just the write. A failure to record never fails the write."""
    svc = _ACTIVE
    found = svc.project_for_path(path) if svc is not None else None
    if found is None or getattr(_tls, "suppress", 0):
        yield
        return
    proj, rel = found
    store = proj.store
    if store is None or proj.closed or not proj.ready:
        yield
        return
    with store.lock:
        yield
        try:
            svc._record_after_write(proj, rel, attrib)
        except Exception:  # noqa: BLE001
            logger.warning("file history: could not record %s", rel, exc_info=True)


def write_and_record(path: str, data: bytes, attrib: dict) -> int:
    """Write ``data`` to ``path`` in place (as the Files panel save does)
    and record the version. Returns bytes written."""
    with recording(path, attrib):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return len(data)
