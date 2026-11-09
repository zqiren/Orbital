# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The pinned worker's own commands, for the composer's ``/`` picker (spec 166).

Every in-scope worker reports what it takes over the protocol Orbital already
speaks to it: Claude Code's ``initialize`` response (``commands``), Codex's
``skills/list``, an ACP agent's ``available_commands_update``. The transports
hand each report to :func:`publish` (the quota store's side-channel pattern);
the ``skill-catalog`` route reads it back with ``?worker=``.

Normal form: ``{name, description, invoke, hint, enabled}``, where ``invoke``
is exactly what the composer holds after a pick (``"/name "`` or
``"$name "``). A parser returns None for a malformed report and ``[]`` only
for a valid empty one; None never overwrites a stored list.

Refresh rule: every worker start (and every ACP re-announcement) overwrites
its entry. A key with no entry is filled by a cold probe where one has no side
effect (Claude Code, Codex); the store runs one probe per key at a time and
backs off after a failure. Persisted to one JSON file in the daemon's data
dir, so an ACP worker's list survives a restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

PROBE_FAILURE_BACKOFF_S = 60.0
_FILE_VERSION = 1

Probe = Callable[[], Awaitable["list[dict] | None"]]


def _key(name: str) -> str:
    return name.strip().lower()


def _command(name: Any, description: Any, invoke_prefix: str, hint: Any = "",
             enabled: bool = True) -> dict | None:
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    return {
        "name": name,
        "description": description if isinstance(description, str) else "",
        "invoke": f"{invoke_prefix}{name} ",
        "hint": hint if isinstance(hint, str) else "",
        "enabled": bool(enabled),
    }


def _dedupe(rows: list[dict | None]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if row is None or _key(row["name"]) in seen:
            continue
        seen.add(_key(row["name"]))
        out.append(row)
    return out


def claude_commands(info: Any) -> list[dict] | None:
    """Claude Code's ``initialize`` response (``get_server_info()``)."""
    commands = info.get("commands") if isinstance(info, dict) else None
    if not isinstance(commands, list):
        return None
    return _dedupe([
        _command(c.get("name"), c.get("description"), "/", c.get("argumentHint"))
        if isinstance(c, dict) else None
        for c in commands
    ])


def codex_commands(result: Any) -> list[dict] | None:
    """Codex's ``skills/list`` result: ``data[].skills[]``. Disabled skills are
    kept (``enabled: False``) so the merge can hide the catalog's copy."""
    data = result.get("data") if isinstance(result, dict) else None
    if not isinstance(data, list):
        return None
    rows: list[dict | None] = []
    for group in data:
        skills = group.get("skills") if isinstance(group, dict) else None
        if not isinstance(skills, list):
            return None
        for s in skills:
            if isinstance(s, dict):
                rows.append(_command(s.get("name"), s.get("description"), "$",
                                     enabled=s.get("enabled", True) is not False))
    return _dedupe(rows)


def acp_commands(available: Any) -> list[dict] | None:
    """An ACP ``available_commands_update``'s ``availableCommands`` (pydantic
    models or their dumped dicts)."""
    if not isinstance(available, list):
        return None
    rows: list[dict | None] = []
    for c in available:
        if hasattr(c, "model_dump"):
            c = c.model_dump(by_alias=True, exclude_none=True)
        if not isinstance(c, dict):
            continue
        hint = c.get("input")
        hint = hint.get("hint") if isinstance(hint, dict) else ""
        rows.append(_command(c.get("name"), c.get("description"), "/", hint))
    return _dedupe(rows)


class WorkerCommandStore:
    """``(slug, realpath(workspace)) -> commands``, persisted to one file."""

    def __init__(self, path: str | None, *, clock: Callable[[], float] = time.monotonic):
        self._path = path
        self._clock = clock
        self._entries: dict[tuple[str, str], list[dict]] = {}
        self._failed_at: dict[tuple[str, str], float] = {}
        self._probes: dict[tuple[str, str], asyncio.Task] = {}
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _k(slug: str, workspace: str) -> tuple[str, str]:
        return (slug, os.path.realpath(workspace) if workspace else "")

    def get(self, slug: str, workspace: str) -> list[dict] | None:
        return self._entries.get(self._k(slug, workspace))

    def publish(self, slug: str, workspace: str, commands: list[dict] | None) -> None:
        if commands is None:
            return
        k = self._k(slug, workspace)
        self._failed_at.pop(k, None)
        if self._entries.get(k) == commands:
            return
        self._entries[k] = list(commands)
        self._write_file()

    async def commands_for(self, slug: str, workspace: str, probe: Probe | None) -> list[dict]:
        """The stored list, or (for a key with none) the cold probe's answer.
        Never raises; ``[]`` when nothing is known."""
        k = self._k(slug, workspace)
        hit = self._entries.get(k)
        if hit is not None or probe is None:
            return hit or []
        task = self._probes.get(k)
        if task is None:
            failed = self._failed_at.get(k)
            if failed is not None and self._clock() - failed < PROBE_FAILURE_BACKOFF_S:
                return []
            task = asyncio.get_running_loop().create_task(self._probe(k, probe))
            self._probes[k] = task
            task.add_done_callback(lambda _t, _k=k: self._probes.pop(_k, None))
        try:
            return await asyncio.shield(task)
        except Exception:  # noqa: BLE001 — display data
            return []

    async def _probe(self, k: tuple[str, str], probe: Probe) -> list[dict]:
        try:
            commands = await probe()
        except Exception as exc:  # noqa: BLE001 — display data, never raise
            logger.info("worker commands: probe for %s failed (%s: %s)",
                        k[0], type(exc).__name__, exc)
            commands = None
        if commands is None:
            self._failed_at[k] = self._clock()
            return []
        self.publish(k[0], k[1], commands)
        return commands

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self._path or not os.path.isfile(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            for row in data.get("entries") or []:
                slug, ws, cmds = row.get("slug"), row.get("workspace"), row.get("commands")
                if isinstance(slug, str) and isinstance(ws, str) and isinstance(cmds, list):
                    self._entries[(slug, ws)] = [c for c in cmds if isinstance(c, dict)]
        except Exception:  # noqa: BLE001 — a bad file is an empty store
            logger.warning("worker commands: ignoring unreadable %s", self._path)

    def _write_file(self) -> None:
        if not self._path:
            return
        try:
            with self._lock:
                payload = {"version": _FILE_VERSION, "entries": [
                    {"slug": s, "workspace": w, "commands": c}
                    for (s, w), c in self._entries.items()]}
                directory = os.path.dirname(os.path.abspath(self._path))
                os.makedirs(directory, exist_ok=True)
                fd, tmp = tempfile.mkstemp(prefix=".worker-commands-", suffix=".tmp",
                                           dir=directory)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    os.replace(tmp, self._path)
                except BaseException:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                    raise
        except Exception:  # noqa: BLE001
            logger.warning("worker commands: could not persist %s", self._path,
                           exc_info=True)


def merge_catalog(entries: list[dict], commands: list[dict]) -> list[dict]:
    """Orbital's catalog with the worker's commands folded in: an enabled
    command replaces the catalog entry with its key, a disabled one only
    hides it, everything else stays (spec 166 §4.2)."""
    if not commands:
        return list(entries)
    worker_keys = {_key(c["name"]) for c in commands}
    kept = [e for e in entries if e.get("key") not in worker_keys]
    added = [{
        "key": _key(c["name"]),
        "name": c["name"],
        "description": c.get("description", ""),
        "source": "worker",
        "path": "",
        "dir_name": "",
        "invoke": c["invoke"],
        "hint": c.get("hint", ""),
    } for c in commands if c.get("enabled", True)]
    return sorted(kept + added, key=lambda e: e["key"])


# ---------------------------------------------------------------------------
# Process-wide store: the transports' entry point
# ---------------------------------------------------------------------------

_store: WorkerCommandStore | None = None


def install(store: WorkerCommandStore | None) -> None:
    """Make ``store`` the daemon-wide store (app factory; tests reset it)."""
    global _store
    _store = store


def get_store() -> WorkerCommandStore | None:
    return _store


def publish(slug: str, workspace: str, commands: list[dict] | None) -> None:
    """Record a worker's report. Never raises; a no-op with no store."""
    store = _store
    if store is None:
        return
    try:
        store.publish(slug, workspace, commands)
    except Exception:  # noqa: BLE001
        logger.warning("worker commands: publish failed", exc_info=True)
