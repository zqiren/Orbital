# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""On-disk state of task sources, under ``{data_dir}/sources/``.

* ``sources.json`` — ``{"version": 1, "sources": [<config>]}``: what the user
  set up. Never a secret (secrets are in the keychain, see ``secrets.py``).
* ``<source_id>.json`` — that source's runtime: the ids it has seen, its
  history entries, connection gaps, failure, timing and the adapter's own
  small state.

Purely additive: an older daemon (v0.22.0) has no reader for this folder and
never touches it. Every write is a temp file + ``os.replace`` in the same
folder, so a crash leaves the previous version whole. A file that does not
parse is moved aside to ``*.corrupt-<n>`` (kept, never deleted) and the source
starts with empty runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile

logger = logging.getLogger(__name__)

DIRNAME = "sources"
CONFIG_FILE = "sources.json"
CONFIG_VERSION = 1
_ID_RE = re.compile(r"^src_[0-9a-f]{8,32}$")


def safe_source_id(source_id: str) -> bool:
    return isinstance(source_id, str) and bool(_ID_RE.match(source_id))


def atomic_write(path: str, text: str) -> None:
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        n = 1
        while os.path.exists(f"{path}.corrupt-{n}"):
            n += 1
        try:
            os.replace(path, f"{path}.corrupt-{n}")
            logger.warning("sources: %s did not parse; kept as %s.corrupt-%d", path, path, n)
        except OSError:
            logger.warning("sources: %s did not parse", path)
        return None


def empty_runtime() -> dict:
    return {
        "seen": {},            # item id -> first-seen time (insertion ordered, capped)
        "entries": [],         # oldest first
        "gaps": [],            # [{"id", "start", "end"}]
        "baselined": [],       # kinds whose current items were recorded silently
        "state": {},           # the adapter's own store
        "failure": None,       # {"code", "params", "at", "retryable"}
        "last_check_at": None,
        "next_check_at": None,
        "fail_count": 0,
        "conn": {"up": False, "down_since": None, "alive_at": None},
        "catch_up_offer": False,
        "active": None,        # {"project_id", "session_id"} of the last send
    }


class SourcesStore:
    """In-memory copy + atomic persistence. All mutation happens on the event
    loop; ``save_*`` snapshot the JSON there and write it off the loop."""

    def __init__(self, data_dir: str):
        self.dir = os.path.join(data_dir, DIRNAME)
        self.config: list[dict] = []
        self.runtime: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # -- load ------------------------------------------------------------

    def load(self) -> None:
        """Blocking read of everything; call once at start, off the loop or
        before serving."""
        raw = _read_json(os.path.join(self.dir, CONFIG_FILE))
        sources = raw.get("sources") if isinstance(raw, dict) else None
        self.config = [s for s in (sources or [])
                       if isinstance(s, dict) and safe_source_id(s.get("id", ""))]
        for s in self.config:
            rt = _read_json(self._runtime_path(s["id"]))
            base = empty_runtime()
            if isinstance(rt, dict):
                base.update(rt)
            self.runtime[s["id"]] = base

    # -- paths -----------------------------------------------------------

    def _runtime_path(self, source_id: str) -> str:
        return os.path.join(self.dir, f"{source_id}.json")

    def _lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    # -- access ----------------------------------------------------------

    def get(self, source_id: str) -> dict | None:
        return next((s for s in self.config if s.get("id") == source_id), None)

    def rt(self, source_id: str) -> dict:
        """The source's runtime. Runtime exists only for a configured source:
        for an id that is not (or no longer) configured, e.g. a check that
        was still running when the source was removed, this returns a
        throwaway empty runtime and stores nothing, so a late writer can
        never bring a removed source back (``save_runtime`` then writes
        nothing either)."""
        rt = self.runtime.get(source_id)
        if rt is None:
            if self.get(source_id) is None:
                return empty_runtime()
            rt = self.runtime[source_id] = empty_runtime()
        return rt

    def add(self, cfg: dict) -> None:
        self.config.append(cfg)
        self.runtime.setdefault(cfg["id"], empty_runtime())

    def remove(self, source_id: str) -> None:
        self.config = [s for s in self.config if s.get("id") != source_id]
        self.runtime.pop(source_id, None)

    # -- persistence -----------------------------------------------------

    async def save_config(self) -> None:
        path = os.path.join(self.dir, CONFIG_FILE)
        async with self._lock(path):
            text = json.dumps({"version": CONFIG_VERSION, "sources": self.config},
                              ensure_ascii=False, indent=1)
            await asyncio.to_thread(atomic_write, path, text)

    async def save_runtime(self, source_id: str) -> None:
        if source_id not in self.runtime or not safe_source_id(source_id):
            return
        path = self._runtime_path(source_id)
        async with self._lock(path):
            rt = self.runtime.get(source_id)
            if rt is None:
                return
            text = json.dumps(rt, ensure_ascii=False)
            await asyncio.to_thread(atomic_write, path, text)

    async def delete_runtime(self, source_id: str) -> None:
        if not safe_source_id(source_id):
            return
        path = self._runtime_path(source_id)
        async with self._lock(path):
            def _rm():
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            await asyncio.to_thread(_rm)
