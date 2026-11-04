# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The per-project history store (spec 065 §5.3, §6).

Plain files under ``{workspace}/orbital/history/``:

- ``blobs/<sha[:2]>/<sha>`` — zlib-compressed content, content-addressed;
  identical content is stored once.
- ``versions.jsonl`` — append-only, one line per version
  ``{v, ts, path, sha, prev_sha, kind, size, added, removed, attrib}`` with
  ``kind`` ∈ baseline | create | modify | delete | restore (a ``delete`` has
  ``sha: null``), plus ``{kind: "name", ts, path, sha, name}`` lines for
  named versions (latest wins; an empty name clears).
- ``index.json`` — current state per path ``{sha, size, mtime_ns}``;
  rebuilt from ``versions.jsonl`` when missing or corrupt.

Every mutation happens under ``self.lock`` (re-entrant): the watcher's
flush, Orbital's own recorded writes and restore all serialise on it.
Additive and self-contained: an older Orbital ignores the folder.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
import zlib
from datetime import datetime, timezone

from agent_os.history import diffing, rules

logger = logging.getLogger(__name__)

STORE_CAP_BYTES = 500 * 1024 * 1024
# Prune down to this fraction of the cap so a store at the cap does not
# rewrite versions.jsonl on every flush.
PRUNE_TARGET_FRACTION = 0.9

VERSION_KINDS = ("baseline", "create", "modify", "delete", "restore")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def epoch_to_iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def iso_to_epoch(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(s)
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def history_dir(workspace: str) -> str:
    from agent_os.agent.project_paths import ProjectPaths
    return os.path.join(ProjectPaths(workspace).orbital_dir, "history")


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Temp file + os.replace, retried briefly (Windows: a reader may hold
    the target open). The store's own writer, separate from the workspace
    restore path."""
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                with contextlib.suppress(OSError):
                    os.remove(tmp)
                raise
            time.sleep(0.05)


class HistoryStore:
    """One project's store. Construct, then ``load()``."""

    def __init__(self, workspace: str, *, cap_bytes: int | None = None):
        self.workspace = workspace
        self.root = history_dir(workspace)
        self.blobs_dir = os.path.join(self.root, "blobs")
        self.versions_path = os.path.join(self.root, "versions.jsonl")
        self.index_path = os.path.join(self.root, "index.json")
        self.cap_bytes = STORE_CAP_BYTES if cap_bytes is None else cap_bytes
        self.lock = threading.RLock()
        self.versions: list[dict] = []
        self.names: dict[tuple[str, str], str] = {}
        self.index: dict[str, dict] = {}
        self.total_bytes = 0
        self._next_v = 1
        self._name_lines: list[dict] = []

    # ── load ──────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return os.path.isfile(self.versions_path)

    def load(self) -> None:
        with self.lock:
            self.versions, self._name_lines, self.names = [], [], {}
            if os.path.isfile(self.versions_path):
                with open(self.versions_path, "rb") as f:
                    for raw in f:
                        try:
                            rec = json.loads(raw)
                        except ValueError:
                            # a crash mid-append leaves a torn last line
                            continue
                        if not isinstance(rec, dict):
                            continue
                        self._ingest(rec)
            self._next_v = max((int(v.get("v") or 0) for v in self.versions), default=0) + 1
            self.index = self._read_index()
            if self.index is None:
                self.index = self._rebuild_index()
                self._save_index_locked()
            self.total_bytes = self._measure()

    def _ingest(self, rec: dict) -> None:
        kind = rec.get("kind")
        if kind == "name":
            self._name_lines.append(rec)
            key = (rec.get("path"), rec.get("sha"))
            if rec.get("name"):
                self.names[key] = rec["name"]
            else:
                self.names.pop(key, None)
        elif kind in VERSION_KINDS and rec.get("path"):
            self.versions.append(rec)

    def _read_index(self) -> dict | None:
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            return None
        return data["files"]

    def _rebuild_index(self) -> dict:
        idx: dict[str, dict] = {}
        for v in self.versions:
            if v.get("sha") is None:
                idx.pop(v["path"], None)
            else:
                idx[v["path"]] = {"sha": v["sha"], "size": v.get("size"), "mtime_ns": None}
        return idx

    def _measure(self) -> int:
        total = 0
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        return total

    # ── blobs ─────────────────────────────────────────────────────────

    def _blob_path(self, sha: str) -> str:
        return os.path.join(self.blobs_dir, sha[:2], sha)

    def has_blob(self, sha: str) -> bool:
        return os.path.isfile(self._blob_path(sha))

    def put_blob(self, data: bytes, sha: str | None = None) -> str:
        sha = sha or sha256(data)
        path = self._blob_path(sha)
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            comp = zlib.compress(data, 6)
            _atomic_write_bytes(path, comp)
            self.total_bytes += len(comp)
        return sha

    def get_blob(self, sha: str) -> bytes:
        with open(self._blob_path(sha), "rb") as f:
            return zlib.decompress(f.read())

    # ── writes ────────────────────────────────────────────────────────

    def _append_line(self, rec: dict) -> None:
        os.makedirs(self.root, exist_ok=True)
        line = (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with open(self.versions_path, "ab") as f:
            f.write(line)
        self.total_bytes += len(line)

    def _save_index_locked(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        payload = json.dumps({"format": 1, "files": self.index}, separators=(",", ":")).encode("utf-8")
        _atomic_write_bytes(self.index_path, payload)

    def save_index(self) -> None:
        with self.lock:
            self._save_index_locked()

    def current(self, path: str) -> dict | None:
        return self.index.get(path)

    def touch(self, path: str, size: int, mtime_ns: int | None) -> None:
        """Content unchanged: refresh the stat the reconcile compares."""
        entry = self.index.get(path)
        if entry is not None:
            entry["size"], entry["mtime_ns"] = size, mtime_ns

    def record(self, path: str, data: bytes | None, *, attrib: dict,
               kind: str | None = None, ts: str | None = None,
               mtime_ns: int | None = None, extra: dict | None = None,
               save_index: bool = True) -> dict | None:
        """Append a version for ``path`` with content ``data`` (None =
        deleted). Equal to the current hash → no version (the stat is
        refreshed). Returns the version line or None."""
        with self.lock:
            cur = self.index.get(path)
            prev_sha = cur["sha"] if cur else None
            if data is None:
                if cur is None:
                    return None
                new_sha, size = None, 0
                kind = kind or "delete"
            else:
                new_sha, size = sha256(data), len(data)
                if new_sha == prev_sha and kind != "restore":
                    self.touch(path, size, mtime_ns)
                    if save_index:
                        self._save_index_locked()
                    return None
                if kind is None:
                    kind = "modify" if cur is not None else "create"
                self.put_blob(data, new_sha)
            added = removed = None
            if not rules.is_office(path):
                prev_data = self._safe_blob(prev_sha) if prev_sha else None
                if kind == "baseline":
                    added, removed = 0, 0
                else:
                    added, removed = diffing.count_changes(prev_data, data)
            rec = {
                "v": self._next_v, "ts": ts or now_iso(), "path": path,
                "sha": new_sha, "prev_sha": prev_sha, "kind": kind, "size": size,
                "added": added, "removed": removed, "attrib": attrib,
            }
            if extra:
                rec.update(extra)
            self._next_v += 1
            self._append_line(rec)
            self.versions.append(rec)
            if new_sha is None:
                self.index.pop(path, None)
            else:
                self.index[path] = {"sha": new_sha, "size": size, "mtime_ns": mtime_ns}
            if save_index:
                self._save_index_locked()
            return rec

    def _safe_blob(self, sha: str | None) -> bytes | None:
        if not sha:
            return None
        try:
            return self.get_blob(sha)
        except (OSError, zlib.error):
            return None

    def set_name(self, path: str, sha: str, name: str) -> dict:
        with self.lock:
            rec = {"kind": "name", "ts": now_iso(), "path": path, "sha": sha, "name": name or ""}
            self._append_line(rec)
            self._ingest(rec)
            return rec

    # ── reads ─────────────────────────────────────────────────────────

    def versions_for(self, path: str) -> list[dict]:
        with self.lock:
            return [v for v in self.versions if v["path"] == path]

    def version_by_id(self, v: int) -> dict | None:
        with self.lock:
            for rec in self.versions:
                if rec.get("v") == v:
                    return rec
            return None

    def has_changes(self) -> bool:
        with self.lock:
            return any(v.get("kind") != "baseline" for v in self.versions)

    # ── retention ─────────────────────────────────────────────────────

    def prune_if_needed(self) -> int:
        """Size cap (§6): over the cap, drop the oldest versions first —
        never a file's current version or a named version — and delete
        blobs no remaining line references. Returns versions dropped."""
        with self.lock:
            if self.total_bytes <= self.cap_bytes:
                return 0
            target = int(self.cap_bytes * PRUNE_TARGET_FRACTION)
            latest: dict[str, int] = {}
            for i, rec in enumerate(self.versions):
                latest[rec["path"]] = i
            protected = set(latest.values())
            for i, rec in enumerate(self.versions):
                if rec.get("sha") and (rec["path"], rec["sha"]) in self.names:
                    protected.add(i)
            refs: dict[str, int] = {}
            for rec in self.versions:
                if rec.get("sha"):
                    refs[rec["sha"]] = refs.get(rec["sha"], 0) + 1
            for entry in self.index.values():
                refs[entry["sha"]] = refs.get(entry["sha"], 0) + 1
            drop: set[int] = set()
            freed: list[str] = []
            est = self.total_bytes
            for i, rec in enumerate(self.versions):
                if est <= target:
                    break
                if i in protected:
                    continue
                drop.add(i)
                est -= 200  # its jsonl line, roughly
                sha = rec.get("sha")
                if sha:
                    refs[sha] -= 1
                    if refs[sha] == 0:
                        try:
                            est -= os.path.getsize(self._blob_path(sha))
                        except OSError:
                            pass
                        freed.append(sha)
            if not drop:
                return 0
            self.versions = [r for i, r in enumerate(self.versions) if i not in drop]
            kept_keys = {(v["path"], v.get("sha")) for v in self.versions}
            self._name_lines = [n for n in self._name_lines
                                if (n.get("path"), n.get("sha")) in kept_keys]
            lines = self.versions + self._name_lines
            lines.sort(key=lambda r: (iso_to_epoch(r.get("ts")) or 0.0))
            body = b"".join(
                (json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                for r in lines)
            _atomic_write_bytes(self.versions_path, body)
            for sha in freed:
                try:
                    os.remove(self._blob_path(sha))
                except OSError:
                    pass
            self.total_bytes = self._measure()
            logger.info("file history: pruned %d versions, %d blobs in %s (now %d bytes)",
                        len(drop), len(freed), self.root, self.total_bytes)
            return len(drop)
