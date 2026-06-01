# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""QueueStore — file-backed CRUD for queue.json.

Atomic writes via tmp + replace, in-process serialization via a threading.Lock.
Designed to be the single source of truth for queue state. The dispatcher
reads and mutates through this; HTTP routes mutate through this; both go
through the same lock.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from agent_os.queue.models import (
    AttemptOutcome,
    AttemptRecord,
    ItemRecord,
    ItemState,
    QueueRunState,
    QueueState,
    Source,
)

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QueueStore:
    """File-backed queue state with atomic writes."""

    def __init__(self, queue_file: str | Path):
        self._path = Path(queue_file)
        self._lock = threading.RLock()
        self._state: Optional[QueueState] = None

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def load(self) -> QueueState:
        with self._lock:
            if self._state is not None:
                return self._state
            if not self._path.exists():
                self._state = QueueState()
                self._save_locked()
                return self._state
            try:
                text = self._path.read_text(encoding="utf-8")
                # Pre-validate migration: rename legacy state values
                # (draining → running, stopped → paused) in the raw JSON so
                # the strict pydantic enum doesn't reject them. Save back to
                # disk so the migration is applied permanently on first read.
                migrated_text, migrated = self._migrate_state_values(text)
                # Seam 3 / decision D3: one-time forward remap of any
                # F1-shaped AttemptRecord.session_id (the retired "sess_…" mint
                # or legacy "default") to the session's uuid, so resume/retry
                # route on the canonical id. Self-retiring: persisted on save,
                # later loads are no-ops.
                remapped_text, remapped = self._remap_attempt_session_ids(migrated_text)
                self._state = QueueState.model_validate_json(remapped_text)
                if migrated or remapped:
                    self._save_locked()
            except (OSError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "queue.json at %s is corrupt; starting from empty",
                    self._path, exc_info=True,
                )
                self._state = QueueState()
                self._save_locked()
            return self._state

    def _remap_attempt_session_ids(self, text: str) -> tuple[str, bool]:
        """Forward-remap F1-shaped attempt session_ids to the session uuid.

        queue.json lives at ``{workspace}/orbital/queue.json``; session JSONLs
        at ``{workspace}/orbital/sessions/{uuid}.jsonl``. We build an F1→uuid
        map from each log's ``session_start`` meta and rewrite any attempt
        session_id that is F1-shaped (``"default"`` or ``sess_…``) to its uuid.
        Unresolvable ids are left as-is (the resolver's instrumented F1-scan
        fallback still covers them). Idempotent. Returns (text, changed).
        """
        try:
            parsed = json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return text, False
        items = parsed.get("items") if isinstance(parsed, dict) else None
        if not items:
            return text, False

        def _is_f1(sid: object) -> bool:
            return isinstance(sid, str) and (sid == "default" or sid.startswith("sess_"))

        # Only build the (potentially I/O-heavy) F1→uuid map if some attempt
        # actually carries an F1-shaped id.
        if not any(_is_f1(a.get("session_id"))
                   for it in items for a in (it.get("attempts") or [])):
            return text, False

        f1_to_uuid: dict[str, str] = {}
        sessions_dir = self._path.parent / "sessions"
        try:
            fnames = os.listdir(sessions_dir)
        except OSError:
            fnames = []
        for fname in fnames:
            if not fname.endswith(".jsonl"):
                continue
            uuid = fname[:-6]
            try:
                with open(sessions_dir / fname, "r", encoding="utf-8") as fh:
                    for raw in fh:
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("role") == "meta" and rec.get("event") == "session_start":
                            f1 = rec.get("session_id")
                            if isinstance(f1, str):
                                f1_to_uuid.setdefault(f1, uuid)
                            break  # meta is the first line; stop scanning this file
            except OSError:
                continue

        changed = False
        for it in items:
            for a in (it.get("attempts") or []):
                sid = a.get("session_id")
                if _is_f1(sid) and sid in f1_to_uuid:
                    a["session_id"] = f1_to_uuid[sid]
                    changed = True
        if not changed:
            return text, False
        return json.dumps(parsed), True

    @staticmethod
    def _migrate_state_values(text: str) -> tuple[str, bool]:
        """Translate legacy queue.state values to the new vocabulary.

        Legacy → new mapping:
          "draining" → "running"
          "stopped"  → "paused"

        Returns (possibly-modified-json-text, was_migrated). Operates on the
        raw JSON before pydantic parsing so the strict QueueRunState enum
        accepts the file. Idempotent: leaves modern values untouched.
        """
        try:
            parsed = json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return text, False
        if not isinstance(parsed, dict):
            return text, False
        legacy = parsed.get("state")
        if legacy == "draining":
            parsed["state"] = "running"
        elif legacy == "stopped":
            parsed["state"] = "paused"
        else:
            return text, False
        return json.dumps(parsed), True

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        if self._state is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            self._state.model_dump_json(indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    # ------------------------------------------------------------------
    # Item CRUD
    # ------------------------------------------------------------------

    def add_item(
        self,
        content: str,
        *,
        file_refs: Optional[list[str]] = None,
        priority: int = 0,
        review_before_advance: bool = False,
        source: str = "user",
        idempotency_key: Optional[str] = None,
    ) -> ItemRecord:
        state = self.load()
        with self._lock:
            if idempotency_key:
                for existing in state.items:
                    if existing.idempotency_key == idempotency_key:
                        return existing
            try:
                src_enum = Source(source)
            except ValueError:
                src_enum = Source.USER
            item = ItemRecord(
                content=content,
                file_refs=file_refs or [],
                priority=priority,
                review_before_advance=review_before_advance,
                source=src_enum,
                idempotency_key=idempotency_key,
            )
            if priority == 1:
                insert_at = sum(
                    1 for it in state.items if it.state == ItemState.RUNNING
                )
                state.items.insert(insert_at, item)
            else:
                state.items.append(item)
            self._save_locked()
            return item

    def edit_item(
        self,
        item_id: str,
        *,
        content: Optional[str] = None,
        file_refs: Optional[list[str]] = None,
        priority: Optional[int] = None,
        review_before_advance: Optional[bool] = None,
    ) -> Optional[ItemRecord]:
        state = self.load()
        with self._lock:
            for item in state.items:
                if item.id == item_id:
                    if item.state != ItemState.QUEUED:
                        return None
                    if content is not None:
                        item.content = content
                    if file_refs is not None:
                        item.file_refs = file_refs
                    if priority is not None:
                        item.priority = priority
                    if review_before_advance is not None:
                        item.review_before_advance = review_before_advance
                    self._save_locked()
                    return item
            return None

    def remove_item(self, item_id: str) -> bool:
        state = self.load()
        with self._lock:
            for idx, item in enumerate(state.items):
                if item.id == item_id:
                    if item.state == ItemState.RUNNING and item.attempts:
                        latest = item.attempts[-1]
                        if latest.outcome is None:
                            latest.outcome = AttemptOutcome.CANCELLED
                            latest.ended_at = _now_iso()
                    state.items.pop(idx)
                    self._save_locked()
                    return True
            return False

    def reorder(self, item_ids: list[str]) -> None:
        state = self.load()
        with self._lock:
            by_id = {it.id: it for it in state.items}
            reordered: list[ItemRecord] = []
            for iid in item_ids:
                if iid in by_id:
                    reordered.append(by_id.pop(iid))
            reordered.extend(by_id.values())
            state.items = reordered
            self._save_locked()

    def move_to_head(self, item_id: str) -> bool:
        """Move an item to the front of the queue (after any RUNNING items).
        Used by reclaim to give an interrupted item priority on retry."""
        state = self.load()
        with self._lock:
            for idx, item in enumerate(state.items):
                if item.id == item_id:
                    state.items.pop(idx)
                    insert_at = sum(
                        1 for it in state.items if it.state == ItemState.RUNNING
                    )
                    state.items.insert(insert_at, item)
                    self._save_locked()
                    return True
            return False

    # ------------------------------------------------------------------
    # Item state transitions
    # ------------------------------------------------------------------

    def set_item_state(self, item_id: str, new_state: ItemState) -> None:
        state = self.load()
        with self._lock:
            for item in state.items:
                if item.id == item_id:
                    item.state = new_state
                    self._save_locked()
                    return

    def append_attempt(self, item_id: str, attempt: AttemptRecord) -> None:
        state = self.load()
        with self._lock:
            for item in state.items:
                if item.id == item_id:
                    item.attempts.append(attempt)
                    self._save_locked()
                    return

    def close_latest_attempt(
        self,
        item_id: str,
        *,
        outcome: AttemptOutcome,
        summary: Optional[str] = None,
        block_reason: Optional[str] = None,
    ) -> None:
        state = self.load()
        with self._lock:
            for item in state.items:
                if item.id == item_id and item.attempts:
                    latest = item.attempts[-1]
                    if latest.outcome is None:
                        latest.outcome = outcome
                        latest.ended_at = _now_iso()
                        if summary is not None:
                            latest.summary = summary
                        if block_reason is not None:
                            latest.block_reason = block_reason
                        self._save_locked()
                    return

    def increment_interrupted(self, item_id: str) -> int:
        state = self.load()
        with self._lock:
            for item in state.items:
                if item.id == item_id:
                    item.interrupted_count += 1
                    self._save_locked()
                    return item.interrupted_count
            return 0

    # ------------------------------------------------------------------
    # Queue-level state
    # ------------------------------------------------------------------

    def set_queue_state(self, new_state: QueueRunState) -> None:
        state = self.load()
        with self._lock:
            state.state = new_state
            self._save_locked()

    def auto_idle_if_empty(self) -> bool:
        """Transition the queue to IDLE when no queueable items remain.

        "Queueable" means items in QUEUED, RUNNING, or BLOCKED state — i.e.
        anything the dispatcher could ever pick up or surface. DONE items are
        terminal and ignored.

        Called by the dispatcher after each advance and on every main-loop
        tick. Triggers regardless of current state (RUNNING → IDLE on the
        last advance; PAUSED → IDLE when the user removes the last item
        from a paused queue). No-op when already IDLE or when queueable
        items remain.

        Returns True if the state was changed, False otherwise.
        """
        state = self.load()
        with self._lock:
            queueable = {ItemState.QUEUED, ItemState.RUNNING, ItemState.BLOCKED}
            if any(it.state in queueable for it in state.items):
                return False
            if state.state == QueueRunState.IDLE:
                return False
            state.state = QueueRunState.IDLE
            self._save_locked()
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def head(self) -> Optional[ItemRecord]:
        state = self.load()
        for item in state.items:
            if item.state in (ItemState.QUEUED, ItemState.RUNNING):
                return item
        return None

    def next_queued(self) -> Optional[ItemRecord]:
        state = self.load()
        for item in state.items:
            if item.state == ItemState.QUEUED:
                return item
        return None

    def snapshot(self) -> dict:
        state = self.load()
        return state.model_dump(mode="json")
