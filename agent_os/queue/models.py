# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pydantic models for the task queue (D2 schema).

Stable across phases; new optional fields are added at the tail so existing
queue.json files keep deserializing. The `version` field on QueueState
allows future migration if a non-additive change is ever needed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Source(str, Enum):
    USER = "user"
    UPLOAD = "upload"


class ItemState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    BLOCKED = "blocked"


class AttemptOutcome(str, Enum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class QueueRunState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    IDLE = "idle"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_item_id() -> str:
    return f"item_{uuid4().hex[:12]}"


class AttemptRecord(BaseModel):
    session_id: str
    started_at: str = Field(default_factory=_now_iso)
    ended_at: Optional[str] = None
    outcome: Optional[AttemptOutcome] = None
    summary: Optional[str] = None
    block_reason: Optional[str] = None


class ItemRecord(BaseModel):
    id: str = Field(default_factory=_new_item_id)
    content: str
    file_refs: List[str] = Field(default_factory=list)
    priority: int = 0
    review_before_advance: bool = False
    state: ItemState = ItemState.QUEUED
    source: Source = Source.USER
    attempts: List[AttemptRecord] = Field(default_factory=list)
    idempotency_key: Optional[str] = None
    interrupted_count: int = 0
    created_at: str = Field(default_factory=_now_iso)


class QueueState(BaseModel):
    # extra="ignore" lets us deserialize older queue.json files that still
    # carry the now-removed chat_session_id field (and any other tail
    # fields added then dropped) without raising ValidationError.
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    state: QueueRunState = QueueRunState.RUNNING
    items: List[ItemRecord] = Field(default_factory=list)
    # When state == PAUSED and the user chose a snooze duration, the absolute
    # UTC ISO deadline after which the dispatcher auto-resumes. None means
    # "paused until explicitly resumed" (the default). Cleared on any
    # transition out of PAUSED.
    paused_until: Optional[str] = None
