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

from pydantic import BaseModel, Field


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
    DRAINING = "draining"
    STOPPED = "stopped"


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
    version: int = 1
    state: QueueRunState = QueueRunState.DRAINING
    items: List[ItemRecord] = Field(default_factory=list)
    chat_session_id: Optional[str] = None
