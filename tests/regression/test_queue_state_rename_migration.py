# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: queue.json with legacy state values (draining/stopped) is
migrated in-place on first load to the new vocabulary (running/paused).

Concern 1 of TASK-queue-architecture-amendments.md: on first read after
deploy, QueueStore.load() must translate persisted legacy state strings to
the modern enum values and re-save the file so subsequent loads stay clean.
"""

from __future__ import annotations

import json

import pytest

from agent_os.queue.models import QueueRunState
from agent_os.queue.store import QueueStore


def _write_legacy(tmp_path, state_value: str) -> None:
    """Write a queue.json with a legacy state value the new pydantic enum
    would reject. Bypasses QueueState.model_dump_json() on purpose."""
    payload = {
        "version": 1,
        "state": state_value,
        "items": [],
        "chat_session_id": None,
    }
    (tmp_path / "queue.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def test_legacy_draining_migrates_to_running(tmp_path):
    _write_legacy(tmp_path, "draining")
    store = QueueStore(tmp_path / "queue.json")

    state = store.load()
    assert state.state == QueueRunState.RUNNING

    # On-disk file must have been rewritten with the new value so the
    # migration is permanent — a second fresh store load sees "running".
    on_disk = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert on_disk["state"] == "running"


def test_legacy_stopped_migrates_to_paused(tmp_path):
    _write_legacy(tmp_path, "stopped")
    store = QueueStore(tmp_path / "queue.json")

    state = store.load()
    assert state.state == QueueRunState.PAUSED

    on_disk = json.loads((tmp_path / "queue.json").read_text(encoding="utf-8"))
    assert on_disk["state"] == "paused"


def test_migration_is_idempotent_for_modern_values(tmp_path):
    """A queue.json already in the new vocabulary loads without rewriting."""
    _write_legacy(tmp_path, "running")
    path = tmp_path / "queue.json"
    mtime_before = path.stat().st_mtime_ns

    store = QueueStore(path)
    state = store.load()
    assert state.state == QueueRunState.RUNNING

    # File should NOT have been re-saved (no migration needed)
    mtime_after = path.stat().st_mtime_ns
    assert mtime_after == mtime_before


def test_migration_preserves_items(tmp_path):
    """Migrating the state value must not damage the items list."""
    payload = {
        "version": 1,
        "state": "draining",
        "items": [
            {
                "id": "item_abc123def456",
                "content": "hello",
                "file_refs": [],
                "priority": 0,
                "review_before_advance": False,
                "state": "queued",
                "source": "user",
                "attempts": [],
                "idempotency_key": None,
                "interrupted_count": 0,
                "created_at": "2026-05-18T00:00:00+00:00",
            }
        ],
        "chat_session_id": None,
    }
    (tmp_path / "queue.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )

    store = QueueStore(tmp_path / "queue.json")
    state = store.load()
    assert state.state == QueueRunState.RUNNING
    assert len(state.items) == 1
    assert state.items[0].content == "hello"
