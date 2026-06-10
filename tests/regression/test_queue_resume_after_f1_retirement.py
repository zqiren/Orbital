# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 4 / decision D3: queue.json one-time forward F1→uuid remap on load.

Existing parked/in-flight queue items recorded an F1 id in
``AttemptRecord.session_id`` (the retired ``sess_…`` mint, or the legacy
``"default"``). After F1 retirement, resume/retry must route on the uuid. On
load, the store remaps any F1-shaped attempt session_id to the session's uuid
(by matching the on-disk JSONL whose session_start meta carries that F1), and
persists the remap on next save. The fix is self-retiring: once rewritten,
later loads are no-ops.
"""

from __future__ import annotations

import json
import os

from agent_os.agent.project_paths import ProjectPaths
from agent_os.queue.store import QueueStore


def _seed_session(workspace, uuid, f1):
    sdir = ProjectPaths(workspace).sessions_dir
    os.makedirs(sdir, exist_ok=True)
    rows = [
        {"role": "meta", "event": "session_start", "session_id": f1,
         "session_uuid": uuid, "origin": "queue", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"role": "user", "content": "do the thing", "session_id": f1,
         "session_uuid": uuid, "timestamp": "2026-01-01T00:00:01+00:00"},
    ]
    with open(os.path.join(sdir, f"{uuid}.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def _seed_queue(workspace, attempt_session_id):
    qpath = os.path.join(ProjectPaths(workspace).orbital_dir, "queue.json")
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    state = {
        "version": 1,
        "state": "paused",
        "items": [
            {"id": "item_1", "content": "do the thing", "state": "running",
             "attempts": [{"session_id": attempt_session_id}]},
        ],
    }
    with open(qpath, "w") as f:
        json.dump(state, f)
    return qpath


def test_sess_f1_attempt_is_remapped_to_uuid_on_load(tmp_path):
    ws = str(tmp_path / "ws")
    _seed_session(ws, uuid="proj_q_aaaa1111", f1="sess_oldqueue")
    qpath = _seed_queue(ws, attempt_session_id="sess_oldqueue")

    store = QueueStore(qpath)
    state = store.load()

    sid = state.items[0].attempts[-1].session_id
    assert sid == "proj_q_aaaa1111", (
        f"D3: parked attempt's F1 must be remapped to the uuid, got {sid!r}"
    )
    # Persisted: a fresh store reads the remapped uuid directly.
    assert json.load(open(qpath))["items"][0]["attempts"][0]["session_id"] == "proj_q_aaaa1111"


def test_default_f1_attempt_is_remapped_to_uuid_on_load(tmp_path):
    ws = str(tmp_path / "ws")
    _seed_session(ws, uuid="proj_q_bbbb2222", f1="default")
    qpath = _seed_queue(ws, attempt_session_id="default")

    state = QueueStore(qpath).load()
    assert state.items[0].attempts[-1].session_id == "proj_q_bbbb2222"


def test_uuid_attempt_is_left_untouched(tmp_path):
    ws = str(tmp_path / "ws")
    _seed_session(ws, uuid="proj_q_cccc3333", f1="sess_whatever")
    # Already a uuid (filename) — no remap, no scan needed.
    qpath = _seed_queue(ws, attempt_session_id="proj_q_cccc3333")
    state = QueueStore(qpath).load()
    assert state.items[0].attempts[-1].session_id == "proj_q_cccc3333"
