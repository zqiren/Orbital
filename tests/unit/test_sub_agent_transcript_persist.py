# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for SubAgentTranscript.open_for_handle — persistent per-(workspace,
handle) transcript reuse (BACKLOG 005 §4b) and the fresh-reset bypass (§4d).

§4b: every respawn used to mint ``str(uuid4())[:8].jsonl`` and rewrite
``.latest`` at an EMPTY file, so the UI's ``.latest`` transcript looked blank
across respawns. ``open_for_handle`` instead REUSES the file ``.latest`` already
points at (append mode), minting a new id only when ``.latest`` is
missing/corrupt or when ``fresh=True`` is requested.
"""

from __future__ import annotations

import os

from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript


def _sub_dir(filepath: str) -> str:
    return os.path.dirname(filepath)


def test_reuses_latest_file_on_respawn(tmp_path):
    ws = str(tmp_path)
    t1 = SubAgentTranscript.open_for_handle(ws, "researcher")
    t1.append({"source": "researcher", "content": "turn 1", "chunk_type": "response"})
    path1 = t1.filepath

    # A respawn for the same (workspace, handle) must REUSE the .latest file,
    # not mint a new uuid file — so the UI transcript stays continuous.
    t2 = SubAgentTranscript.open_for_handle(ws, "researcher")
    assert t2.filepath == path1
    t2.append({"source": "researcher", "content": "turn 2", "chunk_type": "response"})

    # Both turns are in the single continuous file, in order.
    contents = [e["content"] for e in SubAgentTranscript.read(path1)]
    assert contents == ["turn 1", "turn 2"]

    # Exactly ONE .jsonl file exists for the handle (no per-spawn proliferation).
    jsonl = [f for f in os.listdir(_sub_dir(path1)) if f.endswith(".jsonl")]
    assert len(jsonl) == 1


def test_fresh_mints_new_file_even_when_latest_exists(tmp_path):
    ws = str(tmp_path)
    t1 = SubAgentTranscript.open_for_handle(ws, "researcher")
    t1.append({"source": "researcher", "content": "old thread", "chunk_type": "response"})
    path1 = t1.filepath

    # fresh=True (§4d reset) bypasses reuse — a brand-new transcript file.
    t2 = SubAgentTranscript.open_for_handle(ws, "researcher", fresh=True)
    assert t2.filepath != path1

    # The prior transcript stays on disk (archived, not deleted).
    assert os.path.exists(path1)
    # .latest now points at the new file.
    with open(os.path.join(_sub_dir(path1), ".latest"), encoding="utf-8") as f:
        assert f.read().strip() == os.path.basename(t2.filepath)
    # Two distinct .jsonl files now exist for the handle.
    jsonl = [f for f in os.listdir(_sub_dir(path1)) if f.endswith(".jsonl")]
    assert len(jsonl) == 2


def test_mints_new_when_latest_missing(tmp_path):
    # No prior .latest → first spawn mints a fresh uuid file.
    ws = str(tmp_path)
    t1 = SubAgentTranscript.open_for_handle(ws, "fresh-handle")
    assert os.path.exists(t1.filepath)
    assert os.path.basename(t1.filepath).endswith(".jsonl")


def test_mints_new_when_latest_points_at_deleted_file(tmp_path):
    ws = str(tmp_path)
    t1 = SubAgentTranscript.open_for_handle(ws, "researcher")
    path1 = t1.filepath
    os.remove(path1)  # corrupt/missing target — .latest now dangles
    t2 = SubAgentTranscript.open_for_handle(ws, "researcher")
    # Falls back to minting a new file rather than reusing a dead pointer.
    assert t2.filepath != path1
    assert os.path.exists(t2.filepath)
