# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 066 phase 1a — the tool-result archive is content-addressed.

Every large tool result is still archived exactly once per call id, and a
superseded stub still points at a file holding the full content. What
changed: the content is stored once per distinct bytes in the blob store,
and each session keeps one append-only manifest (call id, turn, tool, target,
hash) instead of one JSON file per call. Archives written by earlier versions
(``turn_N_call_<id>.json``) are still honoured and never rewritten.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_os.agent import blob_store
from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import (
    archive_and_supersede_tool_results,
    read_archive_manifest,
)


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path / "ws")


@pytest.fixture
def session(workspace):
    return Session.new("proj_archive1", workspace)


def _fetch(session, call_id, tool_name, arguments, content):
    session.append({
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": tool_name, "arguments": json.dumps(arguments)}}],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


def _session_dir(workspace, uuid="proj_archive1"):
    return os.path.join(workspace, "orbital", "tool-results", uuid)


def _text_blobs(workspace):
    root = blob_store.blobs_dir(os.path.join(workspace, "orbital"))
    out = []
    for dirpath, _d, files in os.walk(root):
        out.extend(os.path.join(dirpath, f) for f in files if f.endswith(".txt"))
    return out


def test_identical_results_in_different_calls_store_one_blob(session, workspace):
    body = "SAME_" + "x" * 5_000
    _fetch(session, "c1", "shell", {"command": "cat a"}, body)
    _fetch(session, "c2", "shell", {"command": "cat a"}, body)
    _fetch(session, "c3", "shell", {"command": "cat b"}, "OTHER_" + "y" * 5_000)

    archive_and_supersede_tool_results(session, iteration=1)

    entries = read_archive_manifest(session)
    assert [e["call_id"] for e in entries] == ["c1", "c2", "c3"]
    assert entries[0]["sha256"] == entries[1]["sha256"] != entries[2]["sha256"]
    assert len(_text_blobs(workspace)) == 2


def test_manifest_keeps_the_per_call_metadata(session, workspace):
    body = "ORIGINAL_" + "X" * 5_000
    _fetch(session, "tc_schema", "shell", {"command": "cat large.log"}, body)
    archive_and_supersede_tool_results(session, iteration=3)

    (entry,) = read_archive_manifest(session)
    assert entry["turn"] == 3
    assert entry["call_id"] == "tc_schema"
    assert entry["tool_name"] == "shell"
    assert entry["key_param"] == "cat large.log"
    assert entry["pre_filter_tokens"] == int(len(body) / 4)
    assert "timestamp" in entry
    path = blob_store.blob_path(os.path.join(workspace, "orbital"), entry["sha256"], "txt")
    with open(path, encoding="utf-8") as f:
        assert f.read() == body


def test_session_dir_holds_one_manifest_not_one_file_per_call(session, workspace):
    for i in range(5):
        _fetch(session, f"c{i}", "read", {"path": f"f{i}.md"}, f"{i}" * 3_000)
    archive_and_supersede_tool_results(session, iteration=1)
    assert os.listdir(_session_dir(workspace)) == ["archive.jsonl"]


def test_archived_once_per_call_id_across_iterations(session, workspace):
    _fetch(session, "tc_once", "read", {"path": "stable.md"}, "S" * 3_000)
    for it in (1, 2, 3):
        archive_and_supersede_tool_results(session, iteration=it)
    entries = read_archive_manifest(session)
    assert [(e["call_id"], e["turn"]) for e in entries] == [("tc_once", 1)]


def test_superseded_stub_path_holds_the_raw_content(session):
    _fetch(session, "tc_p1", "read", {"path": "doc.md"}, "OLDBYTES_" + "O" * 3_000)
    _fetch(session, "tc_p2", "read", {"path": "doc.md"}, "NEWBYTES_" + "W" * 3_000)
    archive_and_supersede_tool_results(session, iteration=6)

    stub = [m for m in session.get_messages() if m.get("tool_call_id") == "tc_p1"][0]["content"]
    marker = "Full result: "
    disk_path = stub[stub.index(marker) + len(marker):].rstrip("]")
    with open(disk_path, encoding="utf-8") as f:
        assert f.read() == "OLDBYTES_" + "O" * 3_000


def test_legacy_json_archive_is_honoured_and_left_alone(session, workspace):
    """A call archived by an earlier version is not archived a second time,
    and its file is never touched."""
    os.makedirs(_session_dir(workspace))
    legacy = os.path.join(_session_dir(workspace), "turn_1_call_old_1.json")
    record = {"turn": 1, "call_id": "old_1", "content": "L" * 3_000}
    with open(legacy, "w", encoding="utf-8") as f:
        json.dump(record, f)
    before = open(legacy, encoding="utf-8").read()

    _fetch(session, "old_1", "read", {"path": "a.md"}, "L" * 3_000)
    _fetch(session, "old_2", "read", {"path": "a.md"}, "M" * 3_000)
    archive_and_supersede_tool_results(session, iteration=2)

    assert [e["call_id"] for e in read_archive_manifest(session)] == ["old_2"]
    assert open(legacy, encoding="utf-8").read() == before
    stub = [m for m in session.get_messages() if m.get("tool_call_id") == "old_1"][0]["content"]
    assert legacy in stub, "the stub of a legacy-archived call points at its legacy file"


def test_manifest_survives_a_torn_last_line(session, workspace):
    _fetch(session, "c1", "read", {"path": "a.md"}, "A" * 3_000)
    archive_and_supersede_tool_results(session, iteration=1)
    with open(os.path.join(_session_dir(workspace), "archive.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"call_id": "half')
    _fetch(session, "c2", "read", {"path": "b.md"}, "B" * 3_000)
    archive_and_supersede_tool_results(session, iteration=2)
    assert [e["call_id"] for e in read_archive_manifest(session)] == ["c1", "c2"]
