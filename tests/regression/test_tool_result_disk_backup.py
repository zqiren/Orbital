# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: full tool results are archived to disk, unconditionally.

Since spec 066 (phase 1a) the archive is content-addressed: each session's
orbital/tool-results/{session_uuid}/archive.jsonl manifest has one line per
archived call (turn, call id, tool, target, hash) and the content itself lives
once per distinct bytes in orbital/tool-results/blobs/. The archive is
DECOUPLED from history rewriting: it happens for every result over the size
threshold whether or not that result is ever stubbed. It is load-bearing
observability (the corpus behind the tool-result lifecycle investigation) and
the recovery path for content a supersession stub replaced.
"""

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
    return str(tmp_path)


@pytest.fixture
def session(workspace):
    return Session.new("disk-backup-test", workspace)


def _add_tool_call_and_result(session, call_id, tool_name, arguments, content):
    """Helper: add an assistant tool_call message and its tool result."""
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            },
        }],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


def _archive(session, workspace, call_id):
    """(manifest entry, archived content) for one call, or (None, None)."""
    for entry in read_archive_manifest(session):
        if entry["call_id"] == call_id:
            path = blob_store.blob_path(
                os.path.join(workspace, "orbital"), entry["sha256"], entry["ext"],
            )
            with open(path, "r", encoding="utf-8") as f:
                return entry, f.read()
    return None, None


class TestToolResultDiskBackup:
    """Verify full tool results are saved to disk."""

    def test_disk_file_exists_for_unstubbed_result(self, session, workspace):
        """Disk archive is created even when history is left untouched."""
        _add_tool_call_and_result(
            session, "tc_disk1", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "D" * 5_000,
        )

        archive_and_supersede_tool_results(session, iteration=3)

        # History is NOT rewritten — a single fetch is never superseded.
        tool_msg = [m for m in session.get_messages() if m.get("role") == "tool"][0]
        assert not tool_msg.get("_stubbed")

        entry, content = _archive(session, workspace, "tc_disk1")
        assert entry is not None, "Expected a disk archive for tc_disk1"
        assert entry["turn"] == 3
        assert content == "D" * 5_000

    def test_disk_file_valid_json_schema(self, session, workspace):
        """Disk backup file contains valid JSON with the correct schema."""
        original_content = "ORIGINAL_" + "X" * 5_000
        _add_tool_call_and_result(
            session, "tc_schema", "shell",
            {"command": "cat large.log"},
            original_content,
        )

        archive_and_supersede_tool_results(session, iteration=1)

        record, content = _archive(session, workspace, "tc_schema")

        # Verify schema fields (manifest line) and the archived content (blob)
        assert record["turn"] == 1
        assert record["call_id"] == "tc_schema"
        assert record["tool_name"] == "shell"
        assert record["key_param"] == "cat large.log"
        assert "timestamp" in record
        assert record["pre_filter_tokens"] == int(len(original_content) / 4)
        assert content == original_content

    def test_disk_content_matches_original(self, session, workspace):
        """Content field in disk backup matches the original pre-filtered content."""
        original = "UNIQUE_CONTENT_FOR_MATCHING_" + "Q" * 3_000
        _add_tool_call_and_result(
            session, "tc_match", "read",
            {"path": "/workspace/data.csv"},
            original,
        )

        archive_and_supersede_tool_results(session, iteration=2)

        _entry, content = _archive(session, workspace, "tc_match")
        assert content == original

    def test_multiple_backups_for_multiple_tools(self, session, workspace):
        """Each tool result gets its own disk backup file."""
        session.append({
            "role": "assistant",
            "tool_calls": [
                {"id": f"tc_multi_{i}", "type": "function",
                 "function": {"name": "read", "arguments": json.dumps({"path": f"f{i}.txt"})}}
                for i in range(3)
            ],
            "source": "management",
        })
        for i in range(3):
            session.append_tool_result(f"tc_multi_{i}", "R" * 2_000)

        archive_and_supersede_tool_results(session, iteration=5)

        for i in range(3):
            entry, _content = _archive(session, workspace, f"tc_multi_{i}")
            assert entry is not None, f"Missing backup for tc_multi_{i}"
            assert entry["turn"] == 5

    def test_superseded_stub_contains_disk_path(self, session, workspace):
        """The one stub that exists carries the path to the archived content."""
        _add_tool_call_and_result(
            session, "tc_path", "browser",
            {"action": "fetch", "url": "https://example.com/api"},
            "F" * 5_000,
        )
        _add_tool_call_and_result(
            session, "tc_path_2", "browser",
            {"action": "fetch", "url": "https://example.com/api"},
            "G" * 5_000,
        )

        archive_and_supersede_tool_results(session, iteration=4)

        messages = session.get_messages()
        stub = [
            m for m in messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "tc_path"
        ][0]["content"]

        assert "Full result:" in stub
        assert "tool-results/blobs/" in stub
        # The path in the stub actually resolves to the archived content.
        marker = "Full result: "
        disk_path = stub[stub.index(marker) + len(marker):].rstrip("]")
        with open(disk_path, "r", encoding="utf-8") as f:
            assert f.read() == "F" * 5_000

    def test_archive_is_written_once_per_call_id(self, session, workspace):
        """Repeated invocations must not duplicate a result under a new turn_N."""
        _add_tool_call_and_result(
            session, "tc_dedupe", "read", {"path": "stable.md"}, "S" * 3_000,
        )

        archive_and_supersede_tool_results(session, iteration=1)
        archive_and_supersede_tool_results(session, iteration=2)

        entries = read_archive_manifest(session)
        assert [(e["call_id"], e["turn"]) for e in entries] == [("tc_dedupe", 1)]

    def test_disk_file_readable_after_session_reload(self, session, workspace):
        """Disk backup is independently readable even after session reload."""
        content = "RELOAD_TEST_" + "M" * 4_000
        _add_tool_call_and_result(
            session, "tc_reload", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            content,
        )

        archive_and_supersede_tool_results(session, iteration=1)

        # Reload session (simulating a new session lifecycle)
        reloaded = Session.load(session._filepath)
        assert reloaded is not None

        # Disk backup still readable independently
        _entry, archived = _archive(reloaded, workspace, "tc_reload")
        assert archived == content
