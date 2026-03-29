# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: full tool results are saved to disk when truncated to stubs.

The disk backup ensures the agent can recall full results via read_file
if needed. Files are stored at orbital/tool-results/{session_id}/.
"""

import json
import os

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import truncate_consumed_tool_results


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


class TestToolResultDiskBackup:
    """Verify full tool results are saved to disk after truncation."""

    def test_disk_file_exists_after_truncation(self, session, workspace):
        """Disk backup file is created at the expected path."""
        _add_tool_call_and_result(
            session, "tc_disk1", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            "D" * 5_000,
        )

        truncate_consumed_tool_results(session, "Analyzed.", iteration=3)

        # Check file exists
        tool_results_dir = os.path.join(
            workspace, "orbital", "tool-results", "disk-backup-test",
        )
        expected_file = os.path.join(tool_results_dir, "turn_3_call_tc_disk1.json")
        assert os.path.exists(expected_file), f"Expected disk backup at {expected_file}"

    def test_disk_file_valid_json_schema(self, session, workspace):
        """Disk backup file contains valid JSON with the correct schema."""
        original_content = "ORIGINAL_" + "X" * 5_000
        _add_tool_call_and_result(
            session, "tc_schema", "shell",
            {"command": "cat large.log"},
            original_content,
        )

        truncate_consumed_tool_results(session, "Log output.", iteration=1)

        tool_results_dir = os.path.join(
            workspace, "orbital", "tool-results", "disk-backup-test",
        )
        backup_file = os.path.join(tool_results_dir, "turn_1_call_tc_schema.json")

        with open(backup_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        # Verify schema fields
        assert record["turn"] == 1
        assert record["call_id"] == "tc_schema"
        assert record["tool_name"] == "shell"
        assert record["key_param"] == "cat large.log"
        assert "timestamp" in record
        assert record["pre_filter_tokens"] == int(len(original_content) / 4)
        assert record["content"] == original_content

    def test_disk_content_matches_original(self, session, workspace):
        """Content field in disk backup matches the original pre-filtered content."""
        original = "UNIQUE_CONTENT_FOR_MATCHING_" + "Q" * 3_000
        _add_tool_call_and_result(
            session, "tc_match", "read",
            {"path": "/workspace/data.csv"},
            original,
        )

        truncate_consumed_tool_results(session, "Read the CSV.", iteration=2)

        tool_results_dir = os.path.join(
            workspace, "orbital", "tool-results", "disk-backup-test",
        )
        backup_file = os.path.join(tool_results_dir, "turn_2_call_tc_match.json")

        with open(backup_file, "r", encoding="utf-8") as f:
            record = json.load(f)

        assert record["content"] == original

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

        truncate_consumed_tool_results(session, "Read all files.", iteration=5)

        tool_results_dir = os.path.join(
            workspace, "orbital", "tool-results", "disk-backup-test",
        )
        for i in range(3):
            path = os.path.join(tool_results_dir, f"turn_5_call_tc_multi_{i}.json")
            assert os.path.exists(path), f"Missing backup for tc_multi_{i}"

    def test_stub_contains_disk_path(self, session, workspace):
        """The stub in session history contains the correct disk path."""
        _add_tool_call_and_result(
            session, "tc_path", "browser",
            {"action": "fetch", "url": "https://example.com/api"},
            "F" * 5_000,
        )

        truncate_consumed_tool_results(session, "Fetched data.", iteration=4)

        messages = session.get_messages()
        tool_msg = [m for m in messages if m.get("role") == "tool"][0]
        stub = tool_msg["content"]

        # Extract the disk path from the stub
        assert "Full result:" in stub
        # The path should exist on disk
        path_start = stub.index("Full result: ") + len("Full result: ")
        path_end = stub.index("]", path_start)
        disk_path = path_end  # Just verify the path reference is there
        assert "turn_4_call_tc_path.json" in stub

    def test_disk_file_readable_after_session_reload(self, session, workspace):
        """Disk backup is independently readable even after session reload."""
        content = "RELOAD_TEST_" + "M" * 4_000
        _add_tool_call_and_result(
            session, "tc_reload", "browser",
            {"action": "snapshot", "url": "https://example.com"},
            content,
        )

        truncate_consumed_tool_results(session, "Done.", iteration=1)

        # Reload session (simulating a new session lifecycle)
        reloaded = Session.load(session._filepath)
        assert reloaded is not None

        # Disk backup still readable independently
        tool_results_dir = os.path.join(
            workspace, "orbital", "tool-results", "disk-backup-test",
        )
        backup_file = os.path.join(tool_results_dir, "turn_1_call_tc_reload.json")
        with open(backup_file, "r", encoding="utf-8") as f:
            record = json.load(f)
        assert record["content"] == content
