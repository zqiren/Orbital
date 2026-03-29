# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for TASK-V5-04: chat history composite timeline."""

import json
import os
import tempfile

import pytest

from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


def _write_session_jsonl(sessions_dir: str, messages: list[dict]) -> str:
    """Write messages to a session JSONL file."""
    os.makedirs(sessions_dir, exist_ok=True)
    fpath = os.path.join(sessions_dir, "session-001.jsonl")
    with open(fpath, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return fpath


def _write_transcript(workspace: str, handle: str, entries: list[dict]) -> str:
    """Write entries to a sub-agent transcript file."""
    transcript = SubAgentTranscript(workspace, handle, "test-txn")
    for entry in entries:
        transcript.append(entry)
    return transcript.filepath


class TestGetAllTranscriptEntries:
    def test_reads_transcript_entries_from_disk(self):
        """Disk scan finds transcript JSONL files and returns entries."""
        with tempfile.TemporaryDirectory() as workspace:
            _write_transcript(workspace, "claude-code", [
                {"source": "claude-code", "content": "Hello", "timestamp": "2026-03-10T10:00:02Z"},
                {"source": "claude-code", "content": "Done", "timestamp": "2026-03-10T10:00:04Z"},
            ])

            # Create manager with project_store that returns this workspace
            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(
                process_manager=None,
                project_store=project_store,
            )

            entries = mgr.get_all_transcript_entries("proj1")
            assert len(entries) == 2
            assert entries[0]["content"] == "Hello"
            assert entries[1]["content"] == "Done"
            assert entries[0]["source"] == "claude-code"

    def test_reads_from_in_memory_transcripts(self):
        """In-memory transcripts are read when disk scan finds nothing extra."""
        with tempfile.TemporaryDirectory() as workspace:
            transcript = SubAgentTranscript(workspace, "test-agent", "txn-001")
            transcript.append({"source": "test-agent", "content": "Hi", "timestamp": "2026-03-10T10:00:01Z"})

            mgr = SubAgentManager(process_manager=None)
            mgr._transcripts[("proj1", "test-agent")] = transcript

            entries = mgr.get_all_transcript_entries("proj1")
            assert len(entries) == 1
            assert entries[0]["content"] == "Hi"

    def test_deduplicates_disk_and_memory(self):
        """Same file found via disk and memory is only read once."""
        with tempfile.TemporaryDirectory() as workspace:
            transcript = SubAgentTranscript(workspace, "claude-code", "txn-001")
            transcript.append({"source": "claude-code", "content": "msg1", "timestamp": "2026-03-10T10:00:01Z"})

            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(
                process_manager=None,
                project_store=project_store,
            )
            mgr._transcripts[("proj1", "claude-code")] = transcript

            entries = mgr.get_all_transcript_entries("proj1")
            # Should have exactly 1 entry, not duplicated
            assert len(entries) == 1

    def test_empty_when_no_transcripts(self):
        """Returns empty list when no transcripts exist."""
        with tempfile.TemporaryDirectory() as workspace:
            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(
                process_manager=None,
                project_store=project_store,
            )
            entries = mgr.get_all_transcript_entries("proj1")
            assert entries == []

    def test_multiple_handles_merged(self):
        """Transcripts from multiple sub-agents are merged."""
        with tempfile.TemporaryDirectory() as workspace:
            _write_transcript(workspace, "claude-code", [
                {"source": "claude-code", "content": "from cc", "timestamp": "2026-03-10T10:00:01Z"},
            ])
            _write_transcript(workspace, "aider", [
                {"source": "aider", "content": "from aider", "timestamp": "2026-03-10T10:00:02Z"},
            ])

            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(
                process_manager=None,
                project_store=project_store,
            )
            entries = mgr.get_all_transcript_entries("proj1")
            sources = {e["source"] for e in entries}
            assert "claude-code" in sources
            assert "aider" in sources


class TestChatCompositeEndpoint:
    """Tests for the merged chat history endpoint logic.

    These test the _read_chat_messages + transcript merge logic
    without spinning up a full FastAPI test client.
    """

    def test_chat_returns_merged_timeline(self):
        """Management messages + transcript entries merged and sorted by timestamp."""
        with tempfile.TemporaryDirectory() as workspace:
            from agent_os.daemon_v2.project_store import project_dir_name
            sessions_dir = os.path.join(workspace, "orbital", project_dir_name("test", "proj1"), "sessions")
            _write_session_jsonl(sessions_dir, [
                {"role": "user", "content": "Hello", "timestamp": "2026-03-10T10:00:01Z"},
                {"role": "assistant", "content": "Hi there", "timestamp": "2026-03-10T10:00:03Z"},
            ])

            _write_transcript(workspace, "claude-code", [
                {"source": "claude-code", "content": "Sub msg 1", "timestamp": "2026-03-10T10:00:02Z"},
                {"source": "claude-code", "content": "Sub msg 2", "timestamp": "2026-03-10T10:00:04Z"},
                {"source": "claude-code", "content": "Sub msg 3", "timestamp": "2026-03-10T10:00:05Z"},
            ])

            # Read management messages
            from agent_os.api.routes.agents_v2 import _read_chat_messages
            mgmt_msgs, _ = _read_chat_messages(sessions_dir, 0, 0)

            # Read transcript entries
            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(process_manager=None, project_store=project_store)
            sub_entries = mgr.get_all_transcript_entries("proj1")
            for entry in sub_entries:
                entry.setdefault("role", "agent")

            # Merge and sort
            all_msgs = mgmt_msgs + sub_entries
            all_msgs.sort(key=lambda m: m.get("timestamp", ""))

            assert len(all_msgs) == 5
            # Verify chronological order
            timestamps = [m["timestamp"] for m in all_msgs]
            assert timestamps == sorted(timestamps)
            # Verify sub-agent messages have role=agent
            sub_msgs = [m for m in all_msgs if m.get("source") == "claude-code"]
            assert all(m["role"] == "agent" for m in sub_msgs)

    def test_chat_works_with_no_transcripts(self):
        """No transcripts → returns management messages only."""
        with tempfile.TemporaryDirectory() as workspace:
            from agent_os.daemon_v2.project_store import project_dir_name
            sessions_dir = os.path.join(workspace, "orbital", project_dir_name("test", "proj1"), "sessions")
            _write_session_jsonl(sessions_dir, [
                {"role": "user", "content": "Hello", "timestamp": "2026-03-10T10:00:01Z"},
            ])

            from agent_os.api.routes.agents_v2 import _read_chat_messages
            msgs, total = _read_chat_messages(sessions_dir, 0, 0)
            assert len(msgs) == 1
            assert total == 1

    def test_chat_works_with_no_session(self):
        """No session files → returns sub-agent messages only."""
        with tempfile.TemporaryDirectory() as workspace:
            _write_transcript(workspace, "claude-code", [
                {"source": "claude-code", "content": "Only sub", "timestamp": "2026-03-10T10:00:01Z"},
            ])

            from agent_os.api.routes.agents_v2 import _read_chat_messages
            sessions_dir = os.path.join(workspace, "orbital", "nonexistent", "sessions")
            mgmt_msgs, _ = _read_chat_messages(sessions_dir, 0, 0)
            assert mgmt_msgs == []

            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(process_manager=None, project_store=project_store)
            sub_entries = mgr.get_all_transcript_entries("proj1")
            for entry in sub_entries:
                entry.setdefault("role", "agent")

            all_msgs = mgmt_msgs + sub_entries
            assert len(all_msgs) == 1
            assert all_msgs[0]["role"] == "agent"
            assert all_msgs[0]["source"] == "claude-code"

    def test_transcript_entries_normalized_to_agent_role(self):
        """Transcript entries without role get role=agent added."""
        with tempfile.TemporaryDirectory() as workspace:
            _write_transcript(workspace, "claude-code", [
                {"source": "claude-code", "content": "No role field", "timestamp": "2026-03-10T10:00:01Z"},
            ])

            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(process_manager=None, project_store=project_store)
            entries = mgr.get_all_transcript_entries("proj1")

            # Before normalization: no role
            assert "role" not in entries[0]

            # After normalization (as done in endpoint):
            for entry in entries:
                entry.setdefault("role", "agent")

            assert entries[0]["role"] == "agent"

    def test_disk_scan_finds_previous_session_transcripts(self):
        """Transcript files on disk (not in _transcripts dict) are found."""
        with tempfile.TemporaryDirectory() as workspace:
            # Write transcript directly to disk without registering in manager
            sub_dir = os.path.join(workspace, "orbital", "sub_agents", "old-agent")
            os.makedirs(sub_dir, exist_ok=True)
            fpath = os.path.join(sub_dir, "old-session.jsonl")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "source": "old-agent",
                    "content": "From previous session",
                    "timestamp": "2026-03-09T10:00:01Z",
                }) + "\n")

            project_store = type("PS", (), {
                "get_project": lambda self, pid: {"workspace": workspace}
            })()
            mgr = SubAgentManager(process_manager=None, project_store=project_store)
            # _transcripts is empty — no active transcripts
            assert len(mgr._transcripts) == 0

            entries = mgr.get_all_transcript_entries("proj1")
            assert len(entries) == 1
            assert entries[0]["source"] == "old-agent"
            assert entries[0]["content"] == "From previous session"
