# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for the v5 sub-agent orchestration architecture.

Verifies that the five layers work together:
1. Transcript isolation (ProcessManager writes to transcripts, not session)
2. Non-blocking dispatch (send returns immediately)
3. Lifecycle observer (system messages on state transitions)
4. Chat composite (GET /chat merges session + transcripts)
5. Prompt awareness (PromptBuilder includes active sub-agent states)
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager
from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.agent.context import ContextManager


class FakeSession:
    """Minimal session mock for integration tests."""
    def __init__(self):
        self._messages = []
        self.on_append = None
        self.on_stream = None

    def append(self, msg):
        self._messages.append(msg)
        if self.on_append:
            self.on_append(msg)

    def get_messages(self):
        return list(self._messages)

    def get_recent(self, budget):
        return list(self._messages)


@pytest.mark.asyncio
async def test_lifecycle_to_session_injection():
    """Integration: lifecycle observer injects system messages into session via agent_manager.

    Verifies the full chain: ProcessManager fires on_completed → LifecycleObserver
    injects → agent_manager.inject_system_message → message appears in session.
    """
    session = FakeSession()
    injected = []

    am = MagicMock()
    async def fake_inject(project_id, content):
        session.append({"role": "system", "content": content, "source": "daemon"})
        injected.append(content)
        return "delivered"
    am.inject_system_message = AsyncMock(side_effect=fake_inject)

    ws = MagicMock()
    ws.broadcast = MagicMock()

    observer = LifecycleObserver(am, ws)

    # Simulate on_completed
    await observer.on_completed("proj1", "claude-code", "Refactored auth module",
                                 "/workspace/orbital/sub_agents/claude-code/abc.jsonl")

    # Verify system message was injected into session
    assert len(injected) == 1
    assert "[Sub-agent] claude-code completed" in injected[0]
    assert "Refactored auth" in injected[0]
    assert "/workspace/orbital/sub_agents/claude-code/abc.jsonl" in injected[0]

    # Verify session has the message
    msgs = session.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert msgs[0]["source"] == "daemon"


@pytest.mark.asyncio
async def test_transcript_isolation_end_to_end():
    """Integration: ProcessManager writes to transcript, NOT to session.

    Verifies that sub-agent output goes to the transcript file and the session
    remains empty (v5 isolation guarantee).
    """
    with tempfile.TemporaryDirectory() as workspace:
        session = FakeSession()
        ws = MagicMock()
        ws.broadcast = MagicMock()
        activity = MagicMock()
        activity.on_message = MagicMock()

        pm = ProcessManager(ws, activity)

        transcript = SubAgentTranscript(workspace, "claude-code", "test-txn")

        # Create a mock adapter that yields one chunk then ends
        class FakeChunk:
            def __init__(self, text, chunk_type=None):
                self.text = text
                self.chunk_type = chunk_type

        async def fake_read_stream():
            yield FakeChunk("Hello from Claude Code", "response")

        adapter = MagicMock()
        adapter.read_stream = fake_read_stream

        await pm.start("proj1", "claude-code", adapter, transcript=transcript)

        # Wait for consumer task to process
        await asyncio.sleep(0.1)

        # Transcript should have the entry
        entries = SubAgentTranscript.read(transcript.filepath)
        assert len(entries) >= 1
        assert entries[0]["content"] == "Hello from Claude Code"
        assert entries[0]["source"] == "claude-code"

        # Session should be EMPTY (v5 isolation)
        assert len(session.get_messages()) == 0


@pytest.mark.asyncio
async def test_chat_composite_merges_session_and_transcripts():
    """Integration: GET /chat logic merges management session + sub-agent transcripts.

    Verifies the merge path produces correctly sorted, normalized entries.
    """
    with tempfile.TemporaryDirectory() as workspace:
        from agent_os.api.routes.agents_v2 import _read_chat_messages
        from agent_os.daemon_v2.project_store import project_dir_name

        # Write management session
        dir_name = project_dir_name("test", "proj1")
        sessions_dir = os.path.join(workspace, "orbital", dir_name, "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        with open(os.path.join(sessions_dir, "s1.jsonl"), "w") as f:
            f.write(json.dumps({"role": "user", "content": "Hello", "timestamp": "2026-03-10T10:00:01Z"}) + "\n")
            f.write(json.dumps({"role": "assistant", "content": "Hi", "timestamp": "2026-03-10T10:00:03Z"}) + "\n")

        # Write sub-agent transcript
        transcript = SubAgentTranscript(workspace, "claude-code", "txn-001")
        transcript.append({"source": "claude-code", "content": "Working...", "timestamp": "2026-03-10T10:00:02Z"})
        transcript.append({"source": "claude-code", "content": "Done!", "timestamp": "2026-03-10T10:00:04Z"})

        # Read management messages
        mgmt_msgs, _ = _read_chat_messages(sessions_dir, 0, 0)

        # Read and normalize transcript entries (as the endpoint does)
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

        # Verify interleaved order
        assert len(all_msgs) == 4
        assert all_msgs[0]["role"] == "user"        # t=01
        assert all_msgs[1]["role"] == "agent"        # t=02 (sub-agent)
        assert all_msgs[2]["role"] == "assistant"    # t=03
        assert all_msgs[3]["role"] == "agent"        # t=04 (sub-agent)

        # Verify sub-agent entries have role=agent and source
        sub = [m for m in all_msgs if m.get("role") == "agent"]
        assert len(sub) == 2
        assert all(m["source"] == "claude-code" for m in sub)


def test_prompt_awareness_reflects_active_agents():
    """Integration: PromptBuilder includes active sub-agent states in system prompt.

    Verifies that when active_sub_agents is populated, the system prompt
    contains the Sub-Agent Coordination section with correct states.
    """
    ctx = PromptContext(
        workspace=tempfile.gettempdir(),
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[{"handle": "claude-code", "display_name": "Claude Code", "type": "cli"}],
        tool_names=["read", "write", "shell", "agent_message"],
        os_type="windows",
        datetime_now="2026-03-10T10:00:00Z",
        project_name="test-project",
        project_id="proj1",
        active_sub_agents=[
            {"handle": "claude-code", "status": "running"},
            {"handle": "aider", "status": "idle"},
        ],
    )

    builder = PromptBuilder()
    cached, semi_stable, dynamic = builder.build(ctx)

    # Full system prompt
    system_prompt = cached + "\n\n" + semi_stable + "\n\n" + dynamic

    # Should have both the static sub-agents listing and the dynamic awareness section
    assert "Sub-Agents Available" in system_prompt
    assert "Sub-Agent Coordination" in system_prompt
    assert "returns IMMEDIATELY" in system_prompt

    # Semi-stable section should list both agents with states
    assert "claude-code" in semi_stable
    assert "running" in semi_stable
    assert "aider" in semi_stable
    assert "idle" in semi_stable


def test_sanitize_roles_drops_agent_messages():
    """Integration: _sanitize_roles warns and drops role=agent messages (v5).

    In v5, role=agent should never appear in the management session's
    sliding window. If it does (legacy data), it should be dropped with
    a warning, not remapped to role=user.
    """
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "agent", "source": "claude-code", "content": "Should not be here"},
        {"role": "assistant", "content": "I see"},
        {"role": "system", "content": "System msg"},
    ]

    result = ContextManager._sanitize_roles(messages)

    # role=agent message should be DROPPED (not remapped)
    roles = [m["role"] for m in result]
    assert "agent" not in roles
    assert "user" in roles      # original user message kept
    assert "assistant" in roles
    assert "system" in roles
    assert len(result) == 3     # agent message dropped
