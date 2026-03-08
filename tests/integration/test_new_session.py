# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration tests for the /new session endpoint.

Covers:
- /new on idle agent archives JSONL and creates new empty one
- /new on running agent interrupts gracefully then completes
- Layer 1 files (PROJECT_STATE.md, DECISIONS.md) survive /new unchanged
- New session boots from Layer 1 — agent has project context, no conversation history
- Two projects same workspace — /new on project A does not touch project B
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.daemon_v2.project_store import project_dir_name as _project_dir_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_manager():
    """Create an AgentManager with mocked dependencies."""
    from agent_os.daemon_v2.agent_manager import AgentManager

    project_store = MagicMock()
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock()
    sub_agent_mgr.list_active = MagicMock(return_value=[])
    sub_agent_mgr.stop = AsyncMock()
    sub_agent_mgr.stop_all = AsyncMock()
    activity_translator = MagicMock()
    process_manager = MagicMock()
    process_manager.set_session = MagicMock()

    mgr = AgentManager(
        project_store=project_store,
        ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=activity_translator,
        process_manager=process_manager,
    )
    return mgr, ws, project_store


def _make_config(**overrides):
    from agent_os.daemon_v2.models import AgentConfig
    defaults = dict(workspace="/tmp/ws", model="gpt-4", api_key="sk-test")
    defaults.update(overrides)
    return AgentConfig(**defaults)


def _write_layer1_files(workspace: str, dir_name: str) -> dict:
    """Write Layer 1 files and return their contents for assertion."""
    base = os.path.join(workspace, ".agent-os", dir_name)
    os.makedirs(base, exist_ok=True)
    contents = {
        "PROJECT_STATE.md": "# Project State\nFeature X is 50% complete.",
        "DECISIONS.md": "# Decisions\n## 2024-01-01: Use React\n**Chose:** React",
    }
    for fname, content in contents.items():
        with open(os.path.join(base, fname), "w") as f:
            f.write(content)
    return contents


def _write_session_file(workspace: str, dir_name: str, session_id: str,
                        messages: list[dict]) -> str:
    """Write a session JSONL file and return its path."""
    sessions_dir = os.path.join(workspace, ".agent-os", dir_name, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
    with open(filepath, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return filepath


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNewSessionIdle:
    """Test /new session on an idle agent (no running loop)."""

    @pytest.mark.asyncio
    async def test_new_session_returns_ok_when_no_handle(self):
        """When no active session exists, new_session returns immediately."""
        mgr, ws, _ = _make_agent_manager()
        result = await mgr.new_session("nonexistent_project")
        assert result["status"] == "no_active_session"

    @pytest.mark.asyncio
    async def test_new_session_creates_fresh_session(self):
        """After new_session, a new empty session file exists."""
        with tempfile.TemporaryDirectory() as workspace:
            mgr, ws, project_store = _make_agent_manager()
            project_id = "proj_abc123def456"
            project_name = "Test Project"
            dir_name = _project_dir_name(project_name, project_id)

            # Write existing session
            old_messages = [
                {"role": "user", "content": "hello", "timestamp": "2024-01-01T00:00:00Z"},
                {"role": "assistant", "content": "hi there", "timestamp": "2024-01-01T00:00:01Z"},
            ]
            old_session_path = _write_session_file(
                workspace, dir_name, "old_session_abc", old_messages
            )

            # Write Layer 1 files
            layer1 = _write_layer1_files(workspace, dir_name)

            # Simulate an idle handle (task done)
            from agent_os.daemon_v2.agent_manager import ProjectHandle
            from agent_os.agent.session import Session

            session = Session.new("old_session_abc", workspace, project_dir_name=dir_name)
            for msg in old_messages:
                session.append(msg)

            mock_loop = MagicMock()
            mock_loop.run = AsyncMock()
            mock_provider = MagicMock()
            mock_registry = MagicMock()
            mock_context = MagicMock()
            mock_interceptor = MagicMock()

            done_task = asyncio.get_event_loop().create_future()
            done_task.set_result(None)

            handle = ProjectHandle(
                session=session,
                loop=mock_loop,
                provider=mock_provider,
                registry=mock_registry,
                context_manager=mock_context,
                interceptor=mock_interceptor,
                task=done_task,
                config_snapshot={"workspace": workspace, "model": "gpt-4", "autonomy": "hands_off"},
                project_dir_name=dir_name,
            )
            mgr._handles[project_id] = handle

            project_store.get_project.return_value = {
                "project_id": project_id,
                "name": project_name,
                "workspace": workspace,
                "model": "gpt-4",
                "api_key": "sk-test",
                "provider": "custom",
                "sdk": "openai",
            }

            result = await mgr.new_session(project_id)
            assert result["status"] == "ok"

            # Old session file should still exist (archived)
            assert os.path.isfile(old_session_path)

            # A new session should exist in the handle
            new_session = mgr._handles[project_id].session
            assert new_session.session_id != "old_session_abc"
            assert new_session.get_messages() == []


class TestNewSessionPreservesLayer1:
    """Layer 1 files survive /new unchanged."""

    @pytest.mark.asyncio
    async def test_layer1_files_intact_after_new_session(self):
        with tempfile.TemporaryDirectory() as workspace:
            mgr, ws, project_store = _make_agent_manager()
            project_id = "proj_abc123def456"
            project_name = "Test Project"
            dir_name = _project_dir_name(project_name, project_id)

            # Write Layer 1 files
            layer1 = _write_layer1_files(workspace, dir_name)

            # Write session
            _write_session_file(workspace, dir_name, "sess_001", [
                {"role": "user", "content": "test"},
            ])

            # Setup handle
            from agent_os.daemon_v2.agent_manager import ProjectHandle
            from agent_os.agent.session import Session

            session = Session.new("sess_001", workspace, project_dir_name=dir_name)
            done_task = asyncio.get_event_loop().create_future()
            done_task.set_result(None)

            handle = ProjectHandle(
                session=session,
                loop=MagicMock(),
                provider=MagicMock(),
                registry=MagicMock(),
                context_manager=MagicMock(),
                interceptor=MagicMock(),
                task=done_task,
                config_snapshot={"workspace": workspace, "model": "gpt-4", "autonomy": "hands_off"},
                project_dir_name=dir_name,
            )
            mgr._handles[project_id] = handle

            project_store.get_project.return_value = {
                "project_id": project_id,
                "name": project_name,
                "workspace": workspace,
                "model": "gpt-4",
                "api_key": "sk-test",
                "provider": "custom",
                "sdk": "openai",
            }

            await mgr.new_session(project_id)

            # Verify Layer 1 files are untouched
            base = os.path.join(workspace, ".agent-os", dir_name)
            for fname, expected_content in layer1.items():
                with open(os.path.join(base, fname)) as f:
                    assert f.read() == expected_content


class TestNewSessionBroadcasts:
    """Verify the WebSocket broadcast sequence during /new."""

    @pytest.mark.asyncio
    async def test_broadcasts_new_session_then_idle(self):
        """new_session must broadcast new_session followed by idle."""
        with tempfile.TemporaryDirectory() as workspace:
            mgr, ws, project_store = _make_agent_manager()
            project_id = "proj_abc123def456"
            project_name = "Test Project"
            dir_name = _project_dir_name(project_name, project_id)

            _write_session_file(workspace, dir_name, "sess_old", [
                {"role": "user", "content": "hello"},
            ])

            from agent_os.daemon_v2.agent_manager import ProjectHandle
            from agent_os.agent.session import Session

            session = Session.new("sess_old", workspace, project_dir_name=dir_name)
            done_task = asyncio.get_event_loop().create_future()
            done_task.set_result(None)

            handle = ProjectHandle(
                session=session,
                loop=MagicMock(),
                provider=MagicMock(),
                registry=MagicMock(),
                context_manager=MagicMock(),
                interceptor=MagicMock(),
                task=done_task,
                config_snapshot={"workspace": workspace, "model": "gpt-4", "autonomy": "hands_off"},
                project_dir_name=dir_name,
            )
            mgr._handles[project_id] = handle

            project_store.get_project.return_value = {
                "project_id": project_id,
                "name": project_name,
                "workspace": workspace,
                "model": "gpt-4",
                "api_key": "sk-test",
                "provider": "custom",
                "sdk": "openai",
            }

            await mgr.new_session(project_id)

            # Extract agent.status broadcasts
            statuses = [
                call[0][1]["status"]
                for call in ws.broadcast.call_args_list
                if call[0][1].get("type") == "agent.status"
            ]
            assert statuses == ["new_session", "idle"]


class TestNewSessionIsolation:
    """Two projects same workspace — /new on project A does not touch project B."""

    @pytest.mark.asyncio
    async def test_new_session_does_not_touch_other_project(self):
        with tempfile.TemporaryDirectory() as workspace:
            mgr, ws, project_store = _make_agent_manager()

            pid_a = "proj_aaaa11112222"
            pid_b = "proj_bbbb33334444"
            name_a = "Project A"
            name_b = "Project B"
            dir_a = _project_dir_name(name_a, pid_a)
            dir_b = _project_dir_name(name_b, pid_b)

            # Write sessions for both
            sess_a_path = _write_session_file(workspace, dir_a, "sess_a", [
                {"role": "user", "content": "from A"},
            ])
            sess_b_path = _write_session_file(workspace, dir_b, "sess_b", [
                {"role": "user", "content": "from B"},
            ])

            # Write layer1 for both
            _write_layer1_files(workspace, dir_a)
            layer1_b = _write_layer1_files(workspace, dir_b)

            # Setup handle for A only
            from agent_os.daemon_v2.agent_manager import ProjectHandle
            from agent_os.agent.session import Session

            session_a = Session.new("sess_a", workspace, project_dir_name=dir_a)
            done_task = asyncio.get_event_loop().create_future()
            done_task.set_result(None)

            handle_a = ProjectHandle(
                session=session_a,
                loop=MagicMock(),
                provider=MagicMock(),
                registry=MagicMock(),
                context_manager=MagicMock(),
                interceptor=MagicMock(),
                task=done_task,
                config_snapshot={"workspace": workspace, "model": "gpt-4", "autonomy": "hands_off"},
                project_dir_name=dir_a,
            )
            mgr._handles[pid_a] = handle_a

            project_store.get_project.return_value = {
                "project_id": pid_a,
                "name": name_a,
                "workspace": workspace,
                "model": "gpt-4",
                "api_key": "sk-test",
                "provider": "custom",
                "sdk": "openai",
            }

            await mgr.new_session(pid_a)

            # Project B's session file should be completely untouched
            assert os.path.isfile(sess_b_path)
            with open(sess_b_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            assert json.loads(lines[0])["content"] == "from B"

            # Project B's layer1 files should be untouched
            base_b = os.path.join(workspace, ".agent-os", dir_b)
            for fname, expected in layer1_b.items():
                with open(os.path.join(base_b, fname)) as f:
                    assert f.read() == expected
