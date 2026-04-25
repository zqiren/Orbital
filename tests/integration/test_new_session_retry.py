# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: HTTP POST /api/v2/agents/{pid}/new-session handles
LLM retry during pre-flush (B1/RC-A).

Drives the FastAPI route directly via fastapi.testclient.TestClient so the
real HTTP path executes (request → route handler → AgentManager.new_session
→ run_session_end_routine → retry loop).

Simulates a dying session where the summarization LLM times out twice and
succeeds on the third attempt. Verifies that:
  - HTTP 200 returned within 200s
  - New session JSONL exists on disk
  - Workspace files reflect the LLM's summary content (not empty fallback)
  - Daemon log contains retry INFO messages but NOT the old timeout warning
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_os.agent import workspace_files as wsf_module
from agent_os.api.routes import agents_v2
from agent_os.daemon_v2.project_store import project_dir_name as _project_dir_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_manager():
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


def _write_session_file(workspace: str, dir_name: str, session_id: str,
                        messages: list[dict]) -> str:
    sessions_dir = os.path.join(workspace, "orbital", dir_name, "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    filepath = os.path.join(sessions_dir, f"{session_id}.jsonl")
    with open(filepath, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return filepath


def _valid_llm_response(tag="retry_success"):
    return json.dumps({
        "project_state": f"# State\nSummary after retries - {tag}",
        "decisions": f"## Decision {tag}\n**Chose:** Retry worked",
        "session_log_entry": f"## Session {tag} -- today\n- Completed with retry",
        "lessons": f"## Lesson {tag}\n**Problem:** slow LLM\n**Fix:** retry",
        "context": f"## People\n- Team {tag}",
    })


def _build_test_client(mgr, project_store):
    """Build a FastAPI TestClient wired to the real agents_v2 router with
    our mocked AgentManager + project_store injected via configure().

    Returns (client, saved_globals) — caller must restore saved_globals
    after the test to avoid bleeding into other tests.
    """
    saved = {
        "_project_store": agents_v2._project_store,
        "_agent_manager": agents_v2._agent_manager,
        "_ws_manager": agents_v2._ws_manager,
        "_sub_agent_manager": agents_v2._sub_agent_manager,
    }
    agents_v2.configure(
        project_store=project_store,
        agent_manager=mgr,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
    )
    app = FastAPI()
    app.include_router(agents_v2.router)
    client = TestClient(app)
    return client, saved


def _restore_agents_v2(saved: dict) -> None:
    agents_v2._project_store = saved["_project_store"]
    agents_v2._agent_manager = saved["_agent_manager"]
    agents_v2._ws_manager = saved["_ws_manager"]
    agents_v2._sub_agent_manager = saved["_sub_agent_manager"]


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_new_session_llm_retry_on_timeout_via_http(caplog):
    """POST /api/v2/agents/{pid}/new-session succeeds (HTTP 200) when LLM
    times out twice and returns on third attempt — within 200s wall clock.

    Workspace files must contain the LLM's actual summary; retry INFO logs
    appear; the old WARNING "pre-flush LLM call timed out" does NOT appear.
    """
    wsf_module._completed_session_ends.clear()

    with tempfile.TemporaryDirectory() as workspace:
        mgr, _ws, project_store = _make_agent_manager()
        project_id = "proj_retry_http_5678"
        project_name = "Retry HTTP Test Project"
        dir_name = _project_dir_name(project_name, project_id)

        # Conversation history on disk.
        old_messages = [
            {"role": "user", "content": "Build me a feature", "timestamp": "2026-04-22T15:00:00Z"},
            {"role": "assistant", "content": "Sure, here is the implementation...", "timestamp": "2026-04-22T15:00:05Z"},
            {"role": "user", "content": "Looks good!", "timestamp": "2026-04-22T15:00:10Z"},
        ]
        _write_session_file(workspace, dir_name, "old_session_http_retry", old_messages)

        from agent_os.daemon_v2.agent_manager import ProjectHandle
        from agent_os.agent.session import Session

        session = Session.new("old_session_http_retry", workspace, project_dir_name=dir_name)
        for msg in old_messages:
            session.append(msg)

        # Mock provider whose complete() times out twice, then succeeds on 3rd.
        llm_response = MagicMock()
        llm_response.text = _valid_llm_response("retry_success")

        mock_provider = AsyncMock()
        mock_provider.complete.side_effect = [
            asyncio.TimeoutError(),
            asyncio.TimeoutError(),
            llm_response,
        ]

        # task=None signals "idle agent" — agent_manager.new_session() at
        # line 1203 short-circuits the loop-stop guard. Avoids cross-loop
        # Future binding (TestClient runs handlers on its own anyio loop).
        handle = ProjectHandle(
            session=session,
            loop=MagicMock(),
            provider=mock_provider,
            registry=MagicMock(),
            context_manager=MagicMock(),
            interceptor=MagicMock(),
            task=None,
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

        # Wire the real agents_v2 router into a TestClient.
        client, saved = _build_test_client(mgr, project_store)
        try:
            with caplog.at_level(logging.DEBUG, logger="agent_os.agent.workspace_files"):
                started = time.monotonic()
                response = client.post(
                    f"/api/v2/agents/{project_id}/new-session"
                )
                elapsed = time.monotonic() - started

            # ---- HTTP-level assertions ----
            assert response.status_code == 200, (
                f"Expected HTTP 200, got {response.status_code}: {response.text}"
            )
            assert elapsed < 200.0, (
                f"new-session must return within 200s, took {elapsed:.2f}s"
            )

            body = response.json()
            assert body.get("status") == "ok", (
                f"Expected status 'ok' in body, got: {body}"
            )

            # ---- New session file on disk ----
            new_handle = mgr._handles[project_id]
            new_session_id = new_handle.session.session_id
            assert new_session_id != "old_session_http_retry"

            new_session_path = os.path.join(
                workspace, "orbital", dir_name, "sessions",
                f"{new_session_id}.jsonl",
            )
            assert os.path.isfile(new_session_path), (
                f"New session JSONL not found at {new_session_path}"
            )

            # ---- Workspace files reflect LLM's actual summary ----
            from agent_os.agent.workspace_files import WorkspaceFileManager
            wfm = WorkspaceFileManager(workspace, project_dir_name=dir_name)

            state = wfm.read("state")
            assert state is not None and "retry_success" in state, (
                f"PROJECT_STATE.md should contain LLM summary, got: {state!r}"
            )

            session_log = wfm.read("session_log") or ""
            assert "retry_success" in session_log, (
                f"SESSION_LOG.md should contain LLM summary, got: {session_log!r}"
            )

            # ---- Retry INFO logs appear ----
            retry_logs = [
                r for r in caplog.records
                if "timed out, retrying" in r.message
            ]
            assert len(retry_logs) >= 2, (
                f"Expected >=2 retry INFO logs, got: "
                f"{[r.message for r in caplog.records]}"
            )
            assert all(r.levelno == logging.INFO for r in retry_logs), (
                "Retry logs must be INFO level"
            )

            # ---- Old "pre-flush LLM call timed out" WARNING absent ----
            old_warning_logs = [
                r for r in caplog.records
                if "pre-flush LLM call timed out" in r.message
                and r.levelno == logging.WARNING
            ]
            assert len(old_warning_logs) == 0, (
                f"Old timeout WARNING should not appear: "
                f"{[r.message for r in old_warning_logs]}"
            )

            # ---- LLM was called exactly 3 times ----
            assert mock_provider.complete.call_count == 3, (
                f"Expected exactly 3 LLM calls, got {mock_provider.complete.call_count}"
            )
        finally:
            client.close()
            _restore_agents_v2(saved)

    wsf_module._completed_session_ends.clear()
