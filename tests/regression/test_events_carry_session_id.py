# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase 2 (seam 3): the three previously-unstamped live events must carry the
canonical session id (the uuid the frontend addresses by), so the frontend can
route them strictly by session_id instead of the viewingHolder heuristic.

Before this phase (REPORT-streaming-status-frontend.md S1):
  - chat.stream_delta        -> no session_id at all
  - management approval.request -> no session_id (interceptor had only project_id)
  - agent.notify             -> no session_id
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_os.daemon_v2.activity_translator import ActivityTranslator
from agent_os.daemon_v2.autonomy import AutonomyInterceptor
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.tools.notify import NotifyTool


CANON = "proj_canon_uuid_0001"   # the canonical session_uuid the frontend views


def _last_payload(ws):
    assert ws.broadcast.call_count >= 1, "expected a broadcast"
    args, _ = ws.broadcast.call_args
    # broadcast(project_id, payload)
    return args[1]


def test_chat_stream_delta_carries_session_id():
    ws = MagicMock()
    tr = ActivityTranslator(ws)
    chunk = MagicMock(); chunk.text = "hello"; chunk.is_final = False
    tr.on_stream_chunk(chunk, "proj_1", "management", session_id=CANON)
    payload = _last_payload(ws)
    assert payload["type"] == "chat.stream_delta"
    assert payload.get("session_id") == CANON


def test_management_approval_request_carries_session_id():
    ws = MagicMock()
    interceptor = AutonomyInterceptor(
        Autonomy.HANDS_OFF, ws, "proj_1", session_id=CANON,
    )
    interceptor.on_intercept(
        {"id": "tc_1", "name": "write_file", "arguments": {"path": "x"}},
        recent_context=[],
    )
    payload = _last_payload(ws)
    assert payload["type"] == "approval.request"
    assert payload.get("session_id") == CANON


def test_agent_notify_carries_session_id():
    ws = MagicMock()
    tool = NotifyTool(ws_manager=ws, project_id="proj_1", session_id=CANON)
    tool.execute(title="Done", body="Task complete", urgency="normal")
    payload = _last_payload(ws)
    assert payload["type"] == "agent.notify"
    assert payload.get("session_id") == CANON
