# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""GET /agents/{pid}/chat does not ship the Responses reasoning state.

Assistant rows from OpenAI Responses models carry ``responses_output``: the
encrypted reasoning items replayed to the provider on later turns (requests
are ``store: false``), a few KB per tool-call turn. The browser and the
mobile relay never use it, so the chat route drops it on the way out. The
session JSONL keeps it, because the next provider request is built from the
stored rows, not from what the UI saw.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock

import pytest

from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.providers.responses_adapter import RESPONSES_STATE_FIELD
from agent_os.api.routes import agents_v2

STATE = {"provider": "openai", "model": "gpt-6.2-sol",
         "items": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "x" * 4096}]}
UUID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def project(tmp_path, monkeypatch):
    sessions = ProjectPaths(str(tmp_path)).sessions_dir
    os.makedirs(sessions, exist_ok=True)
    path = os.path.join(sessions, f"{UUID}.jsonl")
    rows = [
        {"role": "user", "content": "hi", "timestamp": "2026-11-19T00:00:00Z"},
        {"role": "assistant", "content": "hello", "source": "management",
         "timestamp": "2026-11-19T00:00:01Z", RESPONSES_STATE_FIELD: STATE},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    store = MagicMock()
    store.get_project.return_value = {"workspace": str(tmp_path)}
    mgr = MagicMock()
    mgr.list_sessions.return_value = [{"session_id": "sess-1", "session_uuid": UUID}]
    monkeypatch.setattr(agents_v2, "_project_store", store)
    monkeypatch.setattr(agents_v2, "_agent_manager", mgr)
    monkeypatch.setattr(agents_v2, "_sub_agent_manager", None)
    return path


def _get(**kw):
    resp = asyncio.run(agents_v2.chat_history("p1", limit=kw.get("limit", 0),
                                              offset=0, session_id=kw.get("session_id")))
    return json.loads(resp.body)


@pytest.mark.parametrize("kw", [{}, {"limit": 1}, {"session_id": "sess-1"},
                                {"session_id": "sess-1", "limit": 1}])
def test_chat_route_omits_responses_state(project, kw):
    rows = _get(**kw)
    assistant = [r for r in rows if r.get("role") == "assistant"]
    assert assistant and assistant[0]["content"] == "hello"
    assert all(RESPONSES_STATE_FIELD not in r for r in rows)


def test_stored_row_keeps_responses_state(project):
    _get(session_id="sess-1")
    with open(project, encoding="utf-8") as f:
        stored = [json.loads(line) for line in f if line.strip()]
    assert stored[1][RESPONSES_STATE_FIELD] == STATE
