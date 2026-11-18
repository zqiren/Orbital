# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Responses reasoning state survives the agent loop into the session.

Requests on the Responses path are ``store: false``, so the only copy of the
encrypted reasoning items is the assistant row the loop persists. The tool-call
path persists ``raw_message`` whole; the text-only path builds its own row and
must carry the field explicitly (GPT-5.6+ render earlier turns' reasoning by
default, so dropping it there silently costs quality on the next turn).
"""

import json

import pytest

from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.providers.responses_adapter import RESPONSES_SDK, RESPONSES_STATE_FIELD
from agent_os.agent.providers.types import StreamChunk, TokenUsage

from tests.unit.test_empty_response_guard import (
    MockPromptBuilder,
    SimpleToolRegistry,
    _make_base_prompt_context,
)

STATE = {"provider": "openai", "model": "gpt-6.2-sol", "items": [
    {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"},
    {"type": "message", "id": "msg_1", "phase": "final_answer", "text": "done"},
]}


class ResponsesLikeProvider:
    provider = "openai"
    model = "gpt-6.2-sol"
    sdk = RESPONSES_SDK

    async def stream(self, messages, tools=None):
        yield StreamChunk(text="done", reasoning_content="thought")
        yield StreamChunk(is_final=True, usage=TokenUsage(10, 2), finish_reason="stop",
                          raw_extra={RESPONSES_STATE_FIELD: STATE})


@pytest.mark.asyncio
async def test_text_only_turn_keeps_the_responses_state(tmp_path):
    session = Session.new("resp_state", str(tmp_path))
    ctx = ContextManager(session, MockPromptBuilder(), _make_base_prompt_context(str(tmp_path)))
    loop = AgentLoop(session, ResponsesLikeProvider(), SimpleToolRegistry(), ctx,
                     project_dir=str(tmp_path), max_iterations=3,
                     memory_record_enabled=False)
    persist_user_row(loop._session, "hi")
    await loop.run()

    final = [m for m in session.get_messages() if m.get("role") == "assistant"][-1]
    assert final["content"] == "done"
    assert final[RESPONSES_STATE_FIELD] == STATE
    # ...and it is on disk, not just in memory.
    rows = [json.loads(line) for line in open(session._filepath, encoding="utf-8")
            if line.strip()]
    on_disk = [r for r in rows if r.get("role") == "assistant"][-1]
    assert on_disk[RESPONSES_STATE_FIELD] == STATE
