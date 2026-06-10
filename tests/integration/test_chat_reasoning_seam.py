# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test (RULE 2/5): the chat reasoning DATA SEAM.

Owns the data-seam half of the chat-render spec (the React-render assertions
are owned by a separate frontend agent — NOT asserted here).

Drives turns end-to-end through the real seam:

    provider stream → StreamAccumulator → session.append (persist)
                    → ActivityTranslator.on_stream_chunk (WS broadcast)
                    → GET /api/v2/agents/{pid}/chat (refetch)

using a stub provider whose stream emits reasoning. Asserts:

  (a) reasoning-then-answer turn:
      - the broadcast ``chat.stream_delta`` events during the reasoning-only
        phase carry NON-EMPTY ``reasoning_content`` with EMPTY ``text`` (the
        precondition that lets the FE keep the thinking spinner alive), and
      - after completion, GET /chat returns the message with both the reasoning
        and the answer text present.

  (b) reasoning-only / no-answer turn (THE LANDMINE):
      - the completed, persisted message returned by GET /chat carries
        ``reasoning_content`` (NOT dropped). Fails against current code (dropped
        at loop.py text-only persist) and passes after the fix.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_os.api.app import create_app
from agent_os.daemon_v2.activity_translator import ActivityTranslator
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.agent.session import Session
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #

class _SpyWS:
    """Captures every broadcast payload so we can inspect chat.stream_delta."""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def broadcast(self, project_id, payload):
        self.events.append((project_id, payload))

    def broadcast_global(self, payload):  # pragma: no cover - unused
        self.events.append(("__global__", payload))


def _ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=[],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("cached-prefix", "semi-stable", "dynamic")


class _Registry:
    def schemas(self):
        return []

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        return ToolResult(content="ok")

    def tool_names(self):
        return []

    def reset_run_state(self):
        pass


def _reasoning_stream(reasoning: str, answer: str):
    """Streamed text-only turn: reasoning-only deltas (empty text) then answer."""
    def stream(messages, tools=None):
        async def gen():
            # Two reasoning-only deltas — empty text, non-empty reasoning.
            yield StreamChunk(text="", reasoning_content=reasoning[: len(reasoning) // 2])
            yield StreamChunk(text="", reasoning_content=reasoning[len(reasoning) // 2:])
            if answer:
                yield StreamChunk(text=answer)
            yield StreamChunk(is_final=True, usage=TokenUsage(10, 5))
        return gen()
    return stream


class _Provider:
    def __init__(self, stream_fn):
        self._stream = stream_fn

    def stream(self, messages, tools=None):
        return self._stream(messages, tools)


def _run_turn(workspace, session_uuid, session_id, translator, stream_fn, msg):
    """Drive one real AgentLoop turn into the project's session dir, with
    on_stream wired to the real ActivityTranslator (broadcasts go to the spy)."""
    session = Session.new(session_uuid, workspace, session_id=session_id)
    session.on_stream = lambda chunk: translator.on_stream_chunk(
        chunk, "proj", "management", session_id=session_id,
    )
    loop = AgentLoop(
        session, _Provider(stream_fn), _Registry(),
        ContextManager(session, _Builder(), _ctx(workspace)),
    )
    asyncio.run(loop.run(initial_message=msg))


@pytest.fixture
def client(tmp_path):
    app = create_app(data_dir=str(tmp_path / "data"))
    return TestClient(app)


@pytest.fixture
def project(client, tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    resp = client.post("/api/v2/projects", json={
        "name": "ReasonSeam", "workspace": str(ws),
        "model": "m", "api_key": "k",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"], str(ws)


def _stream_deltas(spy):
    return [p for _, p in spy.events if p.get("type") == "chat.stream_delta"]


# --------------------------------------------------------------------------- #
# (a) reasoning-then-answer                                                    #
# --------------------------------------------------------------------------- #

def test_reasoning_then_answer_seam(client, project):
    pid, ws = project
    spy = _SpyWS()
    translator = ActivityTranslator(spy)

    reasoning = "I will reason step by step about the request."
    answer = "Here is the answer."

    _run_turn(
        ws, "seam_reasoning_answer", "seam_reasoning_answer",
        translator, _reasoning_stream(reasoning, answer),
        "please answer",
    )

    deltas = _stream_deltas(spy)
    assert deltas, "no chat.stream_delta broadcast captured"

    # The reasoning-only phase: at least one delta with NON-EMPTY
    # reasoning_content and EMPTY text — the FE precondition for keeping the
    # thinking spinner alive.
    reasoning_only = [
        d for d in deltas
        if (d.get("reasoning_content") or "") and not (d.get("text") or "")
    ]
    assert reasoning_only, (
        "no reasoning-only delta carried reasoning_content with empty text; "
        "deltas=%r" % deltas
    )

    # The answer text reaches the frontend too.
    assert any((d.get("text") or "") == answer for d in deltas), deltas

    # After completion the persisted message carries BOTH reasoning and answer.
    resp = client.get(f"/api/v2/agents/{pid}/chat",
                      params={"session_id": "seam_reasoning_answer"})
    assert resp.status_code == 200, resp.text
    assistant = [m for m in resp.json() if m.get("role") == "assistant"]
    assert len(assistant) == 1, resp.json()
    msg = assistant[0]
    assert msg.get("content") == answer
    assert msg.get("reasoning_content") == reasoning, (
        "persisted message dropped reasoning_content: %r" % msg
    )


# --------------------------------------------------------------------------- #
# (b) reasoning-only / no-answer — THE LANDMINE                               #
# --------------------------------------------------------------------------- #

def test_reasoning_only_no_answer_persists_reasoning(client, project):
    """A completed turn that reasoned but produced NO visible answer must still
    persist reasoning_content. Otherwise the row is content-empty + reasoning-
    dropped + no tool_calls and renders as nothing (silent-vanish).

    This GET-/chat assertion fails against current code (reasoning dropped at the
    loop.py text-only persist) and passes after the fix.
    """
    pid, ws = project
    spy = _SpyWS()
    translator = ActivityTranslator(spy)

    reasoning = "I thought hard but there is nothing visible to say."

    _run_turn(
        ws, "seam_reasoning_only", "seam_reasoning_only",
        translator, _reasoning_stream(reasoning, ""),
        "please answer",
    )

    resp = client.get(f"/api/v2/agents/{pid}/chat",
                      params={"session_id": "seam_reasoning_only"})
    assert resp.status_code == 200, resp.text
    assistant = [m for m in resp.json() if m.get("role") == "assistant"]
    assert len(assistant) == 1, resp.json()
    msg = assistant[0]
    assert msg.get("reasoning_content") == reasoning, (
        "LANDMINE: completed reasoning-only turn dropped reasoning_content "
        "(silent-vanish): %r" % msg
    )
