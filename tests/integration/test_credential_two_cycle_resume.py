# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration (loop harness): two sequential credential cycles in one session.

TASK-credential-contract-fixes 1d — covers the P6 UNOBSERVED back-to-back case:
request → pause → submit → resume → request → pause → submit → resume, with the
final turn typing BOTH credentials via tokens.

Uses the real ToolRegistry (central secret substitution) + real
RequestCredentialTool against a fake keychain-style store, driven by a scripted
provider — the same harness shape as tests/regression/test_yield_turn.py. The
loop pauses via the tool result's `credential_request` meta
(loop.py "Pause if credential was requested"); resume re-runs the loop WITHOUT
re-executing the tool, so the agent's only knowledge of usable token names is
what the PENDING result carried — which is exactly what these tests pin.

Secrecy: the stored sentinel values must never appear in the session
transcript; the probe tool (execution side, post-substitution) must receive
the real values, proving the tokens are usable end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import LLMResponse, StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import Tool, ToolResult
from agent_os.agent.tools.registry import ToolRegistry
from agent_os.agent.tools.request_credential import RequestCredentialTool

SENTINEL_PASSWORD = "SENTINEL-dify-pw-93afE2kQ"
SENTINEL_CODE = "SENTINEL-totp-code-71qQx9"


# --- harness (mirrors tests/regression/test_yield_turn.py) ------------------

def _ctx(ws: str) -> PromptContext:
    return PromptContext(
        workspace=ws, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["request_credential", "type_probe"],
        os_type="linux", datetime_now="2026-06-12T00:00:00",
        context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("sys", "suffix", "runtime")


def _text(text: str) -> LLMResponse:
    return LLMResponse(raw_message={"role": "assistant", "content": text},
                       text=text, tool_calls=[], has_tool_calls=False,
                       finish_reason="stop", status_text=None,
                       usage=TokenUsage(10, 5))


def _tool(tc_id: str, name: str, args: str = "{}") -> LLMResponse:
    tc = {"id": tc_id, "type": "function",
          "function": {"name": name, "arguments": args}}
    return LLMResponse(raw_message={"role": "assistant", "content": None,
                                    "tool_calls": [tc]},
                       text="", tool_calls=[tc], has_tool_calls=True,
                       finish_reason="tool_calls", status_text=None,
                       usage=TokenUsage(10, 5))


class _Provider:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self.stream_call_count = 0

    async def stream(self, messages, tools=None):
        self.stream_call_count += 1
        idx = self._idx
        self._idx += 1
        if idx < len(self._responses):
            resp = self._responses[idx]
            if resp.text:
                yield StreamChunk(text=resp.text)
            for i, tc in enumerate(resp.tool_calls):
                d = dict(tc)
                d["index"] = i
                yield StreamChunk(tool_calls_delta=[d])
            yield StreamChunk(is_final=True, usage=resp.usage)
        else:
            yield StreamChunk(text="default", is_final=True, usage=TokenUsage(1, 1))


class _FakeCredStore:
    """Keychain-shaped fake: metadata + values, settable mid-test like the
    secure modal's POST /credentials."""

    def __init__(self):
        self._values: dict[tuple[str, str], str] = {}
        self._meta: dict[str, dict] = {}

    def get_metadata(self, name):
        return self._meta.get(name)

    def get_value(self, name, field):
        return self._values.get((name, field))

    def store(self, name, domain, fields: dict):
        self._meta[name] = {"domain": domain, "fields": list(fields)}
        for f, v in fields.items():
            self._values[(name, f)] = v


class _TypeProbeTool(Tool):
    """Stands in for browser `type` at the execution boundary: records the
    POST-substitution text it receives and reports a value-free result."""

    name = "type_probe"
    description = "test probe"
    parameters = {"type": "object", "properties": {}}

    def __init__(self):
        self.received_texts: list[str] = []

    def execute(self, **arguments) -> ToolResult:
        self.received_texts.append(arguments.get("text", ""))
        return ToolResult(content="typed into element")

    # Tools NEVER raise; nothing here can.


def _loop(session, provider, registry, ws):
    return AgentLoop(session, provider, registry,
                     ContextManager(session, _Builder(), _ctx(ws)))


def _tool_result_for(session, tc_id: str) -> dict:
    results = [m for m in session.get_messages()
               if m.get("role") == "tool" and m.get("tool_call_id") == tc_id]
    assert len(results) == 1, f"expected exactly one result for {tc_id}"
    return results[0]


@pytest.mark.asyncio
async def test_two_sequential_credential_cycles_resume_with_usable_tokens(tmp_path):
    store = _FakeCredStore()
    registry = ToolRegistry(user_credential_store=store)
    registry.register(RequestCredentialTool(credential_store=store))
    probe = _TypeProbeTool()
    registry.register(probe)

    session = Session.new("cred-two-cycle", str(tmp_path))

    provider = _Provider([
        # Turn 1: first credential request (email-first shape).
        _tool("tc-cred-1", "request_credential", json.dumps({
            "name": "dify", "domain": "cloud.dify.ai",
            "fields": ["email", "password"], "reason": "login",
        })),
        # Turn 2 (after submit+resume): second credential request (a code).
        _tool("tc-cred-2", "request_credential", json.dumps({
            "name": "totp", "domain": "cloud.dify.ai",
            "fields": ["code"], "reason": "2fa challenge",
        })),
        # Turn 3 (after second submit+resume): type BOTH via tokens.
        _tool("tc-type-1", "type_probe", json.dumps({
            "text": "<secret:dify.password> then <secret:totp.code>",
        })),
        _text("logged in"),
    ])

    loop = _loop(session, provider, registry, str(tmp_path))

    # --- cycle 1: request → pause -----------------------------------------
    persist_user_row(loop._session, "log into dify")
    await loop.run()
    assert provider.stream_call_count == 1
    assert session.is_paused()

    result_1 = json.loads(_tool_result_for(session, "tc-cred-1")["content"])
    assert result_1["status"] == "pending"
    # The PENDING result must already carry the exact usable token names —
    # resume never re-executes the tool, so this is all the agent gets.
    assert result_1["tokens"] == {
        "email": "<secret:dify.email>",
        "password": "<secret:dify.password>",
    }

    # --- submit (modal) → resume → cycle 2: request → pause ----------------
    store.store("dify", "cloud.dify.ai",
                {"email": "user@example.com", "password": SENTINEL_PASSWORD})
    session.resume()
    await loop.run()
    assert provider.stream_call_count == 2
    assert session.is_paused()

    result_2 = json.loads(_tool_result_for(session, "tc-cred-2")["content"])
    assert result_2["status"] == "pending"
    assert result_2["tokens"] == {"code": "<secret:totp.code>"}

    # --- second submit → resume → final turn types both via tokens ---------
    store.store("totp", "cloud.dify.ai", {"code": SENTINEL_CODE})
    session.resume()
    await loop.run()

    # The probe (execution side, post-substitution) received the REAL values:
    # the tokens the pending results taught are usable end-to-end.
    assert probe.received_texts == [f"{SENTINEL_PASSWORD} then {SENTINEL_CODE}"]

    # Secrecy: the transcript (full JSONL on disk) never contains the values.
    transcript = "\n".join(
        p.read_text(encoding="utf-8") for p in Path(tmp_path).rglob("*.jsonl")
    )
    assert SENTINEL_PASSWORD not in transcript
    assert SENTINEL_CODE not in transcript
    # The placeholders MAY (and do) appear.
    assert "<secret:dify.password>" in transcript
