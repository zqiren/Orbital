# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 072 — rejected project API key falls back to the global default.

A 401/403 means "this key is bad", not "this run is dead": the loop
infinite-cooldowns every rung sharing the rejected key, lazily appends the
auth-fallback rung (built wholesale from GLOBAL settings by the config
builder), rotates to it, and says so in chat. Nothing is persisted — the next
run tries the project key first again. 400 keeps its cold-ABORT semantics
untouched, and transient errors can never reach the auth rung before an auth
event.
"""

import json
import os

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import LLMError, StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult


# ---------------------------------------------------------------------------
# Helpers (same shapes as tests/regression/test_model_fallback.py, plus an
# api_key identity on every provider — the auth branch compares keys).
# ---------------------------------------------------------------------------

def _make_base_prompt_context(workspace: str) -> PromptContext:
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


class MockPromptBuilder:
    def build(self, context: PromptContext) -> tuple[str, str, str]:
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


class AlwaysErrorProvider:
    """Raises its LLMError on every stream() call."""

    def __init__(self, model: str, error: LLMError, api_key: str = ""):
        self.model = model
        self.api_key = api_key
        self.sdk = "openai"
        self.provider = "custom"
        self._error = error
        self._call_count = 0

    async def stream(self, messages, tools=None):
        self._call_count += 1
        if True:  # noqa: SIM108 — keep this an async generator
            raise self._error
        yield  # unreachable

class SuccessProvider:
    """Always answers with a final text response."""

    def __init__(self, model: str, api_key: str = "",
                 provider: str = "custom"):
        self.model = model
        self.api_key = api_key
        self.sdk = "openai"
        self.provider = provider
        self._call_count = 0

    async def stream(self, messages, tools=None):
        self._call_count += 1
        yield StreamChunk(text="Done from " + self.model)
        yield StreamChunk(
            is_final=True,
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )


class SimpleToolRegistry:
    def schemas(self) -> list[dict]:
        return []

    def is_async(self, name: str) -> bool:
        return False

    def execute(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    async def execute_async(self, name: str, arguments: dict) -> ToolResult:
        return ToolResult(content="ok")

    def tool_names(self) -> list[str]:
        return []

    def reset_run_state(self) -> None:
        pass


def _make_loop(tmp_path, session, primary, *, fallbacks=None, auth=None):
    builder = MockPromptBuilder()
    ctx = _make_base_prompt_context(str(tmp_path))
    context_mgr = ContextManager(session, builder, ctx)
    return AgentLoop(
        session, primary, SimpleToolRegistry(), context_mgr,
        fallback_providers=list(fallbacks or []),
        auth_fallback_provider=auth,
        max_iterations=10,
    )


def _system_texts(session) -> list[str]:
    return [
        m.get("content", "") for m in session.get_messages()
        if m["role"] == "system"
    ]


def _meta_events(tmp_path, stem: str, event: str) -> list[dict]:
    """Meta rows never enter get_messages() — read them off the JSONL."""
    path = os.path.join(str(tmp_path), "orbital", "sessions", f"{stem}.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if r.get("role") == "meta" and r.get("event") == event]


def _err(status: int) -> LLMError:
    return LLMError(f"http {status}", status_code=status)


# ---------------------------------------------------------------------------
# 401/403 with an auth rung available → switch, notice, continue
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_error_rotates_to_global_rung_and_continues(tmp_path, status):
    session = Session.new(f"auth{status}", str(tmp_path))
    primary = AlwaysErrorProvider("proj-model", _err(status), api_key="sk-project")
    auth = SuccessProvider("global-model", api_key="sk-global",
                           provider="tokendance")

    loop = _make_loop(tmp_path, session, primary, auth=auth)
    persist_user_row(session, "hello")
    await loop.run()

    # The run survived and the global rung served it.
    assert not loop._llm_failed
    assert auth._call_count >= 1

    # model_swap meta carries the auth reason.
    swaps = _meta_events(tmp_path, f"auth{status}", "model_swap")
    assert len(swaps) == 1
    assert swaps[0]["model"] == "global-model"
    assert swaps[0]["provider"] == "tokendance"
    assert swaps[0]["reason"] == (
        f"auth: project API key rejected (HTTP {status})"
    )

    # Honest chat notice: names the rejection, the substitute, and that the
    # stored key was NOT touched.
    notices = [t for t in _system_texts(session)
               if f"rejected (HTTP {status})" in t]
    assert len(notices) == 1
    assert "global default" in notices[0]
    assert "tokendance · global-model" in notices[0]
    assert "was not changed" in notices[0]

    # Nothing was mutated on the providers themselves (D2: no auto-repair).
    assert primary.api_key == "sk-project"


# ---------------------------------------------------------------------------
# 400 and transient errors keep today's semantics — the rung is unreachable
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_400_still_cold_aborts_without_touching_auth_rung(tmp_path):
    session = Session.new("cold400", str(tmp_path))
    primary = AlwaysErrorProvider("proj-model", _err(400), api_key="sk-project")
    auth = SuccessProvider("global-model", api_key="sk-global")

    loop = _make_loop(tmp_path, session, primary, auth=auth)
    persist_user_row(session, "hello")
    await loop.run()

    assert loop._llm_failed
    assert primary._call_count == 1  # no retry ladder
    assert auth._call_count == 0
    assert any("non-recoverable" in t for t in _system_texts(session))
    assert _meta_events(tmp_path, "cold400", "model_swap") == []


@pytest.mark.asyncio
async def test_transient_errors_never_reach_the_auth_rung(tmp_path):
    """Before any auth event the rotation list is unchanged: a 503 storm
    exhausts retries and stops without ever calling the global rung."""
    session = Session.new("transient503", str(tmp_path))
    primary = AlwaysErrorProvider("proj-model", _err(503), api_key="sk-project")
    auth = SuccessProvider("global-model", api_key="sk-global")

    loop = _make_loop(tmp_path, session, primary, auth=auth)
    persist_user_row(session, "hello")
    await loop.run()

    assert loop._llm_failed
    assert auth._call_count == 0
    assert any("retries" in t.lower() for t in _system_texts(session))


# ---------------------------------------------------------------------------
# Fallback-chain rungs sharing the rejected key are skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_rungs_sharing_dead_key_are_skipped(tmp_path):
    session = Session.new("sharedkey", str(tmp_path))
    primary = AlwaysErrorProvider("proj-model", _err(401), api_key="sk-shared")
    # Same credential as the primary: would "work" if called, but a rejected
    # key is dead for every rung that carries it.
    sibling = SuccessProvider("proj-fallback", api_key="sk-shared")
    auth = SuccessProvider("global-model", api_key="sk-global")

    loop = _make_loop(tmp_path, session, primary,
                      fallbacks=[sibling], auth=auth)
    persist_user_row(session, "hello")
    await loop.run()

    assert not loop._llm_failed
    assert sibling._call_count == 0
    assert auth._call_count >= 1
    swaps = _meta_events(tmp_path, "sharedkey", "model_swap")
    assert [s["model"] for s in swaps] == ["global-model"]


# ---------------------------------------------------------------------------
# No auth rung → abort exactly as today, naming the global-default status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_401_without_auth_rung_aborts_and_names_global_status(tmp_path):
    session = Session.new("norung", str(tmp_path))
    primary = AlwaysErrorProvider("proj-model", _err(401), api_key="sk-project")

    loop = _make_loop(tmp_path, session, primary, auth=None)
    persist_user_row(session, "hello")
    await loop.run()

    assert loop._llm_failed
    assert loop.last_llm_error is not None
    aborts = [t for t in _system_texts(session) if "non-recoverable" in t]
    assert len(aborts) == 1
    assert "global default" in aborts[0]
    assert _meta_events(tmp_path, "norung", "model_swap") == []


# ---------------------------------------------------------------------------
# 401 on the global rung after fallback → abort with the combined message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_401_on_global_rung_aborts_with_combined_message(tmp_path):
    session = Session.new("bothdead", str(tmp_path))
    primary = AlwaysErrorProvider("proj-model", _err(401), api_key="sk-project")
    auth = AlwaysErrorProvider("global-model", _err(401), api_key="sk-global")

    loop = _make_loop(tmp_path, session, primary, auth=auth)
    persist_user_row(session, "hello")
    await loop.run()

    assert loop._llm_failed
    # One switch happened (to the global rung), then it 401'd too.
    assert auth._call_count == 1
    swaps = _meta_events(tmp_path, "bothdead", "model_swap")
    assert [s["model"] for s in swaps] == ["global-model"]
    aborts = [t for t in _system_texts(session) if "non-recoverable" in t]
    assert len(aborts) == 1
    assert "global default was also rejected" in aborts[0]
