# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 134 Part B (fix B7.3): ``yield_after_batch`` ends the turn after the
WHOLE tool batch, so two background calls in one message both run — and since
spec 172 D so does ``yield_turn`` (a dispatch). Each batch also gets
one id (``current_tool_batch``) so those calls form one job group."""

import json

import pytest

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop, current_tool_batch
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.session import Session
from agent_os.agent.tools.base import ToolResult

_USAGE = TokenUsage(input_tokens=10, output_tokens=5)


class _PromptBuilder:
    def build(self, context):
        return ("a", "b", "c")


def _ctx(ws):
    return PromptContext(workspace=ws, model="m", autonomy=Autonomy.HANDS_OFF,
                         enabled_agents=[], tool_names=[], os_type="linux",
                         datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0)


def _calls(*names):
    return [
        StreamChunk(tool_calls_delta=[{
            "index": i, "id": f"call_{i}", "type": "function",
            "function": {"name": n, "arguments": json.dumps({"i": i})},
        } for i, n in enumerate(names)]),
        StreamChunk(is_final=True, usage=_USAGE, finish_reason="tool_calls"),
    ]


def _reply(text):
    return [StreamChunk(text=text), StreamChunk(is_final=True, usage=_USAGE,
                                                finish_reason="stop")]


class _LLM:
    provider = "test"
    model = "m"
    sdk = "openai"

    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def stream(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        for chunk in self.script[min(idx, len(self.script) - 1)]:
            yield chunk


class _Tools:
    """Sync tools (they run on the tool pool, like ShellTool)."""

    def __init__(self, meta_by_name):
        self.meta_by_name = meta_by_name
        self.ran: list[tuple[str, str | None]] = []

    def schemas(self):
        return [{"type": "function", "function": {
            "name": n, "description": n, "parameters": {"type": "object", "properties": {}},
        }} for n in self.meta_by_name]

    def tool_names(self):
        return list(self.meta_by_name)

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        self.ran.append((name, current_tool_batch()))
        return ToolResult(content=f"{name} ok", meta=self.meta_by_name[name])

    async def execute_async(self, name, arguments):
        raise AssertionError

    def reset_run_state(self):
        pass


def _loop(tmp_path, llm, tools):
    session = Session.new("sess-yab", str(tmp_path))
    ctx = ContextManager(session, _PromptBuilder(), _ctx(str(tmp_path)))
    return session, AgentLoop(session, llm, tools, ctx, project_dir=str(tmp_path),
                              max_iterations=10, memory_record_enabled=False)


async def test_yield_after_batch_runs_every_call_then_ends_the_turn(tmp_path):
    llm = _LLM([_calls("bg", "bg", "other"), _reply("should never be asked")])
    tools = _Tools({"bg": {"yield_after_batch": True}, "other": None})
    session, loop = _loop(tmp_path, llm, tools)
    session.append({"role": "user", "content": "start two jobs"})
    await loop.run()
    assert [n for n, _ in tools.ran] == ["bg", "bg", "other"]
    assert llm.calls == 1                       # no LLM call after the batch
    assert loop._loop_exit_path == "yield_after_batch"
    results = [m for m in session.get_messages() if m.get("role") == "tool"]
    assert [r["content"] for r in results] == ["bg ok", "bg ok", "other ok"]


async def test_one_batch_has_one_id_and_the_next_batch_another(tmp_path):
    llm = _LLM([_calls("a", "a"), _calls("a"), _reply("done")])
    tools = _Tools({"a": None})
    session, loop = _loop(tmp_path, llm, tools)
    session.append({"role": "user", "content": "go"})
    await loop.run()
    ids = [b for _, b in tools.ran]
    assert ids[0] and ids[0] == ids[1]
    assert ids[2] and ids[2] != ids[0]
    assert current_tool_batch() is None          # not leaked outside the batch


async def test_yield_turn_also_ends_the_turn_after_the_batch(tmp_path):
    # Spec 172 D: a dispatch no longer breaks the batch at once (its siblings
    # came back CANCELLED); it ends the turn after it, under its own label.
    llm = _LLM([_calls("disp", "other"), _reply("x")])
    tools = _Tools({"disp": {"yield_turn": True}, "other": None})
    session, loop = _loop(tmp_path, llm, tools)
    session.append({"role": "user", "content": "go"})
    await loop.run()
    assert [n for n, _ in tools.ran] == ["disp", "other"]
    assert llm.calls == 1
    assert loop._loop_exit_path == "yield_turn"


async def test_a_dispatch_beside_a_background_job_keeps_the_dispatch_label(tmp_path):
    # The queue dispatcher's continuation hold keys on ``yield_turn``.
    llm = _LLM([_calls("bg", "disp"), _reply("x")])
    tools = _Tools({"bg": {"yield_after_batch": True}, "disp": {"yield_turn": True}})
    session, loop = _loop(tmp_path, llm, tools)
    session.append({"role": "user", "content": "go"})
    await loop.run()
    assert [n for n, _ in tools.ran] == ["bg", "disp"]
    assert llm.calls == 1
    assert loop._loop_exit_path == "yield_turn"
