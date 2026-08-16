# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for permanent agent-loop diagnostic instrumentation.

Guards the always-on `agent.diag` log points added to make the loop's
per-iteration disposition and exit path visible in logs. The original
motivating failure (a MiniMax-M3 turn that ended at iteration 2 with no
persisted row, no terminal event, and no logged error) was invisible by
design; these records guarantee the next occurrence is captured.

Four log points (see TASK-loop-diagnostic-instrumentation):
  1. response_shape  — per-iteration finalized LLMResponse shape
  2. loop_exit       — named exit path taken (text_complete, tool_continue,
                       task_complete, llm_error, cancel, …, fell_through)
  3. run_terminal    — run-level disposition (normal / error / cancel /
                       none_appended)
  4. _on_loop_done   — logs the exception type/message/traceback before the
                       existing broadcast (agent_manager)

The `none_appended` / `fell_through` assertions fail against pre-instrumentation
code (no such records exist) and pass after.
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from agent_os.agent.providers.types import (
    StreamChunk,
    LLMResponse,
    TokenUsage,
    LLMError,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.prompt_builder import PromptContext, Autonomy
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.loop import AgentLoop
from agent_os.agent.context import ContextManager


DIAG_LOGGER = "agent.diag"


# ---------------------------------------------------------------------------
# Minimal harness (mirrors tests/unit/test_component_a.py)
# ---------------------------------------------------------------------------

def _ctx(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace,
        model="test-model",
        autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[],
        tool_names=["read", "write"],
        os_type="linux",
        datetime_now="2026-01-01T00:00:00",
        context_usage_pct=0.0,
    )


class _Builder:
    def build(self, context):
        return ("cached-prefix", "semi-stable", "dynamic")


def _text_resp(text):
    return LLMResponse(
        raw_message={"role": "assistant", "content": text},
        text=text, tool_calls=[], has_tool_calls=False,
        finish_reason="stop", status_text=None,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


def _tool_resp(tool_calls, text=None):
    return LLMResponse(
        raw_message={"role": "assistant", "content": text, "tool_calls": tool_calls},
        text=text, tool_calls=tool_calls, has_tool_calls=True,
        finish_reason="tool_calls", status_text=None,
        usage=TokenUsage(input_tokens=100, output_tokens=50),
    )


class _Provider:
    """Mock provider. `responses` drives successive stream() calls. A response
    that is an Exception instance is raised from stream() instead of yielded."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def stream(self, messages, tools=None):
        idx = self._idx
        self._idx += 1
        resp = self._responses[idx] if idx < len(self._responses) else _text_resp("default")
        if isinstance(resp, BaseException):
            raise resp
        if resp.text:
            yield StreamChunk(text=resp.text)
        for i, tc in enumerate(resp.tool_calls or []):
            d = dict(tc)
            d["index"] = i
            yield StreamChunk(tool_calls_delta=[d])
        yield StreamChunk(is_final=True, usage=resp.usage)

    async def complete(self, messages, tools=None, *, disable_reasoning=False):
        return _text_resp("default")


class _Registry:
    def __init__(self, results=None):
        self._results = results or {}

    def schemas(self):
        return [{"type": "function", "function": {"name": n}} for n in self._results]

    def is_async(self, name):
        return False

    def execute(self, name, arguments):
        return self._results.get(name, ToolResult(content=f"result of {name}"))

    def tool_names(self):
        return list(self._results.keys())

    def reset_run_state(self):
        pass


def _make_loop(tmp_path, responses, results=None, name="diag"):
    session = Session.new(name, str(tmp_path))
    provider = _Provider(responses)
    registry = _Registry(results=results)
    context_mgr = ContextManager(session, _Builder(), _ctx(str(tmp_path)))
    loop = AgentLoop(session, provider, registry, context_mgr)
    return loop, session


def _diag_records(caplog):
    return [r for r in caplog.records
            if r.name == DIAG_LOGGER and hasattr(r, "diag")]


def _events(caplog, event):
    return [r.diag for r in _diag_records(caplog) if r.diag.get("event") == event]


def _exits(caplog):
    return [d.get("exit") for d in _events(caplog, "loop_exit")]


# ---------------------------------------------------------------------------
# Point 1 — response_shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_response_shape_logged_per_iteration(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    loop, _ = _make_loop(tmp_path, [_text_resp("Hello there!")])
    persist_user_row(loop._session, "hi")
    await loop.run()

    shapes = _events(caplog, "response_shape")
    assert len(shapes) >= 1, "expected at least one response_shape record"
    s = shapes[0]
    assert s["text_len"] == len("Hello there!")
    assert s["tool_call_names"] == []
    assert s["final_usage_chunk_seen"] is True
    assert s["usage_present"] is True
    assert "iteration" in s


# ---------------------------------------------------------------------------
# Point 2 — named exits
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exit_text_complete(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    loop, _ = _make_loop(tmp_path, [_text_resp("done")])
    persist_user_row(loop._session, "hi")
    await loop.run()
    assert "text_complete" in _exits(caplog)


@pytest.mark.asyncio
async def test_exit_tool_continue_then_text_complete(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    tc = [{"id": "call_1", "type": "function",
           "function": {"name": "read", "arguments": '{"path": "x"}'}}]
    loop, _ = _make_loop(
        tmp_path,
        [_tool_resp(tc, text="[STATUS: reading]"), _text_resp("done")],
        results={"read": ToolResult(content="file body")},
    )
    persist_user_row(loop._session, "read x")
    await loop.run()
    exits = _exits(caplog)
    assert "tool_continue" in exits, f"expected tool_continue, got {exits}"
    assert "text_complete" in exits, f"expected text_complete, got {exits}"


@pytest.mark.asyncio
async def test_exit_task_complete(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    tc = [{"id": "call_1", "type": "function",
           "function": {"name": "mark_task_complete",
                        "arguments": '{"summary": "all done"}'}}]
    loop, _ = _make_loop(tmp_path, [_tool_resp(tc, text="[STATUS: wrapping up]")])
    persist_user_row(loop._session, "finish")
    await loop.run()
    assert "task_complete" in _exits(caplog)


@pytest.mark.asyncio
async def test_exit_llm_error(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    # status_code 400 -> ErrorCategory.ABORT -> append_system + break
    loop, _ = _make_loop(tmp_path, [LLMError("bad request", status_code=400)])
    persist_user_row(loop._session, "trigger error")
    await loop.run()
    assert "llm_error" in _exits(caplog)


# ---------------------------------------------------------------------------
# Point 3 — run_terminal none_appended (the original-bug fingerprint)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_yields_none_appended_run_terminal(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    # Stream aborts mid-flight (CancelledError, not a stop) -> break before any
    # row is appended -> the exact silent fingerprint of the original failure.
    loop, _ = _make_loop(tmp_path, [asyncio.CancelledError()])
    persist_user_row(loop._session, "this turn appends nothing")
    await loop.run()

    terminals = _events(caplog, "run_terminal")
    assert len(terminals) == 1, "expected exactly one run_terminal record"
    assert terminals[0]["disposition"] == "none_appended"
    assert terminals[0]["rows_appended"] == 0
    # the named exit for this path is also captured
    assert "cancel" in _exits(caplog)


@pytest.mark.asyncio
async def test_run_terminal_normal_when_rows_appended(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=DIAG_LOGGER)
    loop, _ = _make_loop(tmp_path, [_text_resp("answer")])
    persist_user_row(loop._session, "hi")
    await loop.run()
    terminals = _events(caplog, "run_terminal")
    assert len(terminals) == 1
    assert terminals[0]["disposition"] == "normal"
    assert terminals[0]["rows_appended"] >= 1


# ---------------------------------------------------------------------------
# Point 4 — _on_loop_done logs the exception
# ---------------------------------------------------------------------------

def test_on_loop_done_logs_exception(caplog):
    from agent_os.daemon_v2.agent_manager import AgentManager, ProjectHandle  # noqa

    project_store = MagicMock()
    ws = MagicMock(); ws.broadcast = MagicMock()
    sub_agent_mgr = MagicMock(); sub_agent_mgr.list_active = MagicMock(return_value=[])
    mgr = AgentManager(
        project_store=project_store, ws_manager=ws,
        sub_agent_manager=sub_agent_mgr,
        activity_translator=MagicMock(), process_manager=MagicMock(),
    )
    callback = mgr._on_loop_done("proj", session_id="proj_sess0001")

    boom = RuntimeError("simulated mid-stream blowup")
    task = MagicMock()
    task.exception.return_value = boom

    caplog.set_level(logging.ERROR)
    callback(task)

    matches = [r for r in caplog.records
               if r.levelno >= logging.ERROR
               and "RuntimeError" in r.getMessage()
               and "simulated mid-stream blowup" in r.getMessage()]
    assert matches, "expected _on_loop_done to log the exception type and message"
