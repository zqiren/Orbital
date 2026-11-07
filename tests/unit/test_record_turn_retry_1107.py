# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The memory record turn retries its LLM call once on a busy provider.

Before this, one 429 ended the record turn unrecorded; with parallel sessions
(spec 155) on one key that is more likely. The record turn now waits a short
backoff and tries the same call once more on a 429 / overload / 5xx /
connection error. The backoff is the loop's in-flight task, so spec 142's
``yield_memory_record`` and Stop cancel it exactly like the call, and the
record lock is released. Every other error ends the turn as before.

Real ``AgentLoop.run()`` with a scripted LLM over a real workspace, the
harness of ``test_record_lock_155.py``.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_os.agent import loop as loop_mod
from agent_os.agent import memory_editor
from agent_os.agent.providers.types import LLMError
from tests.unit.test_memory_record_turn import _memory_rows, _reply, _seed_memory, _tool_calls
from tests.unit.test_record_lock_155 import _loop, _state, _until


class _FlakyLLM:
    """One item per ``stream`` call: an exception to raise, or chunks."""

    provider = "test"
    model = "test-model"
    sdk = "openai"

    def __init__(self, items):
        self._items = list(items)
        self.calls = 0

    async def stream(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        item = self._items[min(idx, len(self._items) - 1)]
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        for chunk in item:
            yield chunk

    async def complete(self, messages, tools=None):  # compaction: never here
        raise AssertionError("complete() not expected")


def _record_edit():
    return _tool_calls(("tc1", "edit", {
        "path": "orbital/PROJECT_STATE.md",
        "old_text": "- next: write copy",
        "new_text": "- next: write copy\n- launch date: Friday",
    }))


@pytest.fixture
def fast_backoff(monkeypatch):
    monkeypatch.setattr(loop_mod, "MEMORY_RECORD_RETRY_DELAY_S", 0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    LLMError("rate limited", status_code=429),
    LLMError("overloaded", status_code=529),
    LLMError("bad gateway", status_code=502),
    LLMError("Connection failed"),
    LLMError("Request timed out"),
])
async def test_a_busy_provider_then_success_records(tmp_path, fast_backoff, error):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    llm = _FlakyLLM([_reply("answer"), error, _record_edit(), _reply("recorded")])
    sess, loop, diags = _loop(workspace, llm, "sess-retry")
    await asyncio.wait_for(loop.run(), timeout=5)
    assert "- launch date: Friday" in _state(workspace)
    assert diags[-1]["outcome"] == "edited"
    assert diags[-1]["retried"] is True
    assert llm.calls == 4
    assert not memory_editor.record_lock(workspace).locked()


@pytest.mark.asyncio
async def test_two_busy_errors_end_unrecorded_as_today(tmp_path, fast_backoff):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    busy = LLMError("rate limited", status_code=429)
    llm = _FlakyLLM([_reply("answer"), busy, busy, _record_edit(), _reply("recorded")])
    sess, loop, diags = _loop(workspace, llm, "sess-twice")
    await asyncio.wait_for(loop.run(), timeout=5)
    assert "- launch date: Friday" not in _state(workspace)
    assert diags[-1]["outcome"] == "llm_error"
    assert llm.calls == 3                       # the answer, the call, one retry
    assert not memory_editor.record_lock(workspace).locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    LLMError("invalid key", status_code=401),
    LLMError("bad request", status_code=400),
    LLMError("forbidden", status_code=403),
    LLMError("model not found", status_code=404),
    LLMError("something odd"),
])
async def test_other_errors_are_never_retried(tmp_path, fast_backoff, error):
    workspace = str(tmp_path)
    _seed_memory(workspace)
    llm = _FlakyLLM([_reply("answer"), error, _record_edit(), _reply("recorded")])
    sess, loop, diags = _loop(workspace, llm, "sess-other")
    await asyncio.wait_for(loop.run(), timeout=5)
    assert diags[-1]["outcome"] == "llm_error"
    assert diags[-1]["retried"] is False
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_a_message_during_the_backoff_cancels_it_and_frees_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "MEMORY_RECORD_RETRY_DELAY_S", 30.0)
    workspace = str(tmp_path)
    _seed_memory(workspace)
    llm = _FlakyLLM([_reply("answer"), LLMError("rate limited", status_code=429),
                     _record_edit(), _reply("recorded"), _reply("next answer")])
    sess, loop, diags = _loop(workspace, llm, "sess-yield")
    lock = memory_editor.record_lock(workspace)
    task = asyncio.create_task(loop.run())
    # The record call failed and the loop is in the backoff, holding the lock.
    await _until(lambda: llm.calls == 2 and loop._inflight_stream is not None
                 and lock.locked())
    sess.queue_message("one more thing", nonce="n1")
    assert loop.yield_memory_record() is True
    await asyncio.wait_for(task, timeout=3)     # did not sit out the 30 s
    record_diags = [d for d in diags if d.get("trigger")]
    assert record_diags[0]["outcome"] == "cancelled"
    assert not lock.locked()
    assert "- launch date: Friday" not in _state(workspace)
    # The retry never ran: no third call came from the record turn.
    assert len(_memory_rows(sess)) >= 1


@pytest.mark.asyncio
async def test_stop_during_the_backoff_ends_cleanly_and_frees_the_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "MEMORY_RECORD_RETRY_DELAY_S", 30.0)
    workspace = str(tmp_path)
    _seed_memory(workspace)
    llm = _FlakyLLM([_reply("answer"), LLMError("rate limited", status_code=429),
                     _record_edit(), _reply("recorded")])
    sess, loop, diags = _loop(workspace, llm, "sess-stop")
    lock = memory_editor.record_lock(workspace)
    task = asyncio.create_task(loop.run())
    await _until(lambda: llm.calls == 2 and loop._inflight_stream is not None
                 and lock.locked())
    await loop.terminate()
    try:
        await asyncio.wait_for(task, timeout=3)
    except asyncio.CancelledError:
        pass
    assert not lock.locked()
    assert llm.calls == 2
    assert "- launch date: Friday" not in _state(workspace)
    await asyncio.wait_for(lock.acquire(), timeout=1)
    lock.release()


@pytest.mark.asyncio
async def test_a_message_already_held_when_the_call_fails_skips_the_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "MEMORY_RECORD_RETRY_DELAY_S", 30.0)
    workspace = str(tmp_path)
    _seed_memory(workspace)
    holder = {}

    def queue_then_429():
        holder["sess"].queue_message("one more thing", nonce="n1")
        return LLMError("rate limited", status_code=429)

    llm = _FlakyLLM([_reply("answer"), queue_then_429, _record_edit(),
                     _reply("recorded"), _reply("next answer")])
    sess, loop, diags = _loop(workspace, llm, "sess-held")
    holder["sess"] = sess
    await asyncio.wait_for(loop.run(), timeout=3)   # never sat out the 30 s
    record_diags = [d for d in diags if d.get("trigger")]
    assert record_diags[0]["outcome"] == "llm_error"
    assert record_diags[0]["retried"] is False
    assert not memory_editor.record_lock(workspace).locked()
