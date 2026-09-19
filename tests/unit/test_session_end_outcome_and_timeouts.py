# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: run_session_end_routine outcome reporting + bounded editor
runs, and LLMProvider.reasoning_locked_on.

Incident context (orbital-marketing, 2026-07-03..09 and 07-27): the old
whole-file merge regenerated every Layer-1 file in one response, so its
deadline had to scale with the files and still timed out; every failure fell
back to the deterministic backstop and the agent could not tell. Spec 089
replaced it with the memory editor, whose replies are short id lists — each
call is bounded, and so is the whole run.

Invariants:
  1. run_session_end_routine RETURNS an outcome string so the loop can surface
     it to the agent: "edited" | "no_change" | "backstop_only" | "no_delta" |
     "not_needed" | "in_flight" | "skipped_idempotent".
  2. A hung or failing model never blocks the floor, and never marks the
     files clean (the next pass retries).
  3. LLMProvider.reasoning_locked_on is True exactly when the model reasons
     and no request param can turn it off.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import memory_editor
from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)
from agent_os.agent.providers.openai_compat import LLMProvider
from agent_os.config.provider_registry import ReasoningInfo


def _mock_session(session_id="sess_outcome_test"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = []
    return session


def _provider_answering(text):
    provider = streamable(AsyncMock())
    resp = MagicMock()
    resp.text = text
    resp.raw_message = {"role": "assistant", "content": text}
    resp.tool_calls = []
    provider.complete.return_value = resp
    return provider


def _workspace(tmp_path) -> WorkspaceFileManager:
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("state", "## Now\n- Launch prep.\n- Old note.\n")
    return ws


def _archive_first_state_line(ws) -> str:
    eid = [ln for ln in ws.read("state").split("\n") if "<!--mem id:" in ln][0]
    eid = eid.split("id:")[1].split()[0]
    return json.dumps({"archive": [{"file": "PROJECT_STATE.md", "id": eid, "pointer": "p"}]})


@pytest.fixture(autouse=True)
def _reset_completion_set():
    wsf_module._completed_session_ends.clear()
    memory_editor._RUNNING.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Outcome return values
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_edited_on_an_applied_choice(tmp_path):
    ws = _workspace(tmp_path)
    provider = _provider_answering(_archive_first_state_line(ws))
    outcome = await run_session_end_routine(
        _mock_session("s_out_ok"), provider, ws, session_uuid="s_out_ok", force=True)
    assert outcome == "edited"


@pytest.mark.asyncio
async def test_returns_no_change_when_the_editor_chooses_nothing(tmp_path):
    ws = _workspace(tmp_path)
    outcome = await run_session_end_routine(
        _mock_session("s_nc"), _provider_answering("{}"), ws,
        session_uuid="s_nc", force=True)
    assert outcome == "no_change"


@pytest.mark.asyncio
async def test_returns_backstop_only_on_timeout(tmp_path):
    ws = _workspace(tmp_path)
    provider = streamable(AsyncMock())
    provider.complete.side_effect = asyncio.TimeoutError()
    outcome = await run_session_end_routine(
        _mock_session("s_out_to"), provider, ws, session_uuid="s_out_to", force=True)
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_returns_backstop_only_on_non_timeout_error(tmp_path):
    ws = _workspace(tmp_path)
    provider = streamable(AsyncMock())
    provider.complete.side_effect = ValueError("bad input")
    outcome = await run_session_end_routine(
        _mock_session("s_out_err"), provider, ws, session_uuid="s_out_err", force=True)
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_returns_no_delta_when_nothing_changed(tmp_path):
    ws = _workspace(tmp_path)
    provider = _provider_answering("{}")
    await run_session_end_routine(
        _mock_session("s_nd_1"), provider, ws, session_uuid="s_nd_1", force=True)
    calls_after_first = provider.complete.call_count
    outcome = await run_session_end_routine(
        _mock_session("s_nd_2"), provider, ws,
        session_uuid="s_nd_2", bypass_idempotency=True, force=True,
    )
    assert outcome == "no_delta"
    assert provider.complete.call_count == calls_after_first


@pytest.mark.asyncio
async def test_returns_not_needed_when_under_budget_and_not_forced(tmp_path):
    ws = _workspace(tmp_path)
    provider = _provider_answering("{}")
    outcome = await run_session_end_routine(
        _mock_session("s_nn"), provider, ws, session_uuid="s_nn")
    assert outcome == "not_needed"
    provider.complete.assert_not_called()


@pytest.mark.asyncio
async def test_returns_skipped_idempotent_on_repeat_session(tmp_path):
    ws = _workspace(tmp_path)
    provider = _provider_answering("{}")
    await run_session_end_routine(
        _mock_session("s_idem"), provider, ws, session_uuid="s_idem", force=True)
    outcome = await run_session_end_routine(
        _mock_session("s_idem"), provider, ws, session_uuid="s_idem", force=True)
    assert outcome == "skipped_idempotent"


@pytest.mark.asyncio
async def test_returns_in_flight_while_another_pass_holds_the_project(tmp_path):
    ws = _workspace(tmp_path)
    memory_editor.claim(ws.workspace)
    outcome = await run_session_end_routine(
        _mock_session("s_if"), _provider_answering("{}"), ws,
        session_uuid="s_if", force=True)
    assert outcome == "in_flight"


# ---------------------------------------------------------------------------
# Bounded runs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_hung_model_call_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_editor, "EDITOR_CALL_TIMEOUT_S", 0.05)
    ws = _workspace(tmp_path)
    provider = MagicMock()
    provider.model = "hangs"

    async def _hang(messages, tools=None, **kwargs):
        await asyncio.sleep(3600)

    provider.complete = _hang
    outcome = await run_session_end_routine(
        _mock_session("s_hang"), provider, ws, session_uuid="s_hang", force=True)
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_the_whole_run_is_bounded(tmp_path, monkeypatch):
    """A model that keeps asking for tools, slowly, still ends the run."""
    monkeypatch.setattr(memory_editor, "EDITOR_TOTAL_TIMEOUT_S", 0.2)
    ws = _workspace(tmp_path)
    provider = MagicMock()
    provider.model = "chatty"

    async def _slow_tools(messages, tools=None, **kwargs):
        await asyncio.sleep(0.08)
        resp = MagicMock()
        resp.text = ""
        resp.raw_message = {"tool_calls": [{"id": "c", "function": {"name": "grep", "arguments": "{}"}}]}
        return resp

    provider.complete = _slow_tools
    outcome = await run_session_end_routine(
        _mock_session("s_slow"), provider, ws, session_uuid="s_slow", force=True)
    assert outcome == "backstop_only"


# ---------------------------------------------------------------------------
# Retry is not disarmed by a failed pass.
#
# The cleanup marker used to be written unconditionally — including on the
# timeout path — so every failure stamped all four files "clean" and the next
# checkpoint_state short-circuited to no_delta without trying.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_pass_leaves_files_dirty_so_next_pass_retries(tmp_path):
    ws = _workspace(tmp_path)
    provider = streamable(AsyncMock())
    provider.complete.side_effect = asyncio.TimeoutError()
    first = await run_session_end_routine(
        _mock_session("s_fail_1"), provider, ws,
        session_uuid="s_fail_1", bypass_idempotency=True, force=True,
    )
    assert first == "backstop_only"

    recovered = _provider_answering(_archive_first_state_line(ws))
    second = await run_session_end_routine(
        _mock_session("s_fail_2"), recovered, ws,
        session_uuid="s_fail_2", bypass_idempotency=True, force=True,
    )
    assert second == "edited"


@pytest.mark.asyncio
async def test_successful_pass_still_writes_marker(tmp_path):
    ws = _workspace(tmp_path)
    provider = _provider_answering("{}")
    assert await run_session_end_routine(
        _mock_session("s_ok_1"), provider, ws,
        session_uuid="s_ok_1", bypass_idempotency=True, force=True,
    ) == "no_change"
    calls = provider.complete.call_count
    assert await run_session_end_routine(
        _mock_session("s_ok_2"), provider, ws,
        session_uuid="s_ok_2", bypass_idempotency=True, force=True,
    ) == "no_delta"
    assert provider.complete.call_count == calls


# ---------------------------------------------------------------------------
# LLMProvider.reasoning_locked_on
# ---------------------------------------------------------------------------

def _provider(reasoning) -> LLMProvider:
    return LLMProvider("test-model", "key", None, sdk="openai", reasoning=reasoning)


def test_locked_on_true_for_model_only():
    """The MiniMax-M3 shape from the incident."""
    r = ReasoningInfo(supported=True, enable="model_only")
    assert _provider(r).reasoning_locked_on is True


def test_locked_on_false_when_param_disables():
    r = ReasoningInfo(supported=True, enable="param:thinking.type=enabled")
    assert _provider(r).reasoning_locked_on is False


def test_locked_on_true_for_unrecognized_param():
    r = ReasoningInfo(supported=True, enable="param:mystery.knob=1")
    assert _provider(r).reasoning_locked_on is True


def test_locked_on_false_for_non_reasoning_model():
    # ReasoningInfo defaults: supported=False, enable="model_only" — the
    # default enable value must NOT read as locked-on when the model
    # doesn't reason at all.
    assert _provider(ReasoningInfo()).reasoning_locked_on is False
    assert _provider(None).reasoning_locked_on is False
