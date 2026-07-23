# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: run_session_end_routine outcome reporting + locked-on-reasoning
timeout policy, and LLMProvider.reasoning_locked_on.

Incident context (orbital-marketing, 2026-07-03..09): the consolidation LLM
call ran on MiniMax-M3 (reasoning locked-on, enable='model_only'), where
disable_reasoning is a no-op. Every attempt blew through the 30/60/90s retry
ladder, so every pass fell back to the deterministic backstop — and the agent
had no way to learn any of this.

New invariants:
  1. run_session_end_routine RETURNS an outcome string so the loop can surface
     it to the agent: "llm_merged" | "backstop_only" | "no_delta" |
     "skipped_idempotent".
  2. When the provider's reasoning is locked-on, retrying the same prompt on
     30/60/90s cannot succeed and only burns tokens — use a single attempt
     with one long timeout instead.
  3. LLMProvider.reasoning_locked_on is True exactly when the model reasons
     and no request param can turn it off.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import memory_entries as _mem
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
    session.get_messages.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    return session


def _valid_llm_response(tag="x"):
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## 2026-06-18: Decision {tag}\n**Chose:** A\n\n",
        "lessons": f"1. **Lesson {tag}.** Problem p, fix f.\n",
        "index": f"# INDEX\n- PROJECT_STATE.md — current scratchpad ({tag}).\n",
    })


@pytest.fixture(autouse=True)
def _reset_completion_set():
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Outcome return values
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_llm_merged_on_success(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = _valid_llm_response("ok")
    provider.complete.return_value = resp

    outcome = await run_session_end_routine(
        _mock_session("s_out_ok"), provider, ws, session_uuid="s_out_ok"
    )
    assert outcome == "llm_merged"


@pytest.mark.asyncio
async def test_returns_backstop_only_when_all_attempts_timeout(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    provider = AsyncMock()
    provider.complete.side_effect = asyncio.TimeoutError()

    outcome = await run_session_end_routine(
        _mock_session("s_out_to"), provider, ws, session_uuid="s_out_to"
    )
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_returns_backstop_only_on_non_timeout_error(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    provider = AsyncMock()
    provider.complete.side_effect = ValueError("bad input")

    outcome = await run_session_end_routine(
        _mock_session("s_out_err"), provider, ws, session_uuid="s_out_err"
    )
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_returns_no_delta_when_nothing_changed(tmp_path):
    """First run consolidates and writes the cleanup marker; a second run with
    no file changes must skip (no LLM call) and say so."""
    ws = WorkspaceFileManager(str(tmp_path))
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = _valid_llm_response("nd")
    provider.complete.return_value = resp

    await run_session_end_routine(
        _mock_session("s_nd_1"), provider, ws, session_uuid="s_nd_1"
    )
    calls_after_first = provider.complete.call_count

    outcome = await run_session_end_routine(
        _mock_session("s_nd_2"), provider, ws,
        session_uuid="s_nd_2", bypass_idempotency=True,
    )
    assert outcome == "no_delta"
    assert provider.complete.call_count == calls_after_first  # no new LLM call


@pytest.mark.asyncio
async def test_returns_skipped_idempotent_on_repeat_session(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = _valid_llm_response("idem")
    provider.complete.return_value = resp

    await run_session_end_routine(
        _mock_session("s_idem"), provider, ws, session_uuid="s_idem"
    )
    outcome = await run_session_end_routine(
        _mock_session("s_idem"), provider, ws, session_uuid="s_idem"
    )
    assert outcome == "skipped_idempotent"


# ---------------------------------------------------------------------------
# Locked-on reasoning timeout policy
# ---------------------------------------------------------------------------

def test_dedup_timeouts_default_ladder():
    """Providers without the property (incl. mocks, whose auto-attrs are
    truthy MagicMocks but not True) keep the 30/60/90 ladder."""
    assert wsf_module._dedup_timeouts(AsyncMock()) == [30.0, 60.0, 90.0]
    assert wsf_module._dedup_timeouts(object()) == [30.0, 60.0, 90.0]


def test_dedup_timeouts_locked_on_single_long_attempt():
    provider = MagicMock()
    provider.reasoning_locked_on = True
    assert wsf_module._dedup_timeouts(provider) == [240.0]


@pytest.mark.asyncio
async def test_locked_on_provider_gets_exactly_one_attempt(tmp_path):
    """Reasoning locked-on: retrying the identical prompt cannot get faster —
    one attempt, then straight to the backstop."""
    ws = WorkspaceFileManager(str(tmp_path))
    provider = AsyncMock()
    provider.reasoning_locked_on = True
    provider.complete.side_effect = asyncio.TimeoutError()

    outcome = await run_session_end_routine(
        _mock_session("s_locked"), provider, ws, session_uuid="s_locked"
    )
    assert provider.complete.call_count == 1
    assert outcome == "backstop_only"


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


# ---------------------------------------------------------------------------
# LLM-driven archive: the session-end pass can move entries it cannot condense
# out of DECISIONS/LESSONS into the read-on-demand archive files, mirroring the
# deterministic hard-cap demotion convention (DECISIONS_ARCHIVE.md /
# LESSONS_ARCHIVE.md via _mem.ARCHIVE_OF + FILE_NAMES).
# ---------------------------------------------------------------------------


def _response_with_archive(archive):
    """A valid merge response carrying an optional top-level ``archive`` field.

    The merged "decisions" keeps one entry; the archived entry is what the LLM
    moved out — it must land in the archive file, NOT the kept DECISIONS.md.
    """
    body = {
        "project_state": "# State\ncurrent focus",
        "decisions": "## 2026-07-01: Keep This Decision\n**Chose:** keep it\n",
        "lessons": "1. **Keep lesson.** Problem p, fix f.\n",
        "index": "# INDEX\n- PROJECT_STATE.md — current scratchpad.\n",
    }
    if archive is not None:
        body["archive"] = archive
    return json.dumps(body)


def _provider_returning(text):
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = text
    provider.complete.return_value = resp
    return provider


@pytest.mark.asyncio
async def test_archive_field_writes_dated_decisions_archive_and_index_pointer(tmp_path):
    """archive.decisions -> the DECISIONS archive file gains that text under a
    header carrying today's date, INDEX points to it, and DECISIONS.md holds the
    merged content (kept entry) without the archived entry."""
    ws = WorkspaceFileManager(str(tmp_path))
    provider = _provider_returning(_response_with_archive({
        "decisions": (
            "## 2026-01-05: Old Signing Flow\n"
            "**Chose:** old approach\n**Reason:** superseded by v0.6\n"
        ),
    }))

    outcome = await run_session_end_routine(
        _mock_session("s_arch_ok"), provider, ws, session_uuid="s_arch_ok"
    )
    assert outcome == "llm_merged"

    archive_text = ws.read("decisions_archive")
    assert archive_text is not None, "decisions archive file was not created"
    assert "Old Signing Flow" in archive_text
    today = date.today().isoformat()
    assert today in archive_text, "archived block missing a today-dated header"

    # INDEX points at the archive so the moved entries are discoverable.
    assert "DECISIONS_ARCHIVE.md" in (ws.read("index") or "")

    # DECISIONS.md holds the merged (kept) content; the archived entry is gone
    # from the live file (it was MOVED, not copied).
    decisions_text = ws.read("decisions") or ""
    assert "Keep This Decision" in decisions_text
    assert "Old Signing Flow" not in decisions_text


@pytest.mark.asyncio
async def test_no_archive_field_leaves_archives_untouched(tmp_path):
    """Response WITHOUT an archive field: no archive file is created or touched;
    behavior is identical to before the feature (backward-compat lock)."""
    ws = WorkspaceFileManager(str(tmp_path))
    provider = _provider_returning(_response_with_archive(None))

    outcome = await run_session_end_routine(
        _mock_session("s_arch_none"), provider, ws, session_uuid="s_arch_none"
    )
    assert outcome == "llm_merged"
    assert not ws.exists("decisions_archive")
    assert not ws.exists("lessons_archive")


@pytest.mark.asyncio
async def test_archive_write_oserror_does_not_crash_routine(tmp_path, monkeypatch):
    """An OSError while writing the archive is swallowed: the routine still
    completes, writes the cleanup marker, and leaves no archive file behind."""
    ws = WorkspaceFileManager(str(tmp_path))
    provider = _provider_returning(_response_with_archive({
        "lessons": "5. **Dormant lesson.** Problem q, fix g.\n",
    }))

    attempted = {"archive": False}
    orig_write = ws.write

    def failing_write(file_key, content):
        if file_key.endswith("_archive"):
            attempted["archive"] = True
            raise OSError("simulated disk full")
        return orig_write(file_key, content)

    monkeypatch.setattr(ws, "write", failing_write)

    outcome = await run_session_end_routine(
        _mock_session("s_arch_err"), provider, ws, session_uuid="s_arch_err"
    )
    # The archive write was attempted (feature is wired) but its OSError did not
    # crash the loop.
    assert attempted["archive"] is True
    assert outcome == "llm_merged"
    # Cleanup marker still written — the routine ran to completion.
    assert os.path.exists(wsf_module._cleanup_marker_path(ws))
    # The failed write left no archive file.
    assert ws.read("lessons_archive") is None


@pytest.mark.asyncio
async def test_archive_wrong_type_ignored(tmp_path):
    """archive present but a string (wrong type) is ignored gracefully — no
    crash, no archive file."""
    ws = WorkspaceFileManager(str(tmp_path))
    provider = _provider_returning(
        _response_with_archive("this should be an object but is a string")
    )

    outcome = await run_session_end_routine(
        _mock_session("s_arch_wrong"), provider, ws, session_uuid="s_arch_wrong"
    )
    assert outcome == "llm_merged"
    assert not ws.exists("decisions_archive")
    assert not ws.exists("lessons_archive")
