# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: the consolidation pass must actually COMPRESS, and must survive
a multi-minute generation.

Incident (orbital-marketing, 2026-07-27). Two separate defects, found by
measuring a real pass end to end rather than trusting the outcome string:

  1. A pass that SUCCEEDED returned every file bigger than it started
     (decisions 7355 -> 7652, lessons 5251 -> 5463) and archived nothing. The
     prompt states the target but never the file's CURRENT size — it hands the
     model "(~4 chars/token)" and asks it to estimate its own output length,
     which models cannot do. With no reliable signal and "archiving is a last
     resort", the model always concludes it is fine.

  2. The non-streaming call held an idle connection for 15 minutes while the
     server generated, and the connection dropped ("Connection error." at 901s,
     inside both our deadline and the SDK's). Streaming keeps bytes flowing and
     turns the deadline into a no-progress check instead of a total-duration
     guess.

Invariants:
  - consolidation_target(key) sits a real margin BELOW the soft budget, so a
    pass that lands on target does not re-trip the flag on the next append.
  - The prompt carries MEASURED sizes and an explicit archive quota.
  - The merge streams, and gives up on absence of progress, not on duration.
  - The deterministic backstop enforces the same target when the LLM declines.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import memory_entries as _mem
from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


def _mock_session(session_id="sess_compress"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = []
    return session


def _merge_response(tag="x"):
    return json.dumps({
        "project_state": f"# State\nstate-{tag}",
        "decisions": f"## 2026-06-18: Decision {tag}\n**Chose:** A\n\n",
        "lessons": f"1. **Lesson {tag}.** Problem p, fix f.\n",
        "index": f"# INDEX\n- PROJECT_STATE.md — scratchpad ({tag}).\n",
    })


def _streaming_provider(text, *, chunk_size=40, locked_on=True):
    """A provider whose stream() yields `text` in small deltas."""
    provider = MagicMock()
    provider.reasoning_locked_on = locked_on
    provider.model = "test-model"

    async def _stream(messages, tools=None, **kwargs):
        for i in range(0, len(text), chunk_size):
            yield StreamChunk(text=text[i:i + chunk_size])
        yield StreamChunk(
            is_final=True, usage=TokenUsage(input_tokens=1, output_tokens=1)
        )

    provider.stream = _stream
    provider.complete = AsyncMock(
        side_effect=AssertionError("merge must stream, not use complete()")
    )
    return provider


@pytest.fixture(autouse=True)
def _reset_completion_set():
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


def _over_soft_workspace(tmp_path) -> WorkspaceFileManager:
    """decisions and lessons both over their soft budgets, like the incident."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "".join(
        f"## 2026-07-{d:02d}: Decision {d}\n**Chose:** {'x' * 1100}\n\n"
        for d in range(1, 27)
    ))
    ws.write("lessons", "".join(
        f"{i}. **Lesson {i}.** {'y' * 1150}\n" for i in range(1, 19)
    ))
    return ws


# ---------------------------------------------------------------------------
# 1. Consolidation target carries headroom below the soft budget
# ---------------------------------------------------------------------------

def test_target_sits_a_full_headroom_below_soft_for_durable_files():
    """Landing exactly on the soft budget means the very next appended entry
    re-trips the flag. The pass has to buy room."""
    for key in ("decisions", "lessons"):
        soft = _mem.FILE_BUDGETS[key]["soft"]
        target = _mem.consolidation_target(key)
        assert soft - target >= _mem.CONSOLIDATION_HEADROOM_TOKENS


def test_target_never_demands_an_absurd_cut_from_a_small_file():
    """A flat 1000-token headroom would gut PROJECT_STATE (soft 1800) and
    INDEX (soft 1500), so the target is floored proportionally."""
    for key in ("state", "index"):
        soft = _mem.FILE_BUDGETS[key]["soft"]
        target = _mem.consolidation_target(key)
        assert 0 < target < soft
        assert target >= soft * 0.5


def test_target_is_below_soft_for_every_layer1_file():
    for key in ("state", "decisions", "lessons", "index"):
        assert _mem.consolidation_target(key) < _mem.FILE_BUDGETS[key]["soft"]


# ---------------------------------------------------------------------------
# 2. The prompt states MEASURED sizes and an explicit quota
# ---------------------------------------------------------------------------

def test_prompt_states_measured_size_and_quota_for_over_budget_files(tmp_path):
    ws = _over_soft_workspace(tmp_path)
    prompt = ws.build_session_end_prompt({"recent_messages": [], "files_modified": []})

    dec_now = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    dec_target = _mem.consolidation_target("decisions")

    assert "MEASURED" in prompt.upper()
    # The real number, not an invitation to estimate.
    assert f"{int(dec_now)}" in prompt
    assert f"{dec_target}" in prompt
    # And an explicit quota to remove.
    assert f"{int(dec_now) - dec_target}" in prompt


def test_prompt_does_not_ask_the_model_to_estimate_its_own_size(tmp_path):
    """The '(~4 chars/token)' self-estimate is what made archiving never fire."""
    ws = _over_soft_workspace(tmp_path)
    prompt = ws.build_session_end_prompt({"recent_messages": [], "files_modified": []})
    assert "4 chars/token" not in prompt


def test_prompt_marks_within_target_files_as_needing_no_action(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "## 2026-07-01: Tiny\n**Chose:** A\n")
    prompt = ws.build_session_end_prompt({"recent_messages": [], "files_modified": []})
    assert "within target" in prompt.lower()


# ---------------------------------------------------------------------------
# 3. The merge streams
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_merge_streams_and_accumulates_the_response(tmp_path):
    ws = _over_soft_workspace(tmp_path)
    provider = _streaming_provider(_merge_response("streamed"))

    outcome = await run_session_end_routine(
        _mock_session("s_stream"), provider, ws, session_uuid="s_stream"
    )
    assert outcome == "llm_merged"
    provider.complete.assert_not_called()
    assert "state-streamed" in (ws.read("state") or "")


@pytest.mark.asyncio
async def test_merge_gives_up_on_absence_of_progress_not_duration(tmp_path, monkeypatch):
    """A stream that stalls mid-flight must abort on the idle deadline, well
    before any total-duration cap."""
    monkeypatch.setattr(wsf_module, "_MERGE_IDLE_TIMEOUT_S", 0.05)
    ws = _over_soft_workspace(tmp_path)

    provider = MagicMock()
    provider.reasoning_locked_on = True
    provider.model = "test-model"

    async def _stalling_stream(messages, tools=None, **kwargs):
        yield StreamChunk(text='{"decisions": "')
        await asyncio.sleep(10)          # never delivers another chunk
        yield StreamChunk(text='"}')

    provider.stream = _stalling_stream
    provider.complete = AsyncMock(side_effect=AssertionError("must stream"))

    outcome = await run_session_end_routine(
        _mock_session("s_stall"), provider, ws, session_uuid="s_stall"
    )
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_a_slow_but_progressing_stream_is_not_killed(tmp_path, monkeypatch):
    """The whole point of the idle deadline: a genuinely slow generation that
    keeps producing must be allowed to finish."""
    monkeypatch.setattr(wsf_module, "_MERGE_IDLE_TIMEOUT_S", 0.5)
    ws = _over_soft_workspace(tmp_path)
    text = _merge_response("slow")

    provider = MagicMock()
    provider.reasoning_locked_on = True
    provider.model = "test-model"

    async def _slow_stream(messages, tools=None, **kwargs):
        for i in range(0, len(text), 60):
            await asyncio.sleep(0.05)     # slow, but always progressing
            yield StreamChunk(text=text[i:i + 60])
        yield StreamChunk(is_final=True, usage=TokenUsage(1, 1))

    provider.stream = _slow_stream
    provider.complete = AsyncMock(side_effect=AssertionError("must stream"))

    outcome = await run_session_end_routine(
        _mock_session("s_slow"), provider, ws, session_uuid="s_slow"
    )
    assert outcome == "llm_merged"


# ---------------------------------------------------------------------------
# 4. Deterministic floor enforces the same target when the LLM declines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backstop_demotes_to_target_when_the_merge_declines(tmp_path):
    """The LLM returning the file unchanged (the incident's exact behaviour)
    must NOT leave it over budget — the deterministic pass is the floor."""
    ws = _over_soft_workspace(tmp_path)
    before = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    assert before > _mem.FILE_BUDGETS["decisions"]["soft"]  # guard

    # Merge "succeeds" but hands back exactly what it was given.
    unchanged = json.dumps({
        "project_state": "", "index": "",
        "decisions": ws.read("decisions"),
        "lessons": ws.read("lessons"),
    })
    provider = _streaming_provider(unchanged)

    await run_session_end_routine(
        _mock_session("s_floor"), provider, ws, session_uuid="s_floor"
    )

    after = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    assert after <= _mem.consolidation_target("decisions")


@pytest.mark.asyncio
async def test_demoted_entries_are_moved_to_archive_not_deleted(tmp_path):
    ws = _over_soft_workspace(tmp_path)
    provider = _streaming_provider(json.dumps({
        "project_state": "", "index": "",
        "decisions": ws.read("decisions"), "lessons": ws.read("lessons"),
    }))

    await run_session_end_routine(
        _mock_session("s_arch"), provider, ws, session_uuid="s_arch"
    )

    archive = ws.read("decisions_archive") or ""
    assert archive.strip(), "over-budget entries must land in the archive"
    # And INDEX gains a pointer so the archive is discoverable.
    assert "DECISIONS_ARCHIVE.md" in (ws.read("index") or "")


def test_backstop_leaves_a_file_already_under_target_alone(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", "## 2026-07-01: Small\n**Chose:** A\n")
    before = ws.read("decisions")          # as persisted (write adds the format header)
    wsf_module._apply_hard_caps(ws)
    assert ws.read("decisions") == before
    assert not (ws.read("decisions_archive") or "").strip()
