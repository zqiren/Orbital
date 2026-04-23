# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: DECISIONS.md and CONTEXT.md use full-file write semantics.

Previously DECISIONS and CONTEXT used append() at session-end, relying on
the LLM to avoid duplicates via prompt discipline alone. Duplicates piled
up in production. This test suite locks in the new contract:

  - DECISIONS: LLM returns COMPLETE updated file; code writes it atomically.
  - CONTEXT: same.
  - LESSONS: same (already full-file, now with explicit empty=preserve).
  - Empty string response from the LLM => preserve existing file unchanged.
  - Running twice with identical LLM output => byte-identical file.

Idempotency note: run_session_end_routine short-circuits on repeat
session_id (Task 2). Tests that invoke the routine twice on the same
WorkspaceFileManager use DISTINCT session_ids and reset the completion
set via an autouse fixture.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(session_id="sess_full_file"):
    session = MagicMock()
    session.session_id = session_id
    session.get_messages.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    return session


def _mock_provider(response_text):
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = response_text
    provider.complete.return_value = resp
    return provider


def _llm_response(
    *,
    project_state="# State\nok",
    decisions="",
    session_log_entry="## Session log-entry -- today\n- stuff",
    lessons="",
    context="",
):
    return json.dumps({
        "project_state": project_state,
        "decisions": decisions,
        "session_log_entry": session_log_entry,
        "lessons": lessons,
        "context": context,
    })


_SEED_DECISIONS = (
    "## 2026-01-01: Use FastAPI\n"
    "**Chose:** FastAPI\n**Reason:** async native\n**Rejected:** Flask\n\n"
    "## 2026-01-05: Use Vite\n"
    "**Chose:** Vite\n**Reason:** fast HMR\n**Rejected:** webpack\n\n"
    "## 2026-01-10: Use pytest\n"
    "**Chose:** pytest\n**Reason:** async fixtures\n**Rejected:** unittest\n"
)

_SEED_CONTEXT = (
    "## Railway\n- Relay hosted on Railway\n\n"
    "## Anthropic API\n- Claude models via console.anthropic.com\n\n"
    "## Moonshot\n- Kimi utility model at api.moonshot.cn\n"
)

_SEED_LESSONS = (
    "## Atomic writes on Windows\n"
    "**Problem:** os.replace may raise PermissionError\n"
    "**Fix:** retry loop with 50ms sleep\n\n"
    "## React 19 batching\n"
    "**Problem:** closures mutated in setState read stale\n"
    "**Fix:** use flushSync or restructure\n"
)


@pytest.fixture(autouse=True)
def _reset_completion_set():
    """Clear the module-level idempotency set between tests."""
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Test 1: DECISIONS full-file write carries forward prior entries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decisions_full_file_write_carries_forward(tmp_path):
    """LLM returns complete DECISIONS (3 prior + 1 new); on-disk file has
    exactly 4 entries with no duplication from an old append-style artifact."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("decisions", _SEED_DECISIONS)

    full_updated = _SEED_DECISIONS + (
        "\n## 2026-04-24: Full-file write\n"
        "**Chose:** overwrite\n**Reason:** dedup\n**Rejected:** append\n"
    )
    session = _mock_session(session_id="s_dec_carry")
    provider = _mock_provider(_llm_response(decisions=full_updated))

    await run_session_end_routine(session, provider, ws, session_id="s_dec_carry")

    on_disk = ws.read("decisions") or ""
    # Exactly 4 "## " entry headers (one per decision)
    header_count = sum(1 for line in on_disk.splitlines() if line.startswith("## "))
    assert header_count == 4, (
        f"expected 4 decision headers, got {header_count}\n---\n{on_disk}"
    )
    # Original titles preserved
    assert "Use FastAPI" in on_disk
    assert "Use Vite" in on_disk
    assert "Use pytest" in on_disk
    # New entry added
    assert "Full-file write" in on_disk
    # No append-style artifact: the file should not contain the seed twice
    assert on_disk.count("Use FastAPI") == 1


# ---------------------------------------------------------------------------
# Test 2: DECISIONS empty response preserves existing file (byte-identical)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decisions_empty_response_preserves_existing(tmp_path):
    """LLM returns decisions=''; existing DECISIONS.md is untouched."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("decisions", _SEED_DECISIONS)
    before = ws.read("decisions")

    session = _mock_session(session_id="s_dec_empty")
    provider = _mock_provider(_llm_response(decisions=""))

    await run_session_end_routine(session, provider, ws, session_id="s_dec_empty")

    after = ws.read("decisions")
    assert after == before, (
        "DECISIONS.md changed after empty-response session-end"
    )


# ---------------------------------------------------------------------------
# Test 3: CONTEXT full-file write carries forward prior entries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_full_file_write_carries_forward(tmp_path):
    """LLM returns complete CONTEXT (3 prior + 1 new); on-disk file has 4."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("context", _SEED_CONTEXT)

    full_updated = _SEED_CONTEXT + (
        "\n## GitHub\n- Repo at github.com/orbital/orbital\n"
    )
    session = _mock_session(session_id="s_ctx_carry")
    provider = _mock_provider(_llm_response(context=full_updated))

    await run_session_end_routine(session, provider, ws, session_id="s_ctx_carry")

    on_disk = ws.read("context") or ""
    header_count = sum(1 for line in on_disk.splitlines() if line.startswith("## "))
    assert header_count == 4, (
        f"expected 4 context headers, got {header_count}\n---\n{on_disk}"
    )
    assert "Railway" in on_disk
    assert "Anthropic API" in on_disk
    assert "Moonshot" in on_disk
    assert "GitHub" in on_disk
    assert on_disk.count("## Railway") == 1


# ---------------------------------------------------------------------------
# Test 4: CONTEXT empty response preserves existing file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_empty_response_preserves_existing(tmp_path):
    """LLM returns context=''; existing CONTEXT.md is untouched."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("context", _SEED_CONTEXT)
    before = ws.read("context")

    session = _mock_session(session_id="s_ctx_empty")
    provider = _mock_provider(_llm_response(context=""))

    await run_session_end_routine(session, provider, ws, session_id="s_ctx_empty")

    after = ws.read("context")
    assert after == before, (
        "CONTEXT.md changed after empty-response session-end"
    )


# ---------------------------------------------------------------------------
# Test 5: DECISIONS idempotent on same input (no cosmetic drift)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decisions_idempotent_on_same_input(tmp_path):
    """Run session-end twice with different session_ids but identical LLM
    output — final DECISIONS.md must be byte-identical to the first run.
    Guards against main-agent prefix cache invalidation caused by drift."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("decisions", _SEED_DECISIONS)

    full_updated = _SEED_DECISIONS + (
        "\n## 2026-04-24: Idempotency matters\n"
        "**Chose:** byte-identical writes\n**Reason:** cache\n**Rejected:** drift\n"
    )
    llm_body = _llm_response(decisions=full_updated)

    session1 = _mock_session(session_id="s_idem_1")
    provider1 = _mock_provider(llm_body)
    await run_session_end_routine(session1, provider1, ws, session_id="s_idem_1")
    first_run_bytes = ws.read("decisions")

    session2 = _mock_session(session_id="s_idem_2")
    provider2 = _mock_provider(llm_body)
    await run_session_end_routine(session2, provider2, ws, session_id="s_idem_2")
    second_run_bytes = ws.read("decisions")

    assert second_run_bytes == first_run_bytes, (
        "DECISIONS.md drifted between identical-input session-end runs"
    )


# ---------------------------------------------------------------------------
# Test 6: Prompt contains the new instruction phrases
# ---------------------------------------------------------------------------

def test_prompt_contains_new_decisions_instructions(tmp_path):
    """build_session_end_prompt emits the revised instructions for DECISIONS,
    LESSONS, and CONTEXT (caps, carry-forward, exclusion list)."""
    ws = WorkspaceFileManager(str(tmp_path))
    summary = {
        "message_count": 2,
        "tool_calls_count": 0,
        "files_modified": [],
        "recent_messages": [{"role": "user", "content": "hi"}],
    }
    prompt = ws.build_session_end_prompt(summary)
    # Collapse intra-line whitespace so we match phrases that wrap across
    # lines in the verbatim prompt template.
    normalized = " ".join(prompt.split())

    # Decisions block
    assert "COMPLETE updated DECISIONS.md" in normalized, (
        "prompt missing 'COMPLETE updated DECISIONS.md'"
    )
    assert "Cap: 30 entries" in normalized, "prompt missing 'Cap: 30 entries'"

    # Context exclusion list
    assert "Exclusions (do NOT include)" in normalized, (
        "prompt missing CONTEXT exclusion-list header"
    )


# ---------------------------------------------------------------------------
# Test 7: LESSONS empty response preserves existing file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lessons_empty_response_preserves_existing(tmp_path):
    """LLM returns lessons=''; existing LESSONS.md is untouched."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("lessons", _SEED_LESSONS)
    before = ws.read("lessons")

    session = _mock_session(session_id="s_les_empty")
    provider = _mock_provider(_llm_response(lessons=""))

    await run_session_end_routine(session, provider, ws, session_id="s_les_empty")

    after = ws.read("lessons")
    assert after == before, (
        "LESSONS.md changed after empty-response session-end"
    )
