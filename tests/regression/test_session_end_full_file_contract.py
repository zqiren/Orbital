# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: DECISIONS.md and INDEX.md use full-file (merge-not-append) semantics.

Previously DECISIONS and CONTEXT used append() at session-end, relying on the
LLM to avoid duplicates via prompt discipline alone. Duplicates piled up in
production. This suite locks in the merge-not-append contract under the Layer-1
memory redesign, where CONTEXT.md was promoted to INDEX.md:

  - DECISIONS: LLM returns the COMPLETE updated file; the persist path STAMPS
    system-managed metadata (``<!--mem id:.. created:.. touched:..-->``) onto
    each entry header and writes it atomically. MERGE-AND-SUPERSEDE — prior
    entries are carried forward exactly once, never appended-with-duplication.
  - INDEX (formerly CONTEXT): same full-file/merge contract; INDEX is an
    overwrite navigation map, so the LLM's complete file is written verbatim.
  - LESSONS: same full-file contract (now renumbered + stamped on persist).
  - Empty string ("") response for a field => preserve the existing file
    unchanged (no LLM-driven write for that key).
  - Running twice with identical LLM output => byte-identical file.

JSON contract: the session-end LLM now returns the keys
``project_state``/``decisions``/``lessons``/``index`` ONLY. There is no
``session_log_entry`` and no ``context`` key anymore (SESSION_LOG was removed
and CONTEXT was renamed to INDEX).

No-delta gate: run_session_end_routine is a no-op (no LLM call) when no Layer-1
file changed since the last cleanup marker. Each test seeds a file (which makes
a delta present and the marker absent), so the routine runs. Tests that invoke
the routine twice use DISTINCT session_uuids and bypass idempotency.
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
    session.session_uuid = session_id
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
    lessons="",
    index="",
):
    """The new session-end JSON contract: project_state/decisions/lessons/index.

    No ``session_log_entry`` (SESSION_LOG removed) and no ``context``
    (renamed to ``index``). "" preserves the existing file.
    """
    return json.dumps({
        "project_state": project_state,
        "decisions": decisions,
        "lessons": lessons,
        "index": index,
    })


_SEED_DECISIONS = (
    "## 2026-01-01: Use FastAPI\n"
    "**Chose:** FastAPI\n**Reason:** async native\n**Rejected:** Flask\n\n"
    "## 2026-01-05: Use Vite\n"
    "**Chose:** Vite\n**Reason:** fast HMR\n**Rejected:** webpack\n\n"
    "## 2026-01-10: Use pytest\n"
    "**Chose:** pytest\n**Reason:** async fixtures\n**Rejected:** unittest\n"
)

# Formerly _SEED_CONTEXT — same content, now seeded into INDEX (the navigation
# map that replaced CONTEXT.md in the Layer-1 redesign).
_SEED_INDEX = (
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
# Test 1: DECISIONS full-file write carries forward prior entries (merge-not-append)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decisions_full_file_write_carries_forward(tmp_path):
    """LLM returns complete DECISIONS (3 prior + 1 new); on-disk file has
    exactly 4 entries with no duplication from an old append-style artifact.

    The persist path now STAMPS metadata onto each header line, but stamping
    keeps one ``## `` header per decision and preserves each title's body
    exactly once — so the merge-not-append invariant still holds byte-wise on
    the entry titles."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("decisions", _SEED_DECISIONS)

    full_updated = _SEED_DECISIONS + (
        "\n## 2026-04-24: Full-file write\n"
        "**Chose:** overwrite\n**Reason:** dedup\n**Rejected:** append\n"
    )
    session = _mock_session(session_id="s_dec_carry")
    provider = _mock_provider(_llm_response(decisions=full_updated))

    await run_session_end_routine(
        session, provider, ws, session_uuid="s_dec_carry", bypass_idempotency=True
    )

    on_disk = ws.read("decisions") or ""
    # Exactly 4 "## " entry headers (one per decision) — metadata comments live
    # on the same header line, so they do not inflate this count.
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
    # Persist path stamped system-managed metadata onto the entries.
    assert "<!--mem id:" in on_disk


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

    await run_session_end_routine(
        session, provider, ws, session_uuid="s_dec_empty", bypass_idempotency=True
    )

    after = ws.read("decisions")
    assert after == before, (
        "DECISIONS.md changed after empty-response session-end"
    )


# ---------------------------------------------------------------------------
# Test 3: INDEX full-file write carries forward prior entries (was CONTEXT)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_full_file_write_carries_forward(tmp_path):
    """LLM returns complete INDEX (3 prior + 1 new); on-disk file has 4.

    INDEX (the navigation map that replaced CONTEXT.md) is an overwrite file,
    so the LLM's complete file is written verbatim — carry-forward with no
    append-style duplication."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("index", _SEED_INDEX)

    full_updated = _SEED_INDEX + (
        "\n## GitHub\n- Repo at github.com/orbital/orbital\n"
    )
    session = _mock_session(session_id="s_idx_carry")
    provider = _mock_provider(_llm_response(index=full_updated))

    await run_session_end_routine(
        session, provider, ws, session_uuid="s_idx_carry", bypass_idempotency=True
    )

    on_disk = ws.read("index") or ""
    header_count = sum(1 for line in on_disk.splitlines() if line.startswith("## "))
    assert header_count == 4, (
        f"expected 4 index headers, got {header_count}\n---\n{on_disk}"
    )
    assert "Railway" in on_disk
    assert "Anthropic API" in on_disk
    assert "Moonshot" in on_disk
    assert "GitHub" in on_disk
    assert on_disk.count("## Railway") == 1


# ---------------------------------------------------------------------------
# Test 4: INDEX empty response preserves existing file (was CONTEXT)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_index_empty_response_preserves_existing(tmp_path):
    """LLM returns index=''; existing INDEX.md is untouched."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("index", _SEED_INDEX)
    before = ws.read("index")

    session = _mock_session(session_id="s_idx_empty")
    provider = _mock_provider(_llm_response(index=""))

    await run_session_end_routine(
        session, provider, ws, session_uuid="s_idx_empty", bypass_idempotency=True
    )

    after = ws.read("index")
    assert after == before, (
        "INDEX.md changed after empty-response session-end"
    )


# ---------------------------------------------------------------------------
# Test 5: DECISIONS idempotent on same input (no cosmetic drift)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decisions_idempotent_on_same_input(tmp_path):
    """Run session-end twice with different session_uuids but identical LLM
    output — final DECISIONS.md must be byte-identical to the first run.
    Guards against main-agent prefix cache invalidation caused by drift.

    Stamping is deterministic for a fixed ``today`` and unchanged bodies (the
    second pass matches entries by id/title, preserves created, and leaves
    touched put), so the two runs converge to the same bytes."""
    ws = WorkspaceFileManager(str(tmp_path))
    ws.write("decisions", _SEED_DECISIONS)

    full_updated = _SEED_DECISIONS + (
        "\n## 2026-04-24: Idempotency matters\n"
        "**Chose:** byte-identical writes\n**Reason:** cache\n**Rejected:** drift\n"
    )
    llm_body = _llm_response(decisions=full_updated)

    session1 = _mock_session(session_id="s_idem_1")
    provider1 = _mock_provider(llm_body)
    await run_session_end_routine(
        session1, provider1, ws, session_uuid="s_idem_1", bypass_idempotency=True
    )
    first_run_bytes = ws.read("decisions")

    session2 = _mock_session(session_id="s_idem_2")
    provider2 = _mock_provider(llm_body)
    await run_session_end_routine(
        session2, provider2, ws, session_uuid="s_idem_2", bypass_idempotency=True
    )
    second_run_bytes = ws.read("decisions")

    assert second_run_bytes == first_run_bytes, (
        "DECISIONS.md drifted between identical-input session-end runs"
    )


# ---------------------------------------------------------------------------
# Test 6: Prompt contains the new instruction phrases
# ---------------------------------------------------------------------------

def test_prompt_contains_new_decisions_instructions(tmp_path):
    """build_session_end_prompt emits the revised instructions for DECISIONS,
    LESSONS, and INDEX (merge-and-supersede, navigation-only, archive pointer).

    The old caps-based phrasing ("Cap: 30 entries") and CONTEXT framing are
    gone; the redesign teaches DECISIONS as merge-and-supersede and INDEX as a
    navigation-only map that points to the archives."""
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

    # Decisions block: complete-file + merge-and-supersede (no append).
    assert "COMPLETE updated DECISIONS.md" in normalized, (
        "prompt missing 'COMPLETE updated DECISIONS.md'"
    )
    assert "MERGE AND SUPERSEDE" in normalized, (
        "prompt missing the merge-and-supersede directive"
    )

    # INDEX replaced CONTEXT: it is now a navigation-only map that points to the
    # archives, not a token-capped project context blob.
    assert "COMPLETE updated INDEX.md" in normalized, (
        "prompt missing 'COMPLETE updated INDEX.md'"
    )
    assert "NAVIGATION ONLY" in normalized, (
        "prompt missing INDEX navigation-only framing"
    )
    assert "DECISIONS_ARCHIVE.md" in normalized, (
        "prompt missing the archive pointer instruction"
    )

    # The JSON contract is keyed project_state/decisions/lessons/index — no
    # session_log_entry, no context.
    assert "project_state" in normalized and '"index"' in normalized
    assert "session_log_entry" not in normalized
    assert '"context"' not in normalized


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

    await run_session_end_routine(
        session, provider, ws, session_uuid="s_les_empty", bypass_idempotency=True
    )

    after = ws.read("lessons")
    assert after == before, (
        "LESSONS.md changed after empty-response session-end"
    )
