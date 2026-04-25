# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: code-side sanity checks on session-end LLM output.

Post-LLM defense-in-depth for LESSONS, DECISIONS, CONTEXT:

  - Remove byte-identical duplicate entries (preserves first occurrence).
  - Enforce per-file entry caps (LESSONS 20 keep-first, DECISIONS 30
    keep-last, CONTEXT 25 keep-last).
  - Preserve LLM output byte-for-byte when neither check triggers, to
    avoid spurious main-agent prefix-cache invalidation.
  - Fall back to writing the LLM output verbatim with a warning if the
    entry-boundary regex does not parse the content.

Dedup runs BEFORE cap so exact duplicates do not consume cap slots that
unique entries could have filled.

Idempotency note: Task 2's guard short-circuits repeat invocations with
the same session_id. Tests clear the completion set via an autouse
fixture (mirrors the pattern from test_session_end_idempotency.py).
"""

import json
import logging
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

def _mock_session(session_id="sess_sanity"):
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
    session_log_entry="## Session sanity -- 2026-04-24\n- Ran tests",
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


def _numbered_lessons(n: int, *, prefix: str = "Lesson") -> str:
    """Build a numbered LESSONS body with n distinct entries."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"{i}. {prefix} number {i} text.")
    return "\n".join(lines) + "\n"


def _dashed_context(n: int, *, prefix: str = "Entity") -> str:
    """Build a dash-bulleted CONTEXT body with n distinct entries."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"- {prefix}-{i}: description for entity {i}")
    return "\n".join(lines) + "\n"


def _dated_decisions(n: int) -> str:
    """Build a DECISIONS body with n entries using ## YYYY-MM-DD: Title."""
    parts = []
    for i in range(1, n + 1):
        parts.append(
            f"## 2026-04-{i:02d}: Decision {i}\n"
            f"**Chose:** option-{i}\n"
            f"**Reason:** reason-{i}\n"
            f"**Rejected:** alternative-{i}\n"
        )
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def _reset_completion_set():
    """Clear the module-level idempotency set between tests."""
    wsf_module._completed_session_ends.clear()
    yield
    wsf_module._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# Test 1: LESSONS cap enforced when LLM returns over the cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lessons_cap_enforced_when_llm_returns_over_cap(tmp_path, caplog):
    """LLM returns 25 numbered lessons; on-disk file has 20 (first 20)."""
    ws = WorkspaceFileManager(str(tmp_path))
    llm_lessons = _numbered_lessons(25)

    session = _mock_session(session_id="s_cap_lessons")
    provider = _mock_provider(_llm_response(lessons=llm_lessons))

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws, session_id="s_cap_lessons",
        )

    on_disk = ws.read("lessons") or ""
    # Count "1. ", "2. ", ... markers at line starts.
    import re
    markers = re.findall(r"(?m)^\d+\.\s", on_disk)
    assert len(markers) == 20, (
        f"expected 20 lesson entries, got {len(markers)}\n---\n{on_disk}"
    )
    # keep="first" for LESSONS: entries 1..20 present, 21..25 dropped.
    assert "1. Lesson number 1 text." in on_disk
    assert "20. Lesson number 20 text." in on_disk
    assert "21. Lesson number 21 text." not in on_disk
    assert "25. Lesson number 25 text." not in on_disk
    # Cap-enforcement INFO log.
    assert any(
        "lessons" in r.message and "cap enforced" in r.message
        for r in caplog.records
    ), f"expected cap-enforced INFO log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 2: LESSONS exact duplicate removed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lessons_exact_duplicate_removed(tmp_path, caplog):
    """LLM emits two byte-identical lesson entries; dedup drops one."""
    ws = WorkspaceFileManager(str(tmp_path))
    # Two entries with the same marker ("2.") and the same body — the
    # exact-dedup key is the full marker+body after rstrip, so markers
    # must match as well as body text. Numbering is controlled by the
    # LLM; the sanity check does NOT renumber.
    llm_lessons = (
        "1. Alpha lesson body.\n"
        "2. Beta lesson body.\n"
        "2. Beta lesson body.\n"   # byte-identical to the prior entry
        "3. Gamma lesson body.\n"
    )

    session = _mock_session(session_id="s_dedup_lessons")
    provider = _mock_provider(_llm_response(lessons=llm_lessons))

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws, session_id="s_dedup_lessons",
        )

    on_disk = ws.read("lessons") or ""
    # The duplicate "2. Beta lesson body." should appear exactly once.
    assert on_disk.count("Beta lesson body.") == 1, (
        f"duplicate not removed:\n{on_disk}"
    )
    assert "Alpha lesson body." in on_disk
    assert "Gamma lesson body." in on_disk
    # Dedup INFO log present.
    assert any(
        "lessons" in r.message and "duplicates removed" in r.message
        for r in caplog.records
    ), f"expected duplicates-removed INFO log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 3: DECISIONS cap 30, keep="last"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decisions_cap_30(tmp_path):
    """LLM returns 35 decisions; on-disk file has the last 30."""
    ws = WorkspaceFileManager(str(tmp_path))
    llm_decisions = _dated_decisions(35)

    session = _mock_session(session_id="s_cap_decisions")
    provider = _mock_provider(_llm_response(decisions=llm_decisions))

    await run_session_end_routine(
        session, provider, ws, session_id="s_cap_decisions",
    )

    on_disk = ws.read("decisions") or ""
    header_count = sum(
        1 for line in on_disk.splitlines() if line.startswith("## ")
    )
    assert header_count == 30, (
        f"expected 30 decision headers, got {header_count}\n---\n{on_disk[:500]}"
    )
    # keep="last": entries 6..35 present, 1..5 dropped.
    assert "Decision 1\n" not in on_disk
    assert "Decision 5\n" not in on_disk
    assert "Decision 6\n" in on_disk
    assert "Decision 35\n" in on_disk


# ---------------------------------------------------------------------------
# Test 4: CONTEXT cap 25, keep="last"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_cap_25(tmp_path):
    """LLM returns 30 context entries; on-disk file has the last 25."""
    ws = WorkspaceFileManager(str(tmp_path))
    llm_context = _dashed_context(30)

    session = _mock_session(session_id="s_cap_context")
    provider = _mock_provider(_llm_response(context=llm_context))

    await run_session_end_routine(
        session, provider, ws, session_id="s_cap_context",
    )

    on_disk = ws.read("context") or ""
    import re
    markers = re.findall(r"(?m)^-\s", on_disk)
    assert len(markers) == 25, (
        f"expected 25 context entries, got {len(markers)}\n---\n{on_disk[:400]}"
    )
    # keep="last": entries 6..30 present, 1..5 dropped.
    assert "Entity-1:" not in on_disk
    assert "Entity-5:" not in on_disk
    assert "Entity-6:" in on_disk
    assert "Entity-30:" in on_disk


# ---------------------------------------------------------------------------
# Test 5: parse failure falls back to raw write + warning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parse_failure_falls_back_to_raw_write(tmp_path, caplog):
    """LLM returns LESSONS as a plain multi-line paragraph (no numbered
    markers). File is written as-is; a parse-failure warning is logged."""
    ws = WorkspaceFileManager(str(tmp_path))
    llm_lessons = (
        "This is a plain paragraph describing several lessons.\n"
        "It does not follow the numbered-entry format at all.\n"
        "There are no list markers, no hashes, nothing to split on.\n"
    )

    session = _mock_session(session_id="s_parse_fail")
    provider = _mock_provider(_llm_response(lessons=llm_lessons))

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        await run_session_end_routine(
            session, provider, ws, session_id="s_parse_fail",
        )

    on_disk = ws.read("lessons")
    assert on_disk == llm_lessons, (
        "parse-failure fallback must write LLM output verbatim"
    )
    assert any(
        "parse failed" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected parse-failure WARNING log, got: {[(r.levelno, r.message) for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 6: no-op path preserves LLM bytes exactly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_duplicates_no_cap_exceeded_preserves_bytes_exactly(tmp_path):
    """LLM returns 10 numbered lessons with no duplicates; on-disk content
    is byte-identical to the LLM output (no whitespace drift, no reorder).
    Guards the main-agent prefix cache from spurious invalidation."""
    ws = WorkspaceFileManager(str(tmp_path))
    # Mixed inter-entry whitespace: some entries separated by blank lines,
    # some not. The check must not normalize this.
    llm_lessons = (
        "1. First lesson body.\n"
        "\n"
        "2. Second lesson body.\n"
        "3. Third lesson body.\n"
        "\n"
        "4. Fourth lesson body.\n"
        "5. Fifth lesson body.\n"
        "6. Sixth lesson body.\n"
        "7. Seventh lesson body.\n"
        "\n"
        "8. Eighth lesson body.\n"
        "9. Ninth lesson body.\n"
        "10. Tenth lesson body.\n"
    )

    session = _mock_session(session_id="s_noop_bytes")
    provider = _mock_provider(_llm_response(lessons=llm_lessons))

    await run_session_end_routine(
        session, provider, ws, session_id="s_noop_bytes",
    )

    on_disk = ws.read("lessons")
    assert on_disk == llm_lessons, (
        "byte-identity violated: on-disk content differs from LLM output\n"
        f"---LLM---\n{llm_lessons!r}\n---DISK---\n{on_disk!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: dedup runs before cap (ordering proof)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dedup_before_cap(tmp_path):
    """21 numbered lessons with entries 5 and 6 byte-identical. After
    dedup there are 20 unique entries; cap=20 is NOT triggered. If cap
    had run first (21 -> 20 first-kept, both entries 5 and 6 retained),
    then dedup would have left 19 — so on-disk count of 20 proves the
    ordering."""
    ws = WorkspaceFileManager(str(tmp_path))
    # Build 21 entries with entries 5 and 6 identical (same marker, same
    # body). Both live within the first-20 slice, so cap-first would
    # leave both in and dedup-after would drop to 19.
    entries = []
    for i in range(1, 22):
        if i == 6:
            # Duplicate of entry 5 verbatim.
            entries.append("5. Duplicate body for ordering test.")
        elif i == 5:
            entries.append("5. Duplicate body for ordering test.")
        else:
            entries.append(f"{i}. Unique body for entry {i}.")
    llm_lessons = "\n".join(entries) + "\n"

    session = _mock_session(session_id="s_order")
    provider = _mock_provider(_llm_response(lessons=llm_lessons))

    await run_session_end_routine(
        session, provider, ws, session_id="s_order",
    )

    on_disk = ws.read("lessons") or ""
    import re
    markers = re.findall(r"(?m)^\d+\.\s", on_disk)
    assert len(markers) == 20, (
        f"expected 20 entries after dedup (cap not triggered), "
        f"got {len(markers)}\n---\n{on_disk}"
    )
    # Dedup kept first occurrence of the duplicate.
    assert on_disk.count("Duplicate body for ordering test.") == 1
    # All unique entries (1..4, 7..21) still present.
    for i in list(range(1, 5)) + list(range(7, 22)):
        assert f"{i}. Unique body for entry {i}." in on_disk, (
            f"unique entry {i} missing from on-disk output:\n{on_disk}"
        )


# ---------------------------------------------------------------------------
# Test 8: empty LLM response preserves the existing file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_llm_response_still_preserves_existing_file(tmp_path):
    """Seed LESSONS with 3 entries; LLM returns ''. File untouched.
    Task 3 already skips the write on empty; this asserts that sanity
    checks do not interfere with that contract."""
    ws = WorkspaceFileManager(str(tmp_path))
    seed = (
        "1. Seeded lesson one.\n"
        "2. Seeded lesson two.\n"
        "3. Seeded lesson three.\n"
    )
    ws.write("lessons", seed)
    before = ws.read("lessons")

    session = _mock_session(session_id="s_empty_preserves")
    provider = _mock_provider(_llm_response(lessons=""))

    await run_session_end_routine(
        session, provider, ws, session_id="s_empty_preserves",
    )

    after = ws.read("lessons")
    assert after == before, (
        "LESSONS.md changed after empty-response session-end"
    )
