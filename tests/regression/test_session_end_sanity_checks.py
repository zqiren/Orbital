# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: code-side sanity-check helpers for memory-file LLM output.

Post-LLM defense-in-depth for LESSONS, DECISIONS, CONTEXT lives in three pure
helpers:

  - ``_dedupe_exact`` — remove byte-identical duplicate entries (preserves the
    first occurrence).
  - ``_cap_entries`` / ``_apply_sanity_checks`` — enforce per-file entry caps
    (LESSONS 20 keep-first, DECISIONS 30 keep-last), preserve LLM bytes when
    neither check triggers, and fall back to writing the LLM output verbatim
    (with a warning) when the entry-boundary regex does not parse the content.
  - ``_cap_context_tokens`` — token-cap the (formerly entry-based) project map.

Dedup runs BEFORE cap so exact duplicates do not consume cap slots that unique
entries could have filled.

Layer-1 redesign note: these helpers STILL EXIST and are still the correct unit
to test for the dedup/cap/parse-failure contract, but they are NO LONGER called
inside ``run_session_end_routine``. Session-end now stamps metadata onto
DECISIONS/LESSONS (``memory_entries.stamp``) and enforces size via a
deterministic demotion-to-archive backstop (``_apply_hard_caps`` /
``memory_entries.split_for_demotion``) rather than the 20/30 entry cap. The
session-end-level "size is bounded" intent is therefore re-expressed here against
demotion; the byte-level dedup/cap/parse contract is exercised against the
helpers directly (which is where that logic now lives, and where it is reachable
without the stamping/budget machinery in the way).
"""

import json
import logging
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import memory_entries as M
from agent_os.agent import workspace_files as wsf_module
from agent_os.agent.workspace_files import (
    WorkspaceFileManager,
    run_session_end_routine,
    _apply_sanity_checks,
    _cap_context_tokens,
    _CONTEXT_TOKEN_CAP,
    _DECISIONS_ENTRY_PATTERN,
    _DECISIONS_CAP,
    _LESSONS_ENTRY_PATTERN,
    _LESSONS_CAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_session(session_id="sess_sanity"):
    session = MagicMock()
    session.session_id = session_id
    session.session_uuid = session_id
    session.get_messages.return_value = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    return session


def _mock_provider(response_text):
    provider = streamable(AsyncMock())
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
    """Session-end LLM JSON shape after the Layer-1 redesign.

    Keys are project_state / decisions / lessons / index. There is no
    ``session_log_entry`` and no ``context`` key anymore.
    """
    return json.dumps({
        "project_state": project_state,
        "decisions": decisions,
        "lessons": lessons,
        "index": index,
    })


def _numbered_lessons(n: int, *, prefix: str = "Lesson") -> str:
    """Build a numbered LESSONS body with n distinct entries."""
    lines = []
    for i in range(1, n + 1):
        lines.append(f"{i}. {prefix} number {i} text.")
    return "\n".join(lines) + "\n"


def _dashed_context(n: int, *, prefix: str = "Entity") -> str:
    """Build a dash-bulleted CONTEXT/INDEX body with n distinct lines."""
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
# Test 1: LESSONS cap enforced (helper) when output is over the cap
# ---------------------------------------------------------------------------

def test_lessons_cap_enforced_when_over_cap(caplog):
    """25 numbered lessons → _apply_sanity_checks trims to the first 20.

    Migrated from a session-end test: session-end no longer applies the
    20-entry cap (it stamps + demotes by token budget). The 20-keep-first cap
    contract now lives entirely in the helper, so we exercise it there.
    """
    llm_lessons = _numbered_lessons(25)

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        on_disk = _apply_sanity_checks(
            llm_lessons,
            _LESSONS_ENTRY_PATTERN,
            _LESSONS_CAP,
            keep="first",
            filename="LESSONS.md",
        )

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
        "LESSONS" in r.message and "cap enforced" in r.message
        for r in caplog.records
    ), f"expected cap-enforced INFO log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 2: LESSONS exact duplicate removed (helper)
# ---------------------------------------------------------------------------

def test_lessons_exact_duplicate_removed(caplog):
    """Two byte-identical lesson entries; _apply_sanity_checks drops one."""
    # Two entries with the same marker ("2.") and the same body — the
    # exact-dedup key is the full marker+body after rstrip, so markers must
    # match as well as body text. The sanity check does NOT renumber.
    llm_lessons = (
        "1. Alpha lesson body.\n"
        "2. Beta lesson body.\n"
        "2. Beta lesson body.\n"   # byte-identical to the prior entry
        "3. Gamma lesson body.\n"
    )

    with caplog.at_level(logging.INFO, logger="agent_os.agent.workspace_files"):
        on_disk = _apply_sanity_checks(
            llm_lessons,
            _LESSONS_ENTRY_PATTERN,
            _LESSONS_CAP,
            keep="first",
            filename="LESSONS.md",
        )

    # The duplicate "2. Beta lesson body." should appear exactly once.
    assert on_disk.count("Beta lesson body.") == 1, (
        f"duplicate not removed:\n{on_disk}"
    )
    assert "Alpha lesson body." in on_disk
    assert "Gamma lesson body." in on_disk
    # Dedup INFO log present.
    assert any(
        "LESSONS" in r.message and "duplicates removed" in r.message
        for r in caplog.records
    ), f"expected duplicates-removed INFO log, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 3: DECISIONS cap 30, keep="last" (helper)
# ---------------------------------------------------------------------------

def test_decisions_cap_30():
    """35 decisions → _apply_sanity_checks keeps the last 30."""
    llm_decisions = _dated_decisions(35)

    on_disk = _apply_sanity_checks(
        llm_decisions,
        _DECISIONS_ENTRY_PATTERN,
        _DECISIONS_CAP,
        keep="last",
        filename="DECISIONS.md",
    )

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
# Test 4: CONTEXT is token-capped (not entry-capped) via _cap_context_tokens
# ---------------------------------------------------------------------------

def test_context_no_longer_entry_capped():
    """The project map is token-capped, not entry-capped: under-budget content
    (30 short lines) passes through entirely — no 25-entry truncation.

    ``_cap_context_tokens`` is exercised directly. Session-end no longer routes
    a ``context`` key through it (the roster key is now ``index``), but the
    helper's token-cap contract is unchanged."""
    body = _dashed_context(30)  # ~30 short lines, well under the token cap

    on_disk = _cap_context_tokens(body, filename="INDEX.md")

    markers = re.findall(r"(?m)^-\s", on_disk)
    assert len(markers) == 30, (
        f"expected all 30 entries preserved (no entry cap), got {len(markers)}\n"
        f"---\n{on_disk[:400]}"
    )
    assert "Entity-1:" in on_disk
    assert "Entity-30:" in on_disk
    # Under budget → returned unchanged (same object → prefix-cache friendly).
    assert on_disk is body


def test_context_token_capped_when_oversized():
    """Over-budget content is truncated at a line boundary (~1500 token cap
    => ~6000 char budget)."""
    char_budget = _CONTEXT_TOKEN_CAP * 4
    body = _dashed_context(400)  # ~18k chars, ~3x over budget
    assert len(body) > char_budget

    on_disk = _cap_context_tokens(body, filename="INDEX.md")

    assert 0 < len(on_disk) < len(body)
    # truncation lands on a line boundary — every kept line is whole
    src_lines = set(body.splitlines())
    for ln in on_disk.splitlines():
        assert ln in src_lines


# ---------------------------------------------------------------------------
# Test 5: parse failure falls back to raw passthrough + warning (helper)
# ---------------------------------------------------------------------------

def test_parse_failure_falls_back_to_raw_passthrough(caplog):
    """A plain multi-line paragraph (no numbered markers) is returned as-is and
    a parse-failure warning is logged."""
    llm_lessons = (
        "This is a plain paragraph describing several lessons.\n"
        "It does not follow the numbered-entry format at all.\n"
        "There are no list markers, no hashes, nothing to split on.\n"
    )

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.workspace_files"):
        out = _apply_sanity_checks(
            llm_lessons,
            _LESSONS_ENTRY_PATTERN,
            _LESSONS_CAP,
            keep="first",
            filename="LESSONS.md",
        )

    assert out == llm_lessons, (
        "parse-failure fallback must return the input verbatim"
    )
    assert any(
        "parse failed" in r.message and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected parse-failure WARNING log, got: {[(r.levelno, r.message) for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Test 6: no-op path preserves bytes exactly (helper)
# ---------------------------------------------------------------------------

def test_no_duplicates_no_cap_exceeded_preserves_bytes_exactly():
    """10 numbered lessons, no duplicates, under cap → helper returns the input
    byte-identical (no whitespace drift, no reorder). Guards the main-agent
    prefix cache from spurious invalidation.

    (Note: at session-end the bytes DO change now — DECISIONS/LESSONS get a
    ``<!--mem ...-->`` metadata stamp — so the byte-identity contract only holds
    at the helper level, which is what this test pins.)"""
    # Mixed inter-entry whitespace: some entries separated by blank lines,
    # some not. The helper must not normalize this.
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

    out = _apply_sanity_checks(
        llm_lessons,
        _LESSONS_ENTRY_PATTERN,
        _LESSONS_CAP,
        keep="first",
        filename="LESSONS.md",
    )

    assert out == llm_lessons, (
        "byte-identity violated: output differs from input\n"
        f"---IN---\n{llm_lessons!r}\n---OUT---\n{out!r}"
    )
    # Same object — the cheap-identity guarantee.
    assert out is llm_lessons


# ---------------------------------------------------------------------------
# Test 7: dedup runs before cap (ordering proof, helper)
# ---------------------------------------------------------------------------

def test_dedup_before_cap():
    """21 numbered lessons with entries 5 and 6 byte-identical. After dedup
    there are 20 unique entries; cap=20 is NOT triggered. If cap had run first
    (21 -> 20 first-kept, both 5 and 6 retained), then dedup would have left 19
    — so an output count of 20 proves the ordering."""
    # Build 21 entries with entries 5 and 6 identical (same marker, same body).
    # Both live within the first-20 slice, so cap-first would leave both in and
    # dedup-after would drop to 19.
    entries = []
    for i in range(1, 22):
        if i in (5, 6):
            # Duplicate body (verbatim, same marker) for entries 5 and 6.
            entries.append("5. Duplicate body for ordering test.")
        else:
            entries.append(f"{i}. Unique body for entry {i}.")
    llm_lessons = "\n".join(entries) + "\n"

    out = _apply_sanity_checks(
        llm_lessons,
        _LESSONS_ENTRY_PATTERN,
        _LESSONS_CAP,
        keep="first",
        filename="LESSONS.md",
    )

    markers = re.findall(r"(?m)^\d+\.\s", out)
    assert len(markers) == 20, (
        f"expected 20 entries after dedup (cap not triggered), "
        f"got {len(markers)}\n---\n{out}"
    )
    # Dedup kept first occurrence of the duplicate.
    assert out.count("Duplicate body for ordering test.") == 1
    # All unique entries (1..4, 7..21) still present.
    for i in list(range(1, 5)) + list(range(7, 22)):
        assert f"{i}. Unique body for entry {i}." in out, (
            f"unique entry {i} missing from output:\n{out}"
        )


# ---------------------------------------------------------------------------
# Test 8: session-end size backstop is demotion-to-archive (not the 20/30 cap)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_end_demotes_oversized_decisions_to_archive(
    tmp_path, monkeypatch
):
    """LLM returns many decisions over the hard budget; session-end's
    deterministic backstop DEMOTES the coldest to DECISIONS_ARCHIVE.md (never
    deletes), protecting the oldest 3. This replaces the old "session-end
    enforces a 30-entry cap" assertion — the cap moved to a token-budget
    demotion."""
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()

    # 12 stamped decisions, oldest-first, with explicit touched dates so the
    # MIDDLE ones are coldest and demote (not the protected oldest 3).
    def _decisions(n):
        out = []
        for i in range(1, n + 1):
            touched = "2025-01-01" if 4 <= i <= 9 else "2026-06-18"
            out.append(
                f"## 2026-01-{i:02d}: Decision {i} "
                f"<!--mem id:d{i} created:2026-01-{i:02d} touched:{touched}-->\n"
                f"**Chose:** option {i}\n**Reason:** {'x' * 20}\n\n"
            )
        return "".join(out)

    # The LLM merely returns the same (already-stamped) content; the point is
    # the deterministic demotion that runs afterwards.
    llm_decisions = _decisions(12)
    session = _mock_session(session_id="s_demote")
    provider = _mock_provider(_llm_response(decisions=llm_decisions))

    await run_session_end_routine(
        session, provider, ws,
        session_uuid="s_demote", bypass_idempotency=True, project_id="p1",
    )

    kept = ws.read("decisions") or ""
    archive = ws.read("decisions_archive") or ""
    index = ws.read("index") or ""

    # Live file is within the hard budget.
    assert M.est_tokens(kept) <= M.FILE_BUDGETS["decisions"]["hard"]
    # Oldest-3 protected, never demoted.
    for i in (1, 2, 3):
        assert f"Decision {i} " in kept
    # Something demoted (never deleted), and INDEX points to the archive.
    assert archive and "Decision" in archive
    assert "DECISIONS_ARCHIVE.md" in index
    # The protected entry-1 is NOT in the archive.
    assert "Decision 1 " not in archive


# ---------------------------------------------------------------------------
# Test 9: empty LLM response preserves the existing file (session-end contract)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_llm_response_still_preserves_existing_file(tmp_path):
    """Seed LESSONS with 3 entries; LLM returns ''. The empty-value write guard
    skips the LESSONS write, so the seeded file is untouched (it is well under
    the hard budget, so the deterministic backstop does not demote it either)."""
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
        session, provider, ws,
        session_uuid="s_empty_preserves", bypass_idempotency=True,
        project_id="p1",
    )

    after = ws.read("lessons")
    assert after == before, (
        "LESSONS.md changed after empty-response session-end"
    )
