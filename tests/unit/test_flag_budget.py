# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Flag-aware budget/injection/overflow tests (spec §4.1, Task 3).

Three resolutions under test:
1. ``<!--mem ...-->`` comments (PROJECT_STATE's per-bullet machine metadata)
   are excluded from budget counting for the state file, and stripped
   entirely from ``inject_view`` output — the agent never sees them.
2. Volatile overflow (``trim_volatile``) never deletes a ``[user]``-flagged
   entry: unflagged prose is trimmed first; if flagged entries alone still
   exceed the hard cap, the content is left completely untouched and the
   existing soft-cap hygiene nudge keeps signalling.
3. PROJECT_STATE soft cap is 1800 tokens (was 1500).

DECISIONS/LESSONS carry their own, unrelated ``<!--mem id:...-->`` stamp
(``memory_entries.stamp``) that is always injected — none of the above
changes may touch that behavior, so a couple of pass-through regression
checks are included here too.
"""

from __future__ import annotations

from agent_os.agent import memory_entries as M
from agent_os.agent import user_flags


# ---------------------------------------------------------------------------
# Resolution 3: soft-cap bump.
# ---------------------------------------------------------------------------


def test_state_soft_cap_is_1800():
    assert M.FILE_BUDGETS["state"]["soft"] == 1800
    assert M.FILE_BUDGETS["state"]["hard"] == 2000  # unchanged


# ---------------------------------------------------------------------------
# Resolution 1a: soft-budget check excludes mem-comments (state only).
# ---------------------------------------------------------------------------


def _state_entry(sentence: str, *, evidence_filler: str = "") -> str:
    comment = (
        f'<!--mem id:abc123 from:sess1 evidence:"{evidence_filler}" '
        "confidence:unconfirmed created:2026-07-01 touched:2026-07-01-->"
    )
    return f"- [user] {sentence}\n  {comment}\n"


def test_soft_budget_excludes_mem_comments_for_state(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "state", {"soft": 100, "hard": 300})
    # Visible bracket tag + sentence alone is well under the 100-tok soft
    # cap; the mem-comment's evidence filler alone pushes RAW size past it.
    content = _state_entry("Send the report to the client.", evidence_filler="y" * 400)

    assert M.est_tokens(content) > 100, "fixture must exceed soft cap BEFORE stripping"
    assert len(user_flags.strip_mem_comments(content)) / 4 <= 100, (
        "fixture must fit under soft cap AFTER stripping"
    )

    assert M.soft_flag(content, "state") is None


def test_soft_budget_still_trips_once_visible_text_itself_is_big(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "state", {"soft": 20, "hard": 300})
    content = _state_entry("x" * 400)  # visible sentence alone is over soft
    flag = M.soft_flag(content, "state")
    assert flag is not None
    assert "checkpoint_state" in flag


def test_budget_text_passes_through_decisions_and_lessons_unchanged():
    # DECISIONS/LESSONS have their OWN "<!--mem id:...-->" stamp (a different
    # convention, always injected) — the state-only exclusion must not touch it.
    entry = (
        "## 2026-01-01: Some decision "
        "<!--mem id:d1 created:2026-01-01 touched:2026-01-01-->\n"
        "**Chose:** x\n**Reason:** y\n**Rejected:** z\n\n"
    )
    assert M._budget_text(entry, "decisions") == entry
    assert M._budget_text(entry, "lessons") == entry


# ---------------------------------------------------------------------------
# Resolution 1b: inject_view strips mem-comments for state.
# ---------------------------------------------------------------------------


def test_inject_view_state_strips_mem_comments_keeps_sentence():
    sentence = "Send the report to the client."
    content = (
        "<!--format PROJECT_STATE ...-->\n"
        f"- [user due:2026-07-28] {sentence}\n"
        '  <!--mem id:x7f3a2 from:sess1 evidence:"they said so" '
        "confidence:unconfirmed created:2026-07-01 touched:2026-07-01-->\n"
    )
    view = M.inject_view(content, "state", M.FILE_BUDGETS["state"]["hard"])
    assert view is not None
    assert "<!--mem" not in view
    assert "[user due:2026-07-28]" in view
    assert sentence in view


def test_inject_view_decisions_still_shows_its_own_mem_stamp():
    # Regression: the state-only strip must not leak into decisions/lessons,
    # whose "<!--mem id:...-->" IS the always-injected metadata contract.
    content = (
        "## 2026-01-01: Some decision "
        "<!--mem id:d1 created:2026-01-01 touched:2026-01-01-->\n"
        "**Chose:** x\n**Reason:** y\n**Rejected:** z\n\n"
        "## 2026-01-02: Another decision "
        "<!--mem id:d2 created:2026-01-02 touched:2026-01-02-->\n"
        "**Chose:** a\n**Reason:** b\n**Rejected:** c\n\n"
    )
    view = M.inject_view(content, "decisions", M.FILE_BUDGETS["decisions"]["hard"])
    assert "<!--mem id:d1" in view
    assert "<!--mem id:d2" in view


# ---------------------------------------------------------------------------
# Resolution 2: trim_volatile never deletes a flagged entry.
# ---------------------------------------------------------------------------


def _filler_lines(n: int) -> str:
    return "\n".join(f"- filler line {i} " + "z" * 20 for i in range(1, n + 1))


def test_trim_volatile_protects_flagged_entry_old_behavior_would_have_dropped():
    hard = 50  # tokens -> 200-char budget
    flagged_text = "Send the report to the client."
    comment = "<!--mem id:abc123 created:2026-07-01 touched:2026-07-01-->"
    flagged_block = f"- [user] {flagged_text}\n  {comment}"
    content = _filler_lines(7) + "\n" + flagged_block + "\n"

    assert M.est_tokens(content) > hard, "fixture must exceed the hard budget"

    # Prove the bug this change fixes: the legacy head-keep/tail-drop
    # algorithm (still used for non-flagged content) would cut the tail —
    # exactly where the flagged entry sits — right off.
    legacy = M._head_within(content, hard)
    assert flagged_text not in legacy

    result = M.trim_volatile(content, hard)
    assert flagged_text in result
    assert "id:abc123" in result  # its mem-comment survives too
    # Unflagged filler was trimmed first — not every filler line survives.
    surviving_filler = sum(1 for i in range(1, 8) if f"filler line {i}" in result)
    assert surviving_filler < 7
    # The kept body (i.e. everything but the informational trim note, which
    # — like the pre-existing _head_within note — is not itself budgeted)
    # fits under the comment-stripped budget.
    body = result.rsplit("\n[... older content trimmed", 1)[0]
    assert len(user_flags.strip_mem_comments(body)) <= hard * 4


def test_trim_volatile_flagged_only_overflow_left_completely_untouched():
    hard = 10  # tokens -> 40-char budget; too small for the bracket tag alone
    text = "This obligation absolutely cannot be silently dropped by any means."
    comment = "<!--mem id:abc999 created:2026-07-01 touched:2026-07-01-->"
    content = f"- [user] {text}\n  {comment}\n"

    assert len(user_flags.strip_mem_comments(content)) > hard * 4, (
        "fixture must exceed the budget even after excluding the comment"
    )

    result = M.trim_volatile(content, hard)
    assert result == content


def test_trim_volatile_untouched_overflow_still_signals_hygiene(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "state", {"soft": 5, "hard": 10})
    text = "This obligation absolutely cannot be silently dropped by any means."
    comment = "<!--mem id:abc999 created:2026-07-01 touched:2026-07-01-->"
    content = f"- [user] {text}\n  {comment}\n"

    result = M.trim_volatile(content, M.FILE_BUDGETS["state"]["hard"])
    assert result == content

    flag = M.soft_flag(content, "state")
    assert flag is not None
    assert "checkpoint_state" in flag


def test_trim_volatile_no_flagged_entries_falls_back_to_legacy_head_trim():
    # INDEX.md (or a state file that hasn't adopted the grammar at all) must
    # trim byte-identically to before this change.
    hard = 20
    content = "\n".join(f"- path/to/file{i}.py — does thing {i}" for i in range(1, 15))
    assert M.trim_volatile(content, hard) == M._head_within(content, hard)


def test_trim_volatile_fits_already_returns_unchanged():
    content = "- [user] short\n  <!--mem id:z created:2026-07-01 touched:2026-07-01-->\n"
    result = M.trim_volatile(content, 1000)
    assert result == content
