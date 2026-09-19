# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Budget/injection tests for PROJECT_STATE's machine metadata (spec §4.1).

1. ``<!--mem ...-->`` comments (PROJECT_STATE's per-bullet machine metadata)
   are excluded from budget counting for the state file, and stripped
   entirely from ``inject_view`` output — the agent never sees them.
2. Spec 089 retired flag-based protection: overflow is age-based now (the
   floor and the injected view both leave the OLDEST lines first — see
   test_memory_floor.py). What stays here is that a ``[user]`` tag earns no
   special treatment and a lone oversized entry is never cut.
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
    assert "automatically" in flag


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
# Resolution 2 (spec 089): no flag protection; age decides.
# ---------------------------------------------------------------------------


def _filler_lines(n: int, created: str) -> str:
    return "\n".join(
        f"- filler line {i} " + "z" * 20
        + f"\n  <!--mem id:f{i:05d} created:{created} touched:{created}-->"
        for i in range(1, n + 1)
    )


def test_flagged_tag_earns_no_protection_in_the_view():
    """An OLD flagged line leaves the over-budget view before newer plain
    lines — the [user] tag no longer means anything to the budget."""
    hard = 50
    content = (
        "- [user] Old flagged question from July.\n"
        "  <!--mem id:old001 created:2026-07-01 touched:2026-07-01-->\n"
        + _filler_lines(7, "2026-09-10") + "\n"
    )
    view = M.inject_view(content, "state", hard)
    assert "Old flagged question" not in view
    assert "filler line 7" in view
    assert "<!--mem" not in view


def test_trim_volatile_ignores_flags_now():
    """INDEX-style trim: a [user] tag is just text; only *_ARCHIVE.md lines
    are pinned."""
    hard = 20
    content = "\n".join(f"- path/to/file{i}.py — does thing {i}" for i in range(1, 15))
    content += "\n- [user] a flagged line at the tail"
    out = M.trim_volatile(content, hard)
    assert "a flagged line at the tail" not in out


def test_trim_volatile_fits_already_returns_unchanged():
    content = "- [user] short\n  <!--mem id:z created:2026-07-01 touched:2026-07-01-->\n"
    result = M.trim_volatile(content, 1000)
    assert result == content


def test_state_floor_leaves_a_lone_oversized_recent_entry_untouched():
    text = "This obligation absolutely cannot be silently dropped by any means."
    content = f"- [user] {text}\n  <!--mem id:abc999 created:2026-09-18 touched:2026-09-18-->\n"
    r = M.floor_state(content, 5, "2026-09-19")
    assert r.content is content
    assert r.over_by > 0


def test_inject_view_state_everything_fits_is_just_comment_stripped():
    content = "- [user] short\n  <!--mem id:z created:2026-07-01 touched:2026-07-01-->\n"
    assert M.inject_view(content, "state", 1000) == "- [user] short\n"


def test_inject_view_state_unstamped_overflow_head_trims_deterministically():
    hard = 10
    content = (
        "- First obligation that is quite long indeed.\n"
        "- Second obligation that is also rather long.\n"
    )
    view = M.inject_view(content, "state", hard)
    assert view is not None
    assert M.inject_view(content, "state", hard) == view
    assert len(view) <= hard * 4 + 120
