# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: a complete overflow lifecycle for all four Layer-1 files.

Incident (orbital-marketing, 2026-07-27). Overflow trimming of PROJECT_STATE
kept every ``[user]``-flagged bullet and deleted the unflagged prose around
them. What survived was::

    - [user] **选方案 A / B / C**（默认 A）？

and what was destroyed was the briefing that made it answerable — what A, B and
C were, the 19-24h cost split, and why B was advised against. The question
outlived its own answer, and PROJECT_STATE has no archive, so it was gone.

Two defects, one lifecycle:

  1. The protected unit was the flagged LINE. Supporting prose is by definition
     unflagged, and here it sat in the same ``##`` section separated by blank
     lines — so a block-level rule would not have saved it either. The unit is
     now the SECTION: if any line in a ``##`` section is flagged, the whole
     section survives together, in the live file.

  2. Volatile files DELETED on overflow (no ``ARCHIVE_OF`` entry). Now
     PROJECT_STATE demotes to a Layer-2 ``PROJECT_STATE_ARCHIVE.md`` — never
     injected, never budgeted, read on demand — so nothing is destroyed.

Plus: INDEX carries the pointers that make every archive discoverable, and its
``LESSONS_ARCHIVE`` pointer was the LAST line of the file — first to go under a
tail-dropping trim, which would strand the archive on disk. Pointer lines are
now pinned.
"""

from __future__ import annotations

import pytest

from agent_os.agent import memory_entries as _mem
from agent_os.agent.workspace_files import FILE_NAMES, WorkspaceFileManager


# The shape that lost the data: a heading, prose and key facts, a blank line,
# then the flagged questions those facts exist to support.
BRIEFED_SECTION = """\
# Project State

## Current Status
Shipped v0.8.0. Nothing pending here.
Filler line to make this section droppable and worth dropping.
More filler so the section has real weight in the budget, well beyond the
slack the test allows, so that dropping it is the only way to fit. Padding
padding padding padding padding padding padding padding padding padding.
Padding padding padding padding padding padding padding padding padding.
DROPPABLE-TAIL-SENTINEL at the end of the unflagged section.

## Hero GIF discussion (2026-07-27)
claude-code recommends option A: relay layout, 0-9 lines.

**Key facts:**
- all three options need no on-camera presence
- cost: hero 7.5-9.5h, video 11-14h, total 19-24h
- option B has a competitor-bashing risk and is not reproducible
- option C is downgraded to act three, not the hero

**Decisions needed:**

- [user] **pick option A / B / C** (default A)?
  <!--mem id:4148d8 created:2026-07-27 touched:2026-07-27-->
- [user] **video length 5 / 8 / 12 min** (default 8)?
  <!--mem id:44191b created:2026-07-27 touched:2026-07-27-->
"""


def _budget_forcing_trim(content: str) -> int:
    """A budget the PROTECTED content fits inside, but the whole file does not.

    Sizing matters: below the protected content's own size, ``trim_volatile``
    hits its escape hatch and returns the file untouched — which would make
    these tests pass without trimming anything.
    """
    lines = content.split("\n")
    protected = _mem._flagged_sections(content)
    protected_text = "\n".join(lines[i] for i in sorted(protected))
    budget = int(_mem.est_tokens(protected_text)) + 5
    assert budget < int(_mem.est_tokens(content)), "budget must force a trim"
    return budget


# ---------------------------------------------------------------------------
# 1. A [user] question keeps its briefing, in the live file
# ---------------------------------------------------------------------------

def test_flagged_section_keeps_its_supporting_prose():
    """The exact regression: the question must not outlive its own answer."""
    out = _mem.trim_volatile(BRIEFED_SECTION, _budget_forcing_trim(BRIEFED_SECTION))
    assert out != BRIEFED_SECTION, "guard: this must actually have trimmed"
    assert "pick option A / B / C" in out                      # the question
    assert "option B has a competitor-bashing risk" in out     # why not B
    assert "cost: hero 7.5-9.5h" in out                        # what it costs
    assert "claude-code recommends option A" in out            # the recommendation


def test_unflagged_section_is_still_droppable():
    """Protection must be earned by a flag, or trimming can never reclaim
    anything and the budget stops meaning something. Dropping is tail-first,
    so the end of the unflagged section is what goes first."""
    out = _mem.trim_volatile(BRIEFED_SECTION, _budget_forcing_trim(BRIEFED_SECTION))
    assert "DROPPABLE-TAIL-SENTINEL" not in out
    assert len(out) < len(BRIEFED_SECTION)


def test_blank_lines_do_not_split_a_section():
    """The briefing sat two blank lines above its questions — a blank-line
    block rule would have dropped it. The unit is the ## section."""
    lines = BRIEFED_SECTION.split("\n")
    hero = next(i for i, l in enumerate(lines) if l.startswith("## Hero GIF"))
    facts = next(i for i, l in enumerate(lines) if "competitor-bashing" in l)
    flagged = next(i for i, l in enumerate(lines) if "pick option A" in l)
    assert "" in lines[facts:flagged]          # guard: blank line really is between
    assert hero in _mem._flagged_sections(BRIEFED_SECTION)
    assert facts in _mem._flagged_sections(BRIEFED_SECTION)


def test_flagged_only_content_is_returned_unchanged():
    """Existing escape hatch: when protected content alone busts the budget,
    nothing is silently deleted — the soft flag keeps signalling instead."""
    content = "## S\n- [user] a\n- [user] b\n"
    assert _mem.trim_volatile(content, 1) == content


def test_file_with_no_flags_still_head_trims():
    plain = "# X\n" + "".join(f"- line {i}\n" for i in range(200))
    out = _mem.trim_volatile(plain, 50)
    assert len(out) < len(plain)


# ---------------------------------------------------------------------------
# 2. PROJECT_STATE has a Layer-2 archive — overflow MOVES, never deletes
# ---------------------------------------------------------------------------

def test_state_has_an_archive_destination():
    assert _mem.ARCHIVE_OF["state"] == "state_archive"
    assert FILE_NAMES["state_archive"] == "PROJECT_STATE_ARCHIVE.md"


def test_archives_are_layer2_never_injected():
    """Layer 2 = read-on-demand. An archive with a budget would be injected
    every turn, which defeats the point of demoting to it."""
    for archive_key in _mem.ARCHIVE_OF.values():
        assert archive_key not in _mem.FILE_BUDGETS


def test_state_overflow_moves_prose_to_archive_instead_of_deleting(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    # Volatile trimming keeps the HEAD (current status) and drops the tail, so
    # the sentinel goes last — that is the content actually at risk.
    doomed = "UNIQUE-SENTINEL-PROSE-THAT-MUST-SURVIVE"
    ws.write("state", "# S\n\n## Current\n"
             + "".join(f"filler {i}\n" for i in range(4000))
             + "\n## Old\n" + doomed + "\n")

    from agent_os.agent import workspace_files as wsf
    wsf._apply_hard_caps(ws)

    assert doomed not in (ws.read("state") or ""), "should have been trimmed out"
    assert doomed in (ws.read("state_archive") or ""), "must be MOVED, not deleted"


def test_state_archive_gets_an_index_pointer(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("state", "# S\n\n## Old\n" + "".join(f"filler {i}\n" for i in range(4000)))
    from agent_os.agent import workspace_files as wsf
    wsf._apply_hard_caps(ws)
    assert "PROJECT_STATE_ARCHIVE.md" in (ws.read("index") or "")


# ---------------------------------------------------------------------------
# 3. INDEX pointers are pinned — an archive can never be orphaned
# ---------------------------------------------------------------------------

def test_archive_pointers_survive_a_tail_dropping_trim():
    """On the real project the LESSONS_ARCHIVE pointer was line 81 of 81 —
    the first thing a tail-drop destroys. Losing it leaves the archive on
    disk and invisible to the agent."""
    index = (
        "# INDEX\n"
        + "".join(f"- orbital/f{i}.md — filler entry {i}\n" for i in range(300))
        + "- DECISIONS_ARCHIVE.md — superseded decisions (read on demand).\n"
        + "- LESSONS_ARCHIVE.md — older entries demoted from the live file.\n"
    )
    out = _mem.trim_volatile(index, 100)
    assert len(out) < len(index)                     # it did trim
    assert "DECISIONS_ARCHIVE.md" in out
    assert "LESSONS_ARCHIVE.md" in out


def test_pointer_pinning_does_not_protect_ordinary_lines():
    index = "# INDEX\n" + "".join(f"- orbital/f{i}.md — filler {i}\n" for i in range(300))
    out = _mem.trim_volatile(index, 50)
    assert "orbital/f299.md" not in out


# ---------------------------------------------------------------------------
# 4. Every Layer-1 file ends up with a defined overflow destination
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["state", "decisions", "lessons", "index"])
def test_every_layer1_file_has_a_defined_overflow_behaviour(key):
    """INDEX is the one file with no archive — it is regenerable navigation,
    so stale pointers are noise rather than history. Every other file moves
    its overflow somewhere readable."""
    assert key in _mem.FILE_BUDGETS
    assert _mem.consolidation_target(key) < _mem.FILE_BUDGETS[key]["soft"]
    if key == "index":
        assert key not in _mem.ARCHIVE_OF
    else:
        assert _mem.ARCHIVE_OF[key] in FILE_NAMES
