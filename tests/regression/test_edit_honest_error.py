# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: EditTool's not-found error must be honest and actionable.

The old message was ``Error: '<first 50 chars>...' not found`` — it gave the
model nothing to recover with (so it guessed again) and, because only 50 chars
of old_text were echoed, two genuinely different failed edits produced the
SAME string. The new message names the file, explains the likely cause
(edited-since-read / truncated-from-context), instructs a re-read, and echoes
the full old_text so distinct failures produce distinct strings.

Matching stays EXACT — no fuzzy matching, no byte-showing of the file.
"""

import pytest

from agent_os.agent.tools.edit import EditTool


@pytest.fixture
def workspace(tmp_path):
    f = tmp_path / "shot-list.md"
    f.write_text("## Section 4: Demo (wiki)\n\n> a callout line\n", encoding="utf-8")
    return str(tmp_path)


def test_not_found_names_file_and_instructs_reread(workspace):
    tool = EditTool(workspace=workspace)
    result = tool.execute(
        path="shot-list.md",
        old_text="## Section 4: Demo (wiki)\n\n| shot | type |",  # not present
        new_text="whatever",
    )
    content = result.content
    low = content.lower()
    assert "shot-list.md" in content
    assert "not found" in low
    assert "re-read" in low


def test_not_found_echoes_full_old_text_not_50_char_preview(workspace):
    tool = EditTool(workspace=workspace)
    long_old = "X" * 80  # >50 chars; old code truncated to 50
    result = tool.execute(path="shot-list.md", old_text=long_old, new_text="y")
    assert long_old in result.content  # full text present, not truncated


def test_two_different_failures_produce_different_errors(workspace):
    """old_texts sharing a 50-char prefix but differing after must yield
    different error strings (old 50-char preview collapsed them)."""
    tool = EditTool(workspace=workspace)
    a = "A" * 50 + "_alpha"
    b = "A" * 50 + "_beta"
    ra = tool.execute(path="shot-list.md", old_text=a, new_text="y")
    rb = tool.execute(path="shot-list.md", old_text=b, new_text="y")
    assert ra.content != rb.content


def test_exact_matching_unchanged_success(workspace):
    """A unique, exact old_text still replaces successfully (matching untouched)."""
    import json

    tool = EditTool(workspace=workspace)
    result = tool.execute(
        path="shot-list.md", old_text="a callout line", new_text="a NEW callout line",
    )
    data = json.loads(result.content)
    assert data["status"] == "success"
    assert data["replacements"] == 1


def test_multiple_matches_error_unchanged(workspace, tmp_path):
    """The multiple-matches branch is out of scope and must be unchanged."""
    (tmp_path / "dup.txt").write_text("foo foo", encoding="utf-8")
    tool = EditTool(workspace=workspace)
    result = tool.execute(path="dup.txt", old_text="foo", new_text="bar")
    assert "multiple matches" in result.content.lower()
