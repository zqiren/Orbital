# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 188 §2.5.1 — the first bubble of a routed task parses the framing.

``tests/fixtures/source_task_samples.json`` is written once and committed.
Each sample's ``text`` must equal ``framing.task_text(**inputs)`` exactly, so
a change to the framing fails here, and the same ``text`` is what
``web/src/lib/source-task-parsing.test.ts`` parses. Update both on purpose,
never by regenerating from a test run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_os.sources import framing

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "source_task_samples.json"
SAMPLES = json.loads(FIXTURE.read_text(encoding="utf-8"))["samples"]
FRAMED = [s for s in SAMPLES if "inputs" in s]


def test_fixture_covers_the_spec_cases():
    names = {s["name"] for s in SAMPLES}
    for need in ("github_label_long_zh_title", "github_assigned_with_link", "feishu_dm_no_sender",
                 "feishu_group_at_with_sender", "group_name_with_separator", "source_named_like_kind_word",
                 "calendar_before_zh", "website_down", "command_source", "three_merged_parts",
                 "instruction_present", "body_with_defused_tag", "plain_text_starting_like_header"):
        assert need in names
    # Plain user text never parses.
    assert all(s["parsed"] is None for s in SAMPLES if "inputs" not in s)


@pytest.mark.parametrize("sample", FRAMED, ids=[s["name"] for s in FRAMED])
def test_task_text_matches_committed_sample(sample):
    assert framing.task_text(**sample["inputs"]) == sample["text"]


def test_kind_words_match_the_parser_table():
    """The web parser maps these exact English words back to a kind; a new
    or reworded kind must reach source-task-parsing.ts too."""
    src = (Path(__file__).resolve().parents[2] / "web" / "src" / "lib" / "source-task-parsing.ts").read_text(encoding="utf-8")
    for word in framing.KIND_WORDS.values():
        assert f"'{word}'" in src, word
    assert framing.QUOTE_NOTE in src
