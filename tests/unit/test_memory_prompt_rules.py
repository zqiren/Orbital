# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runtime-agent memory rules (spec 089 §4.5).

The Antler failure was not a memory failure: the rule naming the source of
truth was in the prompt verbatim, and the agent drafted from a derivative it
described as aligned and cited a file it never opened. These rules say what to
do with a pointer — to a source of truth, or to an archived entry.
"""

from agent_os.agent.prompt_builder import (
    _TOOL_DESCRIPTIONS as TOOL_DESCRIPTIONS, Autonomy, PromptBuilder, PromptContext,
)
from agent_os.agent.tools.checkpoint_state import CheckpointStateTool


def _ctx(**overrides) -> PromptContext:
    base = dict(
        workspace="/ws", model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "grep"], os_type="macos",
        datetime_now="2026-09-19T12:00",
    )
    base.update(overrides)
    return PromptContext(**base)


def _memory(**overrides) -> str:
    return PromptBuilder()._memory(_ctx(**overrides))


def test_follow_the_pointer_rule():
    text = _memory()
    assert "source of truth" in text
    assert "open that file before producing" in text
    assert "does not substitute" in text


def test_never_cite_an_unread_source():
    assert "Never name a file as your source unless you read it in this session" in _memory()


def test_archive_recall_by_id_then_topic_never_whole():
    text = _memory()
    assert "[archived" in text and "id:X" in text
    assert "grep" in text and "offset/limit" in text
    assert "Never read an archive file whole" in text
    assert "PROJECT_STATE_ARCHIVE.md" in text


def test_memory_upkeep_wording_matches_the_editor():
    text = _memory()
    assert "merges duplicates at session end" not in text
    assert "automatically" in text


def test_scratch_projects_get_none_of_it():
    text = _memory(is_scratch=True)
    assert "Never read an archive file whole" not in text
    assert "source of truth" not in text


def test_checkpoint_state_descriptions_say_it_is_automatic_and_records_nothing():
    for desc in (TOOL_DESCRIPTIONS["checkpoint_state"], CheckpointStateTool(lambda: None).description):
        assert "automatically" in desc
        assert "never records new facts" in desc.lower() or "does not record new facts" in desc.lower()


def test_status_line_names_the_editor():
    line = PromptBuilder()._state_checkpoint_status(_ctx(
        refresh_in_flight=True, refresh_in_flight_since_turn=3))
    assert "memory editor" in line.lower()
