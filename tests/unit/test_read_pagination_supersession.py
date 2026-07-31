# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""The seam between ReadTool pagination and tool-result supersession.

These two landed together and interact: `read` gained `offset`/`limit`, and
supersession stubs an earlier result when a later one targets the same thing.
Keyed on the path alone, page 2 would supersede page 1 — so an agent walking a
large file would watch each page it had already collected get stubbed out from
under it, which is exactly what pagination exists to make possible.

A paged read therefore targets a RANGE, not a file. Same range → supersedes;
different range → both survive.
"""

import json

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import (
    _supersession_target,
    archive_and_supersede_tool_results,
)


@pytest.fixture
def session(tmp_path):
    return Session.new("read-pagination-supersession", str(tmp_path))


def _fetch(session, call_id, arguments, content):
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": "read", "arguments": json.dumps(arguments)},
        }],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


def _live(session):
    return {
        m["tool_call_id"]: m
        for m in session.get_messages()
        if m.get("role") == "tool" and not m.get("_stubbed")
    }


# ---------------------------------------------------------------------------
# End to end: paging a file must not eat its own pages
# ---------------------------------------------------------------------------

def test_paging_one_file_keeps_every_page(session):
    """Walking a file with offset/limit leaves all pages live in history."""
    for i, call_id in enumerate(("tc_p1", "tc_p2", "tc_p3")):
        _fetch(
            session, call_id,
            {"path": "big.md", "offset": i * 100, "limit": 100},
            f"PAGE_{i}_MARKER" + "x" * 2_000,
        )
    archive_and_supersede_tool_results(session, iteration=1)

    live = _live(session)
    assert set(live) == {"tc_p1", "tc_p2", "tc_p3"}
    for i, call_id in enumerate(("tc_p1", "tc_p2", "tc_p3")):
        assert f"PAGE_{i}_MARKER" in live[call_id]["content"]


def test_same_range_reread_still_supersedes(session):
    """Re-reading the SAME range is a genuine re-fetch — prior copy goes."""
    args = {"path": "big.md", "offset": 100, "limit": 100}
    _fetch(session, "tc_a", args, "STALE_COPY" + "x" * 2_000)
    _fetch(session, "tc_b", args, "FRESH_COPY" + "y" * 2_000)
    archive_and_supersede_tool_results(session, iteration=1)

    live = _live(session)
    assert set(live) == {"tc_b"}
    assert "FRESH_COPY" in live["tc_b"]["content"]


def test_whole_file_reread_still_supersedes(session):
    """The unpaginated path is unchanged: same file twice, newest wins."""
    _fetch(session, "tc_a", {"path": "notes.md"}, "OLD" + "x" * 2_000)
    _fetch(session, "tc_b", {"path": "notes.md"}, "NEW" + "y" * 2_000)
    archive_and_supersede_tool_results(session, iteration=1)

    assert set(_live(session)) == {"tc_b"}


def test_page_does_not_supersede_whole_file_read(session):
    """A ranged read and a whole-file read are different targets."""
    _fetch(session, "tc_whole", {"path": "big.md"}, "WHOLE" + "x" * 2_000)
    _fetch(session, "tc_page",
           {"path": "big.md", "offset": 50, "limit": 10}, "PAGE" + "y" * 2_000)
    archive_and_supersede_tool_results(session, iteration=1)

    assert set(_live(session)) == {"tc_whole", "tc_page"}


# ---------------------------------------------------------------------------
# Range normalization must match ReadTool._window, or equal ranges look unequal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    # models send "3" as often as 3
    ({"offset": 0, "limit": 100}, {"offset": "0", "limit": "100"}),
    # a negative offset behaves as 0 in ReadTool._window
    ({"offset": -5}, {"offset": 0}),
    # a non-positive or junk limit means "to EOF", same as omitting it
    ({}, {"limit": 0}),
    ({}, {"limit": -1}),
    ({}, {"limit": "not-a-number"}),
    # omitted offset is offset 0
    ({}, {"offset": 0}),
])
def test_equivalent_ranges_share_a_target(a, b):
    left = _supersession_target("read", {"path": "f.md", **a})
    right = _supersession_target("read", {"path": "f.md", **b})
    assert left == right


@pytest.mark.parametrize("a,b", [
    ({"offset": 0, "limit": 100}, {"offset": 100, "limit": 100}),
    ({"offset": 0, "limit": 100}, {"offset": 0, "limit": 200}),
    ({"offset": 0}, {"offset": 0, "limit": 100}),
])
def test_distinct_ranges_do_not_share_a_target(a, b):
    left = _supersession_target("read", {"path": "f.md", **a})
    right = _supersession_target("read", {"path": "f.md", **b})
    assert left != right


def test_range_is_scoped_to_the_path():
    """Same range on different files must not collide."""
    args = {"offset": 0, "limit": 100}
    assert (_supersession_target("read", {"path": "a.md", **args})
            != _supersession_target("read", {"path": "b.md", **args}))


@pytest.mark.parametrize("tool,args", [
    ("browser", {"url": "https://example.com"}),
    ("shell", {"command": "ls -la"}),
])
def test_non_read_tools_ignore_range_args(tool, args):
    """offset/limit are a read concept — they must not alter other targets."""
    assert (_supersession_target(tool, args)
            == _supersession_target(tool, {**args, "offset": 40, "limit": 5}))
