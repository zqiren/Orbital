# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Supersession: a re-fetch of the same target stubs the earlier copy.

This is the ONLY case in which tool results are rewritten in history. It is not
an eviction policy — no budget, no water marks, no ordering rules. It fires on
a factual condition: this target was fetched again, so the prior copy is
superseded by definition (identical bytes → pure duplication; different bytes →
the prior copy describes a state that no longer exists).

The newest copy is always the survivor, which is also why supersession cannot
strand the agent: the surviving copy was just appended.
"""

import json
import os

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import (
    SIZE_THRESHOLD,
    archive_and_supersede_tool_results,
)


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path)


@pytest.fixture
def session(workspace):
    return Session.new("supersession-test", workspace)


def _fetch(session, call_id, tool_name, arguments, content):
    """Append an assistant tool_call message plus its tool result."""
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments),
            },
        }],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


def _tool_msgs(session):
    return {
        m["tool_call_id"]: m
        for m in session.get_messages() if m.get("role") == "tool"
    }


def _archive_dir(workspace):
    return os.path.join(
        workspace, "orbital", "tool-results", "supersession-test",
    )


# ---------------------------------------------------------------------------
# The supersession condition
# ---------------------------------------------------------------------------

def test_same_target_refetch_stubs_prior_keeps_newest(session):
    """Two browser fetches of one URL: the first is stubbed, the second lives."""
    page_v1 = "OLD_PAGE_STATE_" + "A" * 5_000
    page_v2 = "NEW_PAGE_STATE_" + "B" * 5_000
    _fetch(session, "tc_1", "browser",
           {"action": "snapshot", "url": "https://example.com/dash"}, page_v1)
    _fetch(session, "tc_2", "browser",
           {"action": "snapshot", "url": "https://example.com/dash"}, page_v2)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert msgs["tc_1"].get("_stubbed") is True
    assert "OLD_PAGE_STATE_" not in msgs["tc_1"]["content"]
    # The newest copy is untouched — full content, no stub marker.
    assert not msgs["tc_2"].get("_stubbed")
    assert msgs["tc_2"]["content"] == page_v2


def test_same_path_reread_stubs_prior_keeps_newest(session):
    """read is superseded on path, exactly like browser on url."""
    _fetch(session, "tc_r1", "read", {"path": "notes.md"}, "FIRST_" + "X" * 3_000)
    _fetch(session, "tc_r2", "read", {"path": "notes.md"}, "SECOND_" + "Y" * 3_000)

    archive_and_supersede_tool_results(session, iteration=2)

    msgs = _tool_msgs(session)
    assert msgs["tc_r1"].get("_stubbed") is True
    assert not msgs["tc_r2"].get("_stubbed")
    assert msgs["tc_r2"]["content"].startswith("SECOND_")


def test_different_targets_both_kept(session):
    """Different targets are not each other's supersession — both stay live."""
    _fetch(session, "tc_a", "read", {"path": "alpha.md"}, "A" * 3_000)
    _fetch(session, "tc_b", "read", {"path": "beta.md"}, "B" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert not msgs["tc_a"].get("_stubbed")
    assert not msgs["tc_b"].get("_stubbed")
    assert msgs["tc_a"]["content"] == "A" * 3_000
    assert msgs["tc_b"]["content"] == "B" * 3_000


def test_same_path_different_tool_is_not_supersession(session):
    """The target key includes the tool, so read and read_file never collide."""
    _fetch(session, "tc_read", "read", {"path": "data.csv"}, "R" * 3_000)
    _fetch(session, "tc_read_file", "read_file", {"path": "data.csv"}, "S" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert not msgs["tc_read"].get("_stubbed")
    assert not msgs["tc_read_file"].get("_stubbed")


def test_single_fetch_is_never_stubbed(session):
    """A large result with no re-fetch stays in history verbatim."""
    content = "SOLE_COPY_" + "Z" * 20_000
    _fetch(session, "tc_only", "browser",
           {"action": "snapshot", "url": "https://example.com/only"}, content)

    archive_and_supersede_tool_results(session, iteration=1)

    msg = _tool_msgs(session)["tc_only"]
    assert not msg.get("_stubbed")
    assert msg["content"] == content


def test_three_fetches_leave_exactly_one_live_copy(session):
    """N fetches of one target collapse to the newest; N-1 become stubs."""
    for i in range(3):
        _fetch(session, f"tc_v{i}", "browser",
               {"action": "snapshot", "url": "https://example.com/live"},
               f"VERSION_{i}_" + "Q" * 4_000)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    live = [m for m in msgs.values() if not m.get("_stubbed")]
    assert len(live) == 1
    assert live[0]["tool_call_id"] == "tc_v2"
    assert live[0]["content"].startswith("VERSION_2_")
    assert msgs["tc_v0"].get("_stubbed") is True
    assert msgs["tc_v1"].get("_stubbed") is True


def test_supersession_across_separate_invocations(session):
    """The re-fetch may land turns later — the older copy is stubbed then."""
    _fetch(session, "tc_t1", "read", {"path": "spec.md"}, "OLD_" + "M" * 3_000)
    archive_and_supersede_tool_results(session, iteration=1)
    assert not _tool_msgs(session)["tc_t1"].get("_stubbed")

    _fetch(session, "tc_t2", "read", {"path": "spec.md"}, "NEW_" + "N" * 3_000)
    archive_and_supersede_tool_results(session, iteration=7)

    msgs = _tool_msgs(session)
    assert msgs["tc_t1"].get("_stubbed") is True
    assert not msgs["tc_t2"].get("_stubbed")


def test_already_stubbed_copy_is_not_reprocessed(session):
    """A superseded stub is stable across later invocations."""
    _fetch(session, "tc_s1", "read", {"path": "a.md"}, "1" * 3_000)
    _fetch(session, "tc_s2", "read", {"path": "a.md"}, "2" * 3_000)
    archive_and_supersede_tool_results(session, iteration=1)
    first_stub = _tool_msgs(session)["tc_s1"]["content"]

    archive_and_supersede_tool_results(session, iteration=2)

    msgs = _tool_msgs(session)
    assert msgs["tc_s1"]["content"] == first_stub
    # The survivor is still the survivor — it is never stubbed by its own stub.
    assert not msgs["tc_s2"].get("_stubbed")


# ---------------------------------------------------------------------------
# Boundaries: what supersession must NOT touch
# ---------------------------------------------------------------------------

def test_sub_threshold_results_untouched(session):
    """Repeated small results are below the threshold — nothing is rewritten.

    Uses `read`, which IS supersedable, so size is the only thing keeping these
    two results alive.
    """
    small = "line one\nline two\n"
    assert len(small) < SIZE_THRESHOLD
    _fetch(session, "tc_p1", "read", {"path": "tiny.md"}, small)
    _fetch(session, "tc_p2", "read", {"path": "tiny.md"}, small)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert not msgs["tc_p1"].get("_stubbed")
    assert not msgs["tc_p2"].get("_stubbed")
    assert msgs["tc_p1"]["content"] == small


def test_multimodal_results_untouched(session, workspace):
    """List (multimodal) content is skipped entirely — not stubbed, not archived."""
    blocks = [{"type": "text", "text": "T" * 4_000}]
    _fetch(session, "tc_m1", "browser",
           {"action": "screenshot", "url": "https://example.com/shot"}, "P" * 3_000)
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": "tc_m2", "type": "function",
            "function": {
                "name": "browser",
                "arguments": json.dumps(
                    {"action": "screenshot", "url": "https://example.com/shot"},
                ),
            },
        }],
        "source": "management",
    })
    session.append_tool_result("tc_m2", blocks)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    # The multimodal result is invisible to supersession, so it neither gets
    # stubbed nor supersedes the earlier text result.
    assert msgs["tc_m2"]["content"] == blocks
    assert not msgs["tc_m2"].get("_stubbed")
    assert not msgs["tc_m1"].get("_stubbed")
    assert not os.path.exists(
        os.path.join(_archive_dir(workspace), "turn_1_call_tc_m2.json"),
    )


def test_untargeted_tool_never_supersedes(session):
    """A tool whose key_param is not a target identity is never superseded.

    `grep` falls through to the generic "first argument value" key — the same
    pattern searched in two different directories is not the same target, so
    supersession must not fire on it.
    """
    _fetch(session, "tc_g1", "grep",
           {"pattern": "TODO", "path": "src/"}, "G" * 3_000)
    _fetch(session, "tc_g2", "grep",
           {"pattern": "TODO", "path": "tests/"}, "H" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert not msgs["tc_g1"].get("_stubbed")
    assert not msgs["tc_g2"].get("_stubbed")


def test_shell_is_never_superseded(session):
    """Shell output is a measurement in time, not a view of current state.

    Comparing two measurements is a legitimate workload — `git diff` before and
    after an edit, a test run before and after a fix. Superseding the earlier
    one deletes exactly what the agent was comparing against, which is the same
    failure mode as the blanket stubbing this module exists to remove.
    (Paged reads are covered in test_read_pagination_supersession.py.)
    """
    _fetch(session, "tc_sh1", "shell", {"command": "git diff"},
           "BEFORE_" + "1" * 3_000)
    _fetch(session, "tc_sh2", "shell", {"command": "git diff"},
           "AFTER_" + "2" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert not msgs["tc_sh1"].get("_stubbed")
    assert not msgs["tc_sh2"].get("_stubbed")
    assert msgs["tc_sh1"]["content"].startswith("BEFORE_")


def test_long_url_prefix_collision_does_not_supersede(session):
    """Supersession keys on the FULL argument, not the 80-char display key."""
    prefix = "https://example.com/a/very/long/path/that/keeps/going/for/a/while/before/it/ends"
    assert len(prefix) >= 80
    assert (prefix + "-one")[:80] == (prefix + "-two")[:80]
    _fetch(session, "tc_c1", "browser", {"url": prefix + "-one"}, "1" * 3_000)
    _fetch(session, "tc_c2", "browser", {"url": prefix + "-two"}, "2" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    msgs = _tool_msgs(session)
    assert not msgs["tc_c1"].get("_stubbed")
    assert not msgs["tc_c2"].get("_stubbed")


# ---------------------------------------------------------------------------
# The archive is independent of supersession
# ---------------------------------------------------------------------------

def test_all_large_results_archived_even_when_not_stubbed(session, workspace):
    """Disk export is unconditional: it does not depend on being stubbed."""
    _fetch(session, "tc_d1", "read", {"path": "one.md"}, "1" * 3_000)
    _fetch(session, "tc_d2", "read", {"path": "two.md"}, "2" * 3_000)

    archive_and_supersede_tool_results(session, iteration=4)

    msgs = _tool_msgs(session)
    assert not any(m.get("_stubbed") for m in msgs.values())
    for call_id in ("tc_d1", "tc_d2"):
        path = os.path.join(_archive_dir(workspace), f"turn_4_call_{call_id}.json")
        assert os.path.exists(path), f"missing archive for {call_id}"


def test_every_superseded_copy_is_archived_before_it_is_stubbed(session, workspace):
    """The content a stub replaces is recoverable from disk."""
    for i in range(3):
        _fetch(session, f"tc_h{i}", "browser",
               {"action": "snapshot", "url": "https://example.com/hist"},
               f"SNAPSHOT_{i}_" + "K" * 4_000)

    archive_and_supersede_tool_results(session, iteration=2)

    for i in range(3):
        path = os.path.join(_archive_dir(workspace), f"turn_2_call_tc_h{i}.json")
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        assert record["content"].startswith(f"SNAPSHOT_{i}_")


def test_archive_written_once_per_call_id(session, workspace):
    """Repeated invocations must not re-archive the same result per turn."""
    _fetch(session, "tc_once", "read", {"path": "stable.md"}, "S" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)
    archive_and_supersede_tool_results(session, iteration=2)
    archive_and_supersede_tool_results(session, iteration=3)

    files = sorted(os.listdir(_archive_dir(workspace)))
    assert files == ["turn_1_call_tc_once.json"]


def test_superseded_stub_points_at_the_archived_copy(session, workspace):
    """The stub carries a disk path that actually exists and holds the content."""
    _fetch(session, "tc_p1", "read", {"path": "doc.md"}, "OLDBYTES_" + "O" * 3_000)
    _fetch(session, "tc_p2", "read", {"path": "doc.md"}, "NEWBYTES_" + "W" * 3_000)

    archive_and_supersede_tool_results(session, iteration=6)

    stub = _tool_msgs(session)["tc_p1"]["content"]
    marker = "Full result: "
    disk_path = stub[stub.index(marker) + len(marker):].rstrip("]")
    assert os.path.exists(disk_path)
    with open(disk_path, "r", encoding="utf-8") as f:
        record = json.load(f)
    assert record["content"].startswith("OLDBYTES_")


# ---------------------------------------------------------------------------
# Stub wording
# ---------------------------------------------------------------------------

def test_superseded_stub_leads_with_the_absence(session):
    """The stub must not read like a receipt for a successful retrieval."""
    _fetch(session, "tc_w1", "read", {"path": "wording.md"}, "1" * 3_000)
    _fetch(session, "tc_w2", "read", {"path": "wording.md"}, "2" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    stub = _tool_msgs(session)["tc_w1"]["content"]
    assert stub.startswith("[SUPERSEDED")
    low = stub.lower()
    assert "not the content" in low
    assert "newer read result for the same target" in low
    assert "wording.md" in stub


def test_superseded_stub_carries_no_model_narration(session):
    """No assistant text may leak into the stub as a faux content summary."""
    narration = "The doc says the header is exactly '## Section 4: Demo (wiki)'."
    _fetch(session, "tc_n1", "read", {"path": "narr.md"}, "1" * 3_000)
    session.append({"role": "assistant", "content": narration,
                    "source": "management"})
    _fetch(session, "tc_n2", "read", {"path": "narr.md"}, "2" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    stub = _tool_msgs(session)["tc_n1"]["content"]
    assert narration not in stub
    assert narration[:40] not in stub
    assert "Agent summary:" not in stub


def test_stub_persists_to_jsonl(session):
    """Supersession is durable — a reloaded session shows the stub."""
    _fetch(session, "tc_j1", "read", {"path": "persist.md"}, "MARKER_1_" + "1" * 3_000)
    _fetch(session, "tc_j2", "read", {"path": "persist.md"}, "MARKER_2_" + "2" * 3_000)

    archive_and_supersede_tool_results(session, iteration=1)

    reloaded = Session.load(session._filepath)
    msgs = {
        m["tool_call_id"]: m
        for m in reloaded.get_messages() if m.get("role") == "tool"
    }
    assert msgs["tc_j1"].get("_stubbed") is True
    assert "MARKER_1_" not in msgs["tc_j1"]["content"]
    assert msgs["tc_j2"]["content"].startswith("MARKER_2_")
