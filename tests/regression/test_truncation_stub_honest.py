# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: a tool-result stub must be honest, not a faux content summary.

Bug (c201e14): when a tool result was evicted from context, the replacement
stub's body was the first ~350 chars of the MODEL'S OWN narration — which reads
like a summary of the file. The model then built exact ``old_text`` for an
``edit`` from that "summary" and the exact-match edit failed, because the stub
is not the file.

The blanket eviction that produced those stubs is gone; the only stub left is
the supersession stub (a later fetch of the same target replaced this copy).
The honesty guard still applies to it, and is now stronger: the entry point
takes no assistant text at all, so narration cannot reach a stub even by
accident, and the wording leads with the ABSENCE rather than reading like a
receipt for a successful retrieval.
"""

import inspect
import json

import pytest

from agent_os.agent.session import Session
from agent_os.agent.tool_result_lifecycle import (
    archive_and_supersede_tool_results,
)


@pytest.fixture
def session(tmp_path):
    return Session.new("honest-stub-test", str(tmp_path))


def _add_tool_call_and_result(session, call_id, tool_name, arguments, content):
    session.append({
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(arguments)},
        }],
        "source": "management",
    })
    session.append_tool_result(call_id, content)


def _refetch(session, path, narration=None):
    """Read `path` twice, optionally narrating in between. Returns the stub."""
    _add_tool_call_and_result(session, "tc_old", "read", {"path": path}, "Z" * 4000)
    if narration is not None:
        session.append({
            "role": "assistant", "content": narration, "source": "management",
        })
    _add_tool_call_and_result(session, "tc_new", "read", {"path": path}, "Y" * 4000)
    archive_and_supersede_tool_results(session, iteration=1)
    return _stub_for(session, "tc_old")


def _stub_for(session, call_id):
    for m in session.get_messages():
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id:
            return m["content"]
    raise AssertionError(f"no tool result for {call_id}")


def test_stub_does_not_embed_model_narration(session):
    """The model's narration must NOT appear in the stub as a faux summary."""
    narration = "I read the file; the header is exactly '## Section 4: Demo (wiki)'."
    stub = _refetch(session, "shot-list.md", narration)

    assert "Agent summary:" not in stub
    assert narration not in stub
    assert narration[:40] not in stub


def test_entry_point_takes_no_assistant_text(session):
    """Structural guard: narration cannot reach a stub, because it is not an input.

    The old signature carried the LLM's response text purely for "signature
    stability" — one edit away from being re-embedded. It is gone.
    """
    params = list(
        inspect.signature(archive_and_supersede_tool_results).parameters,
    )
    assert params == ["session", "iteration"]


def test_stub_states_not_content_and_instructs_recovery(session):
    """The stub must say it is not the content and how to get the content back."""
    stub = _refetch(session, "shot-list.md")

    low = stub.lower()
    assert "not the content" in low
    assert "re-read" in low
    # Points at the surviving newer copy as the thing to use.
    assert "newer read result for the same target" in low


def test_stub_leads_with_absence_not_a_receipt(session):
    """It must not open like a successful retrieval record."""
    stub = _refetch(session, "shot-list.md")

    assert stub.startswith("[SUPERSEDED")
    assert "GONE" in stub
    first_line = stub.splitlines()[0].lower()
    assert "not the content" in first_line


def test_stub_preserves_metadata_header(session):
    """Useful metadata (tool, target, size, disk path) is still present."""
    _add_tool_call_and_result(
        session, "tc_old", "read", {"path": "content-bank/index.md"}, "X" * 4000,
    )
    _add_tool_call_and_result(
        session, "tc_new", "read", {"path": "content-bank/index.md"}, "W" * 4000,
    )
    archive_and_supersede_tool_results(session, iteration=3)
    stub = _stub_for(session, "tc_old")

    assert "[Tool: read" in stub
    assert "Target: content-bank/index.md" in stub
    assert "Original: 1000 tokens" in stub
    assert "Full result:" in stub
    assert ".json" in stub


def test_threshold_unchanged(session):
    """Small (<500 char) results are still never rewritten; large ones can be."""
    _add_tool_call_and_result(
        session, "tc_small_1", "shell", {"command": "echo hi"}, "small output",
    )
    _add_tool_call_and_result(
        session, "tc_small_2", "shell", {"command": "echo hi"}, "small output",
    )
    _add_tool_call_and_result(
        session, "tc_big_1", "read", {"path": "big.md"}, "B" * 600,
    )
    _add_tool_call_and_result(
        session, "tc_big_2", "read", {"path": "big.md"}, "C" * 600,
    )
    archive_and_supersede_tool_results(session, iteration=1)

    tool_msgs = {
        m["tool_call_id"]: m
        for m in session.get_messages() if m.get("role") == "tool"
    }
    assert not tool_msgs["tc_small_1"].get("_stubbed")
    assert tool_msgs["tc_small_1"]["content"] == "small output"
    assert tool_msgs["tc_big_1"].get("_stubbed") is True
    assert not tool_msgs["tc_big_2"].get("_stubbed")


def test_unsuperseded_result_is_not_stubbed_at_all(session):
    """The honest stub is only reachable via supersession — never on its own."""
    _add_tool_call_and_result(
        session, "tc_solo", "read", {"path": "solo.md"}, "S" * 4000,
    )
    archive_and_supersede_tool_results(session, iteration=1)

    msg = [m for m in session.get_messages() if m.get("role") == "tool"][0]
    assert not msg.get("_stubbed")
    assert msg["content"] == "S" * 4000
