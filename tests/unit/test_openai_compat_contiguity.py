# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Strict OpenAI-compatible providers (MiniMax api.minimaxi.com) require every
``role:"tool"`` result to sit in one unbroken block immediately after the
assistant message carrying the matching ``tool_calls``. A persisted session can
contain an interleaved non-tool row, e.g. when a turn-yielding dispatch tool is
batched with other tool calls that get cancelled:

    assistant(tool_calls=[_1, _2, _3])
    tool(_1)                       # the dispatch's own result
    system("[Sub-agent] Message sent to claude-code: …")   # <-- interleaved
    tool(_2)  CANCELLED            # now orphaned
    tool(_3)  CANCELLED            # now orphaned

MiniMax rejects that split block with 400 "invalid params, tool call result
does not follow tool call (2013)" on EVERY resume, permanently wedging the
session (DIAGNOSIS-minimax-tool-block-contiguity-wedge.md — CONFIRMED:
relocating the system row below the tool block flipped session e42ec0c9 from
400 to 200).

``_ensure_tool_result_contiguity`` repairs this at the wire boundary by
*reordering only*: matched tool results are pulled up to be contiguous, and any
non-tool message interleaved within the block is relocated to just after it. It
never fabricates, drops, or merges messages, and never mutates persisted state.
"""

from types import SimpleNamespace

import pytest

from agent_os.agent.providers.openai_compat import (
    LLMProvider,
    _ensure_tool_result_contiguity,
)


def _provider():
    return LLMProvider("MiniMax-M3", "key", base_url="https://api.minimaxi.com/v1", sdk="openai")


def _tc(tid):
    return {"id": tid, "type": "function", "function": {"name": "f", "arguments": "{}"}}


def _roles_ids(msgs):
    """Compact (role, tool_call_id-or-marker) sequence for order assertions."""
    out = []
    for m in msgs:
        if m.get("role") == "tool":
            out.append(("tool", m.get("tool_call_id")))
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            out.append(("assistant", tuple(tc["id"] for tc in m["tool_calls"])))
        else:
            out.append((m.get("role"), m.get("content")))
    return out


# --- PRIMARY: the proven poisoned sequence (mirror of e42ec0c9.preexp-bak) ---


def test_proven_poisoned_sequence_reordered_via_prepare():
    """The exact shape that 400'd in production, run through the real provider
    prep, must come out as assistant → tool(_1/_2/_3) → system(echo)."""
    msgs = [
        {"role": "assistant", "content": "[STATUS] dispatch",
         "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "Dispatched… Awaiting completion"},
        {"role": "system", "content": "[Sub-agent] Message sent to claude-code: …"},
        {"role": "tool", "tool_call_id": "_2", "content": "CANCELLED: This tool call was not executed."},
        {"role": "tool", "tool_call_id": "_3", "content": "CANCELLED: This tool call was not executed."},
    ]
    out = _provider()._prepare_messages_openai(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_1"),
        ("tool", "_2"),
        ("tool", "_3"),
        ("system", "[Sub-agent] Message sent to claude-code: …"),
    ]
    assert len(out) == len(msgs)


def test_pure_helper_reorders_single_interleaved_system():
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "system", "content": "echo"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
        {"role": "tool", "tool_call_id": "_3", "content": "c"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_1"), ("tool", "_2"), ("tool", "_3"),
        ("system", "echo"),
    ]


# --- REGRESSION: shapes that already worked must be byte-identical -----------


def test_solo_dispatch_unchanged():
    """Single tool call + trailing system echo (the dispatch #1 shape that
    always returned 200): already contiguous, must be untouched."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1")]},
        {"role": "tool", "tool_call_id": "_1", "content": "Dispatched…"},
        {"role": "system", "content": "[Sub-agent] Message sent to claude-code: …"},
    ]
    assert _ensure_tool_result_contiguity(msgs) == msgs


def test_already_contiguous_multi_block_unchanged():
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
    ]
    assert _ensure_tool_result_contiguity(msgs) == msgs


def test_no_tool_calls_unchanged():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _ensure_tool_result_contiguity(msgs) == msgs


def test_assistant_empty_tool_calls_list_unchanged():
    msgs = [
        {"role": "assistant", "content": "x", "tool_calls": []},
        {"role": "user", "content": "y"},
    ]
    assert _ensure_tool_result_contiguity(msgs) == msgs


# --- EDGE CASES --------------------------------------------------------------


def test_multiple_displaced_relocated_preserving_order():
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "system", "content": "sysA"},
        {"role": "system", "content": "sysB"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
        {"role": "tool", "tool_call_id": "_3", "content": "c"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_1"), ("tool", "_2"), ("tool", "_3"),
        ("system", "sysA"), ("system", "sysB"),
    ]


def test_two_independent_blocks_no_cross_reorder():
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("a1")]},
        {"role": "tool", "tool_call_id": "a1", "content": "x"},
        {"role": "system", "content": "echo1"},
        {"role": "assistant", "tool_calls": [_tc("b1"), _tc("b2")]},
        {"role": "tool", "tool_call_id": "b1", "content": "y"},
        {"role": "system", "content": "echo2"},
        {"role": "tool", "tool_call_id": "b2", "content": "z"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("a1",)),
        ("tool", "a1"),
        ("system", "echo1"),
        ("assistant", ("b1", "b2")),
        ("tool", "b1"), ("tool", "b2"),
        ("system", "echo2"),
    ]


def test_post_block_messages_not_pulled_into_block():
    """Non-tool messages AFTER the last tool result are normal conversation
    flow — they must NOT be relocated."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "system", "content": "summary"},
        {"role": "user", "content": "next question"},
    ]
    assert _ensure_tool_result_contiguity(msgs) == msgs


def test_missing_tool_result_only_reorders_present_ones():
    """If a tool_call_id has no result at all, leave it absent (different
    defect). Reorder only the present results; do not fabricate."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "system", "content": "echo"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
        {"role": "user", "content": "next"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_1"), ("tool", "_2"),
        ("system", "echo"),
        ("user", "next"),
    ]
    # _3 was never fabricated.
    assert sum(1 for m in out if m.get("role") == "tool") == 2


# --- INVARIANTS --------------------------------------------------------------


def test_message_count_preserved_and_no_fabrication():
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "system", "content": "echo"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
        {"role": "tool", "tool_call_id": "_3", "content": "c"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert len(out) == len(msgs)
    # Same multiset of messages — pure reorder, nothing added or dropped.
    assert sorted(map(repr, out)) == sorted(map(repr, msgs))


def test_input_not_mutated():
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "system", "content": "echo"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
    ]
    snapshot = [dict(m) for m in msgs]
    order_before = list(msgs)
    _ensure_tool_result_contiguity(msgs)
    assert msgs == order_before          # input list order untouched
    assert [dict(m) for m in msgs] == snapshot  # input dicts untouched


# --- REGRESSION PINS: verified-correct shapes, locked against silent refactors


def test_results_emitted_in_encounter_order_not_declared_order():
    """When results arrive scrambled vs the tool_calls declaration order (and the
    dispatch isn't first), they stay in ENCOUNTER order. Locks that contract so a
    future change doesn't start sorting by declared order."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
        {"role": "system", "content": "echo"},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "tool", "tool_call_id": "_3", "content": "c"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_2"), ("tool", "_1"), ("tool", "_3"),
        ("system", "echo"),
    ]


def test_two_poisoned_blocks_each_repaired_independently():
    """A multi-dispatch resume where BOTH tool blocks are split: each is healed
    on its own, with no cross-block reordering."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("a1"), _tc("a2")]},
        {"role": "tool", "tool_call_id": "a1", "content": "x"},
        {"role": "system", "content": "e1"},
        {"role": "tool", "tool_call_id": "a2", "content": "y"},
        {"role": "assistant", "tool_calls": [_tc("b1"), _tc("b2")]},
        {"role": "tool", "tool_call_id": "b1", "content": "p"},
        {"role": "system", "content": "e2"},
        {"role": "tool", "tool_call_id": "b2", "content": "q"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("a1", "a2")),
        ("tool", "a1"), ("tool", "a2"), ("system", "e1"),
        ("assistant", ("b1", "b2")),
        ("tool", "b1"), ("tool", "b2"), ("system", "e2"),
    ]


def test_interleaved_user_row_is_relocated_not_wedged():
    """A non-system interleaved row (e.g. an approval-flow user turn) is relocated
    below the tool block — deliberate: recovering the session beats a permanent
    400 wedge, and the displaced row's position is otherwise preserved."""
    msgs = [
        {"role": "assistant", "tool_calls": [_tc("_1"), _tc("_2")]},
        {"role": "tool", "tool_call_id": "_1", "content": "a"},
        {"role": "user", "content": "approve?"},
        {"role": "tool", "tool_call_id": "_2", "content": "b"},
    ]
    out = _ensure_tool_result_contiguity(msgs)
    assert _roles_ids(out) == [
        ("assistant", ("_1", "_2")),
        ("tool", "_1"), ("tool", "_2"),
        ("user", "approve?"),
    ]


# --- WIRE BOUNDARY: the repair actually reaches the OpenAI SDK payload -------


@pytest.mark.asyncio
async def test_poisoned_payload_is_contiguous_at_the_sdk_boundary():
    """Capture the exact ``messages`` the OpenAI SDK would receive via the real
    ``stream()`` path and assert the tool block is contiguous — proving the
    repair is on the send path, not bypassed. This is the portable equivalent
    of the daemon smoke test (no MiniMax credentials / live daemon needed)."""
    captured = {}

    class _Completions:
        async def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]

            async def _gen():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="ok", reasoning_content=None, tool_calls=None),
                        finish_reason="stop",
                    )],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )

            return _gen()

    prov = _provider()
    prov._openai_client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    poisoned = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "[STATUS] dispatch",
         "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "Dispatched… Awaiting completion"},
        {"role": "system", "content": "[Sub-agent] Message sent to claude-code: …"},
        {"role": "tool", "tool_call_id": "_2", "content": "CANCELLED: This tool call was not executed."},
        {"role": "tool", "tool_call_id": "_3", "content": "CANCELLED: This tool call was not executed."},
    ]
    async for _ in prov.stream(poisoned):
        pass

    sent = captured["messages"]
    # Every tool result immediately follows an assistant-with-tool_calls or
    # another tool result — i.e. no tool result is preceded by a non-tool row.
    for k, m in enumerate(sent):
        if m.get("role") == "tool":
            prev = sent[k - 1]
            assert prev.get("role") == "tool" or (
                prev.get("role") == "assistant" and prev.get("tool_calls")
            ), f"tool result at {k} does not follow a tool call: prev={prev!r}"
