# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for InlineThinkSplitter — separates inline <think>…</think>
reasoning from visible content, including across streaming-delta boundaries.

Motivated by MiniMax-M3, which emits reasoning inline in `content` wrapped in
<think></think> (with `reasoning_content` empty). The registry's
`ReasoningInfo.field is None` contract means "inline think"; this splitter
implements it.
"""

import pytest

from agent_os.agent.providers.think_splitter import InlineThinkSplitter


def _feed_all(chunks):
    """Feed each chunk; return cumulative (visible, reasoning) incl. flush."""
    s = InlineThinkSplitter()
    vis, reason = [], []
    for c in chunks:
        v, r = s.feed(c)
        vis.append(v); reason.append(r)
    v, r = s.flush()
    vis.append(v); reason.append(r)
    return "".join(vis), "".join(reason)


def test_plain_text_no_think():
    vis, reason = _feed_all(["hello world"])
    assert vis == "hello world"
    assert reason == ""


def test_single_block_one_feed():
    vis, reason = _feed_all(["<think>secret reasoning</think>the answer"])
    assert vis == "the answer"
    assert reason == "secret reasoning"


def test_text_before_and_after():
    vis, reason = _feed_all(["hi <think>r</think>bye"])
    assert vis == "hi bye"
    assert reason == "r"


def test_open_tag_split_across_deltas():
    # "<think>" arrives in pieces, then reasoning, then "</think>" in pieces.
    vis, reason = _feed_all(["<thi", "nk>rea", "son</thi", "nk>ans", "wer"])
    assert vis == "answer"
    assert reason == "reason"


def test_close_tag_split_across_deltas():
    vis, reason = _feed_all(["<think>abc", "de</thin", "k>xyz"])
    assert vis == "xyz"
    assert reason == "abcde"


def test_unclosed_think_flushes_as_reasoning():
    # Stream ends mid-think (no </think>) — everything after <think> is reasoning.
    vis, reason = _feed_all(["<think>still thinking when the stream died"])
    assert vis == ""
    assert reason == "still thinking when the stream died"


def test_lone_angle_bracket_is_visible():
    vis, reason = _feed_all(["a < b > c"])
    assert vis == "a < b > c"
    assert reason == ""


def test_trailing_partial_tag_prefix_buffered_then_visible():
    # A trailing "<" that never becomes a tag must surface as visible on flush.
    vis, reason = _feed_all(["done<"])
    assert vis == "done<"
    assert reason == ""


def test_minimax_shape_with_status_marker():
    # Realistic M3 content: <think> block, then a [STATUS:] line + answer.
    raw = "<think>\nThe user asked X. I will Y.\n</think>\n[STATUS: working]\nHere is the answer."
    vis, reason = _feed_all([raw])
    assert "<think>" not in vis and "</think>" not in vis
    assert "[STATUS: working]" in vis
    assert "Here is the answer." in vis
    assert "The user asked X. I will Y." in reason


def test_no_reasoning_emits_no_reasoning_chars():
    # Visible-only deltas must not leak any characters into reasoning.
    s = InlineThinkSplitter()
    v1, r1 = s.feed("partial answer ")
    v2, r2 = s.feed("continues")
    vf, rf = s.flush()
    assert (r1 + r2 + rf) == ""
    assert (v1 + v2 + vf) == "partial answer continues"
