# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: MiniMax-M3 reasoning leaks into visible chat content.

M3 emits chain-of-thought inline as ``<mm:think>…</mm:think>``, but the
``InlineThinkSplitter`` historically matched only the literal ``<think>`` tag.
Result: M3's entire monologue (and any degenerate repetition loop inside it)
rendered verbatim as visible chat content — the "wall of [STATUS: …] lines"
symptom from session orbital-marketing_e42ec0c9.

Fix: the splitter's reasoning tag is configurable (default ``think``), sourced
from ``ReasoningInfo.inline_tag`` in providers.json; M3 declares ``mm:think``.
"""

import pytest

from agent_os.agent.providers.think_splitter import InlineThinkSplitter
from agent_os.config.provider_registry import ReasoningInfo, _parse_reasoning_entry


def _feed_all(chunks, tag="think"):
    """Feed each chunk through a splitter with the given tag; return cumulative
    (visible, reasoning) including the final flush."""
    s = InlineThinkSplitter(tag=tag)
    vis, reason = [], []
    for c in chunks:
        v, r = s.feed(c)
        vis.append(v)
        reason.append(r)
    v, r = s.flush()
    vis.append(v)
    reason.append(r)
    return "".join(vis), "".join(reason)


# --- The bug: M3's <mm:think> must be peeled out when tag=mm:think -----------

def test_mmthink_block_peeled_when_configured():
    vis, reason = _feed_all(
        ["<mm:think>internal reasoning</mm:think>visible answer"], tag="mm:think"
    )
    assert vis == "visible answer"
    assert reason == "internal reasoning"


def test_mmthink_flood_does_not_leak_into_visible():
    """The exact failure shape: a repetition loop inside the think block plus a
    trailing answer. With mm:think configured, the loop stays in reasoning and
    only the answer is visible."""
    flood = "[STATUS: read file]\n" * 40
    raw = f"<mm:think>{flood}</mm:think>\nDone."
    vis, reason = _feed_all([raw], tag="mm:think")
    assert "<mm:think>" not in vis and "</mm:think>" not in vis
    assert "[STATUS: read file]" not in vis
    assert vis.strip() == "Done."
    assert "[STATUS: read file]" in reason


def test_mmthink_split_across_deltas():
    # Tag arrives in pieces across streaming deltas.
    vis, reason = _feed_all(
        ["<mm:th", "ink>rea", "son</mm:th", "ink>ans", "wer"], tag="mm:think"
    )
    assert vis == "answer"
    assert reason == "reason"


def test_mmthink_unclosed_flushes_as_reasoning():
    vis, reason = _feed_all(["<mm:think>still thinking"], tag="mm:think")
    assert vis == ""
    assert reason == "still thinking"


# --- Back-compat: default tag stays <think> ----------------------------------

def test_default_tag_is_think():
    vis, reason = _feed_all(["<think>r</think>answer"])
    assert vis == "answer"
    assert reason == "r"


def test_default_splitter_ignores_mmthink():
    # With the default <think> tag, a <mm:think> tag is NOT a reasoning tag and
    # stays visible (this is the pre-fix behavior, preserved for non-M3 models).
    vis, reason = _feed_all(["<mm:think>x</mm:think>y"])
    assert "<mm:think>" in vis
    assert reason == ""


# --- Provider config carries the tag -----------------------------------------

def test_reasoning_info_defaults_to_think():
    assert ReasoningInfo().inline_tag == "think"


def test_parse_reasoning_entry_reads_inline_tag():
    info = _parse_reasoning_entry(
        {"supported": True, "field": None, "inline_tag": "mm:think"}
    )
    assert info.inline_tag == "mm:think"


def test_parse_reasoning_entry_defaults_inline_tag():
    info = _parse_reasoning_entry({"supported": True, "field": None})
    assert info.inline_tag == "think"


def test_minimax_m3_uses_default_think_tag():
    """M3's standard reasoning tag is <think>. (The v0.6.1 mm:think override was
    a misdiagnosis — M3 normally emits <think>…</think>; mm:think appeared once
    as a malformed lone close tag. Reverted in v0.6.2.)"""
    from agent_os.config.provider_registry import ProviderRegistry

    info = ProviderRegistry().get_model_info("minimax", "MiniMax-M3")
    assert info.reasoning.supported is True
    assert info.reasoning.field is None  # inline-think contract
    assert info.reasoning.inline_tag == "think"


def test_minimax_m3_peels_plain_think_block():
    """REGRESSION (v0.6.1 → v0.6.2): M3 emits standard <think>…</think>. Its
    CONFIGURED splitter must peel that, not leak it. v0.6.1 set inline_tag=
    mm:think, so the splitter missed <think> and the monologue leaked verbatim
    into chat (observed in session orbital-marketing_273196db: 6 leaked <think>
    blocks, 0 reasoning_content)."""
    from agent_os.config.provider_registry import ProviderRegistry

    info = ProviderRegistry().get_model_info("minimax", "MiniMax-M3")
    vis, reason = _feed_all(
        ["<think>internal M3 reasoning</think>visible answer"],
        tag=info.reasoning.inline_tag,
    )
    assert vis == "visible answer"
    assert reason == "internal M3 reasoning"
