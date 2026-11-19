# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Anthropic-protocol stream usage when the upstream reports input in message_delta.

OpenCode Go serves minimax-m3 over /messages. On the wire (2026-11-19) its
``message_start`` carries ``input_tokens: 0`` and null cache fields; the real
counts arrive only in the final ``message_delta.usage``:

    message_start usage: {"input_tokens": 0, "output_tokens": 0,
                          "cache_read_input_tokens": null, ...}
    message_delta usage: {"input_tokens": 1652, "output_tokens": 3,
                          "cache_read_input_tokens": 128,
                          "cache_creation_input_tokens": null, ...}

The adapter used to read only ``output_tokens`` from message_delta, so every
turn ledgered 0 input tokens. Anthropic documents message_delta usage as
cumulative, so a value there replaces (never adds to) the message_start one.
"""

from types import SimpleNamespace as NS

from agent_os.agent.providers.anthropic_adapter import StreamState, translate_stream_event
from agent_os.budget.normalize import ProviderSemantics, normalize_usage


def _usage(**kw):
    base = dict(input_tokens=None, output_tokens=None, cache_read_input_tokens=None,
                cache_creation_input_tokens=None)
    base.update(kw)
    return NS(**base)


def _run(start_usage, delta_usage):
    state = StreamState()
    events = [
        NS(type="message_start", message=NS(usage=start_usage)),
        NS(type="content_block_start", index=0, content_block=NS(type="text")),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="PONG")),
        NS(type="content_block_stop", index=0),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=delta_usage),
        NS(type="message_stop"),
    ]
    final = None
    for ev in events:
        chunk = translate_stream_event(ev, state)
        if chunk is not None and chunk.is_final:
            final = chunk
    assert final is not None
    return final.usage


def test_opencode_go_input_and_cache_read_only_in_message_delta():
    usage = _run(
        _usage(input_tokens=0, output_tokens=0),
        _usage(input_tokens=1652, output_tokens=3, cache_read_input_tokens=128),
    )
    assert usage.input_tokens == 1652
    assert usage.cache_read_tokens == 128
    assert usage.cache_write_tokens == 0
    assert usage.output_tokens == 3
    n = normalize_usage(usage, ProviderSemantics.ANTHROPIC)
    assert (n.uncached_input, n.cache_read, n.output) == (1652, 128, 3)


def test_opencode_go_warm_cache_round():
    usage = _run(
        _usage(input_tokens=0, output_tokens=0),
        _usage(input_tokens=1, output_tokens=3, cache_read_input_tokens=1779),
    )
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (1, 1779, 3)


def test_null_input_in_message_start_is_filled_from_delta():
    usage = _run(_usage(), _usage(input_tokens=700, output_tokens=5))
    assert usage.input_tokens == 700


def test_real_anthropic_delta_without_input_fields_keeps_message_start_counts():
    usage = _run(
        _usage(input_tokens=50, output_tokens=1, cache_read_input_tokens=1000,
               cache_creation_input_tokens=200),
        NS(output_tokens=10),  # older wire: message_delta carries output only
    )
    assert usage.input_tokens == 50
    assert usage.cache_read_tokens == 1000
    assert usage.cache_write_tokens == 200
    assert usage.output_tokens == 11  # unchanged pre-existing behaviour


def test_real_anthropic_cumulative_delta_is_not_double_counted():
    usage = _run(
        _usage(input_tokens=50, output_tokens=1, cache_read_input_tokens=1000,
               cache_creation_input_tokens=200),
        _usage(input_tokens=50, output_tokens=10, cache_read_input_tokens=1000,
               cache_creation_input_tokens=200),
    )
    assert usage.input_tokens == 50
    assert usage.cache_read_tokens == 1000
    assert usage.cache_write_tokens == 200
