# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests for stream delta seq counter in ActivityTranslator.

Pre-Fix instrumentation: each chat.stream_delta must have a monotonic
`seq` number per project that resets on is_final.
"""

from unittest.mock import MagicMock

from agent_os.daemon_v2.activity_translator import ActivityTranslator


class FakeChunk:
    """Minimal stream chunk for testing."""
    def __init__(self, text: str, is_final: bool = False):
        self.text = text
        self.is_final = is_final


class TestStreamSeqCounter:
    def setup_method(self):
        self.ws = MagicMock()
        self.translator = ActivityTranslator(self.ws)

    def test_seq_increments_per_chunk(self):
        """Each non-final chunk should have an incrementing seq."""
        for i in range(5):
            self.translator.on_stream_chunk(
                FakeChunk(f"chunk{i}"), "proj_1", "management",
            )

        calls = self.ws.broadcast.call_args_list
        assert len(calls) == 5
        seqs = [c[0][1]["seq"] for c in calls]
        assert seqs == [1, 2, 3, 4, 5]

    def test_seq_resets_after_final(self):
        """After is_final, the seq counter should reset for the next response."""
        self.translator.on_stream_chunk(FakeChunk("a"), "proj_1", "management")
        self.translator.on_stream_chunk(FakeChunk("b"), "proj_1", "management")
        self.translator.on_stream_chunk(FakeChunk("", is_final=True), "proj_1", "management")

        # Start new response
        self.translator.on_stream_chunk(FakeChunk("c"), "proj_1", "management")
        self.translator.on_stream_chunk(FakeChunk("d"), "proj_1", "management")

        calls = self.ws.broadcast.call_args_list
        seqs = [c[0][1]["seq"] for c in calls]
        # First response: 1, 2, 3 (final)
        # Second response: 1, 2 (reset)
        assert seqs == [1, 2, 3, 1, 2]

    def test_seq_independent_per_project(self):
        """Different projects should have independent seq counters."""
        self.translator.on_stream_chunk(FakeChunk("a"), "proj_1", "management")
        self.translator.on_stream_chunk(FakeChunk("b"), "proj_2", "management")
        self.translator.on_stream_chunk(FakeChunk("c"), "proj_1", "management")

        calls = self.ws.broadcast.call_args_list
        # proj_1 calls: seq 1, 2
        # proj_2 calls: seq 1
        proj1_calls = [c[0][1] for c in calls if c[0][0] == "proj_1"]
        proj2_calls = [c[0][1] for c in calls if c[0][0] == "proj_2"]
        assert [p["seq"] for p in proj1_calls] == [1, 2]
        assert [p["seq"] for p in proj2_calls] == [1]

    def test_is_final_included_in_payload(self):
        """The is_final field should be correctly set in the broadcast payload."""
        self.translator.on_stream_chunk(FakeChunk("a"), "proj_1", "management")
        self.translator.on_stream_chunk(FakeChunk("", is_final=True), "proj_1", "management")

        calls = self.ws.broadcast.call_args_list
        assert calls[0][0][1]["is_final"] is False
        assert calls[1][0][1]["is_final"] is True
