# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ACP tool activity must survive the whole way into a post-hoc capsule row.

The pipe already existed end to end — ``ACPSDKTransport`` converts ACP
``ToolCallStart``/``ToolCallProgress`` into ``tool_use`` events, which become
``tool_activity`` chunks, which ProcessManager appends to the sub-agent
transcript, which the capsule replays. Its last inch was broken for EVERY ACP
agent: the capsule recovers tool names by matching ``[Using tool: X]`` against
the entry's ``content`` (``sub_agent_transcript.py:17-19``), transcript entries
persist ``content`` with no metadata, and the transport wrote a bare title. So
ACP tool updates produced zero capsule rows — cursor's included, long before
dsh existed.

These tests pin the text format as the contract it actually is, and walk one
``ToolCallStart`` through every real hop rather than asserting on the string in
isolation.
"""

import json

import pytest

from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
from agent_os.agent.transports.base import transport_event_to_chunk
from agent_os.daemon_v2.sub_agent_transcript import (
    SubAgentTranscript,
    _summarize_turn,
    read_sub_agent_summary,
)

acp_schema = pytest.importorskip("acp.schema")


def _tool_call_start(title="bash", **kwargs):
    return acp_schema.ToolCallStart(
        sessionUpdate="tool_call", toolCallId="call-1", title=title, **kwargs
    )


def _tool_call_progress(status="completed", **kwargs):
    return acp_schema.ToolCallProgress(
        sessionUpdate="tool_call_update", toolCallId="call-1", status=status, **kwargs
    )


def _entry(chunk, timestamp):
    """The exact dict ProcessManager appends (process_manager.py:371-375)."""
    return {
        "source": "dsh",
        "content": chunk.text,
        "timestamp": timestamp,
        "chunk_type": chunk.chunk_type,
    }


class TestTransportEmitsTheCapsuleFormat:
    def test_tool_call_start_raw_text_is_the_capsule_format(self):
        event = ACPSDKTransport()._session_update_to_event(_tool_call_start("bash"))

        assert event.event_type == "tool_use"
        assert event.raw_text == "[Using tool: bash]", (
            "the capsule parser matches this exact text and transcript entries "
            "carry no metadata, so the format is the contract"
        )

    def test_structured_tool_name_is_preserved_alongside_the_text(self):
        event = ACPSDKTransport()._session_update_to_event(_tool_call_start("bash"))

        assert event.data["tool_name"] == "bash"
        assert event.data["tool_call_id"] == "call-1"

    def test_progress_updates_also_carry_the_format(self):
        event = ACPSDKTransport()._session_update_to_event(
            _tool_call_progress(status="completed", title="bash")
        )

        assert event.raw_text == "[Using tool: bash]"

    def test_falls_back_to_kind_then_to_a_generic_name(self):
        """A title-less update must still parse — never an unnamed row."""
        by_kind = ACPSDKTransport()._session_update_to_event(
            _tool_call_progress(kind="execute")
        )
        assert by_kind.raw_text == "[Using tool: execute]"

        untitled = ACPSDKTransport()._session_update_to_event(_tool_call_progress())
        assert untitled.raw_text == "[Using tool: tool]"


class TestChunkConversion:
    def test_tool_use_becomes_a_tool_activity_chunk_carrying_the_text(self):
        event = ACPSDKTransport()._session_update_to_event(_tool_call_start("bash"))
        chunk = transport_event_to_chunk(event)

        assert chunk.chunk_type == "tool_activity"
        assert chunk.text == "[Using tool: bash]"


class TestCapsuleRows:
    def test_an_acp_tool_call_becomes_a_named_capsule_row(self):
        """The regression this whole change exists for: zero rows before it."""
        start = transport_event_to_chunk(
            ACPSDKTransport()._session_update_to_event(_tool_call_start("bash"))
        )
        entries = [
            _entry(start, "2026-08-14T00:00:00+00:00"),
            {
                "source": "dsh",
                "content": "Done.",
                "timestamp": "2026-08-14T00:00:03+00:00",
                "chunk_type": "response",
            },
        ]

        summary = _summarize_turn(entries)

        assert summary["tools_used"] == ["bash"]
        assert [row["name"] for row in summary["tool_rows"]] == ["bash"]
        assert summary["tool_rows"][0]["duration_seconds"] == 3.0
        assert summary["response"] == "Done."

    def test_several_tools_keep_their_names_and_first_seen_order(self):
        chunks = [
            transport_event_to_chunk(
                ACPSDKTransport()._session_update_to_event(_tool_call_start(name))
            )
            for name in ("bash", "read_file", "bash")
        ]
        entries = [
            _entry(chunk, f"2026-08-14T00:00:0{index}+00:00")
            for index, chunk in enumerate(chunks)
        ]

        summary = _summarize_turn(entries)

        assert [row["name"] for row in summary["tool_rows"]] == [
            "bash", "read_file", "bash",
        ]
        assert summary["tools_used"] == ["bash", "read_file"]

    def test_full_transcript_round_trip(self, tmp_path):
        """Through a real JSONL transcript, not a hand-built entry list."""
        transcript = SubAgentTranscript(str(tmp_path), "dsh", "t1")
        start = transport_event_to_chunk(
            ACPSDKTransport()._session_update_to_event(_tool_call_start("bash"))
        )
        transcript.append(_entry(start, "2026-08-14T00:00:00+00:00"))
        transcript.append({
            "source": "dsh",
            "content": "Done.",
            "timestamp": "2026-08-14T00:00:01+00:00",
            "chunk_type": "response",
        })
        transcript.append({
            "source": "dsh",
            "content": "",
            "timestamp": "2026-08-14T00:00:02+00:00",
            "chunk_type": "turn_complete",
        })

        turns = read_sub_agent_summary(transcript.filepath)

        assert turns is not None and len(turns) == 1
        assert turns[0]["tools_used"] == ["bash"]

    def test_a_bare_title_would_have_produced_no_rows(self):
        """Pins WHY the format matters — this is the pre-fix behaviour."""
        entries = [{
            "source": "dsh",
            "content": "bash",
            "timestamp": "2026-08-14T00:00:00+00:00",
            "chunk_type": "tool_activity",
        }]

        assert _summarize_turn(entries)["tool_rows"] == []


class TestShimPayloadCompatibility:
    """The composition-local activity shim emits exactly this payload.

    Guards the shim's wire contract from the Orbital side: if the ACP schema
    or the transport's handling drifts, this fails here rather than silently
    costing tool rows in a live dispatch. The literal below is copied from a
    frame the shim actually put on stdout during structural verification.
    """

    SHIM_START_FRAME = (
        '{"sessionUpdate": "tool_call", "toolCallId": "spike-call-1",'
        ' "title": "bash", "status": "in_progress"}'
    )

    def test_the_shims_payload_parses_and_names_the_tool(self):
        update = acp_schema.ToolCallStart.model_validate(
            json.loads(self.SHIM_START_FRAME)
        )

        event = ACPSDKTransport()._session_update_to_event(update)

        assert event.raw_text == "[Using tool: bash]"
        assert event.data["status"] == "in_progress"

    def test_one_dispatched_tool_call_yields_exactly_one_capsule_row(self):
        """Why the shim is start-only, pinned as a test.

        The capsule appends a row per ``tool_activity`` chunk and never
        de-duplicates by ``toolCallId`` — ``chatTransform.ts`` even synthesizes
        each row's id from its INDEX. So emitting a terminal
        ``tool_call_update`` alongside the start would render every single call
        twice (``bash · 3.2s`` then ``bash · 0.0s``). One frame per invocation
        is also exactly what claude-code does (``sdk_transport.py:636``), which
        is the parity bar. If a terminal frame is ever restored, the capsule
        must learn to de-duplicate in the same change — and this test is where
        that shows up.
        """
        start = transport_event_to_chunk(
            ACPSDKTransport()._session_update_to_event(
                acp_schema.ToolCallStart.model_validate(
                    json.loads(self.SHIM_START_FRAME)
                )
            )
        )
        terminal = transport_event_to_chunk(
            ACPSDKTransport()._session_update_to_event(
                _tool_call_progress(status="completed", title="bash")
            )
        )

        shipped = _summarize_turn([_entry(start, "2026-08-14T00:00:00+00:00")])
        assert len(shipped["tool_rows"]) == 1

        doubled = _summarize_turn([
            _entry(start, "2026-08-14T00:00:00+00:00"),
            _entry(terminal, "2026-08-14T00:00:03+00:00"),
        ])
        assert len(doubled["tool_rows"]) == 2, (
            "the capsule does not de-duplicate by toolCallId, which is the "
            "whole reason the shim emits only the start frame"
        )

    def test_the_capsule_has_no_status_affordance_to_lose(self):
        """The dropped terminal status costs the user nothing today."""
        terminal = ACPSDKTransport()._session_update_to_event(
            _tool_call_progress(status="failed", title="bash")
        )
        chunk = transport_event_to_chunk(terminal)

        summary = _summarize_turn([_entry(chunk, "2026-08-14T00:00:00+00:00")])

        assert set(summary["tool_rows"][0]) == {
            "name", "timestamp", "duration_seconds",
        }, "a capsule row carries no status field for a terminal frame to fill"
