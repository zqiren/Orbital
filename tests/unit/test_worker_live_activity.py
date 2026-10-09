# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 100: a pinned worker's tool calls and thinking, live and after reload.

Every transport already parsed (or could parse) the detail; it died at
ProcessManager, which persisted chunks without metadata and broadcast only
response text. These tests walk each transport's events through the real
hops: transport parse → chunk → ProcessManager (transcript row + broadcast)
→ transcript reader → /chat interleave.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_os.agent.transports.base import TransportEvent, transport_event_to_chunk
from agent_os.daemon_v2.activity_translator import (
    ActivityTranslator,
    worker_display_meta,
)
from agent_os.daemon_v2.process_manager import ProcessManager
from agent_os.daemon_v2.sub_agent_transcript import (
    SubAgentTranscript,
    _summarize_turn,
    read_sub_agent_summary,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _WS:
    def __init__(self):
        self.events: list[dict] = []

    def broadcast(self, project_id, event):
        self.events.append(event)

    def of(self, type_):
        return [e for e in self.events if e.get("type") == type_]


def _row(chunk_type, content="", *, ts="2026-09-27T00:00:00+00:00", meta=None,
         dispatch_id=None, source="w"):
    row = {"source": source, "content": content, "timestamp": ts, "chunk_type": chunk_type}
    if meta:
        row["metadata"] = meta
    if dispatch_id:
        row["dispatch_id"] = dispatch_id
    return row


def _ts(sec):
    return f"2026-09-27T00:00:{sec:02d}+00:00"


# ---------------------------------------------------------------------------
# SDK (claude-code): ThinkingBlock + ToolResultBlock parsing
# ---------------------------------------------------------------------------

sdk_types = pytest.importorskip("claude_agent_sdk.types")


def _sdk():
    from agent_os.agent.transports.sdk_transport import SDKTransport
    return SDKTransport()


class TestSDKParse:
    def test_mixed_assistant_message_keeps_order_and_drops_empty_thinking(self):
        msg = sdk_types.AssistantMessage(
            content=[
                sdk_types.ThinkingBlock(thinking="", signature="s"),
                sdk_types.ThinkingBlock(thinking="   ", signature="s"),
                sdk_types.ThinkingBlock(thinking="Plan: read then write.", signature="s"),
                sdk_types.TextBlock(text="On it."),
                sdk_types.ToolUseBlock(id="tu1", name="Read", input={"file_path": "a.txt"}),
            ],
            model="claude-x",
        )

        events = _sdk()._message_to_events(msg)

        assert [e.event_type for e in events] == ["thinking", "message", "tool_use"]
        assert events[0].data == {"text": "Plan: read then write."}
        assert events[1].raw_text == "On it."
        # The load-bearing text contract is untouched.
        assert events[2].raw_text == "[Using tool: Read]"
        assert events[2].data["tool_id"] == "tu1"

    def test_tool_result_blocks_become_paired_tool_result_events(self):
        msg = sdk_types.UserMessage(content=[
            sdk_types.ToolResultBlock(tool_use_id="tu1", content="hello\n", is_error=False),
            sdk_types.ToolResultBlock(
                tool_use_id="tu2",
                content=[{"type": "text", "text": "line a"},
                         {"type": "image", "source": {}},
                         {"type": "text", "text": "line b"}],
                is_error=True),
            sdk_types.ToolResultBlock(tool_use_id="tu3", content=None),
        ])

        events = _sdk()._message_to_events(msg)

        assert [e.event_type for e in events] == ["tool_result"] * 3
        assert events[0].data == {"tool_call_id": "tu1", "content": "hello\n", "is_error": False}
        assert events[1].data["content"] == "line a\nline b"
        assert events[1].data["is_error"] is True
        assert events[2].data["content"] == ""
        assert all(e.raw_text == "" for e in events)

    def test_plain_user_message_text_emits_nothing(self):
        assert _sdk()._message_to_events(sdk_types.UserMessage(content="hi")) == []

    def test_new_event_types_keep_their_chunk_types(self):
        """Not the "response" default: that would make a tool result the
        turn's summary and a chat.sub_agent_message bubble."""
        thinking = transport_event_to_chunk(TransportEvent("thinking", {"text": "t"}, "t"))
        result = transport_event_to_chunk(TransportEvent("tool_result", {"tool_call_id": "x"}))
        assert thinking.chunk_type == "thinking"
        assert result.chunk_type == "tool_result"


# ---------------------------------------------------------------------------
# codex app-server: the tool item types it dropped + reasoning summaries
# ---------------------------------------------------------------------------

def _codex():
    from agent_os.agent.transports.codex_transport import CodexTransport
    return CodexTransport()


def _drain(transport) -> list[TransportEvent]:
    out = []
    while not transport._event_queue.empty():
        out.append(transport._event_queue.get_nowait())
    return out


async def _item(t, phase, item):
    await t._route_server_message({"jsonrpc": "2.0", "method": phase, "params": {"item": item}})


# Field shapes pinned to `codex app-server generate-json-schema` (0.144.5; additive-only through 0.157.1).
CODEX_ITEMS = {
    "mcpToolCall": (
        {"type": "mcpToolCall", "id": "m1", "server": "fs", "tool": "list",
         "arguments": {"path": "."}, "status": "inProgress"},
        {"type": "mcpToolCall", "id": "m1", "server": "fs", "tool": "list",
         "arguments": {"path": "."}, "status": "completed",
         "result": {"content": [{"type": "text", "text": "a.txt\nb.txt"}]}},
        "fs.list", {"path": "."}, "a.txt\nb.txt",
    ),
    "dynamicToolCall": (
        {"type": "dynamicToolCall", "id": "d1", "namespace": "ns", "tool": "t",
         "arguments": {"q": 1}, "status": "inProgress"},
        {"type": "dynamicToolCall", "id": "d1", "namespace": "ns", "tool": "t",
         "arguments": {"q": 1}, "status": "completed", "success": True,
         "contentItems": [{"type": "inputText", "text": "done"}]},
        "ns.t", {"q": 1}, "done",
    ),
    "webSearch": (
        {"type": "webSearch", "id": "w1", "query": "orbital"},
        {"type": "webSearch", "id": "w1", "query": "orbital",
         "action": {"type": "search", "query": "orbital"}},
        "webSearch", {"query": "orbital"}, "",
    ),
    "imageView": (
        {"type": "imageView", "id": "i1", "path": "/tmp/x.png"},
        {"type": "imageView", "id": "i1", "path": "/tmp/x.png"},
        "imageView", {"path": "/tmp/x.png"}, "",
    ),
    "imageGeneration": (
        {"type": "imageGeneration", "id": "g1", "result": "", "status": "inProgress",
         "revisedPrompt": "a cat"},
        {"type": "imageGeneration", "id": "g1", "result": "iVBORw0KGgo=BASE64",
         "status": "completed", "revisedPrompt": "a cat", "savedPath": "/tmp/cat.png"},
        "imageGeneration", {"prompt": "a cat"}, "/tmp/cat.png",
    ),
    "collabAgentToolCall": (
        {"type": "collabAgentToolCall", "id": "c1", "tool": "spawnAgent",
         "prompt": "help", "status": "inProgress", "agentsStates": {},
         "receiverThreadIds": [], "senderThreadId": "T"},
        {"type": "collabAgentToolCall", "id": "c1", "tool": "spawnAgent",
         "prompt": "help", "status": "completed", "agentsStates": {},
         "receiverThreadIds": [], "senderThreadId": "T"},
        "spawnAgent", {"prompt": "help"}, "",
    ),
}


class TestCodexToolItems:
    @pytest.mark.parametrize("itype", sorted(CODEX_ITEMS))
    def test_started_then_completed_pair_by_item_id(self, itype):
        started, completed, name, args_subset, result = CODEX_ITEMS[itype]
        t = _codex()

        async def run():
            await _item(t, "item/started", started)
            await _item(t, "item/completed", completed)
        asyncio.run(run())
        events = _drain(t)

        assert [e.event_type for e in events] == ["tool_use", "tool_use"]
        first, last = events
        assert first.data["tool_name"] == name
        assert first.data["tool_id"] == last.data["tool_id"] == started["id"]
        for key, value in args_subset.items():
            assert first.data["tool_input"][key] == value
        assert "result" not in first.data
        assert last.data["result"] == result
        assert last.data["is_error"] is False
        # The transcript text keeps the `[Running …]` style.
        assert first.raw_text.startswith("[Running ")

    def test_image_generation_never_ships_the_image_payload(self):
        _, completed, *_ = CODEX_ITEMS["imageGeneration"]
        t = _codex()
        asyncio.run(_item(t, "item/completed", completed))
        (event,) = _drain(t)
        assert "BASE64" not in json.dumps(event.data)

    def test_failed_mcp_call_carries_its_error(self):
        t = _codex()
        asyncio.run(_item(t, "item/completed", {
            "type": "mcpToolCall", "id": "m2", "server": "fs", "tool": "read",
            "arguments": {}, "status": "failed", "error": {"message": "denied"}}))
        (event,) = _drain(t)
        assert event.data["result"] == "denied"
        assert event.data["is_error"] is True

    @pytest.mark.parametrize("item", [
        {"type": "mcpToolCall", "id": "bad", "server": None, "tool": None,
         "arguments": "not-a-dict", "status": "completed", "result": "garbage"},
        {"type": "dynamicToolCall", "id": "bad2", "contentItems": "nope"},
    ])
    def test_malformed_items_degrade_never_raise(self, item):
        t = _codex()
        asyncio.run(_item(t, "item/completed", item))  # must not raise
        for event in _drain(t):
            assert event.event_type == "tool_use"

    def test_unknown_and_non_tool_items_emit_nothing(self):
        t = _codex()
        async def run():
            for itype in ("plan", "sleep", "contextCompaction", "somethingNew"):
                await _item(t, "item/completed", {"type": itype, "id": "x"})
        asyncio.run(run())
        assert _drain(t) == []

    def test_command_completion_carries_its_output_as_the_result(self):
        t = _codex()
        asyncio.run(_item(t, "item/completed", {
            "type": "commandExecution", "id": "c", "command": "ls", "cwd": "/w",
            "status": "completed", "exitCode": 0, "aggregatedOutput": "a\nb"}))
        (event,) = _drain(t)
        assert event.data["result"] == "a\nb"
        assert event.data["aggregated_output"] == "a\nb"  # old key kept
        assert event.raw_text == "[Command finished (exit 0): ls]"


async def _notify(t, method, **params):
    await t._route_server_message({"jsonrpc": "2.0", "method": method, "params": params})


class TestCodexReasoning:
    def test_summary_deltas_stream_as_thinking_with_part_breaks(self):
        t = _codex()

        async def run():
            await _notify(t, "item/reasoning/summaryPartAdded", itemId="r1", summaryIndex=0)
            await _notify(t, "item/reasoning/summaryTextDelta", itemId="r1", summaryIndex=0, delta="Look")
            await _notify(t, "item/reasoning/summaryTextDelta", itemId="r1", summaryIndex=0, delta=" first")
            await _notify(t, "item/reasoning/summaryPartAdded", itemId="r1", summaryIndex=1)
            await _notify(t, "item/reasoning/summaryTextDelta", itemId="r1", summaryIndex=1, delta="Then act")
            await _notify(t, "item/reasoning/textDelta", itemId="r1", contentIndex=0, delta="RAW")
            await _item(t, "item/completed", {
                "type": "reasoning", "id": "r1",
                "summary": ["Look first", "Then act"], "content": ["RAW"]})
        asyncio.run(run())
        events = _drain(t)

        assert all(e.event_type == "thinking" for e in events)
        assert "".join(e.data["text"] for e in events) == "Look first\n\nThen act"
        assert all(e.data.get("delta") for e in events)  # not repeated whole

    def test_a_summary_seen_only_on_completion_is_emitted_whole(self):
        t = _codex()
        asyncio.run(_item(t, "item/completed", {
            "type": "reasoning", "id": "r2", "summary": ["One", "Two"], "content": []}))
        (event,) = _drain(t)
        assert event.event_type == "thinking"
        assert event.data == {"text": "One\n\nTwo"}

    def test_no_summary_renders_nothing(self):
        t = _codex()
        asyncio.run(_item(t, "item/completed", {"type": "reasoning", "id": "r3"}))
        assert _drain(t) == []


# ---------------------------------------------------------------------------
# ACP: start/progress pairing and thought chunks
# ---------------------------------------------------------------------------

acp_schema = pytest.importorskip("acp.schema")


def _acp_event(update):
    from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
    return ACPSDKTransport()._session_update_to_event(update)


class TestACP:
    def test_thought_chunks_are_thinking_deltas(self):
        event = _acp_event(acp_schema.AgentThoughtChunk(
            sessionUpdate="agent_thought_chunk",
            content=acp_schema.TextContentBlock(type="text", text="hmm")))
        assert event.event_type == "thinking"
        assert event.data == {"text": "hmm", "delta": True}

    def test_empty_thought_chunks_emit_nothing(self):
        assert _acp_event(acp_schema.AgentThoughtChunk(
            sessionUpdate="agent_thought_chunk",
            content=acp_schema.TextContentBlock(type="text", text=""))) is None

    def test_progress_with_output_is_the_rows_result(self):
        event = _acp_event(acp_schema.ToolCallProgress(
            sessionUpdate="tool_call_update", toolCallId="k1", status="completed",
            content=[acp_schema.ContentToolCallContent(
                type="content",
                content=acp_schema.TextContentBlock(type="text", text="3 files"))]))
        assert event.event_type == "tool_result"
        assert event.data["tool_call_id"] == "k1"
        assert event.data["content"] == "3 files"
        assert event.raw_text == ""

    def test_raw_output_is_the_fallback_result(self):
        event = _acp_event(acp_schema.ToolCallProgress(
            sessionUpdate="tool_call_update", toolCallId="k1",
            rawOutput={"exitCode": 0, "stdout": "ok"}))
        assert event.event_type == "tool_result"
        assert json.loads(event.data["content"]) == {"exitCode": 0, "stdout": "ok"}

    def test_progress_without_output_updates_the_row_in_place(self):
        event = _acp_event(acp_schema.ToolCallProgress(
            sessionUpdate="tool_call_update", toolCallId="k1", status="in_progress",
            rawInput={"command": "ls"}))
        assert event.event_type == "tool_use"
        assert event.raw_text == ""  # never counted as a second call
        assert event.data["tool_input"] == {"command": "ls"}

    def test_many_progress_updates_one_row_with_the_result(self):
        start = acp_schema.ToolCallStart(
            sessionUpdate="tool_call", toolCallId="k1", title="Terminal", status="pending")
        updates = [
            acp_schema.ToolCallProgress(sessionUpdate="tool_call_update", toolCallId="k1",
                                        status="in_progress", rawInput={"command": "ls"}),
            acp_schema.ToolCallProgress(sessionUpdate="tool_call_update", toolCallId="k1",
                                        status="in_progress"),
            acp_schema.ToolCallProgress(sessionUpdate="tool_call_update", toolCallId="k1",
                                        status="completed", rawOutput="a\nb"),
        ]
        rows = []
        for sec, update in enumerate([start, *updates]):
            chunk = transport_event_to_chunk(_acp_event(update))
            rows.append(_row(chunk.chunk_type, chunk.text, ts=_ts(sec),
                             meta=worker_display_meta(chunk.chunk_type, chunk.metadata)))

        summary = _summarize_turn(rows)

        assert len(summary["tool_rows"]) == 1
        row = summary["tool_rows"][0]
        assert row["name"] == "Terminal"
        assert row["tool_call_id"] == "k1"
        assert row["arguments"] == {"command": "ls"}
        assert row["result_preview"] == "a\nb"
        assert row["duration_seconds"] == 3.0


# ---------------------------------------------------------------------------
# display metadata normalization
# ---------------------------------------------------------------------------

class TestDisplayMeta:
    def test_sdk_tool_use(self):
        assert worker_display_meta("tool_activity", {
            "tool_name": "Bash", "tool_id": "t1", "tool_input": {"command": "ls"},
        }) == {"tool_call_id": "t1", "tool_name": "Bash", "arguments": {"command": "ls"}}

    def test_tool_result_is_capped_like_the_capsule(self):
        big = "\n".join(f"line {i}" for i in range(100))
        out = worker_display_meta("tool_result", {
            "tool_call_id": "t1", "content": big, "is_error": False})
        assert out["tool_call_id"] == "t1"
        assert out["result_preview"].count("\n") == 11
        assert out["result_total_lines"] == 100
        assert out["is_error"] is False

    def test_pi_completion_status_marks_error(self):
        out = worker_display_meta("tool_activity", {
            "tool_name": "bash", "tool_id": "p1", "tool_input": {}, "status": "error",
            "result": "boom"})
        assert out["result_preview"] == "boom"
        assert out["is_error"] is True

    def test_long_argument_strings_are_cut(self):
        out = worker_display_meta("tool_activity", {
            "tool_name": "Write", "tool_id": "w",
            "tool_input": {"file_path": "a.txt", "content": "x" * 50_000,
                           "nested": {"list": list(range(500))}}})
        assert out["arguments"]["file_path"] == "a.txt"
        assert len(out["arguments"]["content"]) <= 1001
        assert len(out["arguments"]["nested"]["list"]) == 50

    def test_non_dict_input_is_wrapped(self):
        out = worker_display_meta("tool_activity", {"tool_call_id": "a", "tool_input": "ls"})
        assert out["arguments"] == {"input": "ls"}

    def test_thinking(self):
        assert worker_display_meta("thinking", {"text": "t", "delta": True}) == {"delta": True}
        assert worker_display_meta("thinking", {"text": "t"}) == {}


# ---------------------------------------------------------------------------
# ActivityTranslator.on_worker_chunk — live broadcast + thinking throttle
# ---------------------------------------------------------------------------

def _translator():
    ws = _WS()
    return ActivityTranslator(ws), ws


def _call(tr, chunk_type, display, text=""):
    tr.on_worker_chunk(chunk_type, display, text, "p1", session_id="s1", handle="claude-code")


class TestTranslatorTools:
    def test_tool_call_rides_agent_output_so_old_frontends_drop_it(self):
        tr, ws = _translator()
        _call(tr, "tool_activity", {"tool_call_id": "t1", "tool_name": "Read",
                                    "arguments": {"file_path": "a"}})
        (event,) = ws.events
        assert event["type"] == "agent.activity"
        assert event["category"] == "agent_output"
        assert event["worker_event"] == "tool_call"
        assert event["tool_name"] == "Read"
        assert event["arguments"] == {"file_path": "a"}
        assert event["tool_call_id"] == "t1"
        assert event["source"] == "claude-code"
        assert event["session_id"] == "s1"

    def test_result_pairs_by_id(self):
        tr, ws = _translator()
        _call(tr, "tool_activity", {"tool_call_id": "t1", "tool_name": "Read"})
        _call(tr, "tool_result", {"tool_call_id": "t1", "result_preview": "hi", "is_error": False})
        result = ws.events[-1]
        assert result["worker_event"] == "tool_result"
        assert result["tool_call_id"] == "t1"
        assert result["tool_name"] == "Read"
        assert result["result_preview"] == "hi"
        assert len(ws.events) == 2

    def test_codex_completion_is_a_result_not_a_second_row(self):
        tr, ws = _translator()
        _call(tr, "tool_activity", {"tool_call_id": "c1", "tool_name": "commandExecution",
                                    "arguments": {"command": "ls"}, "status": "inProgress"})
        _call(tr, "tool_activity", {"tool_call_id": "c1", "tool_name": "commandExecution",
                                    "arguments": {"command": "ls"}, "status": "completed",
                                    "result_preview": "a", "is_error": False})
        assert [e["worker_event"] for e in ws.events] == ["tool_call", "tool_result"]

    def test_status_only_progress_is_not_rebroadcast(self):
        tr, ws = _translator()
        _call(tr, "tool_activity", {"tool_call_id": "k", "tool_name": "Terminal"})
        _call(tr, "tool_activity", {"tool_call_id": "k", "status": "in_progress"})
        _call(tr, "tool_activity", {"tool_call_id": "k", "status": "in_progress"})
        assert len(ws.events) == 1

    def test_late_arguments_update_the_row(self):
        tr, ws = _translator()
        _call(tr, "tool_activity", {"tool_call_id": "k", "tool_name": "Terminal"})
        _call(tr, "tool_activity", {"tool_call_id": "k", "arguments": {"command": "ls"}})
        assert [e["worker_event"] for e in ws.events] == ["tool_call", "tool_call"]
        assert ws.events[-1]["tool_call_id"] == "k"
        assert ws.events[-1]["arguments"] == {"command": "ls"}

    def test_never_a_sub_agent_message(self):
        tr, ws = _translator()
        _call(tr, "tool_activity", {"tool_call_id": "t1", "tool_name": "Read"})
        _call(tr, "tool_result", {"tool_call_id": "t1", "result_preview": ""})
        _call(tr, "thinking", {}, "thought")
        assert not ws.of("chat.sub_agent_message")


class TestTranslatorThinking:
    def test_thinking_rides_stream_delta_marked_as_worker(self):
        tr, ws = _translator()
        _call(tr, "thinking", {}, "Plan it.")  # no running loop: flushes now
        (event,) = ws.events
        assert event["type"] == "chat.stream_delta"
        assert event["text"] == ""
        assert event["reasoning_content"] == "Plan it."
        assert event["source"] == "claude-code"
        assert event["worker"] is True
        assert event["is_final"] is False

    def test_deltas_are_coalesced_to_about_four_a_second(self):
        tr, ws = _translator()

        async def run():
            for i in range(20):
                _call(tr, "thinking", {"delta": True}, f"{i},")
            await asyncio.sleep(0.4)
        asyncio.run(run())

        deltas = ws.of("chat.stream_delta")
        assert len(deltas) == 2  # the first at once, the rest in one deferred flush
        assert "".join(d["reasoning_content"] for d in deltas) == "".join(
            f"{i}," for i in range(20))

    def test_pending_thinking_flushes_before_the_next_tool_row(self):
        tr, ws = _translator()

        async def run():
            _call(tr, "thinking", {"delta": True}, "a")
            _call(tr, "thinking", {"delta": True}, "b")  # deferred
            _call(tr, "tool_activity", {"tool_call_id": "t", "tool_name": "Read"})
        asyncio.run(run())

        kinds = [(e["type"], e.get("reasoning_content")) for e in ws.events]
        assert kinds == [("chat.stream_delta", "a"), ("chat.stream_delta", "b"),
                         ("agent.activity", None)]

    def test_turn_close_flushes_and_resets(self):
        tr, ws = _translator()

        async def run():
            _call(tr, "thinking", {"delta": True}, "a")
            _call(tr, "thinking", {"delta": True}, "b")
            tr.on_worker_turn_closed("p1", session_id="s1", handle="claude-code")
            await asyncio.sleep(0.3)  # the cancelled timer must not re-send
        asyncio.run(run())

        assert [e["reasoning_content"] for e in ws.of("chat.stream_delta")] == ["a", "b"]
        assert tr._workers == {}

    def test_back_to_back_whole_blocks_are_paragraphs(self):
        tr, ws = _translator()
        _call(tr, "thinking", {}, "one")
        _call(tr, "thinking", {}, "two")
        assert "".join(e["reasoning_content"] for e in ws.events) == "one\n\ntwo"


# ---------------------------------------------------------------------------
# ProcessManager: the choke point, end to end through a real transcript
# ---------------------------------------------------------------------------

class _Chunk:
    def __init__(self, text, chunk_type, metadata=None):
        self.text = text
        self.chunk_type = chunk_type
        self.timestamp = ""
        self.metadata = metadata or {}


class _Adapter:
    def __init__(self, chunks, gate=None):
        self._chunks = chunks
        self._gate = gate
        self._transport = None

    async def read_stream(self):
        for chunk in self._chunks:
            if chunk == "GATE":
                await self._gate.wait()
                continue
            yield chunk


class _Lifecycle:
    def __init__(self):
        self.completed = []

    async def on_completed(self, p, h, summary, transcript_path, *, session_id=None):
        self.completed.append(summary)

    async def on_error(self, *a, **k):
        pass

    async def on_thread_update(self, *a, **k):
        pass


def _sdk_run_chunks():
    events = [
        TransportEvent("thinking", {"text": "Read first."}, "Read first."),
        TransportEvent("tool_use", {"tool_name": "Read", "tool_id": "tu1",
                                    "tool_input": {"file_path": "notes.txt"}},
                       "[Using tool: Read]"),
        TransportEvent("tool_result", {"tool_call_id": "tu1", "content": "hello",
                                       "is_error": False}),
        TransportEvent("message", {"text": "It says hello."}, "It says hello."),
    ]
    return [transport_event_to_chunk(e) for e in events]


class TestProcessManagerRoundTrip:
    def test_rows_persist_metadata_and_broadcast_live_without_status_refetch_traps(self, tmp_path):
        ws = _WS()
        translator = ActivityTranslator(ws)
        lifecycle = _Lifecycle()
        pm = ProcessManager(ws, translator, lifecycle)
        transcript = SubAgentTranscript(str(tmp_path), "claude-code", "t1")
        pm.set_active_dispatch("p1", "claude-code", "D1", session_id="s1")
        chunks = _sdk_run_chunks() + [
            _Chunk("", "turn_complete", {"cause": "success"})]

        async def run():
            await pm.start("p1", "claude-code", _Adapter(chunks),
                           transcript=transcript, session_id="s1")
            await pm._tasks[pm._key("p1", "s1", "claude-code")]
        asyncio.run(run())

        rows = SubAgentTranscript.read(transcript.filepath)
        by_type = {r["chunk_type"]: r for r in rows}
        # The text contract stays exactly as it was.
        assert by_type["tool_activity"]["content"] == "[Using tool: Read]"
        assert by_type["tool_activity"]["metadata"] == {
            "tool_call_id": "tu1", "tool_name": "Read",
            "arguments": {"file_path": "notes.txt"}}
        assert by_type["tool_result"]["metadata"]["result_preview"] == "hello"
        assert by_type["thinking"]["content"] == "Read first."
        assert by_type["response"].get("metadata") is None
        assert all(r.get("dispatch_id") == "D1" for r in rows)

        # Live: one sub_agent_message (the reply), tool rows + thinking on
        # agent.activity / stream_delta only.
        assert [e["content"] for e in ws.of("chat.sub_agent_message")] == ["It says hello."]
        workers = [e for e in ws.of("agent.activity") if e.get("worker_event")]
        assert [e["worker_event"] for e in workers] == ["tool_call", "tool_result"]
        assert [e["reasoning_content"] for e in ws.of("chat.stream_delta")] == ["Read first."]
        # The summary is still the reply, never a tool result or a thought.
        assert lifecycle.completed == ["It says hello."]

        (turn,) = read_sub_agent_summary(transcript.filepath)
        assert turn["dispatch_id"] == "D1"
        assert turn["tool_rows"][0]["arguments"] == {"file_path": "notes.txt"}
        assert turn["tool_rows"][0]["result_preview"] == "hello"
        assert turn["thinking"] == [{"content": "Read first.", "after_tool": 0}]
        assert turn["response"] == "It says hello."

    def test_mid_run_reload_reads_the_in_flight_turn(self, tmp_path):
        ws = _WS()
        pm = ProcessManager(ws, ActivityTranslator(ws), _Lifecycle())
        transcript = SubAgentTranscript(str(tmp_path), "claude-code", "t1")
        pm.set_active_dispatch("p1", "claude-code", "D2", session_id="s1")
        gate = asyncio.Event()

        async def run():
            chunks = _sdk_run_chunks()[:3] + ["GATE"]
            await pm.start("p1", "claude-code", _Adapter(chunks, gate),
                           transcript=transcript, session_id="s1")
            await asyncio.sleep(0.05)
            mid_run = read_sub_agent_summary(transcript.filepath, include_in_flight=True)
            # Without the flag: the legacy flat-file turn, joinable to nothing.
            assert [t["dispatch_id"] for t in read_sub_agent_summary(
                transcript.filepath)] == [None]
            gate.set()
            await pm._tasks[pm._key("p1", "s1", "claude-code")]
            return mid_run
        (turn,) = asyncio.run(run())

        assert turn["in_flight"] is True
        assert turn["dispatch_id"] == "D2"
        assert [r["name"] for r in turn["tool_rows"]] == ["Read"]
        assert turn["tool_rows"][0]["result_preview"] == "hello"


# ---------------------------------------------------------------------------
# transcript reader: metadata-first rows, regex fallback, legacy thoughts
# ---------------------------------------------------------------------------

class TestSummarizeTurn:
    def test_codex_rows_appear_from_metadata(self):
        """The latent bug: `[Running command: …]` never matched the regex, so
        codex capsules had zero rows."""
        rows = [
            _row("tool_activity", "[Running command: ls]", ts=_ts(0), meta={
                "tool_call_id": "c1", "tool_name": "commandExecution",
                "arguments": {"command": "ls", "cwd": "/w"}, "status": "inProgress"}),
            _row("tool_activity", "[Command finished (exit 0): ls]", ts=_ts(2), meta={
                "tool_call_id": "c1", "tool_name": "commandExecution",
                "arguments": {"command": "ls", "cwd": "/w"}, "status": "completed",
                "result_preview": "a\nb", "is_error": False}),
            _row("response", "Two files.", ts=_ts(3)),
        ]

        summary = _summarize_turn(rows)

        assert len(summary["tool_rows"]) == 1
        row = summary["tool_rows"][0]
        assert row["name"] == "commandExecution"
        assert row["arguments"]["command"] == "ls"
        assert row["result_preview"] == "a\nb"
        assert row["duration_seconds"] == 2.0
        assert summary["tools_used"] == ["commandExecution"]

    def test_a_completion_carrying_fuller_arguments_wins(self):
        """Live smoke: codex's webSearch starts with an empty query and
        completes with the real one."""
        rows = [
            _row("tool_activity", "[Running webSearch]", ts=_ts(0), meta={
                "tool_call_id": "w", "tool_name": "webSearch", "arguments": {"query": ""}}),
            _row("tool_activity", "[webSearch finished]", ts=_ts(2), meta={
                "tool_call_id": "w", "tool_name": "webSearch",
                "arguments": {"query": "OpenAI Codex CLI"}, "result_preview": "", "is_error": False}),
            _row("tool_activity", "", ts=_ts(3), meta={"tool_call_id": "w", "status": "done"}),
        ]
        (row,) = _summarize_turn(rows)["tool_rows"]
        assert row["arguments"] == {"query": "OpenAI Codex CLI"}

    def test_legacy_rows_still_use_the_regex(self):
        rows = [
            _row("tool_activity", "[Using tool: Write]", ts=_ts(0)),
            _row("response", "ok", ts=_ts(4)),
        ]
        (row,) = _summarize_turn(rows)["tool_rows"]
        # Spec 104: the row is shared with ``stream_rows`` and carries its kind.
        assert row == {"kind": "tool", "name": "Write", "timestamp": _ts(0), "duration_seconds": 4.0}

    def test_thinking_blocks_sit_between_tool_rows(self):
        rows = [
            _row("thinking", "Look", meta={"delta": True}),
            _row("thinking", " around", meta={"delta": True}),
            _row("tool_activity", "[Using tool: Glob]", meta={"tool_call_id": "a", "tool_name": "Glob"}),
            _row("thinking", "First block."),
            _row("thinking", "Second block."),
            _row("thinking", "   "),
        ]
        assert _summarize_turn(rows)["thinking"] == [
            {"content": "Look around", "after_tool": 0},
            {"content": "First block.\n\nSecond block.\n\n   ", "after_tool": 1},
        ]

    def test_legacy_acp_thought_status_rows_are_backfilled(self):
        rows = [
            _row("status", "I should ", source="cursor"),
            _row("status", "list files.", source="cursor"),
            _row("status", "Plan updated", source="cursor"),
            _row("status", "", source="cursor"),
            _row("tool_activity", "[Using tool: bash]", source="cursor"),
            _row("status", "done", source="cursor"),
        ]
        assert _summarize_turn(rows)["thinking"] == [
            {"content": "I should list files.", "after_tool": 0},
            {"content": "done", "after_tool": 1},
        ]

    @pytest.mark.parametrize("handle", ["claude-code", "aider", "gemini-cli", "worker:f1-0", "unknown-agent"])
    def test_status_rows_of_non_acp_agents_are_never_thoughts(self, handle):
        """pty/pipe output parsers emit "Thinking..." / "Loading..." placeholder
        status rows; only an ACP agent's status rows were thought chunks."""
        rows = [
            _row("status", "Thinking...", source=handle),
            _row("status", "Loading project...", source=handle),
            _row("response", "ok", source=handle),
        ]
        assert _summarize_turn(rows)["thinking"] == []

    def test_every_bundled_acp_agent_is_recognized(self):
        from agent_os.daemon_v2.sub_agent_transcript import _acp_handles
        assert {"cursor", "dsh", "codebuddy"} <= _acp_handles()
        assert not {"claude-code", "codex", "pi", "aider"} & _acp_handles()

    def test_a_result_for_an_unseen_call_adds_no_row(self):
        rows = [_row("tool_result", "", meta={"tool_call_id": "x", "result_preview": "r"})]
        assert _summarize_turn(rows)["tool_rows"] == []


class TestInFlightRead:
    def _write(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_unstamped_trailing_rows_stay_dropped(self, tmp_path):
        p = tmp_path / "t.jsonl"
        self._write(p, [
            _row("response", "one"),
            {"source": "w", "content": "", "chunk_type": "turn_complete",
             "timestamp": _ts(1), "dispatch_id": "A"},
            _row("tool_activity", "[Using tool: Read]"),
        ])
        assert [t["dispatch_id"] for t in read_sub_agent_summary(str(p), include_in_flight=True)] == ["A"]

    def test_only_the_latest_stamp_is_in_flight(self, tmp_path):
        p = tmp_path / "t.jsonl"
        self._write(p, [
            _row("tool_activity", "[Using tool: Old]", dispatch_id="DEAD"),
            _row("tool_activity", "[Using tool: New]", dispatch_id="LIVE"),
        ])
        (turn,) = read_sub_agent_summary(str(p), include_in_flight=True)
        assert turn["dispatch_id"] == "LIVE"
        assert [r["name"] for r in turn["tool_rows"]] == ["New"]


# ---------------------------------------------------------------------------
# /chat interleave: the in-flight capsule and threaded thinking
# ---------------------------------------------------------------------------

def _marker(path, dispatch_id, handle="codex"):
    return {"role": "system", "source": "daemon", "timestamp": _ts(0),
            "content": f'[Sub-agent] Message sent to {handle}: "x". Transcript: {path}',
            "_meta": {"dispatch_id": dispatch_id, "handle": handle,
                      "transcript_path": str(path)}}


class TestInterleave:
    def _write(self, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_in_flight_dispatch_renders_its_capsule_so_far(self, tmp_path):
        from agent_os.api.routes.agents_v2 import _interleave_sub_agent_summaries
        p = tmp_path / "codex.jsonl"
        self._write(p, [
            _row("thinking", "Plan.", dispatch_id="D"),
            _row("tool_activity", "[Running command: ls]", dispatch_id="D", meta={
                "tool_call_id": "c1", "tool_name": "commandExecution",
                "arguments": {"command": "ls"}}),
            _row("response", "Halfway there", dispatch_id="D"),
        ])

        out = _interleave_sub_agent_summaries([_marker(p, "D")])

        (sub,) = [m for m in out if m.get("source") == "sub_agent"]
        assert sub["sub_agent_in_flight"] is True
        assert sub["content"] == ""  # partial text is not the answer
        assert sub["sub_agent_tool_rows"][0]["arguments"] == {"command": "ls"}
        assert sub["sub_agent_thinking"] == [{"content": "Plan.", "after_tool": 0}]

    def test_completed_turn_threads_thinking_only_when_present(self, tmp_path):
        from agent_os.api.routes.agents_v2 import _interleave_sub_agent_summaries
        p = tmp_path / "cc.jsonl"
        self._write(p, [
            _row("tool_activity", "[Using tool: Read]"),
            _row("response", "Done."),
            {"source": "w", "content": "", "chunk_type": "turn_complete",
             "timestamp": _ts(2), "dispatch_id": "D"},
        ])

        out = _interleave_sub_agent_summaries([_marker(p, "D", "claude-code")])

        (sub,) = [m for m in out if m.get("source") == "sub_agent"]
        assert sub["content"] == "Done."
        assert "sub_agent_thinking" not in sub
        assert "sub_agent_in_flight" not in sub
