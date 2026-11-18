# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenAI Responses API as a third wire protocol (sdk ``openai-responses``).

The bug: ``gpt-6.2-sol`` on the OpenAI provider answered every agent turn with

    400 - Function tools with reasoning_effort are not supported for
    gpt-6.2-sol in /v1/chat/completions. To use function tools, use
    /v1/responses or set reasoning_effort to 'none'.

and for that model ``reasoning_effort: "none"`` is rejected too, so there is no
Chat Completions request that works. Starting with GPT-5.4 OpenAI refuses tools
+ reasoning on Chat Completions; the Responses API is the only path that keeps
reasoning on. These tests pin the transport against a mocked client: request
shape, reasoning-item continuity, stream-event mapping, and that a session can
move between protocols in either direction and still produce a valid request.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.responses import Response

from agent_os.agent.providers.openai_compat import LLMProvider, _SUMMARY_REJECTED
from agent_os.agent.providers.responses_adapter import (
    RESPONSES_SDK,
    RESPONSES_STATE_FIELD,
    messages_to_responses_input,
    tools_to_responses,
)
from agent_os.agent.providers.types import (
    ContextOverflowError,
    LLMError,
    StreamAccumulator,
)
from agent_os.config.provider_registry import ProviderRegistry, ReasoningInfo


TOOLS = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]},
    },
}]

REASONING_ON = ReasoningInfo(
    supported=True, field="reasoning_content", echo_back="none",
    enable="param:reasoning_effort=medium",
    efforts=("low", "medium", "high", "xhigh", "max"),
)
REASONING_NONE_OK = ReasoningInfo(
    supported=True, field="reasoning_content", echo_back="none",
    enable="param:reasoning_effort=medium",
    efforts=("none", "low", "medium", "high", "xhigh", "max"),
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _response(output: list, *, status="completed", usage=None, incomplete=None,
              model="gpt-6.2-sol") -> Response:
    return Response.model_validate({
        "id": "resp_1", "object": "response", "created_at": 1, "model": model,
        "status": status, "output": output, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [],
        "incomplete_details": incomplete,
        "usage": usage or {
            "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 100},
            "output_tokens_details": {"reasoning_tokens": 12},
        },
    })


REASONING_ITEM = {
    "type": "reasoning", "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Need to read the file."}],
    "encrypted_content": "ENC-1",
}
CALL_ITEM = {
    "type": "function_call", "id": "fc_1", "call_id": "call_1",
    "name": "read", "arguments": "{\"path\": \"a.txt\"}", "status": "completed",
}
MESSAGE_ITEM = {
    "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
    "phase": "commentary",
    "content": [{"type": "output_text", "text": "Reading it.", "annotations": []}],
}


def _provider(model="gpt-6.2-sol", provider="openai", reasoning=REASONING_ON,
              capabilities=None):
    p = LLMProvider(model, "sk-test", "https://api.openai.com/v1", sdk=RESPONSES_SDK,
                    reasoning=reasoning, provider=provider, capabilities=capabilities)
    calls: list[dict] = []
    p._openai_client = MagicMock()
    p._openai_client.chat.completions.create = AsyncMock(
        side_effect=AssertionError("Chat Completions must not be called"))
    return p, calls


def _capture(p, calls, result):
    async def create(**kwargs):
        calls.append(kwargs)
        if isinstance(result, Exception):
            raise result
        if callable(result):
            return result(kwargs)
        return result
    p._openai_client.responses.create = AsyncMock(side_effect=create)


class _Events:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        ev = self._events.pop(0)
        if isinstance(ev, Exception):
            raise ev
        return ev


def _ev(d: dict):
    """A real SDK event object, so attribute names are the SDK's, not ours."""
    from openai.types.responses import ResponseStreamEvent
    from pydantic import TypeAdapter
    return TypeAdapter(ResponseStreamEvent).validate_python(d)


def _tool_turn_events():
    return [
        _ev({"type": "response.created", "sequence_number": 0,
             "response": _response([], status="in_progress").model_dump()}),
        _ev({"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
             "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        _ev({"type": "response.reasoning_summary_text.delta", "sequence_number": 2,
             "item_id": "rs_1", "output_index": 0, "summary_index": 0,
             "delta": "Need to read"}),
        _ev({"type": "response.reasoning_summary_text.delta", "sequence_number": 3,
             "item_id": "rs_1", "output_index": 0, "summary_index": 0,
             "delta": " the file."}),
        _ev({"type": "response.output_item.done", "sequence_number": 4, "output_index": 0,
             "item": REASONING_ITEM}),
        _ev({"type": "response.output_item.added", "sequence_number": 5, "output_index": 1,
             "item": {**MESSAGE_ITEM, "status": "in_progress", "content": []}}),
        _ev({"type": "response.output_text.delta", "sequence_number": 6, "item_id": "msg_1",
             "output_index": 1, "content_index": 0, "delta": "Reading ", "logprobs": []}),
        _ev({"type": "response.output_text.delta", "sequence_number": 7, "item_id": "msg_1",
             "output_index": 1, "content_index": 0, "delta": "it.", "logprobs": []}),
        _ev({"type": "response.output_item.done", "sequence_number": 8, "output_index": 1,
             "item": MESSAGE_ITEM}),
        _ev({"type": "response.output_item.added", "sequence_number": 9, "output_index": 2,
             "item": {**CALL_ITEM, "arguments": "", "status": "in_progress"}}),
        _ev({"type": "response.function_call_arguments.delta", "sequence_number": 10,
             "item_id": "fc_1", "output_index": 2, "delta": "{\"path\": "}),
        _ev({"type": "response.function_call_arguments.delta", "sequence_number": 11,
             "item_id": "fc_1", "output_index": 2, "delta": "\"a.txt\"}"}),
        _ev({"type": "response.function_call_arguments.done", "sequence_number": 12,
             "item_id": "fc_1", "output_index": 2, "name": "read",
             "arguments": "{\"path\": \"a.txt\"}"}),
        _ev({"type": "response.output_item.done", "sequence_number": 13, "output_index": 2,
             "item": CALL_ITEM}),
        _ev({"type": "response.completed", "sequence_number": 14,
             "response": _response([REASONING_ITEM, MESSAGE_ITEM, CALL_ITEM]).model_dump()}),
    ]


async def _run_stream(p, messages, tools=None):
    acc = StreamAccumulator()
    chunks = []
    async for chunk in p.stream(messages, tools=tools):
        chunks.append(chunk)
        acc.add(chunk)
    return acc.finalize(), chunks


HISTORY = [
    {"role": "system", "content": "You are Orbital.", "timestamp": "t", "source": "x"},
    {"role": "user", "content": "Summarise a.txt", "nonce": "n1"},
    {"role": "assistant", "content": "Reading it.", "source": "management",
     "_status": "Using read",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "read", "arguments": "{\"path\": \"a.txt\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "hello world", "_meta": {"x": 1}},
    {"role": "assistant", "content": "It says hello world."},
    {"role": "user", "content": "Thanks"},
]


# --------------------------------------------------------------------------
# The bug
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gpt_6_2_sol_with_tools_goes_to_responses_not_chat_completions():
    """Regression for the 400 'Function tools with reasoning_effort are not
    supported for gpt-6.2-sol in /v1/chat/completions': the catalog routes the
    model to the Responses API, and a turn with tools is sent there."""
    info = ProviderRegistry().get_model_info("openai", "gpt-6.2-sol")
    assert info.sdk == RESPONSES_SDK
    p = LLMProvider("gpt-6.2-sol", "sk-test", info.base_url or "https://api.openai.com/v1",
                    sdk=info.sdk, reasoning=info.reasoning, provider="openai",
                    capabilities=info.capabilities)
    p._openai_client = MagicMock()
    p._openai_client.chat.completions.create = AsyncMock(
        side_effect=AssertionError("Chat Completions must not be called"))
    calls: list = []
    _capture(p, calls, _response([CALL_ITEM]))

    resp = await p.complete(HISTORY, tools=TOOLS)

    p._openai_client.chat.completions.create.assert_not_called()
    assert len(calls) == 1
    body = calls[0]
    assert body["model"] == "gpt-6.2-sol"
    assert body["tools"][0]["name"] == "read"
    assert "messages" not in body and "reasoning_effort" not in body
    assert resp.has_tool_calls and resp.tool_calls[0]["id"] == "call_1"


# --------------------------------------------------------------------------
# Request shape
# --------------------------------------------------------------------------

def test_tools_are_flattened_to_the_responses_function_shape():
    out = tools_to_responses(TOOLS)
    assert out == [{
        "type": "function", "name": "read", "description": "Read a file",
        "parameters": TOOLS[0]["function"]["parameters"],
        # Responses attempts strict mode when strict is omitted; Orbital's
        # schemas are written for Chat Completions' non-strict default.
        "strict": False,
    }]


@pytest.mark.asyncio
async def test_multi_turn_history_with_tool_calls_and_results():
    p, calls = _provider()
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete(HISTORY, tools=TOOLS)
    body = calls[0]
    assert body["store"] is False
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["stream"] is False
    assert body["input"] == [
        {"role": "system", "content": "You are Orbital."},
        {"role": "user", "content": "Summarise a.txt"},
        {"role": "assistant", "content": "Reading it."},
        {"type": "function_call", "call_id": "call_1", "name": "read",
         "arguments": "{\"path\": \"a.txt\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "hello world"},
        {"role": "assistant", "content": "It says hello world."},
        {"role": "user", "content": "Thanks"},
    ]
    # Orbital bookkeeping never reaches the wire.
    wire = json.dumps(body["input"])
    for leaked in ("timestamp", "nonce", "_status", "_meta", "source"):
        assert leaked not in wire


@pytest.mark.asyncio
async def test_tools_are_omitted_when_there_are_none():
    p, calls = _provider()
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete([{"role": "user", "content": "hi"}])
    assert "tools" not in calls[0]


@pytest.mark.asyncio
async def test_images_translate_to_input_image_parts():
    from agent_os.config.provider_registry import ModelCapabilities
    p, calls = _provider(capabilities=ModelCapabilities(vision=True))
    _capture(p, calls, _response([MESSAGE_ITEM]))
    url = "data:image/png;base64,iVBORw0KGgo="
    await p.complete([
        {"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c9", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c9", "content": [
            {"type": "text", "text": "screenshot"},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ])
    items = calls[0]["input"]
    assert items[0]["content"] == [
        {"type": "input_text", "text": "what is this"},
        {"type": "input_image", "image_url": url, "detail": "auto"},
    ]
    assert items[2]["output"] == [
        {"type": "input_text", "text": "screenshot"},
        {"type": "input_image", "image_url": url, "detail": "auto"},
    ]


@pytest.mark.asyncio
async def test_rejected_images_fall_back_to_text_once():
    from agent_os.config.provider_registry import ModelCapabilities
    import openai as _openai
    import httpx
    p, calls = _provider(capabilities=ModelCapabilities(vision=True))
    err = _openai.BadRequestError(
        "Error code: 400 - invalid image", response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
        body=None)
    results = [err, _response([MESSAGE_ITEM])]

    async def create(**kwargs):
        calls.append(kwargs)
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    p._openai_client.responses.create = AsyncMock(side_effect=create)
    await p.complete([{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}])
    assert len(calls) == 2
    assert "input_image" not in json.dumps(calls[1]["input"])


# --------------------------------------------------------------------------
# Reasoning continuity
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reasoning_items_round_trip_in_order():
    """store:false means the server keeps nothing: the encrypted reasoning
    items must be persisted on the assistant message and replayed, in their
    original order, ahead of the function call they belong to."""
    p, calls = _provider()
    _capture(p, calls, _response([REASONING_ITEM, MESSAGE_ITEM, CALL_ITEM]))
    first = await p.complete([{"role": "user", "content": "read a.txt"}], tools=TOOLS)

    state = first.raw_message[RESPONSES_STATE_FIELD]
    assert state["provider"] == "openai" and state["model"] == "gpt-6.2-sol"
    assert first.raw_message["reasoning_content"] == "Need to read the file."
    assert first.text == "Reading it."

    # The loop persists raw_message (+ bookkeeping), then a tool result.
    persisted = {**first.raw_message, "source": "management", "_status": "Using read"}
    history = [
        {"role": "user", "content": "read a.txt"},
        json.loads(json.dumps(persisted)),  # survives the JSONL round trip
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
    ]
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete(history, tools=TOOLS)
    items = calls[-1]["input"]
    assert items[1] == {
        "type": "reasoning", "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Need to read the file."}],
        "encrypted_content": "ENC-1",
    }
    assert items[2] == {
        "type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "Reading it.", "annotations": []}],
    }
    assert items[3] == {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                        "name": "read", "arguments": "{\"path\": \"a.txt\"}"}
    assert items[4] == {"type": "function_call_output", "call_id": "call_1",
                        "output": "hello"}
    # Never sent back as a Chat-Completions field.
    assert "reasoning_content" not in json.dumps(items)


def test_reasoning_from_another_model_is_not_replayed():
    """Encrypted reasoning is bound to the model (and org) that produced it.
    Replayed elsewhere it is either dropped silently or refused, so it is not
    sent at all; the item ids go too, since a function_call carrying its id
    without the reasoning item it belonged to is itself refused."""
    msg = {
        "role": "assistant", "content": "Reading it.",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "read", "arguments": "{}"}}],
        RESPONSES_STATE_FIELD: {
            "provider": "openai", "model": "gpt-6-luna",
            "items": [REASONING_ITEM,
                      {"type": "message", "id": "msg_1", "phase": "commentary",
                       "text": "Reading it."},
                      {"type": "function_call", "id": "fc_1", "call_id": "call_1"}],
        },
    }
    items = messages_to_responses_input([msg], provider="openai", model="gpt-6.2-sol")
    assert items == [
        {"role": "assistant", "content": "Reading it.", "phase": "commentary"},
        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
    ]


def test_reasoning_whose_call_was_dropped_is_not_replayed():
    """A reasoning item must be followed by the item it produced. If history
    repair dropped that call, the reasoning goes with it."""
    msg = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_2", "type": "function",
                        "function": {"name": "read", "arguments": "{}"}}],
        RESPONSES_STATE_FIELD: {
            "provider": "openai", "model": "gpt-6.2-sol",
            "items": [REASONING_ITEM,
                      {"type": "function_call", "id": "fc_1", "call_id": "call_1"},
                      {"type": "function_call", "id": "fc_2", "call_id": "call_2"}],
        },
    }
    items = messages_to_responses_input([msg], provider="openai", model="gpt-6.2-sol")
    assert all(i.get("type") != "reasoning" for i in items)
    assert items == [{"type": "function_call", "call_id": "call_2", "name": "read",
                      "arguments": "{}"}]


@pytest.mark.asyncio
async def test_undecryptable_reasoning_is_retried_without_it():
    """A key moved to another org can no longer decrypt its old reasoning.
    The turn retries once with reasoning (and item ids) left out."""
    import openai as _openai
    import httpx
    p, calls = _provider()
    err = _openai.BadRequestError(
        "Error code: 400 - The encrypted content for item rs_1 could not be verified.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://x/responses")),
        body=None)
    results = [err, _response([MESSAGE_ITEM])]

    async def create(**kwargs):
        calls.append(kwargs)
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    p._openai_client.responses.create = AsyncMock(side_effect=create)
    msg = {"role": "assistant", "content": "ok", RESPONSES_STATE_FIELD: {
        "provider": "openai", "model": "gpt-6.2-sol",
        "items": [REASONING_ITEM, {"type": "message", "id": "msg_1", "text": "ok"}]}}
    await p.complete([{"role": "user", "content": "a"}, msg, {"role": "user", "content": "b"}])
    assert len(calls) == 2
    assert "rs_1" in json.dumps(calls[0]["input"])
    assert "rs_1" not in json.dumps(calls[1]["input"])
    assert "msg_1" not in json.dumps(calls[1]["input"])


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_events_map_to_stream_chunks():
    p, calls = _provider()
    _capture(p, calls, _Events(_tool_turn_events()))
    resp, chunks = await _run_stream(p, [{"role": "user", "content": "read a.txt"}], TOOLS)

    assert calls[0]["stream"] is True
    assert resp.text == "Reading it."
    assert resp.raw_message["reasoning_content"] == "Need to read the file."
    assert resp.tool_calls == [{"id": "call_1", "type": "function",
                                "function": {"name": "read",
                                             "arguments": "{\"path\": \"a.txt\"}"}}]
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.input_tokens == 120
    assert resp.usage.output_tokens == 30
    assert resp.usage.cache_read_tokens == 100
    assert resp.final_usage_chunk_seen is True
    finals = [c for c in chunks if c.is_final]
    assert len(finals) == 1
    state = resp.raw_message[RESPONSES_STATE_FIELD]
    assert [i["type"] for i in state["items"]] == ["reasoning", "message", "function_call"]
    assert state["items"][0]["encrypted_content"] == "ENC-1"


@pytest.mark.asyncio
async def test_stream_function_call_without_argument_deltas():
    """Some gateways send only output_item.done for a call. Its arguments
    still have to reach the accumulator."""
    p, calls = _provider()
    events = [
        _ev({"type": "response.output_item.done", "sequence_number": 1, "output_index": 0,
             "item": CALL_ITEM}),
        _ev({"type": "response.completed", "sequence_number": 2,
             "response": _response([CALL_ITEM]).model_dump()}),
    ]
    _capture(p, calls, _Events(events))
    resp, _ = await _run_stream(p, [{"role": "user", "content": "x"}], TOOLS)
    assert resp.tool_calls[0]["function"]["arguments"] == "{\"path\": \"a.txt\"}"
    assert resp.tool_calls[0]["id"] == "call_1"


@pytest.mark.asyncio
async def test_stream_output_only_in_completed_event_is_not_lost():
    """Observed live through OpenCode Go (gpt-6-luna, 2026-11-19): a turn whose
    only event was response.completed. Its message text, calls and reasoning
    summary must still reach the accumulator, exactly once."""
    p, calls = _provider()
    events = [
        _ev({"type": "response.completed", "sequence_number": 1,
             "response": _response([REASONING_ITEM, MESSAGE_ITEM, CALL_ITEM]).model_dump()}),
    ]
    _capture(p, calls, _Events(events))
    resp, _ = await _run_stream(p, [{"role": "user", "content": "x"}], TOOLS)
    assert resp.text == "Reading it."
    assert resp.raw_message["reasoning_content"] == "Need to read the file."
    assert resp.tool_calls == [{"id": "call_1", "type": "function",
                                "function": {"name": "read",
                                             "arguments": "{\"path\": \"a.txt\"}"}}]
    assert resp.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_streamed_output_is_not_repeated_by_the_completed_event():
    p, calls = _provider()
    _capture(p, calls, _Events(_tool_turn_events()))
    resp, _ = await _run_stream(p, [{"role": "user", "content": "x"}], TOOLS)
    assert resp.text == "Reading it."
    assert resp.raw_message["reasoning_content"] == "Need to read the file."
    assert len(resp.tool_calls) == 1


@pytest.mark.asyncio
async def test_stream_incomplete_reports_length():
    p, calls = _provider()
    events = [
        _ev({"type": "response.output_text.delta", "sequence_number": 1, "item_id": "m",
             "output_index": 0, "content_index": 0, "delta": "partial", "logprobs": []}),
        _ev({"type": "response.incomplete", "sequence_number": 2,
             "response": _response([], status="incomplete",
                                   incomplete={"reason": "max_output_tokens"}).model_dump()}),
    ]
    _capture(p, calls, _Events(events))
    resp, _ = await _run_stream(p, [{"role": "user", "content": "x"}])
    assert resp.finish_reason == "length"
    assert resp.text == "partial"


@pytest.mark.asyncio
async def test_stream_error_event_raises_llm_error():
    p, calls = _provider()
    _capture(p, calls, _Events([_ev({"type": "error", "sequence_number": 1,
                                     "code": "server_error", "message": "boom"})]))
    with pytest.raises(LLMError) as ei:
        await _run_stream(p, [{"role": "user", "content": "x"}])
    assert "boom" in str(ei.value)


@pytest.mark.asyncio
async def test_stream_failed_context_overflow_raises_context_overflow():
    """The SDK builds stream events without validation, so an error code
    outside its Literal list still arrives; a plain object stands in."""
    from types import SimpleNamespace as NS
    p, calls = _provider()
    failed = NS(type="response.failed", sequence_number=1, response=NS(
        status="failed", error=NS(code="context_length_exceeded",
                                  message="Your input exceeds the context window of this model.")))
    _capture(p, calls, _Events([failed]))
    with pytest.raises(ContextOverflowError):
        await _run_stream(p, [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_create_time_context_overflow_raises_context_overflow():
    import openai as _openai
    import httpx
    p, calls = _provider()
    err = _openai.BadRequestError(
        "Error code: 400 - {'error': {'code': 'context_length_exceeded', 'message': "
        "'Your input exceeds the context window of this model.'}}",
        response=httpx.Response(400, request=httpx.Request("POST", "https://x/responses")),
        body=None)
    _capture(p, calls, err)
    with pytest.raises(ContextOverflowError):
        await _run_stream(p, [{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_mid_stream_transport_error_is_classified():
    p, calls = _provider()
    _capture(p, calls, _Events([ConnectionError("reset by peer")]))
    with pytest.raises(LLMError):
        await _run_stream(p, [{"role": "user", "content": "x"}])


# --------------------------------------------------------------------------
# Reasoning summary request
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reasoning_summary_is_requested_and_dropped_if_refused():
    """Summaries make the reasoning visible in the thinking capsule. An org
    that is not verified for summaries is refused; the turn retries without
    and the model stops asking for the rest of the process."""
    import openai as _openai
    import httpx
    _SUMMARY_REJECTED.clear()
    p, calls = _provider(model="gpt-summary-test")
    err = _openai.BadRequestError(
        "Error code: 400 - Your organization must be verified to generate "
        "reasoning summaries.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://x/responses")),
        body=None)
    results = [err, _response([MESSAGE_ITEM]), _response([MESSAGE_ITEM])]

    async def create(**kwargs):
        calls.append(kwargs)
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    p._openai_client.responses.create = AsyncMock(side_effect=create)
    await p.complete([{"role": "user", "content": "x"}])
    await p.complete([{"role": "user", "content": "y"}])
    assert calls[0]["reasoning"] == {"summary": "auto"}
    assert "reasoning" not in calls[1]
    assert "reasoning" not in calls[2]
    _SUMMARY_REJECTED.clear()


# --------------------------------------------------------------------------
# Reasoning-off switch
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_off_switch_sends_effort_none_when_the_model_accepts_it():
    p, calls = _provider(model="gpt-6-luna", reasoning=REASONING_NONE_OK)
    assert p.reasoning_locked_on is False
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete([{"role": "user", "content": "x"}], disable_reasoning=True)
    assert calls[0]["reasoning"] == {"effort": "none"}
    assert "extra_body" not in calls[0]


@pytest.mark.asyncio
async def test_off_switch_is_a_no_op_when_the_model_rejects_none():
    """gpt-6.2-sol refuses effort 'none': reasoning stays on, and the provider
    reports itself locked-on so utility callers budget for it."""
    p, calls = _provider(model="gpt-6.2-sol", reasoning=REASONING_ON)
    assert p.reasoning_locked_on is True
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete([{"role": "user", "content": "x"}], disable_reasoning=True)
    assert calls[0].get("reasoning", {}).get("effort") is None
    assert "extra_body" not in calls[0]


def test_chat_completions_off_switch_is_unchanged():
    p = LLMProvider("gpt-5.2", "sk", "https://api.openai.com/v1", sdk="openai",
                    reasoning=REASONING_ON)
    assert p.reasoning_locked_on is False  # reasoning_effort=minimal mapping


# --------------------------------------------------------------------------
# Cross-protocol history (both directions)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_history_continues_on_responses():
    """A session started on DeepSeek (reasoning_content echo) continues on
    gpt-6.2-sol: its reasoning text is not a Responses item and is dropped,
    everything else translates."""
    p, calls = _provider()
    _capture(p, calls, _response([MESSAGE_ITEM]))
    history = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "reasoning_content": "deepseek thoughts",
         "tool_calls": [{"id": "call_ds", "type": "function",
                         "function": {"name": "read", "arguments": "{\"path\": "}}]},
        {"role": "tool", "tool_call_id": "call_ds", "content": "bad args"},
        {"role": "assistant", "content": "done", "reasoning_content": "more"},
        {"role": "user", "content": "next"},
    ]
    await p.complete(history, tools=TOOLS)
    items = calls[0]["input"]
    assert items == [
        {"role": "user", "content": "go"},
        # Half-written arguments are repaired to valid JSON, as on Chat.
        {"type": "function_call", "call_id": "call_ds", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_ds", "output": "bad args"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]


@pytest.mark.asyncio
async def test_responses_history_continues_on_chat_completions():
    """The reverse switch: a Chat-Completions provider must never see the
    Responses-only field (strict gateways 400 on unknown message keys)."""
    chat = LLMProvider("deepseek-v4-pro", "sk", "https://gw.example/v1", sdk="openai")
    captured: dict = {}

    async def create(**kwargs):
        captured.update(kwargs)
        msg = MagicMock(content="ok", tool_calls=None)
        msg.model_dump.return_value = {"role": "assistant", "content": "ok"}
        choice = MagicMock(message=msg, finish_reason="stop")
        return MagicMock(choices=[choice], usage=None)
    chat._openai_client = MagicMock()
    chat._openai_client.chat.completions.create = AsyncMock(side_effect=create)

    history = [
        {"role": "user", "content": "read a.txt"},
        {"role": "assistant", "content": "Reading it.", "reasoning_content": "x",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}],
         RESPONSES_STATE_FIELD: {"provider": "openai", "model": "gpt-6.2-sol",
                                 "items": [REASONING_ITEM]}},
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
    ]
    await chat.complete(history, tools=TOOLS)
    assert RESPONSES_STATE_FIELD not in json.dumps(captured["messages"])
    assert "ENC-1" not in json.dumps(captured["messages"])


def test_responses_history_continues_on_anthropic():
    from agent_os.agent.providers.anthropic_adapter import translate_messages_to_anthropic
    from agent_os.agent.providers.openai_compat import _strip_to_spec
    from agent_os.agent.providers.responses_adapter import strip_responses_state
    history = [
        {"role": "user", "content": "read a.txt"},
        {"role": "assistant", "content": "Reading it.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}],
         RESPONSES_STATE_FIELD: {"provider": "openai", "model": "m", "items": [REASONING_ITEM]}},
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
    ]
    out = translate_messages_to_anthropic(
        [strip_responses_state(_strip_to_spec(m)) for m in history], TOOLS)
    assert "ENC-1" not in json.dumps(out)


@pytest.mark.asyncio
async def test_anthropic_path_strips_responses_state():
    p = LLMProvider("claude-x", "sk", None, sdk="anthropic")
    captured: dict = {}

    async def create(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")
    p._anthropic_client = MagicMock()
    p._anthropic_client.messages.create = AsyncMock(side_effect=create)
    with pytest.raises(LLMError):
        await p.complete([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", RESPONSES_STATE_FIELD: {
                "provider": "openai", "model": "m", "items": [REASONING_ITEM]}},
            {"role": "user", "content": "c"},
        ])
    assert "ENC-1" not in json.dumps(captured["messages"], default=str)


# --------------------------------------------------------------------------
# Usage semantics
# --------------------------------------------------------------------------

def test_responses_usage_has_subset_cache_semantics():
    """Responses input_tokens INCLUDES cached tokens, like Chat Completions."""
    from agent_os.budget.normalize import ProviderSemantics
    assert ProviderSemantics.from_sdk(RESPONSES_SDK) is ProviderSemantics.OPENAI_COMPAT


@pytest.mark.asyncio
async def test_complete_usage_and_cache_audit(caplog):
    import logging
    p, calls = _provider()
    _capture(p, calls, _response([MESSAGE_ITEM]))
    with caplog.at_level(logging.INFO, logger="orbital.cache_audit"):
        resp = await p.complete([{"role": "user", "content": "x"}])
    assert resp.usage.input_tokens == 120 and resp.usage.cache_read_tokens == 100
    assert sum("[CACHE_AUDIT]" in r.message for r in caplog.records) == 1


@pytest.mark.asyncio
async def test_session_header_and_static_headers_are_sent():
    p = LLMProvider("gpt-6-luna", "sk", "https://opencode.ai/zen/go/v1", sdk=RESPONSES_SDK,
                    reasoning=REASONING_NONE_OK, provider="opencode-go",
                    session_header="x-opencode-session")
    p.bind_session_id(lambda: "sess-1")
    calls: list = []
    p._openai_client = MagicMock()
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete([{"role": "user", "content": "x"}])
    assert calls[0]["extra_headers"] == {"x-opencode-session": "sess-1"}


@pytest.mark.asyncio
async def test_request_dump_covers_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_OS_DUMP_LLM_REQUESTS", str(tmp_path))
    p, calls = _provider()
    _capture(p, calls, _response([MESSAGE_ITEM]))
    await p.complete([{"role": "user", "content": "x"}])
    dumps = list(tmp_path.glob("*.json"))
    assert len(dumps) == 1
    doc = json.loads(dumps[0].read_text())
    assert doc["body"]["input"] == [{"role": "user", "content": "x"}]
    assert "sk-test" not in dumps[0].read_text()
