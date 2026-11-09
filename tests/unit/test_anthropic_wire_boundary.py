# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 165 / issue #64: the Anthropic wire never carries assistant text the
Messages API rejects.

The API answers 400 to an empty text block ("text content blocks must be
non-empty") anywhere, and to a whitespace-only one ("... must contain
non-whitespace text") when it is the turn's only content. The empty-response
guard saves exactly such a row, and it stays in the session, so the fix is in
translation: every call translates the full history, which heals sessions
already on disk without rewriting them.

Whitespace BESIDE a tool_use block (what OpenAI-compatible models write before
their tool calls) is accepted: checked on the wire 2026-11-09 against Claude
Haiku 5.5 over OpenCode Go's /messages endpoint. It is kept byte for byte, so
the cached prefix of those sessions doesn't move.
"""

import copy
import json
from types import SimpleNamespace

import pytest

from agent_os.agent.context import _RUNTIME_MARKER, _RUNTIME_SEP
from agent_os.agent.providers.anthropic_adapter import translate_messages_to_anthropic

SYS = [{"role": "system", "content": "static prompt"},
       {"role": "system", "content": "semi stable"},
       {"role": "system", "content": "layer one"}]
TAIL = f"{_RUNTIME_MARKER}\ntime=1"
NUDGE = "Your previous reply contained no visible text. Answer the user."


def _api_rejections(messages: list[dict]) -> list[str]:
    """The Messages API's text checks, as observed on the wire (2026-11-09)."""
    errors = []
    for i, m in enumerate(messages):
        content = m["content"]
        if isinstance(content, list) and not content:
            errors.append(f"messages.{i}: content must be non-empty")
        blocks = content if isinstance(content, list) else []
        has_tool_use = any(b.get("type") == "tool_use" for b in blocks)
        for b in blocks:
            if b.get("type") != "text":
                continue
            if not b["text"]:
                errors.append(f"messages.{i}: text content blocks must be non-empty")
            elif not b["text"].strip() and not has_tool_use:
                errors.append(f"messages.{i}: text content blocks must contain non-whitespace text")
    return errors


def _markers(t):
    out = []
    for b in t["system"] or []:
        if isinstance(b, dict) and "cache_control" in b:
            out.append(("system", b["text"]))
    for m in t["messages"]:
        for b in (m["content"] if isinstance(m["content"], list) else []):
            if "cache_control" in b:
                out.append((m["role"], b.get("text") or b.get("tool_use_id") or b.get("id")))
    return out


# ---------------------------------------------------------------------------
# 1. The reporter's sequence: blank row, untagged nudge, runtime tail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blank", ["", None, "\n\n", " \t\n"])
@pytest.mark.parametrize("cache", [False, True])
def test_a_blank_assistant_row_is_left_off_the_wire(blank, cache):
    msgs = [
        {"role": "user", "content": "summarise the notes"},
        {"role": "assistant", "content": blank},       # what the guard persists
        {"role": "system", "content": NUDGE},          # the guard's nudge
        {"role": "user", "content": TAIL},
    ]
    t = translate_messages_to_anthropic(msgs, cache=cache)

    assert [m["role"] for m in t["messages"]] == ["user", "user"]
    assert _api_rejections(t["messages"]) == []
    system_text = t["system"] if isinstance(t["system"], str) else "".join(
        b["text"] for b in t["system"])
    assert NUDGE in system_text


def test_a_blank_assistant_row_with_an_empty_tool_calls_list_is_left_off():
    t = translate_messages_to_anthropic([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "user", "content": "hi again"},
    ])
    assert [m["role"] for m in t["messages"]] == ["user", "user"]


# ---------------------------------------------------------------------------
# 2. Whitespace beside a tool call (written by OpenAI-compatible models)
# ---------------------------------------------------------------------------

def _two_reads(text):
    return [
        {"role": "user", "content": "read both"},
        {"role": "assistant", "content": text, "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "a.py"}'}},
            {"id": "t2", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "b.py"}'}},
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": "A"},
        {"role": "tool", "tool_call_id": "t2", "content": "B"},
    ]


_TOOL_USES = [
    {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}},
    {"type": "tool_use", "id": "t2", "name": "read", "input": {"path": "b.py"}},
]


@pytest.mark.parametrize("ws", ["\n\n", "\n", "\n\n\n\n"])
def test_whitespace_beside_tool_calls_goes_out_unchanged(ws):
    """Accepted by the API, so the bytes (and the cached prefix) stay put."""
    t = translate_messages_to_anthropic(_two_reads(ws))

    assert t["messages"][1] == {"role": "assistant", "content": [
        {"type": "text", "text": ws}, *_TOOL_USES]}
    assert [b["tool_use_id"] for b in t["messages"][2]["content"]] == ["t1", "t2"]
    assert _api_rejections(t["messages"]) == []


@pytest.mark.parametrize("empty", ["", None])
def test_no_text_beside_tool_calls_sends_only_the_tool_use_blocks(empty):
    t = translate_messages_to_anthropic(_two_reads(empty))

    assert t["messages"][1] == {"role": "assistant", "content": _TOOL_USES}
    assert t["messages"][2]["role"] == "user"
    assert [b["tool_use_id"] for b in t["messages"][2]["content"]] == ["t1", "t2"]
    assert _api_rejections(t["messages"]) == []


# ---------------------------------------------------------------------------
# 3. Real text keeps its exact bytes
# ---------------------------------------------------------------------------

def test_non_blank_text_with_surrounding_whitespace_passes_through_unstripped():
    t = translate_messages_to_anthropic([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "\nHello \n"},
    ])
    assert t["messages"][1] == {"role": "assistant",
                                "content": [{"type": "text", "text": "\nHello \n"}]}


# ---------------------------------------------------------------------------
# 4. An unaffected conversation translates exactly as before (prompt cache)
# ---------------------------------------------------------------------------

UNAFFECTED = SYS + [
    {"role": "user", "content": "find the bug"},
    {"role": "assistant", "content": "Looking.", "tool_calls": [
        {"id": "t1", "type": "function",
         "function": {"name": "read", "arguments": '{"path": "a.py"}'}}]},
    {"role": "tool", "tool_call_id": "t1", "content": "def f(): pass"},
    {"role": "assistant", "content": "Found it.\n"},
    {"role": "user", "content": "thanks" + _RUNTIME_SEP + TAIL},
]

_UNAFFECTED_MESSAGES = [
    {"role": "user", "content": "find the bug"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}},
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "def f(): pass"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "Found it.\n"}]},
]


def test_an_unaffected_conversation_is_byte_identical_without_cache():
    t = translate_messages_to_anthropic(UNAFFECTED)
    assert t["system"] == "static prompt\nsemi stable\nlayer one"
    assert t["messages"] == _UNAFFECTED_MESSAGES + [
        {"role": "user", "content": "thanks" + _RUNTIME_SEP + TAIL},
    ]


def test_an_unaffected_conversation_is_byte_identical_with_cache():
    eph = {"type": "ephemeral"}
    t = translate_messages_to_anthropic(UNAFFECTED, cache=True)
    assert t["system"] == [
        {"type": "text", "text": "static prompt", "cache_control": eph},
        {"type": "text", "text": "semi stable"},
        {"type": "text", "text": "layer one", "cache_control": eph},
    ]
    assert t["messages"] == _UNAFFECTED_MESSAGES + [
        {"role": "user", "content": [
            {"type": "text", "text": "thanks", "cache_control": eph},
            {"type": "text", "text": _RUNTIME_SEP + TAIL},
        ]},
    ]


# ---------------------------------------------------------------------------
# 5. The rolling breakpoint lands on the nearest surviving block
# ---------------------------------------------------------------------------

def test_the_breakpoint_skips_a_dropped_whitespace_row_and_lands_on_the_question():
    """Before the fix a "\\n\\n" text-only row took the rolling breakpoint and was
    itself rejected; with it off the wire, the user's question carries it."""
    msgs = SYS + [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "\n\n"},
        {"role": "user", "content": TAIL},
    ]
    t = translate_messages_to_anthropic(msgs, cache=True)
    assert _markers(t) == [("system", "static prompt"), ("system", "layer one"), ("user", "X")]
    assert len(_markers(t)) <= 4
    assert _api_rejections(t["messages"]) == []


# ---------------------------------------------------------------------------
# 6. Translation never mutates the session's rows
# ---------------------------------------------------------------------------

def test_translation_does_not_mutate_the_input_rows():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "reasoning_content": "kept"},
        {"role": "system", "content": NUDGE},
        {"role": "assistant", "content": "\n\n", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "r"},
        {"role": "user", "content": TAIL},
    ]
    before = copy.deepcopy(msgs)
    translate_messages_to_anthropic(msgs, cache=True)
    translate_messages_to_anthropic(msgs)
    assert msgs == before


# ---------------------------------------------------------------------------
# Loop-level regression: the guard's retry reaches a real Anthropic request
# ---------------------------------------------------------------------------

def _bad_request(message: str):
    import anthropic
    import httpx
    req = httpx.Request("POST", "https://example.com/v1/messages")
    return anthropic.BadRequestError(
        message, response=httpx.Response(400, request=req),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": message}})


def _text_stream(text: str):
    events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=0,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0))),
    ]
    if text:
        events += [
            SimpleNamespace(type="content_block_start", index=0,
                            content_block=SimpleNamespace(type="text")),
            SimpleNamespace(type="content_block_delta", index=0,
                            delta=SimpleNamespace(type="text_delta", text=text)),
            SimpleNamespace(type="content_block_stop", index=0),
        ]
    events += [
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"),
                        usage=SimpleNamespace(output_tokens=1)),
        SimpleNamespace(type="message_stop"),
    ]

    async def gen():
        for e in events:
            yield e
    return gen()


class _FakeMessagesAPI:
    """Stands in for the Anthropic client: applies the API's request checks and
    replays scripted replies (an empty end_turn, then text)."""

    def __init__(self, replies: list[str]):
        self._replies = replies
        self.requests: list[dict] = []
        self.rejections: list[list[str]] = []

    async def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        errors = _api_rejections(kwargs["messages"])
        self.rejections.append(errors)
        if errors:
            raise _bad_request(errors[0])
        n = sum(1 for e in self.rejections if not e)
        return _text_stream(self._replies[min(n - 1, len(self._replies) - 1)])


def _anthropic_loop(tmp_path, api, session_id="anthropic_guard"):
    from unittest.mock import MagicMock
    from agent_os.agent.providers.openai_compat import LLMProvider
    from tests.unit.test_empty_response_guard import _make_loop

    provider = LLMProvider("claude-x", "k", "https://example.com", sdk="anthropic")
    provider._anthropic_client = MagicMock()
    provider._anthropic_client.messages.create = api.create
    return _make_loop(tmp_path, provider, session_id)


@pytest.mark.asyncio
async def test_an_empty_end_turn_then_text_completes_on_the_anthropic_wire(tmp_path, monkeypatch):
    from agent_os.agent import loop as loop_mod
    from agent_os.agent.session import persist_user_row

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(loop_mod.asyncio, "sleep", _no_sleep)

    api = _FakeMessagesAPI(["", "the summary"])
    loop, session = _anthropic_loop(tmp_path, api)
    persist_user_row(loop._session, "summarise the notes")
    await loop.run()

    assert api.rejections == [[], []], api.rejections
    assert loop._loop_exit_path == "text_complete"
    assistants = [m for m in session.get_messages() if m.get("role") == "assistant"]
    assert assistants[-1]["content"] == "the summary"
    # The blank row is still on disk, unchanged: healing happens on the wire.
    assert any(m.get("role") == "assistant" and not (m.get("content") or "").strip()
               for m in session.get_messages())


@pytest.mark.asyncio
async def test_a_session_already_poisoned_on_disk_answers_on_the_next_message(tmp_path, monkeypatch):
    """Sessions written before the fix hold a blank row and a "\\n\\n"-beside-a-
    tool-call row. The next message must answer, with no rewrite of the file."""
    from agent_os.agent import loop as loop_mod
    from agent_os.agent.session import persist_user_row

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(loop_mod.asyncio, "sleep", _no_sleep)

    api = _FakeMessagesAPI(["back on track"])
    loop, session = _anthropic_loop(tmp_path, api, "poisoned")
    session.append({"role": "user", "content": "check a.py"})
    session.append({"role": "assistant", "content": "\n\n", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.py"}'}}]})
    session.append({"role": "tool", "tool_call_id": "t1", "content": "print(1)"})
    session.append({"role": "assistant", "content": ""})
    session.append({"role": "system", "content": NUDGE})
    with open(session._filepath, encoding="utf-8") as f:
        on_disk_before = f.read()

    persist_user_row(loop._session, "are you there?")
    await loop.run()

    assert api.rejections == [[]], api.rejections
    assert loop._loop_exit_path == "text_complete"
    with open(session._filepath, encoding="utf-8") as f:
        assert f.read().startswith(on_disk_before)


# ---------------------------------------------------------------------------
# Images the API can't take (saved before the read-tool fix) -> a text note
# ---------------------------------------------------------------------------

_ACCEPTED_MEDIA = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _image_rejections(messages: list[dict]) -> list[str]:
    errors = []

    def walk(blocks, where):
        for j, b in enumerate(blocks if isinstance(blocks, list) else []):
            if b.get("type") == "image" and b["source"].get("media_type") not in _ACCEPTED_MEDIA:
                errors.append(f"{where}.{j}.image.source.base64.media_type")
            if b.get("type") == "tool_result":
                walk(b.get("content"), f"{where}.{j}.tool_result.content")

    for i, m in enumerate(messages):
        walk(m["content"], f"messages.{i}.content")
    return errors


def _image_read(tool_call_id, mime, b64="QUJD"):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": [
        {"type": "text", "text": f"Image file: x, 3B, {mime}"},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}},
    ]}


def _read_call(tool_call_id):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": tool_call_id, "type": "function",
         "function": {"name": "read", "arguments": '{"path": "x"}'}}]}


@pytest.mark.parametrize("mime", ["image/svg+xml", "image/bmp", "image/x-icon", "image/tiff"])
@pytest.mark.parametrize("cache", [False, True])
def test_a_saved_unviewable_image_goes_out_as_the_image_omitted_note(mime, cache):
    msgs = [{"role": "user", "content": "look"}, _read_call("t1"), _image_read("t1", mime),
            {"role": "user", "content": TAIL}]
    before = copy.deepcopy(msgs)
    t = translate_messages_to_anthropic(msgs, cache=cache)

    result = t["messages"][2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert result["content"][0]["text"] == f"Image file: x, 3B, {mime}"
    # Same note the OpenAI path writes when it flattens an image.
    assert {k: v for k, v in result["content"][1].items() if k != "cache_control"} == {
        "type": "text", "text": "[image omitted]"}
    assert _image_rejections(t["messages"]) == []
    assert msgs == before


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg", "image/gif", "image/webp"])
def test_an_accepted_image_goes_out_unchanged(mime):
    t = translate_messages_to_anthropic(
        [{"role": "user", "content": "look"}, _read_call("t1"), _image_read("t1", mime)])
    assert t["messages"][2]["content"][0]["content"][1] == {
        "type": "image", "source": {"type": "base64", "media_type": mime, "data": "QUJD"}}


def test_an_anthropic_format_image_with_an_unaccepted_type_is_also_replaced():
    t = translate_messages_to_anthropic([{"role": "user", "content": [
        {"type": "text", "text": "see"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/svg+xml", "data": "QUJD"}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
    ]}])
    assert t["messages"][0]["content"] == [
        {"type": "text", "text": "see"},
        {"type": "text", "text": "[image omitted]"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
    ]


@pytest.mark.asyncio
async def test_a_session_stuck_behind_a_saved_svg_answers_on_the_next_message(tmp_path, monkeypatch):
    from agent_os.agent import loop as loop_mod
    from agent_os.agent.session import persist_user_row

    async def _no_sleep(_s):
        return None
    monkeypatch.setattr(loop_mod.asyncio, "sleep", _no_sleep)

    api = _FakeMessagesAPI(["it draws a circle"])
    original = api.create

    async def create(**kwargs):
        errors = _image_rejections(kwargs["messages"])
        if errors:
            api.rejections.append(errors)
            raise _bad_request(errors[0])
        return await original(**kwargs)
    loop, session = _anthropic_loop(tmp_path, api, "svg_stuck")
    loop._provider._anthropic_client.messages.create = create
    session.append({"role": "user", "content": "what does logo.svg draw?"})
    session.append(_read_call("t1"))
    session.append(_image_read("t1", "image/svg+xml"))
    session.append({"role": "system", "content": "LLM error after 3 retries: ... media_type ..."})

    persist_user_row(loop._session, "try again")
    await loop.run()

    assert api.rejections == [[]], api.rejections
    assert loop._loop_exit_path == "text_complete"
