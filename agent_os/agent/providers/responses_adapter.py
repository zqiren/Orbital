# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""OpenAI Responses API translation (``LLMProvider.sdk == "openai-responses"``).

Why a third protocol: from GPT-5.4 on, OpenAI refuses function tools together
with reasoning on ``/v1/chat/completions``, and GPT-6 Astra / 6.1 Sol / 6.2 Sol
refuse ``reasoning_effort: "none"`` too, so there is no Chat Completions request
an agent turn can make. OpenCode Go/Zen serve their GPT and Grok models only
over Responses.

The rest of Orbital keeps speaking Chat-Completions-shaped history. This module
translates at the wire boundary, both ways:

  history -> ``input`` items   system/developer/user/assistant messages,
                               assistant ``tool_calls`` -> ``function_call``,
                               ``role: tool`` -> ``function_call_output``,
                               ``image_url`` blocks -> ``input_image``
  response -> StreamChunk / LLMResponse, the same shapes the other two paths
                               produce

Reasoning continuity. Requests go out with ``store: false`` (the server keeps
nothing) and ask for ``reasoning.encrypted_content``. OpenAI requires reasoning
items to be passed back with the function calls they produced, in order, so the
output items are persisted on the assistant message under
``RESPONSES_STATE_FIELD``::

    {"provider": "openai", "model": "gpt-6.2-sol",
     "items": [{"type": "reasoning", "id": "rs_..", "summary": [...],
                "encrypted_content": "..."},
               {"type": "message", "id": "msg_..", "phase": "commentary",
                "text": "..."},
               {"type": "function_call", "id": "fc_..", "call_id": "call_.."}]}

Message and call entries are placeholders that fix the ORDER (and keep the ids
and ``phase``); their content is always taken from the message's own
``content`` / ``tool_calls``, which the rest of Orbital may have repaired. The
field is stripped from every Chat Completions and Anthropic request, so a
session can switch protocols mid-way.

Replay is all-or-nothing per assistant message: the reasoning items (with item
ids) go back only when they were produced by the same provider + model and
every anchor they precede is still there; otherwise the message is sent as
plain items without ids. Encrypted reasoning is bound to the model and org that
made it, and a ``function_call`` that carries its id without the reasoning item
it belonged to is itself refused.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_os.agent.providers.types import (
    ContextOverflowError,
    LLMError,
    StreamChunk,
    TokenUsage,
    plan_usage_error,
)

logger = logging.getLogger(__name__)

#: ``LLMProvider.sdk`` / registry ``sdk`` value selecting this protocol.
RESPONSES_SDK = "openai-responses"

#: Assistant-message field holding the Responses output-item state.
RESPONSES_STATE_FIELD = "responses_output"


def _g(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict key (SDK objects and plain dicts alike)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _dump(obj: Any) -> dict:
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if v is not None}
    dump = getattr(obj, "model_dump", None)
    if dump is not None:
        return dump(exclude_none=True)
    return {}


def strip_responses_state(message: dict) -> dict:
    """``message`` without the Responses-only field (copy only when present).
    Every non-Responses request path calls this."""
    if RESPONSES_STATE_FIELD not in message:
        return message
    return {k: v for k, v in message.items() if k != RESPONSES_STATE_FIELD}


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------

def tools_to_responses(tools: list | None) -> list:
    """Chat-Completions tool definitions -> Responses function tools.

    ``strict`` is sent as False unless the definition asked for it: Responses
    attempts strict mode when ``strict`` is omitted, and Orbital's schemas are
    written for Chat Completions' non-strict default.
    """
    out: list = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            entry = {"type": "function", "name": fn.get("name") or ""}
            if fn.get("description"):
                entry["description"] = fn["description"]
            entry["parameters"] = fn.get("parameters") or {"type": "object", "properties": {}}
            entry["strict"] = bool(fn.get("strict", False))
            out.append(entry)
        elif tool.get("type") == "function" and tool.get("name"):
            entry = dict(tool)  # already Responses-shaped
            entry.setdefault("strict", False)
            out.append(entry)
    return out


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "input_text", "output_text"):
                parts.append(block.get("text", "") or "")
        return "\n".join(parts)
    return str(content)


def _input_parts(content: list) -> list:
    """Chat content blocks -> Responses input content parts."""
    parts: list = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in ("text", "input_text"):
            parts.append({"type": "input_text", "text": block.get("text", "") or ""})
        elif kind == "image_url":
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            detail = (image.get("detail") if isinstance(image, dict) else None) or "auto"
            if url:
                parts.append({"type": "input_image", "image_url": url, "detail": detail})
        elif kind == "image":
            src = block.get("source") or {}
            if src.get("type") == "base64" and src.get("data"):
                media = src.get("media_type") or "image/png"
                parts.append({"type": "input_image",
                              "image_url": f"data:{media};base64,{src['data']}",
                              "detail": "auto"})
            else:
                parts.append({"type": "input_text", "text": "[image omitted]"})
        elif kind == "input_image":
            parts.append(block)
    return parts


def _message_content(content: Any) -> Any:
    if isinstance(content, list):
        parts = _input_parts(content)
        return parts if parts else ""
    return "" if content is None else content


def _tool_output(content: Any) -> Any:
    """A tool result's ``output``: a string, or a part list when it carries an
    image (``function_call_output`` accepts input_text / input_image parts)."""
    if isinstance(content, list):
        parts = _input_parts(content)
        if any(p.get("type") == "input_image" for p in parts):
            return parts
        return "\n".join(p.get("text", "") for p in parts if p.get("type") == "input_text")
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _call_item(tc: dict, item_id: str | None = None) -> dict:
    fn = tc.get("function") or {}
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args if args is not None else {})
    item = {"type": "function_call"}
    if item_id:
        item["id"] = item_id
    item.update({"call_id": tc.get("id") or "", "name": fn.get("name") or "",
                 "arguments": args})
    return item


def _message_item(text: str, phase: str | None = None, item_id: str | None = None) -> dict:
    if item_id:
        item = {"type": "message", "role": "assistant", "id": item_id,
                "status": "completed"}
        if phase:
            item["phase"] = phase
        item["content"] = [{"type": "output_text", "text": text, "annotations": []}]
        return item
    item = {"role": "assistant", "content": text}
    if phase:
        item["phase"] = phase
    return item


def _reasoning_item(entry: dict) -> dict:
    item = {"type": "reasoning", "id": entry.get("id") or "",
            "summary": entry.get("summary") or []}
    if entry.get("content"):
        item["content"] = entry["content"]
    item["encrypted_content"] = entry.get("encrypted_content")
    return item


def _assistant_items(msg: dict, *, provider: str, model: str,
                     replay_reasoning: bool) -> list:
    text = _text_of(msg.get("content"))
    calls = [tc for tc in (msg.get("tool_calls") or []) if isinstance(tc, dict)]
    state = msg.get(RESPONSES_STATE_FIELD)
    stored = state.get("items") if isinstance(state, dict) else None

    def plain() -> list:
        out = []
        if text.strip():
            out.append(_message_item(text))
        out.extend(_call_item(tc) for tc in calls)
        return out

    if not isinstance(stored, list) or not stored:
        return plain()

    faithful = (replay_reasoning
                and state.get("provider") == provider
                and state.get("model") == model)
    calls_by_id = {tc.get("id"): tc for tc in calls if tc.get("id")}
    msg_entries = [e for e in stored if isinstance(e, dict) and e.get("type") == "message"]
    per_message = bool(msg_entries) and "".join(
        e.get("text") or "" for e in msg_entries) == text

    # (kind, payload, phase, item_id) in stored order; reasoning held in
    # `pending` until the anchor it precedes is known to survive.
    seq: list[tuple] = []
    pending: list[dict] = []
    used_calls: set = set()
    text_done = False
    intact = True
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("type")
        if kind == "reasoning":
            if entry.get("encrypted_content"):
                pending.append(entry)
            else:
                intact = False
            continue
        if kind == "message":
            if per_message:
                piece = entry.get("text") or ""
            elif not text_done:
                piece = text
            else:
                piece = ""
            if piece.strip():
                seq.extend(("reasoning", r, None, None) for r in pending)
                seq.append(("message", piece, entry.get("phase"), entry.get("id")))
                text_done = True
            elif pending:
                intact = False
            pending = []
        elif kind == "function_call":
            cid = entry.get("call_id")
            tc = calls_by_id.get(cid)
            if tc is not None and cid not in used_calls:
                seq.extend(("reasoning", r, None, None) for r in pending)
                seq.append(("call", tc, None, entry.get("id")))
                used_calls.add(cid)
            elif pending:
                intact = False
            pending = []
    if pending:
        intact = False

    leftovers = False
    if text.strip() and not text_done:
        leftovers = True
        first_call = next((i for i, s in enumerate(seq) if s[0] == "call"), len(seq))
        seq.insert(first_call, ("message", text, None, None))
    for tc in calls:
        if tc.get("id") not in used_calls:
            leftovers = True
            seq.append(("call", tc, None, None))

    full = faithful and intact and not leftovers
    out: list = []
    for kind, payload, phase, item_id in seq:
        if kind == "reasoning":
            if full:
                out.append(_reasoning_item(payload))
        elif kind == "message":
            out.append(_message_item(payload, phase, item_id if full else None))
        else:
            out.append(_call_item(payload, item_id if full else None))
    return out


def messages_to_responses_input(messages: list, *, provider: str, model: str,
                                replay_reasoning: bool = True,
                                system_role: str = "system") -> list:
    """Prepared Chat-shaped rows -> Responses ``input`` items.

    ``messages`` must already be stripped of Orbital bookkeeping and repaired
    (see ``LLMProvider._prepare_rows``). Chat-only extensions on assistant rows
    (``reasoning_content``, Anthropic ``thinking``) are not Responses items and
    are dropped. ``replay_reasoning=False`` sends every assistant message as
    plain items (no reasoning, no ids) — the retry after the server could not
    decrypt old reasoning.

    ``system_role="developer"`` (registry flag ``system_role``) sends system
    rows as developer messages: the Sign in with ChatGPT route refuses
    system-role items ("System messages are not allowed").
    """
    items: list = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "developer"):
            if role == "system" and system_role == "developer":
                role = "developer"
            items.append({"role": role, "content": _message_content(msg.get("content"))})
        elif role == "user":
            items.append({"role": "user", "content": _message_content(msg.get("content"))})
        elif role == "assistant":
            items.extend(_assistant_items(msg, provider=provider, model=model,
                                          replay_reasoning=replay_reasoning))
        elif role == "tool":
            items.append({"type": "function_call_output",
                          "call_id": msg.get("tool_call_id") or "",
                          "output": _tool_output(msg.get("content"))})
    return items


def input_has_images(items: list) -> bool:
    for item in items:
        for key in ("content", "output"):
            val = item.get(key)
            if isinstance(val, list) and any(
                    isinstance(p, dict) and p.get("type") == "input_image" for p in val):
                return True
    return False


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------

def responses_token_usage(usage: Any) -> TokenUsage | None:
    """Responses usage -> TokenUsage. ``input_tokens`` INCLUDES the cached part
    (subset semantics, like Chat Completions)."""
    if usage is None:
        return None
    details = _g(usage, "input_tokens_details")
    cached = _g(details, "cached_tokens", 0) if details is not None else 0
    return TokenUsage(
        input_tokens=int(_g(usage, "input_tokens", 0) or 0),
        output_tokens=int(_g(usage, "output_tokens", 0) or 0),
        cache_read_tokens=int(cached or 0),
        cache_write_tokens=0,
    )


_ERROR_CODE_STATUS = {
    "rate_limit_exceeded": 429,
    "server_error": 500,
    "invalid_prompt": 400,
}


def raise_response_error(code: str | None, message: str | None) -> None:
    """Raise the LLMError / ContextOverflowError for an in-stream failure
    (an ``error`` event, or ``response.failed``)."""
    message = message or "The model response failed"
    mapped = plan_usage_error(code)
    if mapped is not None:
        raise mapped
    text = f"{code}: {message}" if code else message
    probe = f"{code or ''} {message}".lower()
    if "context_length" in probe or "context window" in probe or "context length" in probe:
        raise ContextOverflowError(text)
    raise LLMError(text, status_code=_ERROR_CODE_STATUS.get(code or ""))


def _message_text(item: Any) -> str:
    parts = []
    for part in _g(item, "content", None) or []:
        kind = _g(part, "type")
        if kind == "output_text":
            parts.append(_g(part, "text", "") or "")
        elif kind == "refusal":
            parts.append(_g(part, "refusal", "") or "")
    return "".join(parts)


def _summary_text(item: Any) -> str:
    texts = [(_g(s, "text", "") or "") for s in (_g(item, "summary", None) or [])]
    if not any(texts):
        texts = [(_g(c, "text", "") or "") for c in (_g(item, "content", None) or [])]
    return "\n\n".join(t for t in texts if t)


def state_entry(item: Any) -> dict | None:
    """One output item -> its persisted state entry (None for item types this
    transport does not replay)."""
    kind = _g(item, "type")
    if kind == "reasoning":
        d = _dump(item)
        entry = {"type": "reasoning", "id": d.get("id") or "",
                 "summary": d.get("summary") or []}
        if d.get("content"):
            entry["content"] = d["content"]
        if d.get("encrypted_content"):
            entry["encrypted_content"] = d["encrypted_content"]
        return entry
    if kind == "message":
        entry = {"type": "message", "id": _g(item, "id") or ""}
        phase = _g(item, "phase")
        if phase:
            entry["phase"] = phase
        entry["text"] = _message_text(item)
        return entry
    if kind == "function_call":
        return {"type": "function_call", "id": _g(item, "id") or "",
                "call_id": _g(item, "call_id") or ""}
    return None


def build_state(entries: list, *, provider: str, model: str) -> dict | None:
    """The persisted field, or None when there is nothing worth replaying
    beyond what ``content`` / ``tool_calls`` already carry."""
    entries = [e for e in entries if e]
    if not any(e["type"] == "reasoning" or e.get("phase") for e in entries):
        return None
    return {"provider": provider, "model": model, "items": entries}


def finish_reason_for(status: str | None, incomplete: Any, has_calls: bool) -> str:
    if status == "incomplete":
        reason = _g(incomplete, "reason")
        return "content_filter" if reason == "content_filter" else "length"
    return "tool_calls" if has_calls else "stop"


def response_to_message(response: Any, *, provider: str, model: str) -> dict:
    """A non-streamed ``Response`` -> ``{raw_message, text, tool_calls,
    finish_reason, usage}`` in Chat-Completions vocabulary."""
    error = _g(response, "error")
    if error is not None and _g(response, "status") == "failed":
        raise_response_error(_g(error, "code"), _g(error, "message"))
    output = list(_g(response, "output", None) or [])
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict] = []
    for item in output:
        kind = _g(item, "type")
        if kind == "message":
            text_parts.append(_message_text(item))
        elif kind == "reasoning":
            s = _summary_text(item)
            if s:
                reasoning_parts.append(s)
        elif kind == "function_call":
            tool_calls.append({
                "id": _g(item, "call_id") or "",
                "type": "function",
                "function": {"name": _g(item, "name") or "",
                             "arguments": _g(item, "arguments") or ""},
            })
    text = "".join(text_parts) or None
    raw: dict = {"role": "assistant", "content": text}
    if tool_calls:
        raw["tool_calls"] = tool_calls
    if reasoning_parts:
        raw["reasoning_content"] = "\n\n".join(reasoning_parts)
    state = build_state([state_entry(i) for i in output], provider=provider, model=model)
    if state is not None:
        raw[RESPONSES_STATE_FIELD] = state
    return {
        "raw_message": raw,
        "text": text,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason_for(_g(response, "status"),
                                           _g(response, "incomplete_details"),
                                           bool(tool_calls)),
        "usage": responses_token_usage(_g(response, "usage")) or TokenUsage(0, 0),
    }


class ResponsesStreamState:
    """Translates Responses stream events into StreamChunks.

    Tool calls are numbered in arrival order and emitted as Chat-style
    ``tool_calls_delta`` dicts (``index`` / ``id`` / ``function``), which
    ``StreamAccumulator`` already joins. The terminal event yields ONE final
    chunk carrying usage, the finish reason and, via ``raw_extra``, the
    output-item state the next turn replays.
    """

    def __init__(self, *, provider: str, model: str):
        self.provider = provider
        self.model = model
        self._call_index: dict[int, int] = {}   # output_index -> tool-call index
        self._call_args: dict[int, str] = {}    # tool-call index -> args emitted
        self._call_ids: set[str] = set()
        self._text_seen: dict[int, bool] = {}   # output_index -> any text delta
        self._summary_key: tuple | None = None
        self._reasoning_emitted = False
        self._entries: dict[int, dict] = {}     # output_index -> state entry
        self._has_calls = False
        self.done = False

    # -- helpers --------------------------------------------------------
    def _register_call(self, output_index: int, item: Any) -> list[StreamChunk]:
        if output_index in self._call_index:
            return []
        call_id = _g(item, "call_id") or ""
        if call_id and call_id in self._call_ids:
            return []
        self._call_ids.add(call_id)
        k = len(self._call_index)
        self._call_index[output_index] = k
        self._has_calls = True
        args = _g(item, "arguments", "") or ""
        self._call_args[k] = args
        return [StreamChunk(tool_calls_delta=[{
            "index": k, "id": _g(item, "call_id") or "", "type": "function",
            "function": {"name": _g(item, "name") or "", "arguments": args},
        }])]

    def _reconcile_args(self, output_index: int, full: str | None) -> list[StreamChunk]:
        k = self._call_index.get(output_index)
        if k is None or full is None:
            return []
        seen = self._call_args.get(k, "")
        if full == seen:
            return []
        if full.startswith(seen):
            self._call_args[k] = full
            return [StreamChunk(tool_calls_delta=[
                {"index": k, "function": {"arguments": full[len(seen):]}}])]
        logger.warning("Responses stream: final arguments for call %d differ from "
                       "the streamed deltas; keeping the deltas", k)
        return []

    def _reasoning_delta(self, ev: Any) -> list[StreamChunk]:
        delta = _g(ev, "delta", "") or ""
        if not delta:
            return []
        key = (_g(ev, "output_index"), _g(ev, "summary_index", _g(ev, "content_index")))
        if self._summary_key is not None and key != self._summary_key and self._reasoning_emitted:
            delta = "\n\n" + delta
        self._summary_key = key
        self._reasoning_emitted = True
        return [StreamChunk(reasoning_content=delta)]

    # -- dispatch -------------------------------------------------------
    def handle(self, ev: Any) -> list[StreamChunk]:
        kind = _g(ev, "type")
        if kind == "response.output_text.delta" or kind == "response.refusal.delta":
            delta = _g(ev, "delta", "") or ""
            if delta:
                self._text_seen[_g(ev, "output_index")] = True
                return [StreamChunk(text=delta)]
            return []
        if kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            return self._reasoning_delta(ev)
        if kind == "response.output_item.added":
            item = _g(ev, "item")
            if _g(item, "type") == "function_call":
                return self._register_call(_g(ev, "output_index"), item)
            return []
        if kind == "response.function_call_arguments.delta":
            idx = _g(ev, "output_index")
            if idx not in self._call_index:
                return []
            delta = _g(ev, "delta", "") or ""
            if not delta:
                return []
            k = self._call_index[idx]
            self._call_args[k] = self._call_args.get(k, "") + delta
            return [StreamChunk(tool_calls_delta=[{"index": k, "function": {"arguments": delta}}])]
        if kind == "response.function_call_arguments.done":
            return self._reconcile_args(_g(ev, "output_index"), _g(ev, "arguments"))
        if kind == "response.output_item.done":
            return self._item_done(_g(ev, "output_index"), _g(ev, "item"))
        if kind in ("response.completed", "response.incomplete"):
            return self._finish(_g(ev, "response"))
        if kind == "response.failed":
            response = _g(ev, "response")
            error = _g(response, "error")
            raise_response_error(_g(error, "code"), _g(error, "message"))
        if kind == "error":
            raise_response_error(_g(ev, "code"), _g(ev, "message"))
        return []

    def _item_done(self, output_index: int, item: Any) -> list[StreamChunk]:
        out: list[StreamChunk] = []
        kind = _g(item, "type")
        if kind == "function_call":
            out += self._register_call(output_index, item)
            out += self._reconcile_args(output_index, _g(item, "arguments"))
        elif kind == "message" and not self._text_seen.get(output_index):
            text = _message_text(item)
            if text:
                self._text_seen[output_index] = True
                out.append(StreamChunk(text=text))
        elif kind == "reasoning" and not self._reasoning_emitted:
            summary = _summary_text(item)
            if summary:
                self._reasoning_emitted = True
                out.append(StreamChunk(reasoning_content=summary))
        entry = state_entry(item)
        if entry is not None:
            self._entries[output_index] = entry
        return out

    def _finish(self, response: Any) -> list[StreamChunk]:
        self.done = True
        output = list(_g(response, "output", None) or [])
        # Whatever the terminal response holds that never arrived as its own
        # events (observed live through OpenCode Go: a turn whose only event
        # was response.completed) is emitted here, once.
        out: list[StreamChunk] = []
        for index, item in enumerate(output):
            kind = _g(item, "type")
            if kind == "function_call":
                out += self._register_call(index, item)
                out += self._reconcile_args(index, _g(item, "arguments"))
            elif kind == "message" and not self._text_seen.get(index):
                text = _message_text(item)
                if text:
                    self._text_seen[index] = True
                    out.append(StreamChunk(text=text))
            elif kind == "reasoning" and not self._reasoning_emitted:
                summary = _summary_text(item)
                if summary:
                    self._reasoning_emitted = True
                    out.append(StreamChunk(reasoning_content=summary))
        if output:
            entries = [state_entry(i) for i in output]
            has_calls = self._has_calls or any(
                _g(i, "type") == "function_call" for i in output)
        else:
            entries = [self._entries[k] for k in sorted(self._entries)]
            has_calls = self._has_calls
        state = build_state(entries, provider=self.provider, model=self.model)
        return out + [StreamChunk(
            is_final=True,
            usage=responses_token_usage(_g(response, "usage")),
            finish_reason=finish_reason_for(_g(response, "status"),
                                            _g(response, "incomplete_details"),
                                            has_calls),
            raw_extra={RESPONSES_STATE_FIELD: state} if state is not None else None,
        )]
