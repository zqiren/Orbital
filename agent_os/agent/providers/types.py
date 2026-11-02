# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared response types for the LLM provider layer.

Owned by Component C. Imported by Component A.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class ErrorCategory(str, Enum):
    """Classification of LLM errors for fallback routing decisions."""
    TRANSIENT = "transient"    # 429, 502, 503, timeout, connection failure -> rotate
    RETRY = "retry"            # 500, network reset -> retry same provider
    ABORT = "abort"            # 401, 403, 400 -> stop immediately


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class StreamChunk:
    text: str = ""
    tool_calls_delta: list = field(default_factory=list)
    is_final: bool = False
    usage: TokenUsage | None = None
    reasoning_content: str = ""
    # The provider's own finish/stop reason, normalized to OpenAI vocabulary
    # where the adapter knows the mapping ("stop"/"length"/"tool_calls"/...).
    # None on delta chunks and for providers that never report one.
    finish_reason: str | None = None


@dataclass
class LLMResponse:
    raw_message: dict
    text: str | None
    tool_calls: list[dict]
    has_tool_calls: bool
    finish_reason: str
    status_text: str | None
    usage: TokenUsage
    # Diagnostic: True when the stream delivered a final usage-bearing chunk
    # (the chunk that drives CACHE_AUDIT). False signals a stream that ended
    # without its terminal chunk — the "silent truncation" signature. Defaults
    # False so non-streaming/legacy constructors are unaffected.
    final_usage_chunk_seen: bool = False


class StreamAccumulator:
    """Accumulates StreamChunk deltas into a complete LLMResponse."""

    def __init__(self):
        self.text_parts: list[str] = []
        self.tool_calls_parts: dict[int, dict] = {}
        self.usage: TokenUsage | None = None
        self.reasoning_parts: list[str] = []
        self.finish_reason: str | None = None

    @staticmethod
    def _get_attr(obj, key, default=""):
        """Get attribute or dict key from an object (handles both SDK objects and dicts)."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def add(self, chunk: StreamChunk) -> None:
        if chunk.text:
            self.text_parts.append(chunk.text)
        if chunk.reasoning_content:
            self.reasoning_parts.append(chunk.reasoning_content)
        for tc_delta in chunk.tool_calls_delta:
            # Handle both OpenAI SDK objects (with .index, .id, etc.)
            # and raw dicts (with "index", "id", etc.)
            idx = self._get_attr(tc_delta, "index", 0)
            tc_id = self._get_attr(tc_delta, "id", "")
            tc_type = self._get_attr(tc_delta, "type", "function")
            func_obj = self._get_attr(tc_delta, "function", None)
            if func_obj is None:
                func_name = self._get_attr(tc_delta, "name", "")
                func_args = self._get_attr(tc_delta, "arguments", "")
                if isinstance(func_args, dict):
                    import json as _json
                    func_args = _json.dumps(func_args)
            elif isinstance(func_obj, dict):
                func_name = func_obj.get("name", "")
                func_args = func_obj.get("arguments", "")
                if isinstance(func_args, dict):
                    import json as _json
                    func_args = _json.dumps(func_args)
            else:
                func_name = getattr(func_obj, "name", "") or ""
                func_args = getattr(func_obj, "arguments", "") or ""

            if idx not in self.tool_calls_parts:
                self.tool_calls_parts[idx] = {
                    "id": tc_id,
                    "type": tc_type,
                    "function": {
                        "name": func_name,
                        "arguments": func_args,
                    },
                }
            else:
                entry = self.tool_calls_parts[idx]
                if tc_id:
                    entry["id"] = tc_id
                if tc_type:
                    entry["type"] = tc_type
                if func_name:
                    entry["function"]["name"] = func_name
                if func_args:
                    entry["function"]["arguments"] += func_args
        if chunk.usage:
            self.usage = chunk.usage
        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason

    def _assemble_tool_calls(self) -> list[dict]:
        if not self.tool_calls_parts:
            return []
        return [self.tool_calls_parts[idx] for idx in sorted(self.tool_calls_parts)]

    @staticmethod
    def _extract_status(text: str | None) -> str | None:
        if not text:
            return None
        match = re.search(r'\[STATUS:\s*(.+?)\]', text)
        return match.group(1).strip() if match else None

    def finalize(self) -> LLMResponse:
        text = "".join(self.text_parts) or None
        tool_calls = self._assemble_tool_calls()
        raw_message: dict = {"role": "assistant", "content": text}
        if tool_calls:
            raw_message["tool_calls"] = tool_calls
        if self.reasoning_parts:
            raw_message["reasoning_content"] = "".join(self.reasoning_parts)
        status_text = self._extract_status(text)
        # Prefer the provider's real finish_reason (it distinguishes a
        # truncated response, "length", from a completed one). Synthesize
        # only for providers that never reported one.
        finish_reason = self.finish_reason or (
            "tool_calls" if tool_calls else "stop"
        )
        return LLMResponse(
            raw_message=raw_message,
            text=text,
            tool_calls=tool_calls,
            has_tool_calls=bool(tool_calls),
            finish_reason=finish_reason,
            status_text=status_text,
            usage=self.usage or TokenUsage(0, 0),
            final_usage_chunk_seen=self.usage is not None,
        )


class ContextOverflowError(Exception):
    """Raised when LLM API rejects request due to context length."""


# Phrases that mean the CREDENTIAL was rejected. Everything else a gateway
# refuses a request for — no payment method, a region gate, a quota, an
# unenrolled model — also arrives as 401/403, and calling those "Invalid API
# key" sends the user to re-issue a key that works fine. Reported from the
# field on OpenCode Go: deepseek-v4-flash answers 403 RegionError until the
# workspace opts into China hosting. Auth phrasing is a small stable set; the
# reasons a request can be refused are open-ended, so the allowlist covers
# auth and everything else passes through in the provider's own words.
AUTH_FAILURE_PHRASES = (
    "api key", "apikey", "api-key",
    "authentication", "unauthorized", "invalid token", "invalid credentials",
)


def unwrap_provider_message(raw: str) -> str:
    """Pull the provider's sentence out of an SDK envelope.

    SDK errors arrive as ``Error code: 403 - {'type': 'error', 'error':
    {'message': '...'}}`` — a Python repr, which is unreadable in a red UI
    line. Return the innermost ``message`` when the envelope parses, else the
    raw string unchanged.
    """
    import ast

    body = raw.split(" - ", 1)[1] if " - " in raw else raw
    try:
        parsed = ast.literal_eval(body.strip())
    except (ValueError, SyntaxError):
        return raw
    for _ in range(4):  # {'error': {'error': {...}}} nests at most a little
        if not isinstance(parsed, dict):
            break
        msg = parsed.get("message")
        if isinstance(msg, str) and msg:
            return msg
        nxt = parsed.get("error")
        if nxt is None:
            break
        parsed = nxt
    return raw


def is_rejected_credential(status: int | None, message: str | None) -> bool:
    """True when a refusal means the key itself is bad. Any 401 is (providers
    word those in many languages); a 403 only when it says so, or says nothing."""
    if status == 401:
        return True
    if status != 403:
        return False
    text = (message or "").strip().lower()
    return not text or any(p in text for p in AUTH_FAILURE_PHRASES)


class LLMError(Exception):
    """General LLM API error."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        self.category = self._classify(status_code, message)
        super().__init__(message)

    @staticmethod
    def _classify(status_code: int | None, message: str) -> str:
        """Derive error category from status code and message."""
        if status_code in (429, 502, 503):
            return ErrorCategory.TRANSIENT
        if status_code in (401, 403, 400):
            return ErrorCategory.ABORT
        if status_code == 500:
            return ErrorCategory.RETRY
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ("timed out", "timeout", "connection failed", "dns")):
            return ErrorCategory.TRANSIENT
        return ErrorCategory.RETRY
