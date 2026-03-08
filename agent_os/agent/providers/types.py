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


@dataclass
class StreamChunk:
    text: str = ""
    tool_calls_delta: list = field(default_factory=list)
    is_final: bool = False
    usage: TokenUsage | None = None
    reasoning_content: str = ""


@dataclass
class LLMResponse:
    raw_message: dict
    text: str | None
    tool_calls: list[dict]
    has_tool_calls: bool
    finish_reason: str
    status_text: str | None
    usage: TokenUsage


class StreamAccumulator:
    """Accumulates StreamChunk deltas into a complete LLMResponse."""

    def __init__(self):
        self.text_parts: list[str] = []
        self.tool_calls_parts: dict[int, dict] = {}
        self.usage: TokenUsage | None = None
        self.reasoning_parts: list[str] = []

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
        finish_reason = "tool_calls" if tool_calls else "stop"
        return LLMResponse(
            raw_message=raw_message,
            text=text,
            tool_calls=tool_calls,
            has_tool_calls=bool(tool_calls),
            finish_reason=finish_reason,
            status_text=status_text,
            usage=self.usage or TokenUsage(0, 0),
        )


class ContextOverflowError(Exception):
    """Raised when LLM API rejects request due to context length."""


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
