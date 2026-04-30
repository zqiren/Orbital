# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for session meta-record (session_start, model_swap).

The motivating finding: production session JSONLs all logged ``sdk: openai``
regardless of underlying provider, making provider-specific behavior
investigations (path-doubling, Kimi quirks, Anthropic streaming) impossible
from logs alone. This file pins the contract: every new session begins with
a ``role: meta`` line carrying provider/model/sdk identity, mid-session
fallbacks emit a ``model_swap`` meta line, and meta records never surface
through ``get_messages()`` (so they cannot reach the LLM).
"""

from __future__ import annotations

import json
import os

from agent_os.agent.session import Session


def test_session_start_writes_meta_record(tmp_path):
    """First line of every new session JSONL is a session_start meta record."""
    session = Session.new(
        "test-session", str(tmp_path),
        provider="moonshot", model="kimi-k2-turbo",
        sdk="openai", fallback_models=["claude-sonnet-4-5"],
    )
    with open(session._filepath) as f:
        first_line = f.readline()
    assert first_line.strip(), "session JSONL is empty — meta record not written"
    record = json.loads(first_line)
    assert record["role"] == "meta"
    assert record["event"] == "session_start"
    assert record["provider"] == "moonshot"
    assert record["model"] == "kimi-k2-turbo"
    assert record["sdk"] == "openai"
    assert record["fallback_models"] == ["claude-sonnet-4-5"]
    assert "timestamp" in record


def test_session_start_meta_defaults_to_unknown(tmp_path):
    """Legacy callers that pass only (session_id, workspace) still work; values are 'unknown'."""
    session = Session.new("legacy", str(tmp_path))
    with open(session._filepath) as f:
        first_line = f.readline()
    record = json.loads(first_line)
    assert record["role"] == "meta"
    assert record["event"] == "session_start"
    assert record["provider"] == "unknown"
    assert record["model"] == "unknown"
    assert record["sdk"] == "unknown"
    assert record["fallback_models"] == []


def test_meta_record_not_in_sliding_window(tmp_path):
    """Meta records must NOT appear in get_messages() — they are not conversation."""
    session = Session.new(
        "test", str(tmp_path),
        provider="anthropic", model="claude-sonnet-4-5", sdk="anthropic",
    )
    session.append({"role": "user", "content": "hi"})
    msgs = session.get_messages()
    assert all(m.get("role") != "meta" for m in msgs), (
        "meta record leaked into get_messages(); the LLM must never see it"
    )
    # Still has the user message
    assert any(m.get("role") == "user" and m.get("content") == "hi" for m in msgs)


def test_meta_record_not_in_get_recent(tmp_path):
    """get_recent() must also skip meta records (sliding window for LLM context)."""
    session = Session.new(
        "test_recent", str(tmp_path),
        provider="anthropic", model="claude-sonnet-4-5", sdk="anthropic",
    )
    session.append({"role": "user", "content": "hello"})
    recent = session.get_recent(max_tokens=10_000)
    assert all(m.get("role") != "meta" for m in recent)


def test_append_meta_for_model_swap(tmp_path):
    """append_meta('model_swap', ...) records mid-session provider/model rotation."""
    session = Session.new(
        "test", str(tmp_path),
        provider="anthropic", model="claude-sonnet-4-5", sdk="anthropic",
    )
    session.append_meta(
        "model_swap",
        provider="openai", model="gpt-4o", sdk="openai", reason="rate_limit",
    )
    with open(session._filepath) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    swap_records = [line for line in lines if line.get("event") == "model_swap"]
    assert len(swap_records) == 1
    swap = swap_records[0]
    assert swap["role"] == "meta"
    assert swap["provider"] == "openai"
    assert swap["model"] == "gpt-4o"
    assert swap["sdk"] == "openai"
    assert swap["reason"] == "rate_limit"
    assert "timestamp" in swap


def test_append_meta_does_not_affect_messages(tmp_path):
    """append_meta() must not push records onto self._messages."""
    session = Session.new(
        "test_meta_msgs", str(tmp_path),
        provider="anthropic", model="claude-sonnet-4-5", sdk="anthropic",
    )
    msgs_before = len(session.get_messages())
    session.append_meta("model_swap", provider="openai", model="gpt-4o", sdk="openai")
    msgs_after = len(session.get_messages())
    assert msgs_before == msgs_after, "append_meta() should not grow get_messages()"


def test_load_skips_meta_records(tmp_path):
    """Loading a JSONL with meta records must not surface them as messages.

    Simulates resume of a real session: meta first, then user/assistant. The
    sliding window must contain only the conversation messages.
    """
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "loaded.jsonl"
    records = [
        {"role": "meta", "event": "session_start", "provider": "moonshot",
         "model": "kimi-k2-turbo", "sdk": "openai", "fallback_models": [],
         "timestamp": "2026-04-30T12:00:00+00:00"},
        {"role": "user", "content": "hi", "timestamp": "2026-04-30T12:00:01+00:00"},
        {"role": "assistant", "content": "hello",
         "timestamp": "2026-04-30T12:00:02+00:00"},
        {"role": "meta", "event": "model_swap", "provider": "openai",
         "model": "gpt-4o", "sdk": "openai", "reason": "rate_limit",
         "timestamp": "2026-04-30T12:00:03+00:00"},
        {"role": "user", "content": "still there?",
         "timestamp": "2026-04-30T12:00:04+00:00"},
    ]
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    session = Session.load(str(filepath))
    msgs = session.get_messages()
    assert len(msgs) == 3, (
        f"expected 3 conversation messages (user/assistant/user); got {len(msgs)}"
    )
    assert all(m.get("role") != "meta" for m in msgs)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user"]


def test_load_does_not_warn_on_meta_records(tmp_path, caplog):
    """Loading a JSONL containing meta records emits no warnings."""
    import logging

    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "warn_test.jsonl"
    with open(filepath, "w") as f:
        f.write(json.dumps({
            "role": "meta", "event": "session_start", "provider": "anthropic",
            "model": "claude-sonnet-4-5", "sdk": "anthropic",
            "fallback_models": [], "timestamp": "2026-04-30T12:00:00+00:00",
        }) + "\n")
        f.write(json.dumps({
            "role": "user", "content": "hi",
            "timestamp": "2026-04-30T12:00:01+00:00",
        }) + "\n")

    with caplog.at_level(logging.WARNING, logger="agent_os.agent.session"):
        Session.load(str(filepath))
    # No warnings about corrupted lines or unknown roles
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_messages, f"unexpected warnings during load: {warning_messages}"


def test_session_start_meta_is_first_line_with_other_appends(tmp_path):
    """After several append() calls, the meta record remains first in the JSONL."""
    session = Session.new(
        "ordering", str(tmp_path),
        provider="moonshot", model="kimi-k2-turbo", sdk="openai",
    )
    session.append({"role": "user", "content": "first"})
    session.append({"role": "assistant", "content": "reply"})
    with open(session._filepath) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines[0]["role"] == "meta"
    assert lines[0]["event"] == "session_start"
    assert lines[1]["role"] == "user"
    assert lines[2]["role"] == "assistant"


def test_session_filepath_exists_after_new(tmp_path):
    """Sanity: Session.new still creates the file at the expected path."""
    session = Session.new("path_check", str(tmp_path))
    assert os.path.exists(session._filepath)
