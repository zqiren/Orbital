# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for session naming (TASK-session-naming-and-deletion.md PART 1).

A session name is a human-readable display label derived from the first user
message (truncated, word-boundary-aware). It is stored on the ``session_start``
meta record (Option A) so it lives in the single JSONL file with no sidecar.
Renames rewrite that meta line. Legacy sessions with no name on the meta record
get a name derived lazily in memory at load time (no file rewrite).

Tests 1-5 from the spec.
"""

from __future__ import annotations

import json

from agent_os.agent.session import Session


def _new_session(tmp_path):
    return Session.new(
        "name_test", str(tmp_path),
        provider="anthropic", model="claude-sonnet-4-5", sdk="anthropic",
    )


# Test 1 — Auto-name from first user message
def test_auto_name_from_first_user_message(tmp_path):
    session = _new_session(tmp_path)
    session.append({"role": "user", "content": "Implement the login flow for the dashboard"})
    assert session.name == "Implement the login flow for the dashboard"


def test_auto_name_only_set_by_first_user_message(tmp_path):
    """A second user message does not change the auto-name; only the first wins."""
    session = _new_session(tmp_path)
    session.append({"role": "user", "content": "first message wins"})
    session.append({"role": "user", "content": "second message ignored"})
    assert session.name == "first message wins"


def test_assistant_first_does_not_set_name(tmp_path):
    """Only role == 'user' triggers auto-naming; an assistant message does not."""
    session = _new_session(tmp_path)
    session.append({"role": "assistant", "content": "hello there"})
    assert session.name is None


# Test 2 — Long message truncation
def test_long_message_truncation_word_boundary(tmp_path):
    session = _new_session(tmp_path)
    long_msg = (
        "Please refactor the entire authentication subsystem so that it "
        "supports OAuth, SAML, and passkeys simultaneously without breaking"
    )
    assert len(long_msg) > 100
    session.append({"role": "user", "content": long_msg})
    name = session.name
    # 50 chars + ellipsis ⇒ at most 53 chars.
    assert len(name) <= 53, f"name too long ({len(name)}): {name!r}"
    assert name.endswith("…"), f"truncated name must end with ellipsis: {name!r}"
    # Word-boundary-aware: no truncation in the middle of a word (the char
    # before the ellipsis must not be followed by a non-space in the source).
    stem = name[:-1]
    assert not stem.endswith(" "), "trailing space before ellipsis should be stripped"
    # The stem is a prefix of the original message up to a word boundary.
    assert long_msg.startswith(stem), f"stem {stem!r} is not a prefix of source"
    # The character in the source right after the stem is a space (word break).
    assert long_msg[len(stem)] == " ", "truncation did not occur at a word boundary"


def test_name_persisted_on_meta_record(tmp_path):
    """The auto-generated name is written onto the session_start meta line."""
    session = _new_session(tmp_path)
    session.append({"role": "user", "content": "build a thing"})
    with open(session._filepath) as f:
        first = json.loads(f.readline())
    assert first["role"] == "meta"
    assert first["event"] == "session_start"
    assert first["name"] == "build a thing"


# Test 3 — Rename via session.set_name + persistence across reload
def test_rename_updates_name_and_persists(tmp_path):
    session = _new_session(tmp_path)
    session.append({"role": "user", "content": "auto generated name"})
    assert session.name == "auto generated name"

    session.set_name("my custom name")
    assert session.name == "my custom name"

    # The meta line is rewritten in place — reload from disk and confirm.
    reloaded = Session.load(session._filepath)
    assert reloaded.name == "my custom name"
    # Conversation is intact and meta is still first line / not in messages.
    msgs = reloaded.get_messages()
    assert all(m.get("role") != "meta" for m in msgs)
    assert any(m.get("role") == "user" and m.get("content") == "auto generated name" for m in msgs)


def test_rename_before_first_message_persists_on_creation(tmp_path):
    """Renaming before the file exists updates the pending meta, then persists."""
    session = _new_session(tmp_path)
    session.set_name("named before any message")
    assert session.name == "named before any message"
    session.append({"role": "assistant", "content": "headless work"})
    reloaded = Session.load(session._filepath)
    assert reloaded.name == "named before any message"


# Test 4 — Legacy session backfill (no name on meta / no meta at all)
def test_legacy_backfill_derives_name_from_first_user_message(tmp_path):
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "legacy.jsonl"
    records = [
        {"role": "user", "content": "Fix the broken deploy pipeline",
         "timestamp": "2026-04-30T12:00:00+00:00"},
        {"role": "assistant", "content": "on it",
         "timestamp": "2026-04-30T12:00:01+00:00"},
    ]
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    session = Session.load(str(filepath))
    assert session.name == "Fix the broken deploy pipeline"


def test_legacy_backfill_does_not_rewrite_file(tmp_path):
    """Backfill is in-memory only — Session.load must not rewrite the JSONL."""
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "legacy_norewrite.jsonl"
    records = [
        {"role": "user", "content": "do not rewrite me",
         "timestamp": "2026-04-30T12:00:00+00:00"},
    ]
    raw = "\n".join(json.dumps(r) for r in records) + "\n"
    with open(filepath, "w") as f:
        f.write(raw)

    Session.load(str(filepath))
    with open(filepath) as f:
        after = f.read()
    assert after == raw, "Session.load must not rewrite the legacy JSONL during backfill"


def test_meta_without_name_backfills_from_first_user(tmp_path):
    """A meta line that lacks a name (older session_start records) still
    backfills the derived name from the first user message on load."""
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "meta_no_name.jsonl"
    records = [
        {"role": "meta", "event": "session_start", "provider": "anthropic",
         "model": "claude-sonnet-4-5", "sdk": "anthropic", "fallback_models": [],
         "timestamp": "2026-04-30T12:00:00+00:00"},
        {"role": "user", "content": "derive me from the user message",
         "timestamp": "2026-04-30T12:00:01+00:00"},
    ]
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    session = Session.load(str(filepath))
    assert session.name == "derive me from the user message"


def test_meta_with_name_loads_stored_name(tmp_path):
    """A meta line carrying an explicit name uses it verbatim on load (a rename
    that differs from the first user message must survive)."""
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "meta_with_name.jsonl"
    records = [
        {"role": "meta", "event": "session_start", "name": "Renamed Session",
         "provider": "anthropic", "model": "claude-sonnet-4-5", "sdk": "anthropic",
         "fallback_models": [], "timestamp": "2026-04-30T12:00:00+00:00"},
        {"role": "user", "content": "the original first message",
         "timestamp": "2026-04-30T12:00:01+00:00"},
    ]
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    session = Session.load(str(filepath))
    assert session.name == "Renamed Session"


# Test 5 — No user message (DATA-1 headless case)
def test_no_user_message_name_is_null(tmp_path):
    session = _new_session(tmp_path)
    session.append({"role": "assistant", "content": "headless output, no user"})
    session.append({"role": "system", "content": "lifecycle note"})
    assert session.name is None


def test_no_user_message_loads_null(tmp_path):
    """A loaded session whose JSONL has only assistant/system messages → name None."""
    sessions_dir = tmp_path / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    filepath = sessions_dir / "headless.jsonl"
    records = [
        {"role": "meta", "event": "session_start", "provider": "anthropic",
         "model": "claude-sonnet-4-5", "sdk": "anthropic", "fallback_models": [],
         "timestamp": "2026-04-30T12:00:00+00:00"},
        {"role": "assistant", "content": "headless",
         "timestamp": "2026-04-30T12:00:01+00:00"},
    ]
    with open(filepath, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    session = Session.load(str(filepath))
    assert session.name is None
