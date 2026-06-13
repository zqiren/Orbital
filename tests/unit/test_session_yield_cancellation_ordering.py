# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run-end cancellation of pending tool calls must write its CANCELLED results
*contiguously* after the originating assistant's existing tool results, and
persist that order to disk — not tail-append them after an interleaved non-tool
row (e.g. the lifecycle "[Sub-agent] Message sent…" echo).

The persisted JSONL is what gets replayed on resume; a strict provider (MiniMax)
400s on a tool result that does not immediately follow its tool_calls. TASK 1
repairs this at send time; this writes it correctly the first time so the
on-disk history is conformant for replay/export/debug/frontend.

These tests assert the ON-DISK order after reload (the false-green trap is
asserting in-memory `_messages`). They also pin that meta rows (session_start,
sub_agent_thread) survive — the placement must operate on the full on-disk
content, never a meta-stripped rewrite. (Investigation probe proved the
load-time healer's old `self._messages` rewrite silently dropped meta.)
"""

import json

from agent_os.agent.session import Session


def _tc(tid):
    return {"id": tid, "type": "function", "function": {"name": "f", "arguments": "{}"}}


def _raw(path):
    """All on-disk rows, INCLUDING meta (Session.load() would strip meta)."""
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _shape(rows):
    out = []
    for r in rows:
        role = r.get("role")
        if role == "tool":
            out.append(("tool", r.get("tool_call_id")))
        elif role == "assistant" and r.get("tool_calls"):
            out.append(("assistant", tuple(tc["id"] for tc in r["tool_calls"])))
        elif role == "meta":
            out.append(("meta", r.get("event")))
        else:
            out.append((role, (r.get("content") or "")[:12]))
    return out


def _write_disk(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _session_at(path, messages, pending):
    """A live Session bound to `path` with the given in-memory view + pending
    set, WITHOUT going through load() (which would heal first)."""
    s = Session(path)
    s._messages = [dict(m) for m in messages]
    s.pending_tool_calls = set(pending)
    return s


# --- PRIMARY (RED): run-end resolve persists a contiguous block, meta intact --


def test_resolve_persists_contiguous_block_preserving_meta(tmp_path):
    path = str(tmp_path / "yield.jsonl")
    disk = [
        {"role": "meta", "event": "session_start", "name": "S", "origin": "chat"},
        {"role": "assistant", "content": "dispatch", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "Dispatched… Awaiting completion"},
        {"role": "meta", "event": "sub_agent_thread", "handle": "claude-code", "thread": {"session_id": "abc"}},
        {"role": "system", "content": "[Sub-agent] Message sent to claude-code: …"},
    ]
    _write_disk(path, disk)
    # in-memory view excludes meta (mirrors what load() builds); _2/_3 pending
    s = _session_at(path, [m for m in disk if m["role"] != "meta"], {"_2", "_3"})

    s.resolve_pending_tool_calls()

    assert _shape(_raw(path)) == [
        ("meta", "session_start"),
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_1"), ("tool", "_2"), ("tool", "_3"),
        ("meta", "sub_agent_thread"),
        ("system", "[Sub-agent] "),
    ]
    # resolve-path wording is distinct from the heal path (constraint pin:
    # positive + negative so a DRY refactor collapsing the two strings fails).
    cancels = [r for r in _raw(path) if r["role"] == "tool" and "CANCELLED" in (r.get("content") or "")]
    assert cancels and all("not executed." in c["content"] for c in cancels)
    assert not any("session interrupted" in c["content"] for c in cancels)
    # pending cleared; both meta rows survived; reload yields resume identity.
    assert s.pending_tool_calls == set()
    loaded = Session.load(path)
    assert loaded.sub_agent_threads.get("claude-code", {}).get("session_id") == "abc"
    # LLM-facing (meta-excluded) view is a contiguous tool block.
    llm = [m for m in loaded.get_messages()]
    roles = [(m.get("role"), m.get("tool_call_id")) for m in llm if m.get("role") in ("tool", "system")]
    assert roles == [("tool", "_1"), ("tool", "_2"), ("tool", "_3"), ("system", None)]


# --- HEALER (RED): crash-recovery heal must ALSO preserve meta ---------------


def test_load_time_heal_preserves_meta_and_is_contiguous(tmp_path):
    path = str(tmp_path / "heal.jsonl")
    disk = [
        {"role": "meta", "event": "session_start", "name": "S", "origin": "chat"},
        {"role": "assistant", "content": "d", "tool_calls": [_tc("h1"), _tc("h2"), _tc("h3")]},
        {"role": "tool", "tool_call_id": "h1", "content": "ok"},
        {"role": "meta", "event": "sub_agent_thread", "handle": "claude-code", "thread": {"session_id": "xyz"}},
        {"role": "system", "content": "echo"},
    ]
    _write_disk(path, disk)

    loaded = Session.load(path)  # pending {h2,h3} → heal runs

    assert loaded.pending_tool_calls == set()
    on_disk = _raw(path)
    assert sum(1 for r in on_disk if r["role"] == "meta") == 2, "meta rows must survive heal"
    assert _shape(on_disk) == [
        ("meta", "session_start"),
        ("assistant", ("h1", "h2", "h3")),
        ("tool", "h1"), ("tool", "h2"), ("tool", "h3"),
        ("meta", "sub_agent_thread"),
        ("system", "echo"),
    ]
    # heal keeps its distinct wording
    cancels = [r for r in on_disk if r["role"] == "tool" and "CANCELLED" in (r.get("content") or "")]
    assert all("session interrupted" in c["content"] for c in cancels)


# --- REGRESSION: shapes that already worked stay correct ---------------------


def test_resolve_no_interleaved_row_is_contiguous(tmp_path):
    path = str(tmp_path / "clean.jsonl")
    disk = [
        {"role": "meta", "event": "session_start"},
        {"role": "assistant", "content": "d", "tool_calls": [_tc("_1"), _tc("_2"), _tc("_3")]},
        {"role": "tool", "tool_call_id": "_1", "content": "ok"},
    ]
    _write_disk(path, disk)
    s = _session_at(path, [m for m in disk if m["role"] != "meta"], {"_2", "_3"})
    s.resolve_pending_tool_calls()
    assert _shape(_raw(path)) == [
        ("meta", "session_start"),
        ("assistant", ("_1", "_2", "_3")),
        ("tool", "_1"), ("tool", "_2"), ("tool", "_3"),
    ]


def test_resolve_solo_pending_unchanged(tmp_path):
    path = str(tmp_path / "solo.jsonl")
    disk = [
        {"role": "meta", "event": "session_start"},
        {"role": "assistant", "content": "d", "tool_calls": [_tc("_1")]},
    ]
    _write_disk(path, disk)
    s = _session_at(path, [m for m in disk if m["role"] != "meta"], {"_1"})
    s.resolve_pending_tool_calls()
    assert _shape(_raw(path)) == [
        ("meta", "session_start"),
        ("assistant", ("_1",)),
        ("tool", "_1"),
    ]


def test_resolve_no_pending_does_not_rewrite(tmp_path):
    path = str(tmp_path / "nopend.jsonl")
    disk = [
        {"role": "meta", "event": "session_start"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "yo"},
    ]
    _write_disk(path, disk)
    before = open(path, encoding="utf-8").read()
    s = _session_at(path, [m for m in disk if m["role"] != "meta"], set())
    s.resolve_pending_tool_calls()
    assert open(path, encoding="utf-8").read() == before  # byte-identical, no rewrite


def test_resolve_does_not_double_insert_already_resolved_id(tmp_path):
    """Idempotence: even if pending wrongly claims an already-resolved id (a
    memory/disk desync), resolve must NOT append a second result for it — two
    tool rows for one id is itself a strict-provider violation."""
    path = str(tmp_path / "dup.jsonl")
    disk = [
        {"role": "assistant", "content": "d", "tool_calls": [_tc("_1"), _tc("_2")]},
        {"role": "tool", "tool_call_id": "_1", "content": "ok"},
        {"role": "tool", "tool_call_id": "_2", "content": "ok"},
    ]
    _write_disk(path, disk)
    s = _session_at(path, [m for m in disk if m["role"] != "meta"], {"_1", "_2"})
    before = open(path, encoding="utf-8").read()
    s.resolve_pending_tool_calls()
    assert open(path, encoding="utf-8").read() == before  # no spurious rows
    rows = _raw(path)
    assert sum(1 for r in rows if r.get("tool_call_id") == "_1") == 1
    assert sum(1 for r in rows if r.get("tool_call_id") == "_2") == 1


def test_resolve_fires_on_append_for_inserted_cancellations(tmp_path):
    path = str(tmp_path / "cb.jsonl")
    disk = [
        {"role": "assistant", "content": "d", "tool_calls": [_tc("_1"), _tc("_2")]},
        {"role": "tool", "tool_call_id": "_1", "content": "ok"},
        {"role": "system", "content": "echo"},
    ]
    _write_disk(path, disk)
    s = _session_at(path, [m for m in disk if m["role"] != "meta"], {"_2"})
    seen = []
    s.on_append = lambda m: seen.append(m.get("tool_call_id"))
    s.resolve_pending_tool_calls()
    assert seen == ["_2"]  # WS-activity parity preserved for the cancellation
