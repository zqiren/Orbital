# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 066 phase 1b — sessions stop inlining base64 images on disk.

What changes is the JSONL only. The in-memory conversation — what the model
is sent, what the WS observers see — is exactly what it was: a freshly
appended image keeps its data URL in memory, and a loaded session gets its
images rehydrated from the blob store. Old sessions with inline images keep
loading and keep their rows as written.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from agent_os.agent import blob_store
from agent_os.agent.session import Session


PNG_A = b"\x89PNG\r\n\x1a\n" + b"A" * 4096
PNG_B = b"\x89PNG\r\n\x1a\n" + b"B" * 4096


def _data_url(raw: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"


def _screenshot_content(raw: bytes = PNG_A) -> list:
    return [
        {"type": "text", "text": "Screenshot of https://example.com. Title: Example"},
        {"type": "image_url", "image_url": {"url": _data_url(raw)}},
    ]


def _add_screenshot(session, call_id, raw=PNG_A):
    session.append({
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": "browser",
                                     "arguments": json.dumps({"action": "screenshot"})}}],
        "source": "management",
    })
    session.append_tool_result(
        call_id, _screenshot_content(raw),
        meta={"url": "https://example.com", "screenshot_path": "/x/step_0001.png"},
    )


def _disk_rows(session):
    with open(session._filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _blob_count(workspace):
    root = blob_store.blobs_dir(os.path.join(workspace, "orbital"))
    return sum(len(files) for _d, _s, files in os.walk(root))


@pytest.fixture
def workspace(tmp_path):
    return str(tmp_path / "ws")


@pytest.fixture
def session(workspace):
    return Session.new("proj_0000aaaa", workspace)


def test_jsonl_row_holds_a_reference_not_base64(session, workspace):
    _add_screenshot(session, "call_1")

    raw = open(session._filepath, encoding="utf-8").read()
    assert "base64," not in raw
    tool_row = [r for r in _disk_rows(session) if r.get("role") == "tool"][0]
    assert tool_row["_meta"]["image_blobs"][0]["sha256"]
    assert tool_row["_meta"]["screenshot_path"] == "/x/step_0001.png"
    assert _blob_count(workspace) == 1


def test_in_memory_message_keeps_its_data_url(session):
    """The model and the live WS stream see what they saw before."""
    seen = []
    session.on_append = lambda m: seen.append(m)
    _add_screenshot(session, "call_1")

    tool = [m for m in session.get_messages() if m.get("role") == "tool"][0]
    assert tool["content"] == _screenshot_content()
    assert seen[-1]["content"] == _screenshot_content()


def test_load_rehydrates_the_exact_row(session):
    _add_screenshot(session, "call_1")
    before = [m for m in session.get_messages() if m.get("role") == "tool"][0]

    loaded = Session.load(session._filepath)
    after = [m for m in loaded.get_messages() if m.get("role") == "tool"][0]
    assert json.dumps(after) == json.dumps(before)


def test_model_window_still_carries_the_image_after_reload(session):
    """``get_recent`` is what ContextManager.prepare() windows over — the
    existing media-pruning rules then run on the same data as before."""
    _add_screenshot(session, "call_1")
    loaded = Session.load(session._filepath)
    window = loaded.get_recent(100_000)
    tool = [m for m in window if m.get("role") == "tool"][0]
    assert tool["content"][1]["image_url"]["url"] == _data_url(PNG_A)


def test_same_screenshot_twice_stores_one_blob(session, workspace):
    _add_screenshot(session, "call_1", PNG_A)
    _add_screenshot(session, "call_2", PNG_A)
    _add_screenshot(session, "call_3", PNG_B)
    assert _blob_count(workspace) == 2


def test_legacy_inline_session_loads_and_is_not_rewritten(workspace):
    """An old session with inline base64 opens as before, and a rewrite of the
    file (stub supersession here) keeps its old rows exactly as written."""
    sessions_dir = os.path.join(workspace, "orbital", "sessions")
    os.makedirs(sessions_dir)
    path = os.path.join(sessions_dir, "proj_1111bbbb.jsonl")
    legacy_tool = {"role": "tool", "content": _screenshot_content(), "tool_call_id": "old_1",
                   "source": "management", "timestamp": "2026-01-01T00:00:00+00:00",
                   "_meta": {"image_path": "/u/a.png"}}
    rows = [
        {"role": "meta", "event": "session_start", "session_id": "proj_1111bbbb",
         "session_uuid": "proj_1111bbbb", "origin": "chat"},
        {"role": "user", "content": "look", "timestamp": "2026-01-01T00:00:00+00:00"},
        {"role": "assistant", "tool_calls": [{"id": "old_1", "type": "function",
         "function": {"name": "read", "arguments": "{\"path\": \"a.png\"}"}}],
         "timestamp": "2026-01-01T00:00:00+00:00"},
        legacy_tool,
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    loaded = Session.load(path)
    tool = [m for m in loaded.get_messages() if m.get("role") == "tool"][0]
    assert tool["content"] == _screenshot_content()

    # New image appended to the old session, then a whole-file rewrite.
    _add_screenshot(loaded, "call_new", PNG_B)
    loaded.replace_tool_results_with_stubs({"nonexistent": "x"})  # no-op
    loaded.replace_tool_results_with_stubs({"call_new": "[stub]"})

    disk = {r.get("tool_call_id"): r for r in _disk_rows(loaded) if r.get("role") == "tool"}
    assert json.dumps(disk["old_1"], ensure_ascii=False) == json.dumps(legacy_tool, ensure_ascii=False), (
        "an existing inline row must be written back exactly as it was"
    )
    assert _blob_count(workspace) == 1  # only the NEW image went to the store


def test_whole_file_rewrite_keeps_new_images_as_references(session, workspace):
    """A stub-supersession rewrite regenerates the file from memory, where
    the images are data URLs — they must go back out as references."""
    _add_screenshot(session, "call_1", PNG_A)
    _add_screenshot(session, "call_2", PNG_B)
    session.append({"role": "assistant", "source": "management", "tool_calls": [
        {"id": "call_3", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
    session.append_tool_result("call_3", "T" * 2000)
    session.replace_tool_results_with_stubs({"call_3": "[stub]"})

    raw = open(session._filepath, encoding="utf-8").read()
    assert "base64," not in raw
    loaded = Session.load(session._filepath)
    tools = [m for m in loaded.get_messages() if m.get("role") == "tool"]
    assert tools[0]["content"][1]["image_url"]["url"] == _data_url(PNG_A)
    assert tools[1]["content"][1]["image_url"]["url"] == _data_url(PNG_B)


def test_cancellation_splice_keeps_memory_rehydrated(session):
    """The heal path rebuilds the in-memory list from disk rows — those rows
    must come back with their images, not their placeholders."""
    _add_screenshot(session, "call_1", PNG_A)
    session.append({
        "role": "assistant",
        "tool_calls": [{"id": "pending_1", "type": "function",
                        "function": {"name": "shell", "arguments": "{}"}}],
        "source": "management",
    })
    session.resolve_pending_tool_calls()

    tool = [m for m in session.get_messages() if m.get("tool_call_id") == "call_1"][0]
    assert tool["content"][1]["image_url"]["url"] == _data_url(PNG_A)
    raw = open(session._filepath, encoding="utf-8").read()
    assert "base64," not in raw


def test_missing_blob_loads_as_placeholder_without_crashing(session, workspace):
    _add_screenshot(session, "call_1")
    root = blob_store.blobs_dir(os.path.join(workspace, "orbital"))
    for dirpath, _d, files in os.walk(root):
        for f in files:
            os.remove(os.path.join(dirpath, f))

    loaded = Session.load(session._filepath)
    tool = [m for m in loaded.get_messages() if m.get("role") == "tool"][0]
    assert tool["content"][1]["type"] == "text"
    assert "tool-results/blobs/" in tool["content"][1]["text"]


# ---------------------------------------------------------------------------
# trigger_type: automation sessions are marked on their session_start meta
# ---------------------------------------------------------------------------

def _meta_row(session):
    return [r for r in _disk_rows(session) if r.get("event") == "session_start"][0]


def test_schedule_fired_session_is_stamped(session):
    session.append({"role": "user", "content": "[Triggered by schedule 'Daily' (every day)]\n\nDo it"})
    assert session.trigger_type == "schedule"
    assert _meta_row(session)["trigger_type"] == "schedule"
    assert Session.load(session._filepath).trigger_type == "schedule"


def test_file_watch_fired_session_is_stamped(session):
    session.append({"role": "user", "content": "[Triggered by file_watch 'Inbox']\n\nChanged files: a"})
    assert _meta_row(session)["trigger_type"] == "file_watch"


def test_trigger_stamp_survives_rename(session):
    session.append({"role": "user", "content": "[Triggered by schedule 'Daily']\n\nDo it"})
    session.set_name("My renamed run")
    loaded = Session.load(session._filepath)
    assert loaded.name == "My renamed run"
    assert loaded.trigger_type == "schedule"


def test_plain_chat_is_not_stamped_and_later_triggers_do_not_count(session):
    session.append({"role": "user", "content": "hello"})
    session.append({"role": "user", "content": "[Triggered by schedule 'Daily']\n\nlater"})
    assert session.trigger_type is None
    assert "trigger_type" not in _meta_row(session)


def test_legacy_trigger_session_is_recognised_on_load(workspace):
    sessions_dir = os.path.join(workspace, "orbital", "sessions")
    os.makedirs(sessions_dir)
    path = os.path.join(sessions_dir, "proj_2222cccc.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "meta", "event": "session_start", "origin": "chat"}) + "\n")
        f.write(json.dumps({"role": "user", "content": "[Triggered by file_watch 'X']\n\nhi"}) + "\n")
    before = open(path, encoding="utf-8").read()
    loaded = Session.load(path)
    assert loaded.trigger_type == "file_watch"
    assert open(path, encoding="utf-8").read() == before, "load never rewrites the file"
