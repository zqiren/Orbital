# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 066 phase 1b — the chat-history API returns images exactly as before.

Session rows now store images in the blob store, but /chat rehydrates them
server-side: same JSON shape, base64 in JSON (the relay tunnel mangles binary
bodies and phones run an older frontend). Old inline rows pass through as-is,
and a row whose blob is gone comes back as its placeholder — never an error.
"""

from __future__ import annotations

import base64
import json
import os

from agent_os.agent import blob_store
from agent_os.agent.session import Session
from agent_os.api.routes.agents_v2 import _read_chat_messages, _read_chat_messages_single


PNG = b"\x89PNG\r\n\x1a\n" + b"Z" * 3000


def _content():
    return [
        {"type": "text", "text": "Image file: a.png"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(PNG).decode("ascii"),
            "detail": "low",
        }},
    ]


def _session_with_image(workspace):
    s = Session.new("proj_chat0001", workspace)
    s.append({"role": "user", "content": "look at this"})
    s.append({"role": "assistant", "source": "management", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]})
    s.append_tool_result("c1", _content(), meta={"image_path": "/u/a.png", "mime": "image/png"})
    return s


def _expected_rows(session):
    """What /chat returned before the blob store: the rows as in memory,
    plus the session_start meta line first."""
    with open(session._filepath, encoding="utf-8") as f:
        meta = json.loads(f.readline())
    return [meta] + session.get_messages()


def test_single_session_read_rehydrates_images(tmp_path):
    ws = str(tmp_path / "ws")
    s = _session_with_image(ws)
    assert "base64," not in open(s._filepath, encoding="utf-8").read()

    messages, total = _read_chat_messages_single(s._filepath, 0, 0)

    assert total == 4
    assert json.dumps(messages) == json.dumps(_expected_rows(s))


def test_paginated_read_rehydrates_images(tmp_path):
    ws = str(tmp_path / "ws")
    s = _session_with_image(ws)
    messages, total = _read_chat_messages_single(s._filepath, 1, 0)
    assert total == 4
    assert messages[0]["content"] == _content()


def test_unfiltered_read_rehydrates_images(tmp_path):
    ws = str(tmp_path / "ws")
    s = _session_with_image(ws)
    sessions_dir = os.path.dirname(s._filepath)
    for limit in (0, 2):
        messages, _total = _read_chat_messages(sessions_dir, limit, 0)
        tool = [m for m in messages if m.get("role") == "tool"][0]
        assert tool["content"] == _content()
        assert "image_blobs" not in tool["_meta"]


def test_legacy_inline_rows_pass_through_unchanged(tmp_path):
    sessions_dir = tmp_path / "ws" / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True)
    row = {"role": "tool", "content": _content(), "tool_call_id": "c1", "_meta": {"x": 1}}
    path = sessions_dir / "proj_legacy01.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    messages, _ = _read_chat_messages_single(str(path), 0, 0)
    assert json.dumps(messages) == json.dumps([row])


def test_missing_blob_returns_the_placeholder_row(tmp_path):
    ws = str(tmp_path / "ws")
    s = _session_with_image(ws)
    root = blob_store.blobs_dir(os.path.join(ws, "orbital"))
    for dirpath, _d, files in os.walk(root):
        for f in files:
            os.remove(os.path.join(dirpath, f))

    messages, _ = _read_chat_messages_single(s._filepath, 0, 0)
    tool = [m for m in messages if m.get("role") == "tool"][0]
    assert tool["content"][1]["type"] == "text"
    assert "tool-results/blobs/" in tool["content"][1]["text"]
