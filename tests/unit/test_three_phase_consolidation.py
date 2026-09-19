# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: the memory pass is two phases, and moving is by id.

History. Measured on orbital-marketing (2026-07-27): asking one call to merge
AND archive made its response BIGGER, because archived entries had to come
back VERBATIM. That led to three phases (whole-file merge, archive by id,
deterministic floor). Spec 089 dropped the first: the whole-file merge
regenerated files from a size-capped view and silently lost what it never saw.

The pipeline now:

  P1 EDITOR  (LLM, read-only tools) chooses entries to archive or merge BY ID;
                                    the daemon moves the exact bytes and leaves
                                    a pointer. The reply is a short id list.
  P2 FLOOR   (deterministic, always last) demote/trim toward target — age-based
                                    for PROJECT_STATE, coldest-first for
                                    DECISIONS/LESSONS, pointers every time.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.testutils import streamable

from agent_os.agent import memory_editor
from agent_os.agent import memory_entries as _mem
from agent_os.agent import workspace_files as wsf
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine


def _session(sid="s_phase"):
    s = MagicMock()
    s.session_id = sid
    s.session_uuid = sid
    s.get_messages.return_value = []
    return s


def _scripted_provider(*payloads):
    """A provider whose successive complete() calls answer with the payloads."""
    provider = streamable(AsyncMock())
    provider.prompts = []
    queue = list(payloads)

    async def _complete(messages, tools=None, **kwargs):
        provider.prompts.append(messages[-1]["content"])
        text = queue.pop(0) if queue else "{}"
        resp = MagicMock()
        resp.text = text
        resp.raw_message = {"role": "assistant", "content": text}
        resp.tool_calls = []
        return resp

    provider.complete = AsyncMock(side_effect=_complete)
    return provider


def _entries(n, *, key="decisions", pad=1100, tag=""):
    """n stamped entries, each big enough to matter against the budget."""
    out = []
    for i in range(1, n + 1):
        meta = f'<!--mem id:e{i:02d} created:2026-07-{i:02d} touched:2026-07-{i:02d}{" tag:" + tag if tag else ""}-->'
        if key == "decisions":
            out.append(f"## 2026-07-{i:02d}: Decision {i} {meta}\n**Chose:** {'x' * pad}\n\n")
        else:
            out.append(f"{i}. **Lesson {i}.** {'y' * pad} {meta}\n")
    return "".join(out)


def _over_budget_ws(tmp_path) -> WorkspaceFileManager:
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", _entries(26))
    ws.write("lessons", _entries(18, key="lessons"))
    return ws


def _archive(*ids, key="decisions"):
    name = {"decisions": "DECISIONS.md", "lessons": "LESSONS.md"}[key]
    return json.dumps({"archive": [
        {"file": name, "id": i, "pointer": f"cold {i}"} for i in ids
    ]})


@pytest.fixture(autouse=True)
def _reset():
    wsf._completed_session_ends.clear()
    memory_editor._RUNNING.clear()
    yield
    wsf._completed_session_ends.clear()


# ---------------------------------------------------------------------------
# P1: the editor sees ids and sizes, and returns ids — never bodies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_editor_prompt_carries_ids_and_measured_sizes(tmp_path):
    ws = _over_budget_ws(tmp_path)
    now = int(_mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions")))
    provider = _scripted_provider("{}")
    await run_session_end_routine(_session("s_ids"), provider, ws, session_uuid="s_ids")
    prompt = provider.prompts[0]
    assert "id:e01" in prompt and "id:e26" in prompt
    assert f"{now} tokens" in prompt


@pytest.mark.asyncio
async def test_editor_moves_bytes_exactly(tmp_path):
    ws = _over_budget_ws(tmp_path)
    before = ws.read("decisions")
    provider = _scripted_provider(_archive("e05"))
    await run_session_end_routine(_session("s_bytes"), provider, ws, session_uuid="s_bytes")

    archive = ws.read("decisions_archive") or ""
    entry = "## 2026-07-05: Decision 5" + before.split("## 2026-07-05: Decision 5", 1)[1].split("## 2026-07-06", 1)[0]
    assert entry.rstrip("\n") in archive, "archived text must be byte-identical"


@pytest.mark.asyncio
async def test_editor_pointer_with_id_is_left_behind(tmp_path):
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(json.dumps({"archive": [
        {"file": "DECISIONS.md", "id": "e05", "pointer": "early naming decisions — read before a rebrand"}
    ]}))
    await run_session_end_routine(_session("s_stub"), provider, ws, session_uuid="s_stub")
    assert "id:e05] early naming decisions — read before a rebrand → DECISIONS_ARCHIVE.md" in ws.read("decisions")


@pytest.mark.asyncio
async def test_editor_never_archives_a_pinned_entry(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", _entries(26, tag="pinned"))
    provider = _scripted_provider(_archive("e05"))
    await run_session_end_routine(_session("s_pin"), provider, ws, session_uuid="s_pin")
    assert "Decision 5 <!--mem id:e05" in (ws.read("decisions") or ""), "pinned must stay live"


@pytest.mark.asyncio
async def test_unknown_id_is_ignored_and_nothing_is_lost(tmp_path):
    ws = _over_budget_ws(tmp_path)
    before_entries = ws.read("decisions").count("## 2026-07-")
    provider = _scripted_provider(_archive("does-not-exist"))
    await run_session_end_routine(_session("s_bad"), provider, ws, session_uuid="s_bad")
    live = (ws.read("decisions") or "").count("## 2026-07-")
    archived = (ws.read("decisions_archive") or "").count("## 2026-07-")
    assert live + archived == before_entries, "an unmatched id must not lose entries"


@pytest.mark.asyncio
async def test_malformed_payload_is_ignored(tmp_path):
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(json.dumps({"archive": "not-a-list", "merge": 3}))
    outcome = await run_session_end_routine(
        _session("s_wrongtype"), provider, ws, session_uuid="s_wrongtype")
    assert outcome == "no_change"


# ---------------------------------------------------------------------------
# P2: the floor is the guarantee
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_editor_failure_falls_through_to_the_floor(tmp_path):
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(*(["not json at all"] * memory_editor.EDITOR_MAX_CALLS))
    outcome = await run_session_end_routine(
        _session("s_fail"), provider, ws, session_uuid="s_fail")
    after = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    assert after <= _mem.consolidation_target("decisions")
    assert outcome == "backstop_only"


@pytest.mark.asyncio
async def test_all_four_land_under_target_with_headroom(tmp_path):
    """With nothing recent in the way, every file reaches its target."""
    ws = _over_budget_ws(tmp_path)
    ws.write("state", "# S\n\n## Old\n" + "".join(
        f"- old item {i} " + "o" * 600
        + f"\n  <!--mem id:s{i:05d} created:2026-01-01 touched:2026-01-01-->\n"
        for i in range(20)
    ))
    ws.write("index", "# INDEX\n" + "".join(f"- f{i}.md — x\n" for i in range(900)))

    provider = _scripted_provider(_archive(*[f"e{i:02d}" for i in range(4, 12)]))
    await run_session_end_routine(_session("s_e2e"), provider, ws, session_uuid="s_e2e")

    for key in ("state", "decisions", "lessons", "index"):
        now = _mem.est_tokens(_mem._budget_text(ws.read(key) or "", key))
        assert now <= _mem.consolidation_target(key), f"{key} still over target"


@pytest.mark.asyncio
async def test_archive_write_failure_does_not_crash_or_lose_content(tmp_path, monkeypatch):
    """A failing archive write is a quality loss, never a crash and never a
    content loss: the live file is not shrunk when its archive could not be
    written."""
    ws = _over_budget_ws(tmp_path)
    before = ws.read("decisions")
    real_write = WorkspaceFileManager.write

    def _explode_on_archive(self, file_key, content):
        if file_key.endswith("_archive"):
            raise OSError("disk full")
        return real_write(self, file_key, content)

    monkeypatch.setattr(WorkspaceFileManager, "write", _explode_on_archive)
    provider = _scripted_provider(_archive("e05"))
    outcome = await run_session_end_routine(
        _session("s_oserr"), provider, ws, session_uuid="s_oserr")
    assert outcome in ("no_change", "edited")
    after = ws.read("decisions") or ""
    for i in range(1, 27):
        assert f"Decision {i} <!--mem id:e{i:02d}" in after or f"id:e{i:02d}]" not in after
    assert after.count("## 2026-07-") == before.count("## 2026-07-")
