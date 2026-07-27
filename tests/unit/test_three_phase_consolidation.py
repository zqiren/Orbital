# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Unit tests: merge and archive are separate phases, and archiving is by id.

Measured on orbital-marketing (2026-07-27): asking one call to merge AND
archive made its response BIGGER, because archived entries had to come back
VERBATIM on top of the kept files. The pass ran 17.7 minutes and never
finished. The daemon already holds those exact bytes on disk, so making the
model retype them buys nothing and costs the whole budget.

The pipeline:

  P1 MERGE    (LLM, always)  dedup + supersede only. No archiving, no quota.
                             Shrinks when there is redundancy.
  P2 RELIEVE  (LLM, only if P1 left a file over target) archive BY MEM-ID plus
                             a one-line pointer stub. The response is a short
                             id list, so it cannot time out. "Still over after
                             the model just tried to merge" IS the definition
                             of "nothing left to dedup".
  P3 FLOOR    (deterministic, always last) demote/trim to target. Guarantees
                             the target whatever P1 and P2 did.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import memory_entries as _mem
from agent_os.agent import workspace_files as wsf
from agent_os.agent.providers.types import StreamChunk, TokenUsage
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine


def _session(sid="s_phase"):
    s = MagicMock()
    s.session_id = sid
    s.session_uuid = sid
    s.get_messages.return_value = []
    return s


def _scripted_provider(*responses):
    """A provider whose successive stream() calls yield the given payloads."""
    provider = MagicMock()
    provider.reasoning_locked_on = True
    provider.model = "test-model"
    provider.prompts = []
    queue = list(responses)

    async def _stream(messages, tools=None, **kwargs):
        provider.prompts.append(messages[-1]["content"])
        text = queue.pop(0) if queue else "{}"
        for i in range(0, len(text), 128):
            yield StreamChunk(text=text[i:i + 128])
        yield StreamChunk(is_final=True, usage=TokenUsage(1, 1))

    provider.stream = _stream
    provider.complete = AsyncMock(side_effect=AssertionError("must stream"))
    return provider


def _entries(n, *, key="decisions", pad=1100, tag=""):
    """n stamped entries, each big enough to matter against the budget."""
    # The mem-comment must sit on the entry's FIRST line — that is where
    # _parse_meta reads it. Put it anywhere else and stamp() treats the entry
    # as new and re-derives an id from the title.
    out = []
    for i in range(1, n + 1):
        meta = f'<!--mem id:e{i:02d} created:2026-07-{i:02d} touched:2026-07-{i:02d}{" tag:" + tag if tag else ""}-->'
        if key == "decisions":
            out.append(f"## 2026-07-{i:02d}: Decision {i} {meta}\n**Chose:** {'x' * pad}\n\n")
        else:
            out.append(f"{i}. **Lesson {i}.** {meta} {'y' * pad}\n")
    return "".join(out)


def _over_budget_ws(tmp_path) -> WorkspaceFileManager:
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", _entries(26))
    ws.write("lessons", _entries(18, key="lessons"))
    return ws


@pytest.fixture(autouse=True)
def _reset():
    wsf._completed_session_ends.clear()
    yield
    wsf._completed_session_ends.clear()


def _merge_only(ws, shrink=False):
    """A P1 response: the merged files, verbatim unless ``shrink`` is set.

    ``shrink=True`` models a merge that genuinely found redundancy and got
    BOTH entry files under target on its own — the case where phase 2 must
    not run at all.
    """
    return json.dumps({
        "project_state": "", "index": "",
        "decisions": _entries(3) if shrink else ws.read("decisions"),
        "lessons": _entries(2, key="lessons") if shrink else ws.read("lessons"),
    })


# ---------------------------------------------------------------------------
# P1: merge only — no archive quota, because that is what blew the budget
# ---------------------------------------------------------------------------

def test_merge_prompt_does_not_demand_archiving(tmp_path):
    ws = _over_budget_ws(tmp_path)
    prompt = ws.build_session_end_prompt({"recent_messages": [], "files_modified": []})
    lowered = prompt.lower()
    assert "you must archive at least" not in lowered
    assert "return them here verbatim" not in lowered


def test_merge_prompt_still_states_measured_sizes(tmp_path):
    """Sizes stay — the model cannot judge its own output length, and that is
    what made it archive nothing. Only the verbatim-return quota goes."""
    ws = _over_budget_ws(tmp_path)
    prompt = ws.build_session_end_prompt({"recent_messages": [], "files_modified": []})
    now = int(_mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions")))
    assert "MEASURED" in prompt.upper()
    assert str(now) in prompt


# ---------------------------------------------------------------------------
# P2: only when needed, by id, and cheap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase2_skipped_when_merge_got_under_target(tmp_path):
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(_merge_only(ws, shrink=True))
    await run_session_end_routine(_session("s_skip"), provider, ws, session_uuid="s_skip")
    assert len(provider.prompts) == 1, "no archive pass should have been needed"


@pytest.mark.asyncio
async def test_phase2_runs_when_merge_left_it_over_target(tmp_path):
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(
        _merge_only(ws),                                        # P1: no shrink
        json.dumps({"archive": {"decisions": [                  # P2: by id
            {"id": f"e{i:02d}", "pointer": f"cold decision {i}"} for i in range(1, 13)
        ]}}),
    )
    await run_session_end_routine(_session("s_p2"), provider, ws, session_uuid="s_p2")
    assert len(provider.prompts) == 2
    assert "MANIFEST" in provider.prompts[1].upper()


@pytest.mark.asyncio
async def test_phase2_manifest_carries_ids_not_bodies(tmp_path):
    """The whole point: the response is a short id list, so the call is cheap
    and cannot time out the way the verbatim version did."""
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(_merge_only(ws), json.dumps({"archive": {}}))
    await run_session_end_routine(_session("s_man"), provider, ws, session_uuid="s_man")
    manifest = provider.prompts[1]
    assert "e01" in manifest and "e26" in manifest        # ids present
    assert "x" * 200 not in manifest                       # bodies absent


@pytest.mark.asyncio
async def test_phase2_moves_bytes_exactly(tmp_path):
    ws = _over_budget_ws(tmp_path)
    before = ws.read("decisions")
    provider = _scripted_provider(
        _merge_only(ws),
        json.dumps({"archive": {"decisions": [{"id": "e01", "pointer": "first"}]}}),
    )
    await run_session_end_routine(_session("s_bytes"), provider, ws, session_uuid="s_bytes")

    archive = ws.read("decisions_archive") or ""
    body = before.split("## 2026-07-01: Decision 1", 1)[1].split("## 2026-07-02", 1)[0]
    assert body.strip()[:80] in archive, "archived text must be byte-identical"


@pytest.mark.asyncio
async def test_phase2_pointer_stub_is_left_behind(tmp_path):
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(
        _merge_only(ws),
        json.dumps({"archive": {"decisions": [
            {"id": "e01", "pointer": "early naming decisions — read before a rebrand"}
        ]}}),
    )
    await run_session_end_routine(_session("s_stub"), provider, ws, session_uuid="s_stub")
    assert "read before a rebrand" in (ws.read("decisions") or "")


@pytest.mark.asyncio
async def test_phase2_never_archives_a_pinned_entry(tmp_path):
    ws = WorkspaceFileManager(str(tmp_path))
    ws.ensure_dir()
    ws.write("decisions", _entries(26, tag="pinned"))
    provider = _scripted_provider(
        _merge_only(ws),
        json.dumps({"archive": {"decisions": [{"id": "e01", "pointer": "nope"}]}}),
    )
    await run_session_end_routine(_session("s_pin"), provider, ws, session_uuid="s_pin")
    assert "Decision 1" in (ws.read("decisions") or ""), "pinned must stay live"


@pytest.mark.asyncio
async def test_unknown_id_is_ignored_and_nothing_is_lost(tmp_path):
    ws = _over_budget_ws(tmp_path)
    before_entries = ws.read("decisions").count("## 2026-07-")
    provider = _scripted_provider(
        _merge_only(ws),
        json.dumps({"archive": {"decisions": [{"id": "does-not-exist", "pointer": "x"}]}}),
    )
    await run_session_end_routine(_session("s_bad"), provider, ws, session_uuid="s_bad")
    live = (ws.read("decisions") or "").count("## 2026-07-")
    archived = (ws.read("decisions_archive") or "").count("## 2026-07-")
    assert live + archived >= before_entries, "an unmatched id must not lose entries"


@pytest.mark.asyncio
async def test_phase2_failure_falls_through_to_the_floor(tmp_path):
    """A broken P2 response must not leave the file over budget — P3 is the
    guarantee, P2 is only the quality pass."""
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(_merge_only(ws), "not json at all")
    outcome = await run_session_end_routine(
        _session("s_p2fail"), provider, ws, session_uuid="s_p2fail"
    )
    after = _mem.est_tokens(_mem._budget_text(ws.read("decisions"), "decisions"))
    assert after <= _mem.consolidation_target("decisions")
    assert outcome in ("llm_merged", "partial_merge", "backstop_only")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_four_land_under_target_with_headroom(tmp_path):
    ws = _over_budget_ws(tmp_path)
    ws.write("state", "# S\n\n## Old\n" + "".join(f"filler {i}\n" for i in range(4000)))
    ws.write("index", "# INDEX\n" + "".join(f"- f{i}.md — x\n" for i in range(900)))

    provider = _scripted_provider(
        _merge_only(ws),
        json.dumps({"archive": {"decisions": [
            {"id": f"e{i:02d}", "pointer": f"cold {i}"} for i in range(1, 14)
        ]}}),
    )
    await run_session_end_routine(_session("s_e2e"), provider, ws, session_uuid="s_e2e")

    for key in ("state", "decisions", "lessons", "index"):
        now = _mem.est_tokens(_mem._budget_text(ws.read(key) or "", key))
        assert now <= _mem.consolidation_target(key), f"{key} still over target"


@pytest.mark.asyncio
async def test_archive_write_failure_does_not_crash_the_routine(tmp_path, monkeypatch):
    """Ported from the retired phase-1 archive tests: a failing archive write
    is a quality loss, never a crash and never a content loss — the floor
    still brings the file to target."""
    ws = _over_budget_ws(tmp_path)
    real_write = WorkspaceFileManager.write

    def _explode_on_archive(self, file_key, content):
        if file_key.endswith("_archive"):
            raise OSError("disk full")
        return real_write(self, file_key, content)

    monkeypatch.setattr(WorkspaceFileManager, "write", _explode_on_archive)

    provider = _scripted_provider(
        _merge_only(ws),
        json.dumps({"archive": {"decisions": [{"id": "e01", "pointer": "p"}]}}),
    )
    outcome = await run_session_end_routine(
        _session("s_oserr"), provider, ws, session_uuid="s_oserr"
    )
    assert outcome in ("llm_merged", "llm_merged_archived", "backstop_only")
    assert (ws.read("decisions") or "").count("## 2026-07-") > 0, "nothing lost"


@pytest.mark.asyncio
async def test_malformed_archive_payload_is_ignored(tmp_path):
    """A wrong-typed archive value must not crash the pass."""
    ws = _over_budget_ws(tmp_path)
    provider = _scripted_provider(_merge_only(ws), json.dumps({"archive": "not-a-dict"}))
    outcome = await run_session_end_routine(
        _session("s_wrongtype"), provider, ws, session_uuid="s_wrongtype"
    )
    assert outcome in ("llm_merged", "backstop_only")
