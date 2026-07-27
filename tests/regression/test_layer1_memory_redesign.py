# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""RED-before-GREEN gates for the Layer-1 memory redesign (TASK-mem-1).

Covers the 10 deliverable behaviors: injection cap + oldest-3, metadata survival,
demotion-to-archive, LESSONS not word-trimmed, soft-cap placement, no-delta gate,
contradiction dedup (fake provider), SESSION_LOG removal, registry-derived budget,
roster injection. No live LLM; the session-end pass uses a fake provider.
"""
from __future__ import annotations

import json
import os

import pytest

from agent_os.agent import memory_entries as M
from agent_os.agent import workspace_files as WF
from agent_os.agent.workspace_files import WorkspaceFileManager


def _wf(tmp_path) -> WorkspaceFileManager:
    wf = WorkspaceFileManager(str(tmp_path))
    wf.ensure_dir()
    return wf


def _decisions(n: int, *, touched=None, body="x" * 120) -> str:
    """n stamped DECISIONS entries, oldest-first, with explicit touched dates."""
    out = []
    for i in range(1, n + 1):
        t = touched(i) if touched else f"2026-01-{i:02d}"
        out.append(
            f"## 2026-01-{i:02d}: Decision {i} "
            f"<!--mem id:d{i} created:2026-01-{i:02d} touched:{t}-->\n"
            f"**Chose:** option {i}\n**Reason:** {body}\n**Rejected:** alt {i}\n\n"
        )
    return "".join(out)


# ---------------------------------------------------------------------------
# Gate 1 — injection cap + oldest-3
# ---------------------------------------------------------------------------

def test_injection_cap_keeps_budget_and_oldest_three(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 200, "hard": 400})
    content = _decisions(20, body="x" * 20)
    view = M.inject_view(content, "decisions", 400)
    # oldest-3 (foundational) always present
    for i in (1, 2, 3):
        assert f"Decision {i} " in view
    # newest present, a middle one omitted, with a pointer note
    assert "Decision 20" in view
    assert "Decision 10" not in view
    assert "omitted from this view" in view
    # injected size stays near budget (oldest-3 is the only allowed overshoot)
    assert M.est_tokens(view) <= 400 + 150


def test_injection_unchanged_when_fits(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 7000, "hard": 9000})
    content = _decisions(5)
    view = M.inject_view(content, "decisions", 9000)
    assert view is content  # same object — prefix-cache friendly


# ---------------------------------------------------------------------------
# Gate 2 — metadata survival through a strip-on-overwrite
# ---------------------------------------------------------------------------

def test_metadata_survives_overwrite_with_headers_stripped():
    old = (
        "## 2026-06-15: Keep API base url "
        "<!--mem id:keep-api created:2026-06-15 touched:2026-06-15-->\n"
        "**Chose:** moonshot\n\n"
    )
    # agent overwrites WITHOUT the metadata comment, body unchanged
    new = "## 2026-06-15: Keep API base url\n**Chose:** moonshot\n\n"
    stamped, warns = M.stamp(new, old, "decisions", today="2026-06-18")
    assert "id:keep-api" in stamped          # id preserved
    assert "created:2026-06-15" in stamped    # created preserved
    assert "touched:2026-06-15" in stamped    # body unchanged → touched preserved
    # body changed → touched advances
    new2 = "## 2026-06-15: Keep API base url\n**Chose:** minimax\n\n"
    stamped2, _ = M.stamp(new2, old, "decisions", today="2026-06-18")
    assert "touched:2026-06-18" in stamped2
    assert "created:2026-06-15" in stamped2


def test_lessons_renumbered_contiguously():
    broken = "1. **A.** a.\n\n4. **B.** b.\n\n7. **C.** c.\n"
    stamped, _ = M.stamp(broken, None, "lessons", today="2026-06-18")
    nums = [ln.split(".")[0] for ln in stamped.splitlines() if ln and ln[0].isdigit()]
    assert nums == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Gate 3 — demotion to archive (coldest touched first; protect oldest-3 + pinned)
# ---------------------------------------------------------------------------

def test_hard_cap_demotes_coldest_to_archive(tmp_path, monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    wf = _wf(tmp_path)
    # 10 entries; make the MIDDLE ones (4..7) coldest by touched so they demote,
    # NOT the protected oldest 3.
    def touched(i):
        return "2025-01-01" if i in (4, 5, 6, 7) else "2026-06-18"
    wf.write("decisions", _decisions(10, touched=touched, body="x" * 20))
    WF._apply_hard_caps(wf)

    kept = wf.read("decisions")
    archive = wf.read("decisions_archive")
    index = wf.read("index") or ""
    assert M.est_tokens(kept) <= M.FILE_BUDGETS["decisions"]["hard"]
    # oldest-3 retained in the live file
    for i in (1, 2, 3):
        assert f"Decision {i} " in kept
    # something demoted to archive, and INDEX points to it
    assert archive and "Decision" in archive
    assert "DECISIONS_ARCHIVE.md" in index
    # coldest (a middle entry) went to the archive, not entry 1
    assert "Decision 1" not in archive


def test_hard_cap_protects_pinned(tmp_path, monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 40, "hard": 70})
    wf = _wf(tmp_path)
    entries = _decisions(8)
    # pin entry 5 and make it coldest — it must NOT demote
    entries = entries.replace(
        "id:d5 created:2026-01-05 touched:2026-01-05",
        "id:d5 created:2026-01-05 touched:2000-01-01 tag:pinned",
    )
    wf.write("decisions", entries)
    WF._apply_hard_caps(wf)
    assert "Decision 5 " in (wf.read("decisions") or "")


# ---------------------------------------------------------------------------
# Gate 4 — LESSONS never word-trimmed
# ---------------------------------------------------------------------------

def test_lessons_long_playbook_survives_persist(tmp_path):
    long_body = " ".join(["word"] * 243)
    lesson = f"1. **Slate.js playbook.** {long_body}\n"
    stamped, warns = M.stamp(lesson, None, "lessons", today="2026-06-18")
    assert long_body in stamped  # byte-intact, not shortened
    assert not warns


# ---------------------------------------------------------------------------
# Gate 5 — soft-cap flag (logic + only fires over threshold)
# ---------------------------------------------------------------------------

def test_soft_flag_only_over_threshold(monkeypatch):
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 50, "hard": 80})
    small = _decisions(1, body="x" * 4)
    big = _decisions(20)
    assert M.soft_flag(small, "decisions") is None
    flag = M.soft_flag(big, "decisions")
    assert flag and "checkpoint_state" in flag and "consolidat" in flag and "tok" in flag and "entries" in flag


# ---------------------------------------------------------------------------
# Gate 6 — no-delta gate (no write since last cleanup → no-op marker logic)
# ---------------------------------------------------------------------------

def test_no_delta_gate(tmp_path):
    wf = _wf(tmp_path)
    wf.write("decisions", _decisions(2))
    # fresh project, no marker → delta present
    assert WF._has_cleanup_delta(wf) is True
    WF._write_cleanup_marker(wf)
    # nothing changed → no delta
    assert WF._has_cleanup_delta(wf) is False
    # a write flips it back to delta
    wf.write("decisions", _decisions(3))
    assert WF._has_cleanup_delta(wf) is True


# ---------------------------------------------------------------------------
# Gate 7 — contradiction dedup via the session-end pass (fake provider)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeProvider:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    async def complete(self, messages, disable_reasoning=False):
        self.calls += 1
        return _FakeResp(json.dumps(self._payload))

    async def stream(self, messages, tools=None, **kwargs):
        # The merge streams (a 15-min idle connection is unreliable); replay
        # complete()'s payload as chunks so this test keeps its own subject.
        from agent_os.agent.providers.types import StreamChunk, TokenUsage
        resp = await self.complete(messages)
        text = getattr(resp, "text", "") or ""
        for i in range(0, len(text), 256):
            yield StreamChunk(text=text[i:i + 256])
        yield StreamChunk(is_final=True, usage=TokenUsage(input_tokens=1, output_tokens=1))


class _FakeSession:
    def __init__(self):
        self.session_id = "s1"

    def get_messages(self):
        return [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_session_end_merges_supersedes(tmp_path):
    wf = _wf(tmp_path)
    # the real contradiction pair from the reports
    wf.write("decisions",
        "## 2026-06-15: QClaw PM Identity for Outreach\n**Chose:** Use QClaw PM identity.\n\n"
        "## 2026-06-15: Use Orbital Name (Not QClaw) for Outreach\n**Chose:** Orbital indie builder.\n\n")
    merged = (
        "## 2026-06-15: Use Orbital Name (Not QClaw) for Outreach\n"
        "**Chose:** Orbital indie builder.\n**Reason:** IP cleanliness.\n**Rejected:** QClaw PM identity (superseded).\n\n"
    )
    prov = _FakeProvider({"decisions": merged, "project_state": "", "lessons": "", "index": ""})
    await WF.run_session_end_routine(
        _FakeSession(), prov, wf, utility_provider=None,
        session_uuid="u1", bypass_idempotency=True, project_id="p1",
    )
    out = wf.read("decisions")
    assert "Use Orbital Name" in out
    assert "QClaw PM Identity for Outreach\n" not in out  # superseded entry gone
    assert "id:" in out  # stamped by the persist path
    assert prov.calls == 1


@pytest.mark.asyncio
async def test_session_end_no_delta_skips_llm(tmp_path):
    wf = _wf(tmp_path)
    wf.write("decisions", _decisions(2))
    WF._write_cleanup_marker(wf)  # mark as already cleaned
    prov = _FakeProvider({"decisions": "x"})
    await WF.run_session_end_routine(
        _FakeSession(), prov, wf, session_uuid="u2",
        bypass_idempotency=True, project_id="p1",
    )
    assert prov.calls == 0  # no-delta → no LLM call


@pytest.mark.asyncio
async def test_session_end_survives_llm_timeout_and_still_caps(tmp_path, monkeypatch):
    """LLM times out → deterministic hard cap still runs (no propagation)."""
    monkeypatch.setitem(M.FILE_BUDGETS, "decisions", {"soft": 120, "hard": 180})
    wf = _wf(tmp_path)
    wf.write("decisions", _decisions(12, body="x" * 20))

    class _Timeout:
        async def complete(self, messages, disable_reasoning=False):
            import asyncio
            await asyncio.sleep(999)

    # shrink the retry ladder so the test is fast
    monkeypatch.setattr(WF, "asyncio", WF.asyncio)
    import asyncio as _a
    orig_wait_for = _a.wait_for

    async def fast_wait_for(coro, timeout):
        # close the un-awaited coroutine to avoid a warning, then raise
        coro.close()
        raise _a.TimeoutError()

    monkeypatch.setattr(WF.asyncio, "wait_for", fast_wait_for)
    await WF.run_session_end_routine(
        _FakeSession(), _Timeout(), wf, session_uuid="u3",
        bypass_idempotency=True, project_id="p1",
    )
    monkeypatch.setattr(WF.asyncio, "wait_for", orig_wait_for)
    # deterministic cap ran despite the timeout
    assert M.est_tokens(wf.read("decisions")) <= M.FILE_BUDGETS["decisions"]["hard"]
    assert wf.read("decisions_archive")


# ---------------------------------------------------------------------------
# Gate 8 — SESSION_LOG removed
# ---------------------------------------------------------------------------

def test_session_log_removed_from_roster():
    assert "session_log" not in WF.FILE_NAMES
    assert "index" in WF.FILE_NAMES
    assert "decisions_archive" in WF.FILE_NAMES
    assert WF._RESUME_ORDER == ["state", "decisions", "lessons", "index"]
    assert not hasattr(WF, "_truncate_session_log")
    assert not hasattr(WF, "_get_session_log_lock")


def test_cold_resume_has_no_session_log(tmp_path):
    wf = _wf(tmp_path)
    wf.write("state", "current state")
    wf.write("index", "the map")
    ctx = wf.build_cold_resume_context()
    assert "current state" in ctx and "the map" in ctx
    assert "Session Log" not in ctx


def test_sub_agent_prompt_omits_session_log(tmp_path):
    from agent_os.agent.sub_agent_prompt import render_sub_agent_prompt
    out = render_sub_agent_prompt(str(tmp_path), None, "claude-code", ["claude-code"])
    assert "SESSION_LOG.md" not in out
    assert "INDEX.md" in out


# ---------------------------------------------------------------------------
# Gate 9 — registry-derived budget
# ---------------------------------------------------------------------------

def test_budget_derives_from_window():
    big = M.budgets_for_window(1_000_000)
    assert big["decisions"]["hard"] == 9000   # floor for a large window
    tiny = M.budgets_for_window(40_000)
    assert tiny["decisions"]["hard"] < 9000   # tiny window scales down
    missing = M.budgets_for_window(None)
    assert missing["decisions"]["hard"] == 9000  # missing → floor


def test_active_window_comes_from_registry():
    from agent_os.config.provider_registry import ProviderRegistry
    reg = ProviderRegistry()
    assert reg.get_context_window("minimax", "MiniMax-M3") == 1_000_000


# ---------------------------------------------------------------------------
# Gate 10 — roster injection (INDEX not CONTEXT; SESSION_LOG never)
# ---------------------------------------------------------------------------

def test_layer1_roster():
    from agent_os.agent.context import _LAYER1_FILES
    keys = [k for k, _ in _LAYER1_FILES]
    names = [n for _, n in _LAYER1_FILES]
    assert keys == ["state", "decisions", "lessons", "index"]
    assert "INDEX.md" in names and "CONTEXT.md" not in names
    assert "session_log" not in keys
