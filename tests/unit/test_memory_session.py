# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 124 §3.7 — the Memory session replaces the memory editor.

The pass keeps its deterministic shell (trigger, backup, prepass, id
conservation, outcome log) and swaps the read-only JSON-choices loop for ONE
bounded turn of a real session on the project's own AgentLoop: same system
prompt, same tool registry, Layer-1 rows byte-identical to the triggering
run's snapshot (so the provider prefix hits), edit/write confined to
``orbital/*.md``, a tool cap the daemon enforces, no record turn, and
``caller=memory`` on its cache-audit lines.

These tests cover: the task message (fields filled, bytes fixed), the pass
shell with a fake turn runner (outcomes, conservation, INDEX archive lines,
run record), the loop seams (tool guard, tool cap, per-run stats, seen-set),
write-path conservation per file, the context manager's adopted snapshot,
and the agent manager's slot bypass + session traits. The editor's deleted
surface is asserted gone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_os.agent import context as context_mod
from agent_os.agent import memory_editor as E
from agent_os.agent import memory_entries as M
from agent_os.agent import state_blocks
from agent_os.agent import workspace_files as WF
from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.prompt_builder import Autonomy, PromptContext
from agent_os.agent.providers.types import LLMResponse, StreamChunk, TokenUsage
from agent_os.agent.session import Session, persist_user_row
from agent_os.agent.tools.base import ToolResult
from agent_os.agent.workspace_files import WorkspaceFileManager, run_session_end_routine

TODAY = "2026-10-21"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _bullet(text, eid, created="2026-09-01"):
    return f"- {text}\n  <!--mem id:{eid} created:{created} touched:{created}-->"


STATE = "## Now\n" + "\n".join([
    _bullet("GOAI booth prep — posters printed", "goai01", "2026-09-14"),
    _bullet("Newsjack radar 9/18: no new target", "news18", "2026-09-18"),
    _bullet("Cold-DM automation paused since 08-15", "cdm001", "2026-08-10"),
]) + "\n"
DECISIONS = "# DECISIONS\n\n" + _bullet("TabTin → watchlist", "tabtin") + "\n  **Chose:** A.\n\n" \
    + _bullet("PenguinHarness → watchlist", "penguin") + "\n  **Chose:** B.\n\n" \
    + _bullet("Pinned rule", "pinrule").replace("touched:2026-09-01", "touched:2026-09-01 tag:pinned") + "\n"
LESSONS = "# LESSONS\n\n" + _bullet("Curl beats the browser for GitHub.", "curl") + "\n"
INDEX = "# INDEX\n\n- agent_output/ — reports\n"


@pytest.fixture
def wf(tmp_path):
    w = WorkspaceFileManager(str(tmp_path))
    w.write("state", STATE)
    w.write("decisions", DECISIONS)
    w.write("lessons", LESSONS)
    w.write("index", INDEX)
    return w


@pytest.fixture(autouse=True)
def _clear_single_flight():
    E._RUNNING.clear()
    WF._completed_session_ends.clear()
    yield
    E._RUNNING.clear()


def _over_budget(wf):
    big = "\n".join(_bullet(f"recent item {i} " + "r" * 400, f"big{i:03d}", "2026-09-18") for i in range(25))
    wf.write("state", STATE + big + "\n")


def _turn(**fields) -> E.TurnResult:
    base = dict(mode="snapshot", llm_calls=1, tool_calls=0, input_tokens=100,
                cached_tokens=90, output_tokens=10, ended="text")
    base.update(fields)
    return E.TurnResult(**base)


# ---------------------------------------------------------------------------
# 1. the deleted editor
# ---------------------------------------------------------------------------

def test_the_editor_contract_is_gone():
    for name in ("build_prompt", "_SYSTEM", "apply_choices", "plan_entries", "parse_choices",
                 "build_tools", "EDITOR_TOOL_RESULT_CHARS", "apply_ask_closes", "group_choices",
                 "_complete_once_or_retry", "run_editor", "FilePlan", "_run_tool", "_normalize_calls"):
        assert not hasattr(E, name), name
    # kept: the shell
    for name in ("backup_live_files", "EDITOR_MAX_CALLS", "is_running", "claim", "release",
                 "write_marker", "auto_run_due", "sessions_since", "_keep_archive_lines",
                 "_write_run_record", "run_pass", "build_task_message", "memory_tool_guard"):
        assert hasattr(E, name), name


# ---------------------------------------------------------------------------
# 2. the task message
# ---------------------------------------------------------------------------

def _sizes():
    return {
        "state": (3816, 1800, 1200), "decisions": (6415, 7000, 6000),
        "lessons": (4930, 5000, 4000), "index": (1320, 1500, 900),
    }


def test_task_message_fields_are_filled_and_the_rest_is_fixed():
    sessions = [
        {"session_uuid": "orbital-marketing_c2465b5c", "last_activity_at": "2026-10-20T13:02:11+00:00",
         "name": "观猹展位印刷物料"},
        {"session_uuid": "orbital-marketing_008a87f4", "last_activity_at": "2026-10-20T11:40:00+00:00",
         "name": "Hollyclaw 评估"},
    ]
    text = E.build_task_message(
        sizes=_sizes(), folded=[], over=["state"], forced=False,
        pointer_paths=["agent_output/competitor-dynamics/", "agent_output/2026-10-14-dahuang-proposal.md"],
        last_pass=datetime(2026, 10, 20, 9, 3, tzinfo=timezone.utc), sessions=sessions,
        today="2026-10-20", tool_cap=14,
    )
    assert text.startswith("Tidy this project's memory. You are in the project's own context: the system\n")
    assert "you never touch\n<!--mem …--> lines." in text
    assert "SIZES (bullet lines only; indented detail does not count toward soft)\n" in text
    assert "- PROJECT_STATE.md: 3,816 tokens (soft 1,800, target 1,200) — OVER\n" in text
    assert "- DECISIONS.md: 6,415 (soft 7,000) — within\n" in text
    assert "- LESSONS.md: 4,930 (soft 5,000) — within\n" in text
    assert "- INDEX.md: 1,320 (soft 1,500) — within\n" in text
    assert "Folded in the injected view: none.\n" in text
    assert ("1. PROJECT_STATE.md is over: move entries that are finished or superseded to\n"
            "   PROJECT_STATE_ARCHIVE.md; merge entries that say the same thing; fold\n"
            "   indented detail into its line. The sessions listed below are your evidence\n"
            "   for \"finished\" (read_session with since=…, grep, small limits).\n") in text
    assert ("2. These paths lost their only live mention since the last pass:\n"
            "   agent_output/competitor-dynamics/, agent_output/2026-10-14-dahuang-proposal.md\n"
            "   Decide whether INDEX.md needs a line for them.") in text
    assert "3. If you archive into a file, update that archive's line in INDEX.md" in text
    assert "4. If nothing is finished or superseded, say so and stop. Never force cuts.\n" in text
    assert "`## [archived 2026-10-20]`" in text and "`## [merged 2026-10-20 into id:<new id>]`" in text
    assert "Never touch entries tagged tag:pinned." in text
    assert "Close an ask in orbital/ASKS.md only with the user's own words quoted" in text
    assert "INDEX.md lists folders and topics, not dated files. Keep every *_ARCHIVE.md line.\n" in text
    assert "LAST PASS: 2026-10-20T09:03\n" in text
    assert ("SESSIONS ACTIVE SINCE THEN (newest first):\n"
            "- orbital-marketing_c2465b5c  last_activity=2026-10-20T13:02  观猹展位印刷物料\n"
            "- orbital-marketing_008a87f4  last_activity=2026-10-20T11:40  Hollyclaw 评估\n") in text
    assert text.endswith("At most 14 tool calls. Finish with one line saying what moved and where.")
    # byte-stable
    assert text == E.build_task_message(
        sizes=_sizes(), folded=[], over=["state"], forced=False,
        pointer_paths=["agent_output/competitor-dynamics/", "agent_output/2026-10-14-dahuang-proposal.md"],
        last_pass=datetime(2026, 10, 20, 9, 3, tzinfo=timezone.utc), sessions=sessions,
        today="2026-10-20", tool_cap=14,
    )


def test_task_message_do_items_follow_the_trigger():
    text = E.build_task_message(
        sizes=_sizes(), folded=["DECISIONS.md"], over=["decisions", "index"], forced=False,
        pointer_paths=[], last_pass=None, sessions=[], today=TODAY, tool_cap=12,
    )
    assert "Folded in the injected view: DECISIONS.md.\n" in text
    assert "1. DECISIONS.md is over: move decisions that were superseded in practice to\n   DECISIONS_ARCHIVE.md" in text
    assert "2. INDEX.md is over: merge lines upward into folder or topic lines" in text
    assert "3. DECISIONS.md's bullet lines do not fit the injected view" in text
    assert "lost their only live mention" not in text
    assert "LAST PASS: never\n" in text
    assert "SESSIONS ACTIVE SINCE THEN (newest first):\n(none)\n" in text
    assert text.endswith("At most 12 tool calls. Finish with one line saying what moved and where.")
    # a forced pass with nothing over says so instead of inventing a cut
    forced = E.build_task_message(
        sizes=_sizes(), folded=[], over=[], forced=True, pointer_paths=[],
        last_pass=None, sessions=[], today=TODAY, tool_cap=12,
    )
    assert "1. This pass was requested explicitly (checkpoint_state)" in forced
    assert "If nothing is finished or superseded, say so and stop." in forced


# ---------------------------------------------------------------------------
# 3. the tool guard
# ---------------------------------------------------------------------------

def test_tool_guard_confines_edit_and_write_to_orbital_md(tmp_path):
    guard = E.memory_tool_guard(str(tmp_path))
    ok = {"name": "edit", "arguments": {"path": "orbital/PROJECT_STATE.md"}}
    assert guard(ok) is None
    assert guard({"name": "write", "arguments": {"path": os.path.join(str(tmp_path), "orbital", "INDEX.md")}}) is None
    assert guard({"name": "write", "arguments": {"path": "orbital/ASKS.md"}}) is None
    for bad in ("README.md", "orbital/sessions/x.jsonl", "orbital/notes/x.md", "orbital/x.txt",
                "../orbital/PROJECT_STATE.md", "/etc/passwd"):
        err = guard({"name": "write", "arguments": {"path": bad}})
        assert err and err.startswith("Error") and "orbital/*.md" in err, bad
    # every other tool is unrestricted
    assert guard({"name": "read", "arguments": {"path": "src/main.py"}}) is None
    assert guard({"name": "shell", "arguments": {"command": "ls"}}) is None
    assert guard({"name": "edit", "arguments": {}}) is None   # the tool reports its own missing arg


# ---------------------------------------------------------------------------
# 4. the pass shell with a fake turn runner
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pass_backs_up_prepasses_sends_the_task_and_reports_edited(wf, caplog):
    seen = {}

    async def run_turn(task_text):
        seen["task"] = task_text
        # the agent archives cdm001: archive first, then the live file
        block = _bullet("Cold-DM automation paused since 08-15", "cdm001", "2026-08-10")
        wf.write("state_archive", (wf.read("state_archive") or "# A\n") + f"\n## [archived {TODAY}]\n\n{block}\n")
        wf.write("state", wf.read("state").replace(block + "\n", ""))
        return _turn(tool_calls=3)

    with caplog.at_level(logging.INFO):
        result = await E.run_pass(wf, run_turn, today=TODAY, project_id="p1", over_keys=["state"])
    assert result.ok and result.outcome == "edited"
    assert result.restored == [] and result.written == ["state", "index"]   # index: the daemon's archive line
    assert "cdm001" not in wf.read("state") and "id:cdm001" in wf.read("state_archive")
    assert "- id:cdm001 · " in wf.read("state_archive")                    # contents table
    assert "- orbital/PROJECT_STATE_ARCHIVE.md — 1 archived entry" in wf.read("index")
    assert "PROJECT_STATE.md is over" in seen["task"]
    assert f"At most {E.EDITOR_MAX_CALLS} tool calls." in seen["task"]
    assert os.path.isfile(os.path.join(result.backup_dir, "PROJECT_STATE.md"))
    with open(os.path.join(result.backup_dir, "editor-run.json")) as f:
        record = json.load(f)
    assert record["outcome"] == "edited" and record["restored"] == [] and record["turn"]["tool_calls"] == 3
    summary = [r.getMessage() for r in caplog.records if r.getMessage().startswith("memory session:")]
    assert summary and "mode=snapshot calls=1 input=100 cached=90 output=10 cost=$" in summary[-1]


@pytest.mark.asyncio
async def test_pass_outcomes_no_change_index_only_skipped_failed(wf):
    async def nothing(task_text):
        return _turn()
    r = await E.run_pass(wf, nothing, today=TODAY)
    assert r.ok and r.outcome == "no_change" and r.written == []

    async def index_only(task_text):
        wf.write("index", wf.read("index") + "- docs/ — notes\n")
        return _turn(tool_calls=1)
    r = await E.run_pass(wf, index_only, today=TODAY)
    assert r.ok and r.outcome == "index_only" and r.written == ["index"]

    async def declined(task_text):
        return None
    r = await E.run_pass(wf, declined, today=TODAY)
    assert r.ok and r.outcome == "skipped"

    async def boom(task_text):
        raise RuntimeError("provider 500")
    r = await E.run_pass(wf, boom, today=TODAY)
    assert not r.ok and r.outcome == "failed" and "provider 500" in (r.error or "")


@pytest.mark.asyncio
async def test_pass_restores_an_entry_dropped_without_archiving(wf, caplog):
    async def deleter(task_text):
        text = wf.read("decisions")
        with open(wf._file_path("decisions"), "w", encoding="utf-8") as f:   # bypasses the write guard
            f.write(text.replace(_bullet("TabTin → watchlist", "tabtin") + "\n  **Chose:** A.\n\n", ""))
        return _turn(tool_calls=1)
    with caplog.at_level(logging.WARNING):
        r = await E.run_pass(wf, deleter, today=TODAY)
    assert r.ok and r.outcome == "restored" and r.restored == ["tabtin"]
    assert "- TabTin → watchlist\n  <!--mem id:tabtin" in wf.read("decisions")
    assert any("restored=['tabtin']" in rec.getMessage() for rec in caplog.records)


@pytest.mark.asyncio
async def test_pass_keeps_index_archive_lines_the_agent_dropped(wf):
    WF._append_archive_section(wf, "lessons_archive", f"## [archived {TODAY}]", _bullet("old rule", "old-rule"))
    WF.refresh_index_archive_lines(wf)
    line = [ln for ln in wf.read("index").split("\n") if "LESSONS_ARCHIVE.md" in ln][0]
    described = line.replace("to be described", "old rules about curl")
    wf.write("index", wf.read("index").replace(line, described))

    async def rewriter(task_text):
        wf.write("index", "# INDEX\n\n- src/ — code\n")
        return _turn(tool_calls=1)
    r = await E.run_pass(wf, rewriter, today=TODAY)
    assert r.outcome == "index_only"
    idx = wf.read("index")
    assert "- src/ — code" in idx and "old rules about curl" in idx


@pytest.mark.asyncio
async def test_pass_is_bounded_by_the_total_timeout(wf, monkeypatch):
    monkeypatch.setattr(E, "EDITOR_TOTAL_TIMEOUT_S", 0.05)

    async def hangs(task_text):
        await asyncio.sleep(10)
    r = await E.run_pass(wf, hangs, today=TODAY)
    assert not r.ok and r.outcome == "failed"


# ---------------------------------------------------------------------------
# 5. run_session_end_routine: the shell around the pass
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_routine_runs_the_memory_turn_and_advances_the_watermark(wf):
    _over_budget(wf)
    calls = []

    async def run_turn(task_text):
        calls.append(task_text)
        return _turn()
    out = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True,
                                        run_turn=run_turn, trigger="over_budget")
    assert out == "no_change" and len(calls) == 1
    assert E.read_marker(wf.dir).get("editor_last_run")
    assert await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True,
                                         run_turn=run_turn) == "no_delta"


@pytest.mark.asyncio
async def test_routine_without_a_runner_skips_and_leaves_the_files_dirty(wf):
    _over_budget(wf)
    out = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True)
    assert out == "skipped"
    assert "editor_last_run" not in E.read_marker(wf.dir)


@pytest.mark.asyncio
async def test_routine_failure_is_backstop_only_and_leaves_the_watermark(wf):
    _over_budget(wf)

    async def boom(task_text):
        raise RuntimeError("x")
    out = await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True, run_turn=boom)
    assert out == "backstop_only"
    assert "editor_last_run" not in E.read_marker(wf.dir)


@pytest.mark.asyncio
async def test_routine_gates_not_needed_forced_in_flight(wf):
    async def run_turn(task_text):
        return _turn()
    assert await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True,
                                         run_turn=run_turn) == "not_needed"
    assert await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True,
                                         run_turn=run_turn, force=True) == "no_change"
    _over_budget(wf)
    E.claim(wf.workspace)
    assert await run_session_end_routine(None, None, wf, session_uuid="s", bypass_idempotency=True,
                                         run_turn=run_turn) == "in_flight"


# ---------------------------------------------------------------------------
# 6. write-path conservation per file (coordinator decision 2026-10-21)
# ---------------------------------------------------------------------------

def test_decisions_and_lessons_are_strict_even_with_a_fresh_entry_in_the_gap():
    prev = _bullet("a", "a1") + "\n" + _bullet("b", "b1") + "\n" + _bullet("c", "c1") + "\n"
    new = _bullet("a", "a1") + "\n- b reworded entirely\n" + _bullet("c", "c1") + "\n"
    for key in ("decisions", "lessons"):
        text, restored = M.conserve_ids(prev, new, set(), key=key)
        assert restored == ["b1"], key
        assert "id:b1" in text and "- b reworded entirely" in text
    # archived first → may leave
    text, restored = M.conserve_ids(prev, new, {"b1"}, key="decisions")
    assert restored == [] and text == new


def test_state_with_a_seen_set_protects_only_unseen_ids():
    prev = _bullet("a", "a1") + "\n" + _bullet("b", "b1") + "\n" + _bullet("other writer", "x9") + "\n"
    new = _bullet("a", "a1") + "\n"
    text, restored = M.conserve_ids(prev, new, set(), key="state", seen={"a1", "b1"})
    assert restored == ["x9"]
    assert "id:x9" in text and "id:b1" not in text
    # everything seen may be dropped outright (overwrite contract)
    text, restored = M.conserve_ids(prev, new, set(), key="state", seen={"a1", "b1", "x9"})
    assert restored == [] and text == new


def test_state_without_a_seen_set_keeps_the_gap_rule():
    prev = _bullet("radar 9/18: none", "n18") + "\n" + _bullet("booth prep", "g01") + "\n"
    reworded = "- radar 9/18: one target found\n" + _bullet("booth prep", "g01") + "\n"
    text, restored = M.conserve_ids(prev, reworded, set(), key="state")
    assert restored == []                      # a reworded line in the same gap is an overwrite
    bare = _bullet("booth prep", "g01") + "\n"
    text, restored = M.conserve_ids(prev, bare, set(), key="state")
    assert restored == ["n18"]


def test_process_on_write_uses_the_bound_seen_set_for_state(tmp_path):
    ws = str(tmp_path)
    wf = WorkspaceFileManager(ws)
    wf.write("state", _bullet("a", "a1") + "\n" + _bullet("b", "b1") + "\n")
    path = wf._file_path("state")
    token = M.bind_seen_state_ids({"a1", "b1"})
    try:
        # another writer adds an entry after the snapshot
        with open(path, "a", encoding="utf-8") as f:
            f.write(_bullet("late arrival", "x9") + "\n")
        merged, warnings = M.process_on_write(ws, path, _bullet("a", "a1") + "\n", today=TODAY)
        assert "id:x9" in merged and "id:b1" not in merged
        assert any("x9" in w for w in warnings)
        # the write's own ids are now seen (minus the restored one)
        assert "x9" not in M.seen_state_ids() and "a1" in M.seen_state_ids()
    finally:
        M.unbind_seen_state_ids(token)
    assert M.seen_state_ids() is None
    # a decision drop is strict regardless of any seen-set
    wf.write("decisions", _bullet("d", "d1") + "\n" + _bullet("e", "e1") + "\n")
    merged, _ = M.process_on_write(ws, wf._file_path("decisions"), _bullet("d", "d1") + "\n", today=TODAY)
    assert "id:e1" in merged


# ---------------------------------------------------------------------------
# 7. the loop seams: guard, cap, stats, no record turn, seen-set binding
# ---------------------------------------------------------------------------

class _PromptBuilder:
    def build(self, context):
        return ("cached-system-prefix", "semi-stable-suffix", "dynamic-runtime")


def _prompt_context(workspace: str) -> PromptContext:
    return PromptContext(
        workspace=workspace, model="test-model", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "edit", "write"], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
    )


_USAGE = TokenUsage(input_tokens=100, output_tokens=7, cache_read_tokens=80)


def _reply(text: str) -> list[StreamChunk]:
    return [StreamChunk(text=text), StreamChunk(is_final=True, usage=_USAGE, finish_reason="stop")]


def _tool_calls(*calls) -> list[StreamChunk]:
    deltas = []
    for idx, (call_id, name, args) in enumerate(calls):
        deltas.append(StreamChunk(tool_calls_delta=[{
            "index": idx, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]))
    deltas.append(StreamChunk(is_final=True, usage=_USAGE, finish_reason="tool_calls"))
    return deltas


class _ScriptedLLM:
    provider = "test"
    model = "test-model"
    sdk = "openai"

    def __init__(self, script):
        self._script = script
        self.calls = 0
        self.sent = []

    async def stream(self, messages, tools=None):
        idx = self.calls
        self.calls += 1
        self.sent.append(list(messages))
        for chunk in self._script[min(idx, len(self._script) - 1)]:
            yield chunk

    async def complete(self, messages, tools=None):
        return LLMResponse(raw_message={"role": "assistant", "content": "x"}, text="x",
                           tool_calls=[], usage=_USAGE)


class _Tools:
    def __init__(self, workspace):
        self._ws = workspace
        self.executed = []

    def schemas(self):
        return [{"type": "function", "function": {"name": n, "description": n,
                 "parameters": {"type": "object", "properties": {}}}} for n in ("read", "edit", "write")]

    def tool_names(self):
        return ["read", "edit", "write"]

    def is_async(self, name):
        return False

    def reset_run_state(self):
        pass

    def execute(self, name, arguments):
        self.executed.append((name, dict(arguments)))
        path = os.path.join(self._ws, arguments.get("path", ""))
        if name == "write":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            content = arguments.get("content", "")
            content, _w = M.process_on_write(self._ws, path, content, today=TODAY)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            return ToolResult(content=f"Wrote {arguments.get('path')}")
        if name == "read":
            with open(path, "r", encoding="utf-8") as fh:
                return ToolResult(content=fh.read())
        return ToolResult(content="ok")


def _seed(workspace):
    orb = os.path.join(workspace, "orbital")
    os.makedirs(orb, exist_ok=True)
    files = {
        "PROJECT_STATE.md": "# State\n" + _bullet("focus: launch", "a1") + "\n" + _bullet("next: copy", "b1") + "\n",
        "DECISIONS.md": "# Decisions\n" + _bullet("Vite over CRA", "v1") + "\n",
        "LESSONS.md": "# Lessons\n- never trust a green local build\n",
        "INDEX.md": "# Index\n- docs/ — design notes\n",
    }
    for name, text in files.items():
        with open(os.path.join(orb, name), "w", encoding="utf-8") as fh:
            fh.write(text)


def _rig(tmp_path, llm, *, origin="chat", **loop_kw):
    workspace = str(tmp_path)
    _seed(workspace)
    session = Session.new("sess-mem-0001", workspace)
    session.origin = origin
    tools = _Tools(workspace)
    ctx = ContextManager(session, _PromptBuilder(), _prompt_context(workspace), model_context_limit=200_000)
    loop = AgentLoop(session, llm, tools, ctx, project_dir=workspace, max_iterations=20, **loop_kw)
    persist_user_row(session, "Tidy.")
    return session, tools, loop, ctx


@pytest.mark.asyncio
async def test_tool_guard_refuses_a_write_outside_orbital_without_executing(tmp_path):
    llm = _ScriptedLLM([
        _tool_calls(("c1", "write", {"path": "README.md", "content": "x"}),
                    ("c2", "write", {"path": "orbital/INDEX.md", "content": "# Index\n- src/ — code\n"})),
        _reply("done"),
    ])
    session, tools, loop, _ = _rig(tmp_path, llm, origin="memory", memory_record_enabled=False,
                                   tool_guard=E.memory_tool_guard(str(tmp_path)))
    await loop.run()
    assert [n for n, _ in tools.executed] == ["write"]
    assert not os.path.exists(os.path.join(str(tmp_path), "README.md"))
    results = [m for m in session.get_messages() if m.get("role") == "tool"]
    assert results[0]["content"].startswith("Error") and "orbital/*.md" in results[0]["content"]
    assert "- src/ — code" in open(os.path.join(str(tmp_path), "orbital", "INDEX.md"), encoding="utf-8").read()


@pytest.mark.asyncio
async def test_tool_cap_ends_the_turn_and_skips_the_rest_of_the_batch(tmp_path):
    llm = _ScriptedLLM([
        _tool_calls(("c1", "read", {"path": "orbital/INDEX.md"}), ("c2", "read", {"path": "orbital/INDEX.md"}),
                    ("c3", "read", {"path": "orbital/INDEX.md"})),
        _tool_calls(("c4", "read", {"path": "orbital/INDEX.md"})),
        _reply("never reached"),
    ])
    session, tools, loop, _ = _rig(tmp_path, llm, origin="memory", memory_record_enabled=False, tool_call_cap=2)
    await loop.run()
    assert len(tools.executed) == 2 and llm.calls == 1
    assert loop._loop_exit_path == "tool_cap"
    rows = {m["tool_call_id"]: m["content"] for m in session.get_messages() if m.get("role") == "tool"}
    assert set(rows) == {"c1", "c2", "c3"}
    assert rows["c3"].startswith("Skipped") and "tool budget" in rows["c3"]
    assert loop.last_run_stats["tool_calls"] == 2 and loop.last_run_stats["llm_calls"] == 1
    assert loop.last_run_stats["ended"] == "tool_cap"


@pytest.mark.asyncio
async def test_last_run_stats_count_tokens_and_a_memory_session_gets_no_record_turn(tmp_path):
    llm = _ScriptedLLM([_tool_calls(("c1", "read", {"path": "orbital/INDEX.md"})), _reply("moved nothing")])
    session, tools, loop, _ = _rig(tmp_path, llm, origin="memory", memory_record_enabled=False)
    await loop.run()
    stats = loop.last_run_stats
    assert stats["llm_calls"] == 2 and stats["tool_calls"] == 1
    assert stats["input_tokens"] == 200 and stats["cached_tokens"] == 160 and stats["output_tokens"] == 14
    assert stats["ended"] == "text_complete"
    assert not any(m.get("_kind") == "memory_record" for m in session.get_messages())
    assert not any((m.get("_meta") or {}).get("event") == "memory_record" for m in session.get_messages())


@pytest.mark.asyncio
async def test_run_binds_the_state_ids_it_saw_and_protects_later_arrivals(tmp_path):
    """The loop's own write may drop ids the snapshot showed; an id another
    writer added after the snapshot comes back."""
    state_path = os.path.join(str(tmp_path), "orbital", "PROJECT_STATE.md")

    class _LateWriterLLM(_ScriptedLLM):
        async def stream(self, messages, tools=None):
            if self.calls == 0:
                with open(state_path, "a", encoding="utf-8") as f:
                    f.write(_bullet("added by a worker", "w7") + "\n")
            async for chunk in super().stream(messages, tools=tools):
                yield chunk

    llm = _LateWriterLLM([
        _tool_calls(("c1", "write", {"path": "orbital/PROJECT_STATE.md",
                                     "content": "# State\n" + _bullet("focus: launch", "a1") + "\n"})),
        _reply("ok"),
    ])
    session, tools, loop, _ = _rig(tmp_path, llm)
    await loop.run()
    text = open(state_path, encoding="utf-8").read()
    assert "id:a1" in text and "id:b1" not in text and "id:w7" in text
    assert M.seen_state_ids() is None   # unbound once the run is over


# ---------------------------------------------------------------------------
# 8. the context manager: adopted Layer-1 rows
# ---------------------------------------------------------------------------

def test_adopted_snapshot_is_injected_byte_identical_then_the_next_run_reads_disk(tmp_path):
    workspace = str(tmp_path)
    _seed(workspace)
    main = Session.new("sess-main-0001", workspace)
    main_ctx = ContextManager(main, _PromptBuilder(), _prompt_context(workspace), model_context_limit=200_000)
    main_ctx.begin_run()
    main_ctx.prepare()
    rows = main_ctx.layer1_snapshot()
    assert rows and all(r[1].startswith("[") for r in rows)
    # disk changes after the snapshot
    with open(os.path.join(workspace, "orbital", "INDEX.md"), "a", encoding="utf-8") as f:
        f.write("- later/ — added after the snapshot\n")

    mem = Session.new("sess-mem-0002", workspace)
    mem_ctx = ContextManager(mem, _PromptBuilder(), _prompt_context(workspace), model_context_limit=200_000)
    mem_ctx.adopt_layer1_snapshot(rows)
    mem_ctx.begin_run()                       # what AgentLoop.run() does first
    prepared = mem_ctx.prepare()
    injected = [m["content"] for m in prepared if m["role"] == "system" and m["content"].startswith("[")]
    assert injected[:len(rows)] == [r[1] for r in rows]
    assert "added after the snapshot" not in "\n".join(injected)
    # the next run (no adoption) renders from disk
    mem_ctx.begin_run()
    again = [m["content"] for m in mem_ctx.prepare() if m["role"] == "system"]
    assert any("added after the snapshot" in c for c in again)


# ---------------------------------------------------------------------------
# 9. the agent manager: slot bypass, session traits, the pass launcher
# ---------------------------------------------------------------------------

def _manager(tmp_path, projects=None):
    from agent_os.daemon_v2.agent_manager import AgentManager
    projects = projects or {"p1": {"project_id": "p1", "name": "demo", "workspace": str(tmp_path),
                                   "is_scratch": False}}
    project_store = MagicMock()
    project_store.get_project = MagicMock(side_effect=lambda pid: projects.get(pid))
    project_store.list_projects = MagicMock(return_value=list(projects.values()))
    ws = MagicMock()
    ws.broadcast = MagicMock()
    sub = MagicMock()
    sub.list_active = MagicMock(return_value=[])
    sub.stop = AsyncMock()
    sub.stop_all = AsyncMock()
    pm = MagicMock()
    pm.set_session = MagicMock()
    return AgentManager(project_store=project_store, ws_manager=ws, sub_agent_manager=sub,
                        activity_translator=MagicMock(), process_manager=pm)


def _handle(session, *, running):
    from agent_os.daemon_v2.agent_manager import ProjectHandle
    task = MagicMock()
    task.done = MagicMock(return_value=not running)
    return ProjectHandle(session=session, loop=MagicMock(), provider=MagicMock(), registry=MagicMock(),
                         context_manager=MagicMock(), interceptor=MagicMock(), task=task)


def test_a_running_memory_session_is_never_a_live_session(tmp_path):
    """Spec 155: sessions run in parallel; the Memory session is still never
    reported as a project's live session (hidden, never shows running)."""
    from agent_os.daemon_v2.agent_manager import make_session_key
    mgr = _manager(tmp_path)
    mem = Session.new("demo_mem00001", str(tmp_path), origin="memory")
    mgr._handles[make_session_key("p1", mem.session_uuid)] = _handle(mem, running=True)
    assert mgr.current_holder_session_id("p1") is None
    assert mgr.live_session_ids("p1") == []
    assert mgr._start_rejection("p1", "demo_other0001") is None
    chat = Session.new("demo_chat00001", str(tmp_path))
    mgr._handles[make_session_key("p1", chat.session_uuid)] = _handle(chat, running=True)
    assert mgr.live_session_ids("p1") == [chat.session_uuid]
    # the (idle) memory session may start while the chat runs, and so may a
    # third session: nothing holds a project-wide slot any more
    mgr._handles[make_session_key("p1", mem.session_uuid)].task.done = MagicMock(return_value=True)
    assert mgr._start_rejection("p1", mem.session_uuid, session=mem) is None
    assert mgr._start_rejection("p1", "demo_third00001") is None
    # only the running session itself is refused a second start
    assert mgr._start_rejection("p1", chat.session_uuid) is not None
    # hidden from the sidebar's list, visible to the daemon's own lookup
    assert mem.session_uuid not in {s["session_uuid"] for s in mgr.list_sessions("p1")}
    listed = {s["session_uuid"]: s for s in mgr.list_sessions("p1", include_memory=True)}
    assert listed[mem.session_uuid]["origin"] == "memory" and listed[mem.session_uuid]["status"] == "idle"


@pytest.mark.asyncio
async def test_start_agent_gives_a_memory_session_its_traits_and_skips_the_slot(tmp_path):
    from agent_os.daemon_v2.agent_manager import make_session_key
    from agent_os.daemon_v2.models import AgentConfig
    mgr = _manager(tmp_path)
    chat = Session.new("demo_chat00001", str(tmp_path))
    mgr._handles[make_session_key("p1", chat.session_uuid)] = _handle(chat, running=True)
    mem = Session.new("demo_mem00001", str(tmp_path), origin="memory")
    config = AgentConfig(workspace=str(tmp_path), model="gpt-4", api_key="sk-test")
    rows = [("PROJECT_STATE.md", "[PROJECT_STATE.md — snapshot]\n- focus: launch")]

    with patch("agent_os.daemon_v2.agent_manager.LLMProvider") as MockProv, \
         patch("agent_os.daemon_v2.agent_manager.ToolRegistry") as MockReg, \
         patch("agent_os.daemon_v2.agent_manager.PromptBuilder"), \
         patch("agent_os.daemon_v2.agent_manager.ContextManager") as MockCM, \
         patch("agent_os.daemon_v2.agent_manager.AgentLoop") as MockLoop, \
         patch("agent_os.daemon_v2.agent_manager.AutonomyInterceptor"):
        reg = MagicMock()
        reg.tool_names.return_value = ["read", "write", "edit"]
        MockReg.return_value = reg
        loop = MagicMock()
        loop.run = AsyncMock()
        MockLoop.return_value = loop
        provider = MagicMock()
        MockProv.return_value = provider
        await mgr.start_agent("p1", config, session=mem, layer1_rows=rows, cache_session_id=chat.session_uuid)
        kwargs = MockLoop.call_args.kwargs
        assert kwargs["memory_record_enabled"] is False
        assert kwargs["tool_call_cap"] == E.EDITOR_MAX_CALLS
        assert callable(kwargs["tool_guard"]) and kwargs["tool_guard"](
            {"name": "write", "arguments": {"path": "README.md"}})
        assert kwargs["on_session_end_refresh"] is None     # no self-trigger
        assert provider.cache_audit_caller == "memory"
        MockCM.return_value.adopt_layer1_snapshot.assert_called_once_with(rows)
        # the gateway session header follows the triggering run (prefix routing)
        getter = provider.bind_session_id.call_args.args[0]
        assert getter() == chat.session_uuid
    assert make_session_key("p1", mem.session_uuid) in mgr._handles


@pytest.mark.asyncio
async def test_start_agent_admits_a_second_chat_session(tmp_path):
    """Spec 155: a second chat session of the project starts while the first
    runs (it used to raise "Slot held"). Cut off just past the entry guard."""
    from agent_os.daemon_v2.agent_manager import make_session_key
    from agent_os.daemon_v2.models import AgentConfig
    mgr = _manager(tmp_path)
    chat = Session.new("demo_chat00001", str(tmp_path))
    mgr._handles[make_session_key("p1", chat.session_uuid)] = _handle(chat, running=True)
    other = Session.new("demo_chat00002", str(tmp_path))

    class _PastTheGuard(Exception):
        pass

    mgr._build_llm_providers = MagicMock(side_effect=_PastTheGuard())
    with pytest.raises(_PastTheGuard):
        await mgr.start_agent("p1", AgentConfig(workspace=str(tmp_path), model="m", api_key="k"), session=other)


def test_memory_session_is_created_once_named_memory_and_found_again(tmp_path):
    mgr = _manager(tmp_path)
    first = mgr.memory_session("p1")
    assert first.origin == "memory" and first.name == "Memory"
    first.append({"role": "user", "content": "hello"})           # materialise it
    assert first.name == "Memory"                                  # auto-naming does not override
    again = mgr.memory_session("p1")
    assert again.session_uuid == first.session_uuid
    assert again.origin == "memory" and again.name == "Memory"
    assert not [s for s in mgr.list_sessions("p1") if s["origin"] == "memory"]   # hidden
    listed = [s for s in mgr.list_sessions("p1", include_memory=True) if s["origin"] == "memory"]
    assert len(listed) == 1 and listed[0]["name"] == "Memory"


@pytest.mark.asyncio
async def test_run_memory_turn_appends_the_task_runs_one_turn_and_reports(tmp_path):
    from agent_os.daemon_v2.agent_manager import ProjectHandle, make_session_key
    mgr = _manager(tmp_path)
    mgr._abuild_agent_config_from_project = AsyncMock(return_value=MagicMock())
    started = {}

    async def fake_start(project_id, config, **kw):
        started.update(kw)
        session = kw["session"]
        loop = MagicMock()
        loop.last_run_stats = {"llm_calls": 2, "tool_calls": 3, "input_tokens": 500,
                               "cached_tokens": 450, "output_tokens": 40, "ended": "text"}
        task = asyncio.create_task(asyncio.sleep(0))
        mgr._handles[make_session_key(project_id, session.session_uuid)] = ProjectHandle(
            session=session, loop=loop, provider=MagicMock(), registry=MagicMock(),
            context_manager=MagicMock(), interceptor=MagicMock(), task=task)
    mgr.start_agent = fake_start

    result = await mgr.run_memory_turn("p1", "Tidy this project's memory.", trigger="over_budget",
                                       layer1_rows=[("INDEX.md", "[INDEX.md]\n- x")], cache_session_id="demo_chat1")
    assert result.mode == "snapshot" and result.llm_calls == 2 and result.tool_calls == 3
    assert result.cached_tokens == 450
    assert started["layer1_rows"] == [("INDEX.md", "[INDEX.md]\n- x")]
    assert started["cache_session_id"] == "demo_chat1"
    mem = started["session"]
    rows = mem.get_messages()
    assert rows[-1]["role"] == "user" and rows[-1]["content"] == "Tidy this project's memory."
    assert rows[-1]["source"] == "daemon" and rows[-1]["_meta"]["event"] == "memory_task"
    # a turn already running in the Memory session (the user is typing) → None
    mgr._handles[make_session_key("p1", mem.session_uuid)].task = MagicMock(done=MagicMock(return_value=False))
    assert await mgr.run_memory_turn("p1", "again", trigger="over_budget") is None


@pytest.mark.asyncio
async def test_the_pass_waits_for_the_triggering_runs_first_call(tmp_path):
    """The prefix exists provider-side only once the triggering run's first
    call was served, so the turn waits for it (bounded) and then adopts that
    run's rows."""
    from agent_os.daemon_v2.agent_manager import make_session_key
    mgr = _manager(tmp_path)
    chat = Session.new("demo_chat00001", str(tmp_path))
    sk = make_session_key("p1", chat.session_uuid)
    handle = _handle(chat, running=True)
    handle.loop.is_running = True
    handle.loop.last_run_stats = {"llm_calls": 0}
    mgr._handles[sk] = handle

    async def first_call_lands():
        await asyncio.sleep(0.2)
        handle.loop.last_run_stats = {"llm_calls": 1}
    asyncio.get_running_loop().create_task(first_call_lands())
    started = asyncio.get_running_loop().time()
    await mgr._wait_for_first_call(sk, 5.0)
    waited = asyncio.get_running_loop().time() - started
    assert 0.15 <= waited < 2.0
    # a run that ended first does not hold the pass
    handle.loop.is_running = False
    handle.loop.last_run_stats = {"llm_calls": 0}
    started = asyncio.get_running_loop().time()
    await mgr._wait_for_first_call(sk, 5.0)
    assert asyncio.get_running_loop().time() - started < 0.5


# ---------------------------------------------------------------------------
# 10. follow-up (coordinator 2026-10-21): no memory file through the shell
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("command", [
    "cat orbital/PROJECT_STATE.md",
    "cd /tmp/ws && python3 - <<'EOF'\nopen('orbital/DECISIONS.md','w').write('x')\nEOF",
    "grep -n foo /abs/path/orbital/LESSONS.md",
    "printf 'x' >> ./orbital/INDEX.md",
    "echo '- done abc123' >> orbital/ASKS.md",
    "cp orbital/backups/memory-editor-20261021T024742670407Z/PROJECT_STATE.md /tmp/x",
    "tail -5 orbital/PROJECT_STATE_ARCHIVE.md",
    "sed -i '' 's/a/b/' LESSONS_ARCHIVE.md",
])
def test_shell_commands_naming_a_memory_file_are_refused(tmp_path, command):
    guard = E.memory_tool_guard(str(tmp_path))
    err = guard({"name": "shell", "arguments": {"command": command}})
    assert err and err.startswith("Error")
    assert "Memory files are edited with the edit/write tools only" in err
    assert "run this command without them" in err
    assert "read" in err and "grep" in err          # reads go through the read/grep tools


@pytest.mark.parametrize("command", [
    "ls orbital/",
    "git status",
    "cat README.md",
    "grep -rn index.html src/",
    "python3 -c 'print(1)'",
    "wc -l orbital/sessions/x.jsonl",
])
def test_shell_commands_not_naming_a_memory_file_are_allowed(tmp_path, command):
    guard = E.memory_tool_guard(str(tmp_path))
    assert guard({"name": "shell", "arguments": {"command": command}}) is None
    assert guard({"name": "shell", "arguments": {}}) is None
    assert guard({"name": "edit", "arguments": {"path": "orbital/PROJECT_STATE.md"}}) is None
    assert guard({"name": "write", "arguments": {"path": "orbital/DECISIONS_ARCHIVE.md"}}) is None


def test_task_message_says_edit_and_write_never_the_shell_once():
    text = E.build_task_message(
        sizes=_sizes(), folded=[], over=["state"], forced=False, pointer_paths=[],
        last_pass=None, sessions=[], today=TODAY, tool_cap=12,
    )
    sentence = "Edit these files only with edit and write; never through the shell."
    assert text.count(sentence) == 1
    assert ("<!--mem …--> lines.\n" + sentence + "\n\nSIZES (bullet lines only") in text
