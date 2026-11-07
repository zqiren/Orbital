# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Worker-run onboarding on a keyless project, and the prompt-text lift.

A project whose management agent has no key (``manager_ready`` false) runs
its first-run onboarding — the interview, or the cold-start workspace scan —
on the pinned worker instead of failing with ``missing_api_key``. The worker
gets the SAME interview / scan text the management agent gets, lifted into
``agent_os/agent/onboarding_prompts.py``; the management agent's rendered
system prompt must stay byte-identical (prompt-cache safety), which the
frozen copy of the pre-lift method below pins.
"""

from __future__ import annotations

import json
import os
from pathlib import PurePath

import pytest

from agent_os.agent import onboarding_prompts as op
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from tests.unit.keyless_harness import (
    agent_status, build_client, create_project,
)


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch):
    monkeypatch.delenv("AGENT_OS_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Frozen copy of PromptBuilder._onboarding_or_directive as it stood on main
# (f266ddf) BEFORE the lift, with ONE deliberate contract change since
# (fix/1105-prompts, owner 2026-11-05): onboarding ends with a proposed first
# step and waits for the user's go instead of "begin working", and the step-2
# summary proposes who does what. Spec 160 §5.4 (owner 2026-11-06) adds the
# shared-memory line to the greeting when more than one agent is enabled.
# Edit only for a deliberate contract change: it is the byte-identity
# reference.
# ---------------------------------------------------------------------------

def _frozen_onboarding_or_directive(self, context) -> str:
    """Return onboarding prompt if project_goals.md missing, else directive."""
    from agent_os.agent.project_paths import ProjectPaths
    goals_path = ProjectPaths(context.workspace).project_goals
    content = self._read_truncated(goals_path)
    if content is None and context.cold_start:
        return (
            "## COLD-START WORKSPACE SCAN\n\n"
            "This is an imported project with an existing workspace. The user has\n"
            "consented to a one-time scan. A deterministic [WORKSPACE SKELETON] (every\n"
            "gitignore-respected file + size) has been provided to you as a system\n"
            "message. Work through three stages:\n\n"
            "STAGE 2 — READ (informed by the skeleton sizes):\n"
            "- Use the skeleton's sizes to plan BEFORE opening files. Small total →\n"
            "  read broadly. Large total → read high-signal files (README, config,\n"
            "  entry points, the largest meaningful sources) and sample the rest.\n"
            "- You have NO precise token meter. You will see a coarse 'Context usage:'\n"
            "  line each turn. When it crosses ~70%, STOP reading and move to Stage 3\n"
            "  with what you have. State what you skipped.\n\n"
            "STAGE 3 — PROPOSE → CONFIRM → WRITE:\n"
            "- Propose, in chat, your read of the project (descriptive State) and a\n"
            "  DRAFT of suggested Goals. Report which files you read and which you skipped.\n"
            "- State is yours to assert. Goals are a SUGGESTION the user owns — invite edits.\n"
            "- Do NOT propose or write prescriptive Instructions; a scan cannot infer intent.\n"
            "- Write NOTHING until the user confirms (ok / yes / looks good / any affirmative).\n"
            "- On confirmation: (1) write the agreed Goals to\n"
            f"  {context.workspace}/orbital/instructions/project_goals.md using the `write`\n"
            "  tool (Mission, Triggers, Scope, Rules, Preferences; under 1500 words), then\n"
            "  (2) write PROJECT_STATE.md (current focus, in-progress work, next steps) and\n"
            "     INDEX.md (navigation map) in orbital/ yourself with the `write` tool.\n"
            "- After writing, say in two or three lines what you would do first and which agent\n"
            "  would do it, then stop and wait for the user's go. If the user's reply picks a\n"
            "  different agent, record it under Preferences in project_goals.md."
        )
    if content is None:
        # Scratch (Quick Tasks) keeps no project memory, so the "this
        # project becomes an asset" introduction would be a false promise
        # there — it gets the plain greeting.
        if context.is_scratch:
            greeting = (
                "1. Greet the user briefly. Introduce yourself as their agent for this project.\n"
            )
        else:
            greeting = (
                "1. Greet the user briefly and introduce yourself as their agent for this project.\n"
                "   In that same first message, tell them what they are getting. Plain words, no file\n"
                "   names, no jargon, about 4-6 short lines in total:\n"
                "   - What you can do: name the abilities you actually have, read from your tool list\n"
                "     (for example working with the files in this folder, running commands, using the\n"
                "     web, running things on a schedule or when something changes, handing work to\n"
                "     coding agents). Mention only what is really available to you.\n"
                "   - How Orbital turns this project into an asset: as you work together you keep the\n"
                "     project's goals, where things stand, the decisions made and why, the lessons\n"
                "     learned, and a map of where everything lives — as plain files inside this folder.\n"
                "   - Why that is valuable: every new session starts from that record instead of from\n"
                "     zero, so they never have to re-explain the project. The files are theirs — readable,\n"
                "     editable, and usable by any other AI tool they open in this folder.\n"
                "   - That the work accumulates: every task leaves the project better understood, so you\n"
                "     get faster and more accurate the longer you work together.\n"
            )
            if context.enabled_agents:
                greeting += (
                    "   - Then add this one line exactly as written, in the user's language:\n"
                    "     中文: 这个项目里，不管用哪个 Agent、开几个会话，都用同一份项目记忆。"
                    "随时切换 Agent 或新开会话，不用重新解释。\n"
                    "     English: Every agent and every session in this project works from the "
                    "same project memory. Switch agents or open a new session anytime; no need "
                    "to explain again.\n"
                )
            greeting += (
                "   Write it in the user's language and your own words. Do not recite this list, do not\n"
                "   oversell, and do not let it crowd out the question that follows.\n"
            )
        base = (
            "## ONBOARDING MODE\n\n"
            "This is a new project. No project_goals.md exists yet. Your priority is to understand\n"
            "what the user wants before doing any work.\n\n"
            "The user created this project with:\n"
            f"- Name: {context.project_name}\n"
            f"- Instructions: {context.project_instructions}\n\n"
            "YOUR TASK:\n"
            + greeting +
            "2. If the user's instructions are clear and detailed enough, confirm your understanding\n"
            "   and present a summary of how you'll operate. When sub-agents are available, the\n"
            "   summary names which agent will handle which kind of work, as a proposal the user can\n"
            "   change in one word. Ask if they want to adjust anything.\n"
            "3. If the instructions are vague or missing, ask clarifying questions about:\n"
            "   - What they want you to do (objective)\n"
            "   - How they want you to do it (preferences, constraints)\n"
            "   - What you should NOT do (boundaries)\n"
            "4. Keep it to at most 5 exchanges. Do not over-ask. If the user gives short answers,\n"
            "   work with what you have.\n"
            "   Do not promise to start the work as soon as they answer: say that after their answer\n"
            "   you will come back with a suggested first step for them to approve.\n"
            "5. Once confirmed (user says ok/yes/looks good/any affirmative, OR you've hit 5 exchanges),\n"
            f"   write project_goals.md to {context.workspace}/orbital/instructions/project_goals.md using the structure:\n"
            "   Mission, Triggers, Scope, Rules, Preferences.\n"
            "6. Keep project_goals.md under 1500 words. Distill, don't dump.\n"
        )
        # Scratch (quick-action) projects skip workspace mapping — they are
        # ephemeral and excluded from INDEX.md/persistence machinery
        # (see _memory and _workspace_persistence).
        if context.is_scratch:
            return base + (
                "\nDO NOT use any tools (read, shell, write, edit, browser, etc.) until onboarding "
                "is complete.\n"
                "The only tool call you make during onboarding is the final `write` to create "
                "project_goals.md.\n"
                "After writing project_goals.md, say in two or three lines what you would do first and\n"
                "which agent would do it, then stop and wait for the user's go. Do not start on the\n"
                "project instructions as a task: they describe the project, not an order. If the user's\n"
                "reply picks a different agent, record it under Preferences in project_goals.md."
            )
        return base + (
            "7. After writing project_goals.md, explore the workspace and build INDEX.md:\n"
            "   - List the top-level directory structure and read the key files\n"
            "     (README, config files, entry points) to understand the project\n"
            "   - Create INDEX.md (in orbital/) as a NAVIGATION MAP: the important\n"
            "     files/dirs, ONE sentence each ('path — what it is'). Not prose, a map —\n"
            "     decisions/status/lessons live in their own files, not here.\n"
            "   - If the workspace is empty, still create INDEX.md with a brief note\n"
            "     (\"No files yet — will populate as the project develops.\")\n"
            "8. Once INDEX.md is written, say in two or three lines what you would do first and which\n"
            "   agent would do it, then stop and wait for the user's go.\n\n"
            "DO NOT use any tools until goal-setting is complete. After writing project_goals.md,\n"
            "use read/list tools to explore the workspace, then write INDEX.md. After INDEX.md\n"
            "is written, propose your first step and wait for the user's go. Do not start on the\n"
            "project instructions as a task: they describe the project, not an order. If the user's\n"
            "reply picks a different agent, record it under Preferences in project_goals.md."
        )
    return (
        "## PROJECT DIRECTIVE\n\n"
        f"{content}\n\n"
        "This is your core operating guide. Follow these objectives, rules, and preferences\n"
        "in all your work. If the user asks you to change your approach, update project_goals.md\n"
        "to reflect the change.\n"
        "If you wrote this file earlier in this conversation and the user has not said go since,\n"
        "onboarding is still finishing: write orbital/INDEX.md if it is missing, then say in two\n"
        "or three lines what you would do first and which agent would do it, and wait for the\n"
        "user's go."
    )


# ---------------------------------------------------------------------------
# B5 — the lift keeps the management prompt byte-identical
# ---------------------------------------------------------------------------


def _ctx(workspace, **over):
    base = dict(
        workspace=str(workspace), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[{"handle": "codex", "display_name": "Codex",
                         "type": "cli"}],
        tool_names=["read", "write", "edit", "shell"], os_type="macos",
        datetime_now="2026-10-21T10:00:00+00:00", project_name="Demo",
        project_instructions="Build a thing", agent_name="demo",
        project_id="proj_x",
    )
    base.update(over)
    return PromptContext(**base)


_CASES = {
    "interview": {},
    "interview_single_agent": {"enabled_agents": []},
    "interview_no_instructions": {"project_instructions": ""},
    "interview_scratch": {"is_scratch": True},
    "cold_start": {"cold_start": True},
    "cold_start_scratch": {"cold_start": True, "is_scratch": True},
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_section_is_byte_identical(tmp_path, case):
    ctx = _ctx(tmp_path, **_CASES[case])
    builder = PromptBuilder(str(tmp_path))
    assert builder._onboarding_or_directive(ctx) == \
        _frozen_onboarding_or_directive(builder, ctx)


@pytest.mark.parametrize("cold_start", [False, True])
def test_directive_is_byte_identical(tmp_path, cold_start):
    goals = tmp_path / "orbital" / "instructions" / "project_goals.md"
    goals.parent.mkdir(parents=True)
    goals.write_text("# Mission\nShip it.\n", encoding="utf-8")
    ctx = _ctx(tmp_path, cold_start=cold_start)
    builder = PromptBuilder(str(tmp_path))
    assert builder._onboarding_or_directive(ctx) == \
        _frozen_onboarding_or_directive(builder, ctx)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_full_rendered_prompt_is_byte_identical(tmp_path, monkeypatch, case):
    """The whole three-part system prompt, not just the section."""
    ctx = _ctx(tmp_path, **_CASES[case])
    lifted = PromptBuilder(str(tmp_path)).build(ctx)
    monkeypatch.setattr(PromptBuilder, "_onboarding_or_directive",
                        _frozen_onboarding_or_directive)
    frozen = PromptBuilder(str(tmp_path)).build(ctx)
    assert lifted == frozen


def test_manager_flavours_are_the_section_text(tmp_path):
    ws = str(tmp_path)
    builder = PromptBuilder(ws)
    assert op.interview_instruction(
        ws, is_scratch=False, project_name="Demo",
        project_instructions="Build a thing", multi_agent=True,
    ) == builder._onboarding_or_directive(_ctx(tmp_path))
    assert op.interview_instruction(
        ws, is_scratch=False, project_name="Demo",
        project_instructions="Build a thing",
    ) == builder._onboarding_or_directive(_ctx(tmp_path, enabled_agents=[]))
    assert op.cold_start_instruction(ws, None) == \
        builder._onboarding_or_directive(_ctx(tmp_path, cold_start=True))


class TestOnboardingEndsWithAProposal:
    """Owner 2026-11-05: onboarding ends with a proposed first step and waits
    for the user's go; the project instructions describe the project, they
    are not an order to start on."""

    def _variants(self, ws):
        return {
            "interview": op.interview_instruction(
                ws, is_scratch=False, project_name="Demo",
                project_instructions="Write a weekly report"),
            "interview_scratch": op.interview_instruction(
                ws, is_scratch=True, project_name="Demo",
                project_instructions="Write a weekly report"),
            "cold_start": op.cold_start_instruction(ws, None),
            "worker_interview": op.worker_interview_instruction(
                "codex", ws, project_name="Demo",
                project_instructions="Write a weekly report"),
            "worker_interview_scratch": op.worker_interview_instruction(
                "codex", ws, project_name="Demo",
                project_instructions="", is_scratch=True),
            "worker_cold_start": op.worker_cold_start_instruction(
                "codex", ws, "[WORKSPACE SKELETON] 1 file\nREADME.md 1B"),
        }

    def test_no_variant_says_begin_working(self, tmp_path):
        for name, text in self._variants(str(tmp_path)).items():
            assert "begin working" not in text, name
            assert "announce readiness" not in text, name

    def test_every_variant_waits_for_the_users_go(self, tmp_path):
        for name, text in self._variants(str(tmp_path)).items():
            flat = " ".join(text.split())
            assert "what you would do first and which agent would do it" in flat, name
            assert "wait for the user's go" in flat, name
            assert ("If the user's reply picks a different agent, record it "
                    "under Preferences in project_goals.md.") in flat, name

    def test_interviews_say_instructions_are_not_an_order(self, tmp_path):
        for name, text in self._variants(str(tmp_path)).items():
            if "interview" not in name:
                continue
            flat = " ".join(text.split())
            assert ("Do not start on the project instructions as a task: they "
                    "describe the project, not an order.") in flat, name

    def test_directive_finishes_onboarding_in_the_run_that_wrote_goals(self, tmp_path):
        """The prompt is rebuilt every call and onboarding mode ends the moment
        project_goals.md exists, so the wait-for-go must also ride the
        directive (static text: same bytes every turn)."""
        goals = tmp_path / "orbital" / "instructions" / "project_goals.md"
        goals.parent.mkdir(parents=True)
        goals.write_text("# Mission\nWrite the weekly report.\n", encoding="utf-8")
        builder = PromptBuilder(str(tmp_path))
        flat = " ".join(builder._onboarding_or_directive(_ctx(tmp_path)).split())
        assert ("If you wrote this file earlier in this conversation and the user "
                "has not said go since, onboarding is still finishing") in flat
        assert "wait for the user's go" in flat

    def test_summary_proposes_who_does_what(self, tmp_path):
        text = op.interview_instruction(str(tmp_path), is_scratch=False)
        flat = " ".join(text.split())
        assert ("names which agent will handle which kind of work, as a "
                "proposal the user can change in one word") in flat


class TestWorkerFlavours:
    def test_interview_has_preamble_paths_and_the_manager_text(self, tmp_path):
        ws = str(tmp_path)
        # Prompt paths use forward slashes on every platform (ffa7653).
        ws_posix = PurePath(ws).as_posix()
        text = op.worker_interview_instruction(
            "codex", ws, project_name="Demo", project_instructions="")
        assert text.startswith(
            f"You are codex working directly with the user in Orbital project {ws_posix}. "
            "Orbital's own agent is not configured, so you run this setup. "
            "Use your own file tools. Write files at these absolute paths:")
        for rel in ("orbital/instructions/project_goals.md",
                    "orbital/PROJECT_STATE.md", "orbital/INDEX.md"):
            assert f"{ws_posix}/{rel}" in text
        assert op.interview_instruction(
            ws, is_scratch=False, project_name="Demo",
            project_instructions="") in text

    def test_cold_start_inlines_the_skeleton(self, tmp_path):
        ws = str(tmp_path)
        skeleton = "[WORKSPACE SKELETON] 2 files\nREADME.md 12B\nsrc/app.py 40B"
        text = op.worker_cold_start_instruction("claude-code", ws, skeleton)
        assert text.startswith("You are claude-code working directly with the user")
        assert op.cold_start_instruction(ws, None) in text
        assert skeleton in text

    def test_windows_paths_reach_the_worker_with_forward_slashes(self, monkeypatch):
        monkeypatch.setattr(op, "_WINDOWS", True)
        ws = "C:\\Users\\dev\\proj"
        skeleton = "[WORKSPACE SKELETON] 2 files\nREADME.md\t12\nsrc\\app.py\t40"
        text = op.worker_cold_start_instruction("codex", ws, skeleton)
        head = text.split("\n\n", 1)[0]
        assert "Orbital project C:/Users/dev/proj." in head
        assert "C:/Users/dev/proj/orbital/instructions/project_goals.md" in head
        assert "C:/Users/dev/proj/orbital/PROJECT_STATE.md" in head
        assert "C:/Users/dev/proj/orbital/INDEX.md" in head
        assert "\\" not in head
        assert "src/app.py\t40" in text and "src\\app.py" not in text
        # The manager's text is shared verbatim, not rewritten.
        assert op.cold_start_instruction(ws, None) in text

    @pytest.mark.skipif(os.name == "nt", reason="POSIX pass-through: os.path.join itself yields backslashes on a Windows host")
    def test_posix_paths_are_untouched(self, monkeypatch):
        monkeypatch.setattr(op, "_WINDOWS", False)
        skeleton = "[WORKSPACE SKELETON] 1 files\nodd\\name.txt\t3"
        text = op.worker_cold_start_instruction("codex", "/home/dev/proj", skeleton)
        assert "/home/dev/proj/orbital/INDEX.md" in text
        assert skeleton in text

    def test_skeleton_is_bounded(self, tmp_path):
        skeleton = "[WORKSPACE SKELETON] many files\n" + ("x" * 99 + "\n") * 2000
        text = op.worker_cold_start_instruction("codex", str(tmp_path), skeleton)
        assert len(skeleton) > 150_000
        assert len(text) < op.SKELETON_CHAR_LIMIT + 10_000
        assert "skeleton truncated" in text


# ---------------------------------------------------------------------------
# B6 — the routes take the worker branch on a keyless project
# ---------------------------------------------------------------------------


def _rows(workspace, sid):
    path = os.path.join(workspace, "orbital", "sessions", f"{sid}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _keyless(tmp_path, *, statuses=None, files=None):
    tc, deps = build_client(tmp_path, keyed=False, statuses=statuses)
    ws = tmp_path / "ws"
    ws.mkdir()
    for rel, body in (files or {}).items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    row = create_project(tc, ws)
    return tc, deps, row


class TestWorkerInterview:
    def test_dispatches_to_the_effective_worker(self, tmp_path):
        tc, deps, row = _keyless(tmp_path)
        pid, ws = row["project_id"], row["workspace"]
        r = tc.post(f"/api/v2/agents/{pid}/start-onboarding")
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "ok" and body["runner"] == "claude-code"
        sid = body["session_id"]
        assert body["session_uuid"] == sid

        sub = deps["sub_agent_manager"]
        assert sub.send.await_count == 1
        args, kwargs = sub.send.await_args
        assert args[0] == pid and args[1] == "claude-code"
        assert args[2].startswith("You are claude-code working directly")
        assert "## ONBOARDING MODE" in args[2]
        assert kwargs["session_id"] == sid
        assert kwargs["initiator"] == "user_pinned"

        rows = _rows(ws, sid)
        meta = rows[0]
        assert meta["role"] == "meta" and meta["event"] == "session_start"
        assert meta["pinned_target"] == "claude-code"
        convo = [r for r in rows if r.get("role") != "meta"]
        assert len(convo) == 1, convo
        sys_row = convo[0]
        assert sys_row["role"] == "system" and sys_row["source"] == "daemon"
        assert sys_row["content"] == \
            "[Onboarding] claude-code is setting up this project"
        assert sys_row["timestamp"]
        assert sys_row["_meta"] == {
            "event": "worker_onboarding", "handle": "claude-code",
            "mode": "interview", "suppress_wake": True,
        }
        assert not [r for r in rows if r.get("role") == "user"]

        sessions = tc.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"]
        assert [s["pinned_target"] for s in sessions
                if s["session_id"] == sid] == ["claude-code"]

    def test_the_sticky_pin_decides_the_runner(self, tmp_path):
        tc, deps, row = _keyless(tmp_path)
        pid = row["project_id"]
        tc.put(f"/api/v2/projects/{pid}", json={"default_pinned_target": "codex"})
        body = tc.post(f"/api/v2/agents/{pid}/start-onboarding").json()
        assert body["runner"] == "codex"
        assert deps["sub_agent_manager"].send.await_args.args[1] == "codex"

    def test_no_worker_is_the_typed_400_and_mints_nothing(self, tmp_path):
        tc, deps, row = _keyless(
            tmp_path, statuses=[agent_status("built-in"),
                                agent_status("codex", creds=False)])
        pid = row["project_id"]
        r = tc.post(f"/api/v2/agents/{pid}/start-onboarding")
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "missing_api_key"
        assert deps["sub_agent_manager"].send.await_count == 0
        assert tc.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"] == []

    def test_409_guards_hold(self, tmp_path):
        tc, deps, row = _keyless(tmp_path)
        pid, ws = row["project_id"], row["workspace"]
        assert tc.post(f"/api/v2/agents/{pid}/start-onboarding").status_code == 201
        # Second call: the project now has a session.
        assert tc.post(f"/api/v2/agents/{pid}/start-onboarding").status_code == 409
        # Goals present.
        tc2, _d, row2 = _keyless(tmp_path / "two")
        goals = os.path.join(row2["workspace"], "orbital", "instructions",
                             "project_goals.md")
        os.makedirs(os.path.dirname(goals), exist_ok=True)
        open(goals, "w").write("# Mission\nx\n")
        assert tc2.post(
            f"/api/v2/agents/{row2['project_id']}/start-onboarding").status_code == 409

    def test_scratch_is_409(self, tmp_path):
        tc, deps = build_client(tmp_path, keyed=False)
        ws = tmp_path / "scratch"; ws.mkdir()
        pid = create_project(tc, ws, is_scratch=True)["project_id"]
        assert tc.post(f"/api/v2/agents/{pid}/start-onboarding").status_code == 409
        assert deps["sub_agent_manager"].send.await_count == 0

    def test_a_refused_dispatch_is_reported(self, tmp_path):
        tc, deps, row = _keyless(tmp_path)
        pid, ws = row["project_id"], row["workspace"]
        deps["sub_agent_manager"].send.side_effect = None
        deps["sub_agent_manager"].send.return_value = (
            "Error: 'claude-code' is installed but not signed in")
        r = tc.post(f"/api/v2/agents/{pid}/start-onboarding")
        assert r.status_code == 502
        detail = r.json()["detail"]
        assert detail["code"] == "worker_dispatch_failed"
        rows = _rows(ws, detail["session_id"])
        events = [r.get("_meta", {}).get("event") for r in rows
                  if r.get("role") == "system"]
        assert events == ["worker_onboarding", "sub_agent_dispatch_failed"]


class TestWorkerScan:
    def test_scan_runs_on_the_worker_with_the_skeleton(self, tmp_path):
        tc, deps, row = _keyless(tmp_path, files={
            "README.md": "# Demo\n", "notes.md": "n\n",
            "src/app.py": "print('hi')\n",
        })
        pid, ws = row["project_id"], row["workspace"]
        r = tc.post(f"/api/v2/agents/{pid}/cold-start-scan")
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["runner"] == "claude-code" and body["status"] == "ok"
        text = deps["sub_agent_manager"].send.await_args.args[2]
        assert "## COLD-START WORKSPACE SCAN" in text
        assert "[WORKSPACE SKELETON]" in text and "src/app.py" in text
        sys_rows = [r for r in _rows(ws, body["session_id"])
                    if r.get("role") == "system"]
        assert [r["_meta"]["mode"] for r in sys_rows] == ["scan"]

    def test_scan_without_a_worker_is_the_typed_400(self, tmp_path):
        tc, deps, row = _keyless(
            tmp_path, statuses=[agent_status("built-in")],
            files={"README.md": "x"})
        r = tc.post(f"/api/v2/agents/{row['project_id']}/cold-start-scan")
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "missing_api_key"
        assert deps["sub_agent_manager"].send.await_count == 0


class TestManagerBranchUntouched:
    def _keyed(self, tmp_path, monkeypatch, files=None):
        from unittest.mock import AsyncMock
        tc, deps = build_client(tmp_path, keyed=True)
        ws = tmp_path / "ws"; ws.mkdir()
        for rel, body in (files or {}).items():
            (ws / rel).write_text(body, encoding="utf-8")
        row = create_project(tc, ws)
        start = AsyncMock()
        monkeypatch.setattr(deps["agent_manager"], "start_agent", start)
        return tc, deps, row, start

    def test_interview_runs_on_orbital(self, tmp_path, monkeypatch):
        tc, deps, row, start = self._keyed(tmp_path, monkeypatch)
        r = tc.post(f"/api/v2/agents/{row['project_id']}/start-onboarding")
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "started" and body["runner"] == "orbital"
        assert start.await_count == 1
        assert start.await_args.kwargs["session_id"] == body["session_id"]
        assert deps["sub_agent_manager"].send.await_count == 0

    def test_scan_runs_on_orbital(self, tmp_path, monkeypatch):
        tc, deps, row, start = self._keyed(tmp_path, monkeypatch,
                                           files={"README.md": "x"})
        r = tc.post(f"/api/v2/agents/{row['project_id']}/cold-start-scan")
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "started" and body["runner"] == "orbital"
        assert start.await_args.kwargs["cold_start"] is True
        assert deps["sub_agent_manager"].send.await_count == 0

    def test_a_sticky_pin_does_not_move_onboarding_off_orbital(
            self, tmp_path, monkeypatch):
        tc, deps, row, start = self._keyed(tmp_path, monkeypatch)
        pid = row["project_id"]
        tc.put(f"/api/v2/projects/{pid}", json={"default_pinned_target": "codex"})
        body = tc.post(f"/api/v2/agents/{pid}/start-onboarding").json()
        assert body["runner"] == "orbital"
        assert deps["sub_agent_manager"].send.await_count == 0


# ---------------------------------------------------------------------------
# The "already has sessions" guard ignores unmaterialized imported entries
# ---------------------------------------------------------------------------


def _virtual_import(sid="imp_codex_0123456789abcdef"):
    return {
        "session_id": sid, "session_uuid": sid, "status": "idle",
        "origin": "imported", "name": "old chat", "pinned_target": "codex",
        "imported": {"agent": "codex", "native_id": "0123456789abcdef",
                     "materialized": False, "size_bytes": 10},
    }


class TestGuardIgnoresVirtualImports:
    def _with_imports(self, deps, monkeypatch, extra=()):
        am = deps["agent_manager"]
        real = am.list_sessions
        monkeypatch.setattr(
            am, "list_sessions",
            lambda pid: [_virtual_import(), *extra] + real(pid))

    def test_keyless_interview_and_scan_are_201(self, tmp_path, monkeypatch):
        tc, deps, row = _keyless(tmp_path, files={"README.md": "x"})
        self._with_imports(deps, monkeypatch)
        r = tc.post(f"/api/v2/agents/{row['project_id']}/start-onboarding")
        assert r.status_code == 201, r.text
        tc2, deps2, row2 = _keyless(tmp_path / "scan", files={"README.md": "x"})
        self._with_imports(deps2, monkeypatch)
        r = tc2.post(f"/api/v2/agents/{row2['project_id']}/cold-start-scan")
        assert r.status_code == 201, r.text

    def test_keyed_interview_and_scan_are_201(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock
        for mode in ("start-onboarding", "cold-start-scan"):
            tc, deps = build_client(tmp_path / mode, keyed=True)
            ws = tmp_path / mode / "ws"; ws.mkdir()
            (ws / "README.md").write_text("x", encoding="utf-8")
            pid = create_project(tc, ws)["project_id"]
            monkeypatch.setattr(deps["agent_manager"], "start_agent", AsyncMock())
            self._with_imports(deps, monkeypatch)
            r = tc.post(f"/api/v2/agents/{pid}/{mode}")
            assert r.status_code == 201, (mode, r.text)
            assert r.json()["runner"] == "orbital"

    def test_a_materialized_import_still_counts(self, tmp_path, monkeypatch):
        tc, deps, row = _keyless(tmp_path)
        done = _virtual_import("imp_codex_fedcba9876543210")
        done["imported"] = {**done["imported"], "materialized": True}
        self._with_imports(deps, monkeypatch, extra=[done])
        r = tc.post(f"/api/v2/agents/{row['project_id']}/start-onboarding")
        assert r.status_code == 409


# ---------------------------------------------------------------------------
# Spec 160 round 2 — the import scan on a keyless project's worker
# ---------------------------------------------------------------------------


class TestKeylessImportScan:
    def test_worker_gets_the_guess_text_and_the_session_its_title(self, tmp_path, monkeypatch):
        import asyncio as _asyncio

        from agent_os.api.routes import agents_v2
        from agent_os.onboarding import import_scan

        tc, deps, row = _keyless(tmp_path, files={"README.md": "# lib\n"})
        pid, ws = row["project_id"], row["workspace"]
        deps["project_store"].update_project(pid, {import_scan.LANG_FIELD: "zh"})
        sid = _asyncio.run(agents_v2.start_import_scan(pid, "worker"))
        sub = deps["sub_agent_manager"]
        args, kwargs = sub.send.await_args
        assert args[1] == "claude-code" and kwargs["session_id"] == sid
        text = args[2]
        assert text.startswith("You are claude-code working directly")
        assert "STAGE 3 — WRITE YOUR GUESS → ASK" in text
        assert "这样理解对吗" not in text            # round 4: the daemon asks
        assert "do not suggest a next step or task" in " ".join(text.split())
        assert "README.md" in text and "orbital/" not in text.split("[WORKSPACE SKELETON]")[-1]
        meta = _rows(ws, sid)[0]
        assert meta["name"] == "项目梳理" and "name_provisional" not in meta
        assert meta["origin"] == "cold_start"
