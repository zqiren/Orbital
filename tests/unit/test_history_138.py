# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 138 history core: the Claude Code / Codex readers (stop causes,
files, skills), the scanner cache, the three grouping passes, the noise
filter and the left-off selection. Synthetic transcripts only."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

import pytest

from agent_os.onboarding import history as h
from agent_os.onboarding.history_paths import HistoryEnv
from agent_os.onboarding.history_types import (
    CAUSE_INTERRUPTED, CAUSE_MODEL_UNAVAILABLE, CAUSE_NETWORK_AUTH, CAUSE_NO_REPLY,
    CAUSE_USAGE_LIMIT, HistorySession, StopInfo,
)
from agent_os.onboarding.sources import claude_code, codex
from tests.unit._history_138_builders import native, norm


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(path)


# ---- Claude Code row builders ------------------------------------------------

def cu(text, ts="2026-10-20T10:00:00Z", cwd="/w/app", **kw):
    return {"type": "user", "sessionId": "c-1", "cwd": cwd, "timestamp": ts,
            "entrypoint": kw.pop("entrypoint", "cli"),
            "message": {"role": "user", "content": text}, **kw}


def ca(blocks, ts="2026-10-20T10:00:05Z", cwd="/w/app", **kw):
    if isinstance(blocks, str):
        blocks = [{"type": "text", "text": blocks}]
    return {"type": "assistant", "sessionId": "c-1", "cwd": cwd, "timestamp": ts,
            "message": {"role": "assistant", "model": kw.pop("model", "claude-x"),
                        "content": blocks}, **kw}


def tool_use(name, **inp):
    return {"type": "tool_use", "id": "t1", "name": name, "input": inp}


def tool_result():
    return {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}


def read_claude(tmp_path, rows):
    return claude_code.SOURCE.read(_write(tmp_path / "proj" / "c-1.jsonl", rows))


class TestClaudeReader:
    def test_finished_session_with_question_is_not_unfinished(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("add a login page"),
            ca([tool_use("Write", file_path=native("/w/app/login.tsx"))]),
            cu([tool_result()]),
            ca("Done. Want me to add tests too?"),
        ])
        assert s.stop is None
        assert s.user_turns == 1
        assert s.first_user_text == s.last_user_text == "add a login page"
        assert s.creates == {norm(native("/w/app/login.tsx")): 1}
        assert s.cwd == "/w/app" and s.native_id == "c-1"

    def test_unanswered_last_user_text_is_no_reply(self, tmp_path):
        s = read_claude(tmp_path, [cu("first"), ca("ok"),
                                   cu("second thing", ts="2026-10-20T11:00:00Z")])
        assert s.stop.cause == CAUSE_NO_REPLY
        assert s.last_user_text == "second thing"
        assert s.last_user_at == "2026-10-20T11:00:00+00:00"

    def test_api_error_usage_limit_with_reset(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("refactor the parser"),
            ca("You've hit your session limit · resets 10pm (Asia/Shanghai)",
               ts="2026-10-25T09:00:00Z", model="<synthetic>",
               isApiErrorMessage=True, error="rate_limit"),
        ])
        assert s.stop.cause == CAUSE_USAGE_LIMIT
        assert s.stop.resets_at == "2026-10-25T14:00:00+00:00"

    def test_api_error_without_known_text_uses_the_error_field(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("go"), ca("API Error: something odd", model="<synthetic>",
                         isApiErrorMessage=True, error="authentication_failed")])
        assert s.stop.cause == CAUSE_NETWORK_AUTH

    def test_unplaceable_api_error_still_counts_as_unfinished(self, tmp_path):
        # Same choice as every reader (138-R): an error naming no cause is
        # "interrupted", never a finished turn.
        s = read_claude(tmp_path, [
            cu("go"), ca("API Error: 529 overloaded", model="<synthetic>",
                         isApiErrorMessage=True, error="unknown")])
        assert s.stop.cause == CAUSE_INTERRUPTED
        assert "overloaded" in s.stop.detail

    def test_tool_result_last_is_interrupted_mid_tool(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("run the suite"), ca([tool_use("Bash", command="pytest")]), cu([tool_result()])])
        assert s.stop.cause == CAUSE_INTERRUPTED

    def test_user_interrupt_marker_is_interrupted_not_a_user_turn(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("do it"), ca([tool_use("Bash", command="make")]),
            cu([{"type": "text", "text": "[Request interrupted by user for tool use]"}])])
        assert s.stop.cause == CAUSE_INTERRUPTED
        assert s.user_turns == 1

    def test_injected_prompts_are_cleaned(self, tmp_path):
        recap = ("Conversation so far (earlier messages):\n\nuser: hi\nassistant: yo"
                 "\n\n--- end of conversation so far ---\n\nnow ship it")
        s = read_claude(tmp_path, [
            cu(recap), ca("shipped."),
            cu("[Orbital memory] Your last turn is ending. Decide whether…"),
            ca("Nothing durable."),
        ])
        assert s.first_user_text == s.last_user_text == "now ship it"
        assert s.user_turns == 1
        assert s.stop is None

    def test_trigger_runs_are_automation(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("[Triggered by schedule 'Daily' (every day at 9)]\n\nSummarize the inbox"),
            ca("Summary: …")])
        assert s.automation is True and s.user_turns == 0

    def test_sdk_entrypoint_is_orbital_started(self, tmp_path):
        s = read_claude(tmp_path, [cu("x y", entrypoint="sdk-py"), ca("ok")])
        assert s.orbital_started is True

    def test_slash_command_plumbing_and_sidechains_are_skipped(self, tmp_path):
        s = read_claude(tmp_path, [
            cu("<command-name>/clear</command-name>"),
            cu("real request here"),
            {**ca("sidechain reply"), "isSidechain": True},
            ca("main reply"),
        ])
        assert s.user_turns == 1 and s.stop is None

    def test_files_skills_titles_and_branch(self, tmp_path):
        s = read_claude(tmp_path, [
            {"type": "ai-title", "aiTitle": "Deck work", "sessionId": "c-1"},
            {**cu("make the deck"), "gitBranch": "feat/deck"},
            ca([tool_use("Edit", file_path=native("/w/app/a.py"), old_string="x", new_string="y"),
                tool_use("Read", file_path=native("/Users/x/.claude/skills/pptx/SKILL.md")),
                tool_use("Skill", skill="anthropic-skills:docx"),
                tool_use("Bash", command="cat " + native("/Users/x/notes/plan.md"))]),
            cu([tool_result()]), ca("done"),
        ])
        assert s.title == "Deck work" and s.branch == "feat/deck"
        assert s.edits == {norm(native("/w/app/a.py")): 1}
        assert norm(native("/Users/x/.claude/skills/pptx/SKILL.md")) in s.reads
        assert norm(native("/Users/x/notes/plan.md")) in s.mentions
        assert set(s.skills) == {"pptx", "docx"}

    def test_empty_transcript_is_none(self, tmp_path):
        assert read_claude(tmp_path, [{"type": "summary", "summary": "x"}]) is None


# ---- Codex row builders ------------------------------------------------------

def xmeta(cwd="/w/app", originator="codex_cli_rs", **kw):
    return {"type": "session_meta", "timestamp": "2026-10-20T10:00:00Z",
            "payload": {"id": "019a-x", "cwd": cwd, "originator": originator,
                        "timestamp": "2026-10-20T10:00:00Z", **kw}}


def xu(text, ts="2026-10-20T10:00:01Z"):
    return {"type": "response_item", "timestamp": ts,
            "payload": {"type": "message", "role": "user",
                        "content": [{"type": "input_text", "text": text}]}}


def xa(text, ts="2026-10-20T10:00:09Z"):
    return {"type": "response_item", "timestamp": ts,
            "payload": {"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}}


def xev(ptype, ts="2026-10-20T10:00:10Z", **payload):
    return {"type": "event_msg", "timestamp": ts, "payload": {"type": ptype, **payload}}


def read_codex(tmp_path, rows):
    return codex.SOURCE.read(_write(tmp_path / "sessions" / "rollout-1.jsonl", rows))


class TestCodexReader:
    def test_windows_extended_prefix_is_stripped_from_the_folder(self, tmp_path):
        """Codex writes the extended-length form (a "\\\\?\\" prefix) on Windows;
        grouping and the git lookups must see the plain folder, as the import list does."""
        s = read_codex(tmp_path, [xmeta(cwd="\\\\?\\D:\\proj"), xu("hi"), xa("hello")])
        assert s.cwd == "D:\\proj"

    def test_completed_turn(self, tmp_path):
        s = read_codex(tmp_path, [
            xmeta(git={"branch": "main"}),
            xu("<environment_context>cwd</environment_context>"),
            xu("fix the flaky test"), xa("Fixed. Anything else?"),
            xev("task_complete", last_agent_message="Fixed."),
        ])
        assert s.native_id == "019a-x" and s.cwd == "/w/app" and s.branch == "main"
        assert s.user_turns == 1 and s.stop is None
        assert s.resume_path.endswith("rollout-1.jsonl")

    def test_turn_aborted_is_interrupted(self, tmp_path):
        s = read_codex(tmp_path, [xmeta(), xu("do it"),
                                  xev("turn_aborted", reason="interrupted")])
        assert s.stop.cause == CAUSE_INTERRUPTED

    def test_error_event_model_unavailable(self, tmp_path):
        s = read_codex(tmp_path, [
            xmeta(), xu("go"),
            xev("error", message="400 Bad Request: the 'gpt-x' model is not supported"),
            xev("task_complete")])
        assert s.stop.cause == CAUSE_MODEL_UNAVAILABLE

    def test_unplaceable_error_event_is_interrupted(self, tmp_path):
        s = read_codex(tmp_path, [xmeta(), xu("go"), xev("error", message="stream disconnected")])
        assert s.stop.cause == CAUSE_INTERRUPTED

    def test_usage_limit_error(self, tmp_path):
        s = read_codex(tmp_path, [xmeta(), xu("go"),
                                  xev("error", message="You've hit your usage limit. Try later.")])
        assert s.stop.cause == CAUSE_USAGE_LIMIT

    def test_no_reply(self, tmp_path):
        s = read_codex(tmp_path, [xmeta(), xu("a"), xa("b"), xev("task_complete"), xu("c then")])
        assert s.stop.cause == CAUSE_NO_REPLY

    def test_orbital_originator_and_helper_threads(self, tmp_path):
        assert read_codex(tmp_path, [xmeta(originator="orbital"), xu("x y"), xa("ok")]).orbital_started
        assert read_codex(tmp_path, [xmeta(source={"subagent": {"other": "guardian"}}),
                                     xu("x y"), xa("ok")]) is None

    def test_file_changes_and_patches(self, tmp_path):
        s = read_codex(tmp_path, [
            xmeta(cwd=native("/w/app")), xu("write the report"),
            {"type": "response_item", "timestamp": "2026-10-20T10:00:03Z",
             "payload": {"type": "custom_tool_call", "name": "apply_patch",
                         "input": "*** Begin Patch\n*** Update File: src/a.py\n@@\n*** End Patch"}},
            xev("item_completed", item={"type": "FileChange", "changes": {
                native("/w/app/report.md"): {"type": "add", "content": "x"}}}),
            xa("done"), xev("task_complete"),
        ])
        assert s.edits == {norm(native("/w/app/src/a.py")): 1}
        assert s.creates == {norm(native("/w/app/report.md")): 1}


# ---- Scanner -----------------------------------------------------------------

class _CountingSource:
    agent = "claude-code"

    def __init__(self, root):
        self.root = root
        self.reads = 0

    def session_files(self, home):
        return sorted(os.path.join(self.root, n) for n in os.listdir(self.root))

    def read(self, path):
        self.reads += 1
        return claude_code.SOURCE.read(path)


class TestScanner:
    def test_cache_skips_unchanged_files_and_survives_restart(self, tmp_path):
        root = tmp_path / "hist"
        _write(root / "a.jsonl", [cu("hello world"), ca("hi")])
        _write(root / "b.jsonl", [cu("other thing"), ca("ok")])
        src = _CountingSource(str(root))
        data = tmp_path / "data"
        sc = h.HistoryScanner(str(data), home=str(tmp_path), sources=[src],
                              env=HistoryEnv.for_home(str(tmp_path), temp_roots=("/nope",)))
        assert sc.status()["status"] == "scanning"
        assert sc.wait(10)
        st = sc.status()
        assert st == {"status": "ready", "sessions_total": 2, "by_agent": {"claude-code": 2},
                      "notes_by_agent": {"claude-code": 0}, "notes_by_folder": []}
        assert src.reads == 2
        sc.collect()
        assert src.reads == 2  # unchanged
        _write(root / "b.jsonl", [cu("other thing"), ca("ok"), cu("more please")])
        sc.collect()
        assert src.reads == 3
        # A new scanner over the same data dir reads nothing.
        src2 = _CountingSource(str(root))
        sc2 = h.HistoryScanner(str(data), home=str(tmp_path), sources=[src2])
        assert len(sc2.collect()) == 2 and src2.reads == 0

    def test_a_failing_source_never_raises(self, tmp_path):
        class Boom:
            agent = "codex"

            def session_files(self, home):
                raise OSError("gone")

            def read(self, path):
                raise AssertionError

        sc = h.HistoryScanner(None, home=str(tmp_path), sources=[Boom()])
        assert sc.wait(5)
        assert sc.status() == {"status": "ready", "sessions_total": 0, "by_agent": {"codex": 0},
                               "notes_by_agent": {}, "notes_by_folder": []}


# ---- Grouping ----------------------------------------------------------------

def sess(native_id, cwd, *, agent="claude-code", first="please build the thing", turns=2,
         edits=None, creates=None, reads=None, mentions=None, skills=None,
         ended="2026-10-20T10:00:00+00:00", **kw):
    return HistorySession(agent=agent, native_id=native_id, path=f"/h/{native_id}", cwd=cwd,
                          first_user_text=first, last_user_text=first, user_turns=turns,
                          edits=edits or {}, creates=creates or {}, reads=reads or {},
                          mentions=mentions or {}, skills=skills or [], started_at=ended,
                          ended_at=ended, **kw)


@pytest.fixture
def world(tmp_path):
    home = tmp_path / "home"
    for d in ("Desktop", "Documents", "repo", "plain", "md", "minor", "decks", "decks/assets"):
        (home / d).mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(home / "repo")], check=True)
    subprocess.run(["git", "-C", str(home / "repo"), "-c", "user.email=a@b", "-c", "user.name=a",
                    "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    subprocess.run(["git", "-C", str(home / "repo"), "worktree", "add", "-q",
                    str(home / "repo-wt")], check=True)
    (home / "md" / "CLAUDE.md").write_text("x")
    h.clear_git_cache()
    env = HistoryEnv.for_home(str(home), temp_roots=("/definitely-not-here",))
    return str(home), env


class TestGrouping:
    def test_pass1_folder_rules(self, world):
        home, env = world
        j = lambda *p: os.path.join(home, *p)
        sessions = [
            sess("r1", j("repo")), sess("r2", j("repo-wt")),          # git + worktree fold
            sess("p1", j("plain")), sess("p2", j("plain")), sess("p3", j("plain")),  # 3+
            sess("m1", j("md")),                                        # CLAUDE.md
            sess("n1", j("minor")),                                     # 1 session → minor
            sess("d1", j("Desktop")), sess("g1", j("gone")),           # container / dead
        ]
        a = h.analyze(sessions, env)
        assert set(a.projects) == {j("repo"), j("plain"), j("md"), j("minor")}
        assert a.projects[j("repo")].is_git and not a.projects[j("repo")].minor
        assert {s.native_id for s in a.projects[j("repo")].sessions} == {"r1", "r2"}
        assert not a.projects[j("plain")].minor and a.projects[j("md")].has_agent_md
        assert a.projects[j("minor")].minor
        assert ("claude-code", "d1") not in a.placed and ("claude-code", "g1") not in a.placed

    def test_pass2_attaches_by_files_touched(self, world):
        home, env = world
        repo = os.path.join(home, "repo")
        sessions = [
            sess("r1", repo),
            sess("x", os.path.join(home, "Desktop"),
                 edits={os.path.join(repo, "a.py"): 2}, reads={os.path.join(repo, "b.py"): 1}),
            sess("weak", os.path.join(home, "Desktop"), first="something entirely different",
                 mentions={os.path.join(repo, "c.py"): 1}),
        ]
        a = h.analyze(sessions, env)
        assert a.placed[("claude-code", "x")] == repo
        assert ("claude-code", "weak") not in a.placed  # 1 point < 4

    def test_pass3_groups_leftovers_with_reasons_and_folder(self, world):
        home, env = world
        desk = os.path.join(home, "Desktop")
        deck = os.path.join(home, "decks", "Q3-review.pptx")
        asset = os.path.join(home, "decks", "assets", "chart.png")
        sessions = [
            sess("a", desk, first="write this week's report from the notes", creates={deck: 1},
                 skills=["pptx"], ended="2026-10-21T00:00:00+00:00"),
            sess("b", desk, agent="codex", first="Write this week's report, from notes!",
                 edits={deck: 1, asset: 1}, skills=["pptx"]),
            sess("c", desk, first="an unrelated lonely request here"),
        ]
        a = h.analyze(sessions, env)
        assert len(a.groups) == 1
        g = a.groups[0].to_dict()
        assert g["session_count"] == 2 and g["by_agent"] == {"claude-code": 1, "codex": 1}
        kinds = {r["kind"]: r for r in g["reasons"]}
        assert kinds["same_skill"] == {"kind": "same_skill", "value": "pptx", "count": 2}
        assert kinds["same_files"]["value"] == "Q3-review.pptx"
        assert kinds["same_request"]["count"] == 2
        assert "same_folder" not in kinds  # the Desktop is a container
        assert g["suggested_folder"] == os.path.join(home, "decks")
        assert g["last_activity"] == "2026-10-21T00:00:00+00:00"
        assert {(m["agent"], m["native_id"]) for m in g["members"]} == {
            ("claude-code", "a"), ("codex", "b")}
        # Stable id for the same membership.
        assert h.analyze(list(reversed(sessions)), env).groups[0].id == g["id"]

    def test_group_without_a_real_folder_has_no_suggestion(self, world):
        home, env = world
        desk = os.path.join(home, "Desktop")
        sessions = [sess("a", desk, creates={os.path.join(desk, "x.md"): 1}),
                    sess("b", desk, first="other words entirely here",
                         edits={os.path.join(desk, "x.md"): 1})]
        g = h.analyze(sessions, env).groups[0]
        assert g.suggested_folder is None

    def test_agent_private_files_never_link(self, world):
        home, env = world
        desk = os.path.join(home, "Desktop")
        mem = os.path.join(home, ".claude", "memory.md")
        sessions = [sess("a", desk, first="first distinct request text", edits={mem: 1}),
                    sess("b", desk, first="second distinct request text", edits={mem: 1})]
        assert h.analyze(sessions, env).groups == []


class TestNoise:
    def test_probe_temp_automation_orbital_are_noise(self, world):
        home, env = world
        env_t = HistoryEnv.for_home(home, temp_roots=(os.path.join(home, "tmpx"),))
        assert h.is_noise(sess("p", home, first="PING", turns=1), env_t)
        assert h.is_noise(sess("p", home, first="Reply with exactly the word PONG", turns=1), env_t)
        assert h.is_noise(sess("t", os.path.join(home, "tmpx", "a")), env_t)
        assert h.is_noise(sess("o", home, orbital_started=True), env_t)
        assert h.is_noise(sess("u", home, automation=True, turns=0), env_t)
        assert not h.is_noise(sess("ok", home, first="add a hide list to settings"), env_t)


# ---- Left-off selection ------------------------------------------------------

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def e(sid, at, stop=None, text="did a thing"):
    return h.LeftOffEntry(session_id=sid, agent="codex", user_text=text, at=at,
                          stop=StopInfo(stop) if stop else None)


class TestSelectLeftOff:
    def test_last_time_is_newest_finished_and_unfinished_capped(self):
        picked = h.select_left_off([
            e("s1", "2026-10-27T10:00:00+00:00", CAUSE_USAGE_LIMIT),
            e("s2", "2026-10-26T10:00:00+00:00"),
            e("s3", "2026-10-25T10:00:00+00:00", CAUSE_NO_REPLY),
            e("s4", "2026-10-24T10:00:00+00:00", CAUSE_INTERRUPTED),
            e("s5", "2026-10-23T10:00:00+00:00", CAUSE_NETWORK_AUTH),
            e("old", "2026-09-01T10:00:00+00:00", CAUSE_NO_REPLY),
        ], now=NOW)
        assert picked["last_time"].session_id == "s2"
        assert [x.session_id for x in picked["unfinished"]] == ["s1", "s3", "s4"]

    def test_entries_without_user_words_are_ignored(self):
        picked = h.select_left_off([e("x", "2026-10-27T10:00:00+00:00", text=None)], now=NOW)
        assert picked == {"last_time": None, "unfinished": []}

    def test_resets_at_only_in_the_future(self):
        stop = StopInfo(CAUSE_USAGE_LIMIT, resets_at="2026-10-27T14:00:00+00:00")
        assert h.resets_at_if_future(stop, NOW) == "2026-10-27T14:00:00+00:00"
        assert h.resets_at_if_future(stop, datetime(2026, 10, 28, tzinfo=timezone.utc)) is None


class TestWindowsFolderRules:
    """Spec 138 never-a-project folders with Windows shapes (run on any OS)."""

    def test_appdata_and_agent_dirs(self, tmp_path):
        from agent_os.onboarding.history_paths import HistoryEnv
        home = str(tmp_path / "Users" / "x")
        env = HistoryEnv.for_home(home, temp_roots=None)
        assert env.is_temp(os.path.join(home, "AppData", "Local", "Temp", "a"))
        assert env.is_temp(os.path.join(home, "AppData", "Roaming", "Orbital", "ws"))
        for agent_dir in (".pi", ".kimi-code", ".minimax"):
            assert env.is_private(os.path.join(home, agent_dir, "sessions", "f.md"))

    def test_folder_checks_ignore_case_where_the_os_does(self, tmp_path, monkeypatch):
        from agent_os.onboarding import history_paths
        monkeypatch.setattr(history_paths, "_normcase", str.lower)
        home = str(tmp_path / "Users" / "Kea")
        env = history_paths.HistoryEnv.for_home(home, temp_roots=())
        assert env.is_container(home.upper())
        assert env.is_container(os.path.join(home.lower(), "desktop"))
        assert env.is_private(os.path.join(home.upper(), ".CODEX", "x"))


class TestCodexAppPreambles:
    """The Codex app's own wrappers around the user's words (batch 1026 rc,
    once compressed rollouts were read): the attached-files header became
    every such chat's "opening request" and grouped unrelated chats, and
    Codex automation runs were never marked as automation."""

    def test_attached_files_header_is_not_the_request(self, tmp_path):
        files = native("/Users/x/Downloads/cv.pdf")
        text = (f"# Files mentioned by the user:\n\n## cv.pdf: {files}\n\n"
                "## My request for Codex:\nsubmit this to the job posts below")
        s = read_codex(tmp_path, [xmeta(), xu(text), xa("on it"), xev("task_complete")])
        assert s.first_user_text == "submit this to the job posts below"
        assert norm(files) in s.mentions

    def test_the_newer_request_heading_too(self, tmp_path):
        text = (f"# Files mentioned by the user:\n\n## a.srt: {native('/Users/x/a.srt')}\n\n"
                "## My request:\nfix the caption")
        s = read_codex(tmp_path, [xmeta(), xu(text), xa("done"), xev("task_complete")])
        assert s.first_user_text == "fix the caption"

    def test_codex_automation_runs_are_automation(self, tmp_path):
        text = ("Automation: Daily brief\nAutomation ID: daily-brief\n"
                "Automation memory: $CODEX_HOME/automations/daily-brief/memory.md\n"
                "Last run: 2026-09-25T00:09:34.493Z (1790294974493)\n\n"
                "Give me a morning brief.")
        s = read_codex(tmp_path, [xmeta(), xu(text), xa("brief"), xev("task_complete")])
        assert s.automation is True
        assert h.is_noise(s, HistoryEnv.for_home("/nowhere", temp_roots=("/nope",)))

    def test_a_request_that_merely_mentions_automation_is_not_one(self, tmp_path):
        s = read_codex(tmp_path, [xmeta(), xu("Automation: can you set one up for me?"),
                                  xa("sure"), xev("task_complete")])
        assert not s.automation
