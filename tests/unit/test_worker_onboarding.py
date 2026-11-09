# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 167: no interview, no card-driven scan, no agent-speaks-first start.

A project's goals come from its first run's record turn
(``test_work_first_goals_167.py``). What is left of first-run onboarding is
spec 160's import scan: the manager's guess-first section for that scan's
run, and the same text behind a worker preamble on a keyless project, run
through ``start_import_scan``. This file pins that the removed paths stay
removed and the scan path keeps working.
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
# The manager prompt: directive or nothing (the scan run keeps its section)
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


def _write_goals(ws, text="# Mission\nShip it.\n"):
    goals = ws / "orbital" / "instructions" / "project_goals.md"
    goals.parent.mkdir(parents=True, exist_ok=True)
    goals.write_text(text, encoding="utf-8")


class TestNoInterview:
    @pytest.mark.parametrize("over", [{}, {"enabled_agents": []}, {"is_scratch": True},
                                      {"cold_start": True}])
    def test_no_goals_means_no_section(self, tmp_path, over):
        builder = PromptBuilder(str(tmp_path))
        assert builder._onboarding_or_directive(_ctx(tmp_path, **over)) is None
        _, semi, _ = builder.build(_ctx(tmp_path, **over))
        for gone in ("ONBOARDING MODE", "COLD-START WORKSPACE SCAN", "Greet the user",
                     "PROJECT DIRECTIVE"):
            assert gone not in semi, gone

    def test_blank_goals_file_is_no_goals(self, tmp_path):
        _write_goals(tmp_path, "  \n")
        assert PromptBuilder(str(tmp_path))._onboarding_or_directive(_ctx(tmp_path)) is None

    @pytest.mark.parametrize("scratch", [False, True])
    def test_directive_is_the_goals_with_no_onboarding_tail(self, tmp_path, scratch):
        _write_goals(tmp_path)
        text = PromptBuilder(str(tmp_path))._onboarding_or_directive(
            _ctx(tmp_path, is_scratch=scratch))
        assert text == (
            "## PROJECT DIRECTIVE\n\n# Mission\nShip it.\n\n\n"
            "This is your core operating guide. Follow these objectives, rules, and preferences\n"
            "in all your work. If the user asks you to change your approach, update project_goals.md\n"
            "to reflect the change.")

    @pytest.mark.parametrize("flavour,lang", [("guess:zh", "zh"), ("guess:en", "en")])
    def test_the_import_scan_run_keeps_its_section(self, tmp_path, flavour, lang):
        for goals in (False, True):
            if goals:
                _write_goals(tmp_path)
            text = PromptBuilder(str(tmp_path))._onboarding_or_directive(
                _ctx(tmp_path, cold_start=flavour))
            assert text == op.guess_first_instruction(str(tmp_path), lang=lang, multi_agent=True)

    def test_the_removed_texts_are_gone(self):
        for name in ("interview_instruction", "cold_start_instruction",
                     "worker_interview_instruction", "worker_cold_start_instruction"):
            assert not hasattr(op, name), name


# ---------------------------------------------------------------------------
# The worker's scan text
# ---------------------------------------------------------------------------


class TestWorkerScanText:
    SKELETON = "[WORKSPACE SKELETON] 2 files\nREADME.md 12B\nsrc/app.py 40B"

    def test_preamble_paths_and_the_manager_text(self, tmp_path):
        ws = str(tmp_path)
        ws_posix = PurePath(ws).as_posix()
        text = op.worker_guess_first_instruction("codex", ws, self.SKELETON, lang="en")
        assert text.startswith(
            f"You are codex working directly with the user in Orbital project {ws_posix}. "
            "Orbital's own agent is not configured, so you run this setup. "
            "Use your own file tools. Write files at these absolute paths:")
        for rel in ("orbital/instructions/project_goals.md",
                    "orbital/PROJECT_STATE.md", "orbital/INDEX.md"):
            assert f"{ws_posix}/{rel}" in text
        assert op.guess_first_instruction(ws, lang="en") in text
        assert self.SKELETON in text

    def test_windows_paths_reach_the_worker_with_forward_slashes(self, monkeypatch):
        monkeypatch.setattr(op, "_WINDOWS", True)
        ws = "C:\\Users\\dev\\proj"
        skeleton = "[WORKSPACE SKELETON] 2 files\nREADME.md\t12\nsrc\\app.py\t40"
        text = op.worker_guess_first_instruction("codex", ws, skeleton)
        head = text.split("\n\n", 1)[0]
        assert "Orbital project C:/Users/dev/proj." in head
        assert "C:/Users/dev/proj/orbital/instructions/project_goals.md" in head
        assert "C:/Users/dev/proj/orbital/PROJECT_STATE.md" in head
        assert "C:/Users/dev/proj/orbital/INDEX.md" in head
        assert "\\" not in head
        assert "src/app.py\t40" in text and "src\\app.py" not in text
        # The scan text's own goals line (Stage 3) uses the same form: never
        # the drive path with backslashes joined to a forward-slash tail.
        assert "C:\\Users\\dev\\proj/orbital" not in text
        assert text.count("C:/Users/dev/proj/orbital/instructions/project_goals.md") == 2
        assert "C:\\Users" not in text

    @pytest.mark.skipif(os.name == "nt", reason="POSIX pass-through: os.path.join itself yields backslashes on a Windows host")
    def test_posix_paths_are_untouched(self, monkeypatch):
        monkeypatch.setattr(op, "_WINDOWS", False)
        skeleton = "[WORKSPACE SKELETON] 1 files\nodd\\name.txt\t3"
        text = op.worker_guess_first_instruction("codex", "/home/dev/proj", skeleton)
        assert "/home/dev/proj/orbital/INDEX.md" in text
        assert skeleton in text

    def test_skeleton_is_bounded(self, tmp_path):
        skeleton = "[WORKSPACE SKELETON] many files\n" + ("x" * 99 + "\n") * 2000
        text = op.worker_guess_first_instruction("codex", str(tmp_path), skeleton)
        assert len(skeleton) > 150_000
        assert len(text) < op.SKELETON_CHAR_LIMIT + 10_000
        assert "skeleton truncated" in text


# ---------------------------------------------------------------------------
# The routes that ran the interview and the card scan are gone
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


class TestRemovedRoutes:
    @pytest.mark.parametrize("keyed", [False, True])
    @pytest.mark.parametrize("route", ["start-onboarding", "cold-start-scan"])
    def test_route_is_gone_and_mints_nothing(self, tmp_path, keyed, route):
        tc, deps = build_client(tmp_path, keyed=keyed)
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "README.md").write_text("x", encoding="utf-8")
        pid = create_project(tc, ws)["project_id"]
        r = tc.post(f"/api/v2/agents/{pid}/{route}")
        assert r.status_code in (404, 405), r.text
        assert deps["sub_agent_manager"].send.await_count == 0
        assert tc.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"] == []


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
        # Round 5: the worker records 1-3 next steps for the card, starts none.
        flat = " ".join(text.split())
        assert "`## Next steps` heading (exactly that heading" in flat
        assert "do not start any step" in flat
        assert "README.md" in text and "orbital/" not in text.split("[WORKSPACE SKELETON]")[-1]
        meta = _rows(ws, sid)[0]
        assert meta["name"] == "项目梳理" and "name_provisional" not in meta
        assert meta["origin"] == "cold_start"

    def test_a_refused_dispatch_is_reported_and_keeps_its_session(self, tmp_path):
        import asyncio as _asyncio

        from fastapi import HTTPException

        from agent_os.api.routes import agents_v2

        tc, deps, row = _keyless(tmp_path, files={"README.md": "# lib\n"})
        pid, ws = row["project_id"], row["workspace"]
        deps["sub_agent_manager"].send.side_effect = None
        deps["sub_agent_manager"].send.return_value = (
            "Error: 'claude-code' is installed but not signed in")
        with pytest.raises(HTTPException) as exc:
            _asyncio.run(agents_v2.start_import_scan(pid, "worker"))
        assert exc.value.status_code == 502
        detail = exc.value.detail
        assert detail["code"] == "worker_dispatch_failed"
        events = [r.get("_meta", {}).get("event") for r in _rows(ws, detail["session_id"])
                  if r.get("role") == "system"]
        assert events == ["worker_onboarding", "sub_agent_dispatch_failed"]

    def test_no_worker_is_the_typed_400_and_mints_nothing(self, tmp_path):
        import asyncio as _asyncio

        from fastapi import HTTPException

        from agent_os.api.routes import agents_v2

        tc, deps, row = _keyless(
            tmp_path, statuses=[agent_status("built-in"), agent_status("codex", creds=False)],
            files={"README.md": "x"})
        pid = row["project_id"]
        with pytest.raises(HTTPException) as exc:
            _asyncio.run(agents_v2.start_import_scan(pid, "worker"))
        assert exc.value.status_code == 400
        assert exc.value.detail["code"] == "missing_api_key"
        assert deps["sub_agent_manager"].send.await_count == 0
        assert tc.get(f"/api/v2/projects/{pid}/sessions").json()["sessions"] == []


class TestKeyedImportScan:
    def test_the_manager_runs_the_guess_flavour(self, tmp_path, monkeypatch):
        import asyncio as _asyncio
        from unittest.mock import AsyncMock

        from agent_os.api.routes import agents_v2
        from agent_os.onboarding import import_scan

        tc, deps = build_client(tmp_path, keyed=True)
        ws = tmp_path / "ws"; ws.mkdir()
        (ws / "README.md").write_text("x", encoding="utf-8")
        pid = create_project(tc, ws)["project_id"]
        deps["project_store"].update_project(pid, {import_scan.LANG_FIELD: "zh"})
        start = AsyncMock()
        monkeypatch.setattr(deps["agent_manager"], "start_agent", start)
        sid = _asyncio.run(agents_v2.start_import_scan(pid, "orbital"))
        assert start.await_args.kwargs["session_id"] == sid
        assert start.await_args.kwargs["cold_start"] == "guess:zh"
        assert deps["sub_agent_manager"].send.await_count == 0
