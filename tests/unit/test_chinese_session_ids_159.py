# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 159: a Chinese project name behaves like an ASCII one.

Session ids start with the project name (``中文项目_3ffcfbcb``). Three checks
only accepted ASCII stems (the manager's ``read_session``, a close in ASKS.md
that cites a session, and the worker job launcher), and project create /
import stripped every punctuation mark from the typed name.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from agent_os.agent import asks
from agent_os.agent.project_paths import SESSION_STEM_RE, ProjectPaths
from agent_os.agent.tools.sessions_tool import ReadSessionTool
from agent_os.agent.tools.write import WriteTool
from agent_os.daemon_v2 import job_launcher
from agent_os.daemon_v2.agent_manager import _sanitize_project_name
from agent_os.daemon_v2.project_store import ProjectStore

CN_STEM = "中文项目_3ffcfbcb"
BAD_STEMS = ["../x", "..", "a/b", "a\\b", "", "a\x00b", "a b", "a\nb", "a:b"]


# ---------------------------------------------------------------------------
# The shared rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "中文项目", "我的 项目（测试）、v1.0：草稿!", "改名（第二版）.v2", "导入 测试（一）",
    "Control", "-leading-dash", "日本語プロジェクト", "한국어", "", "!!!",
])
def test_every_minted_stem_passes_the_shared_rule(name):
    stem = f"{_sanitize_project_name(name)}_3ffcfbcb"
    assert SESSION_STEM_RE.fullmatch(stem), stem


def test_worker_stems_pass_the_shared_rule():
    assert SESSION_STEM_RE.fullmatch("worker_worker_deadbeef_0_cafebabe")


@pytest.mark.parametrize("stem", BAD_STEMS + [".hidden", "."])
def test_separators_and_control_bytes_fail_the_shared_rule(stem):
    assert not SESSION_STEM_RE.fullmatch(stem)


# ---------------------------------------------------------------------------
# A: read_session
# ---------------------------------------------------------------------------

def _session_ws(tmp_path, stem, content="我们决定用中文发布。"):
    workspace = tmp_path / "ws"
    sessions = workspace / "orbital" / "sessions"
    sessions.mkdir(parents=True)
    (sessions / f"{stem}.jsonl").write_text(json.dumps({
        "role": "user", "source": "user",
        "timestamp": "2026-11-04T12:00:00Z", "content": content,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    return workspace


def _read_tool(workspace, stem):
    return ReadSessionTool(
        workspace=str(workspace),
        list_sessions=lambda: [{
            "session_id": stem, "session_uuid": stem, "name": None,
            "origin": "chat", "status": "idle",
            "last_activity_at": "2026-11-04T12:00:00Z",
        }],
    )


def test_read_session_opens_a_chinese_stem(tmp_path):
    ws = _session_ws(tmp_path, CN_STEM)
    result = _read_tool(ws, CN_STEM).execute(session_uuid=CN_STEM)
    assert not result.content.startswith("Error"), result.content
    data = json.loads(result.content)
    assert data["session_uuid"] == CN_STEM
    assert data["messages"][0]["content"] == "我们决定用中文发布。"


@pytest.mark.parametrize("stem", BAD_STEMS)
def test_read_session_still_rejects_separators(tmp_path, stem):
    ws = _session_ws(tmp_path, CN_STEM)
    # Even when the lister would "authorize" the bad stem, the shape check
    # refuses it first.
    result = _read_tool(ws, stem).execute(session_uuid=stem)
    assert result.content == "Error: invalid session_uuid"


# ---------------------------------------------------------------------------
# B: an ASKS.md close citing a Chinese session
# ---------------------------------------------------------------------------

ANSWER = "我和合伙人商量好了：网络研讨会用中文，定了。"


@pytest.fixture
def asks_ws(tmp_path):
    workspace = str(tmp_path)
    orbital = os.path.join(workspace, "orbital")
    os.makedirs(orbital, exist_ok=True)
    ask = asks.open_ask(orbital, "网络研讨会用中文还是英文", today="2026-11-03")
    sessions = ProjectPaths(workspace).sessions_dir
    os.makedirs(sessions, exist_ok=True)
    with open(os.path.join(sessions, f"{CN_STEM}.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"role": "user", "content": ANSWER, "source": "user"},
                           ensure_ascii=False) + "\n")
    return workspace, orbital, ask.id


def test_a_cited_chinese_session_verifies_the_quote(asks_ws):
    workspace, _, _ = asks_ws
    sessions = ProjectPaths(workspace).sessions_dir
    note = f'session:{CN_STEM} "网络研讨会用中文，定了"'
    assert asks._cited_quote_verified(note, sessions) == (CN_STEM, True)


def test_a_close_citing_a_chinese_session_closes_the_ask(asks_ws):
    workspace, orbital, ask_id = asks_ws
    current = open(os.path.join(orbital, "ASKS.md"), encoding="utf-8").read()
    out = WriteTool(workspace).execute(
        path="orbital/ASKS.md",
        content=current + f'- done {ask_id} session:{CN_STEM} "网络研讨会用中文，定了"\n',
    )
    assert '"status": "success"' in out.content and "not closed" not in out.content
    ask = {a.id: a for a in asks.read_asks(orbital)}[ask_id]
    assert ask.state == "done" and ask.closed_by == "agent"


@pytest.mark.parametrize("stem", ["../" + CN_STEM, "a/b", "a\\b", ""])
def test_a_cited_path_escape_is_still_not_verified(asks_ws, stem):
    workspace, _, _ = asks_ws
    sessions = ProjectPaths(workspace).sessions_dir
    assert asks._session_user_texts(sessions, stem) == []


# ---------------------------------------------------------------------------
# C: the worker job launcher keeps the Chinese part of the session id
# ---------------------------------------------------------------------------

def _run_bg(ws, **env):
    job_launcher.ensure_launcher(str(ws))
    base = {k: v for k, v in os.environ.items() if not k.startswith("ORBITAL_")}
    out = subprocess.run(
        [os.path.join(str(ws), "orbital", "bin", "bg"), "true"], cwd=str(ws),
        capture_output=True, text=True, timeout=10, env={**base, **env})
    assert out.returncode == 0, out.stderr
    job_id = out.stdout.split("Started background job ")[1].split(" ")[0]
    with open(os.path.join(str(ws), "orbital", "jobs", job_id, "job.json"),
              encoding="utf-8") as f:
        return json.load(f), out.stdout


def test_bg_writes_a_chinese_session_id_intact(tmp_path):
    record, stdout = _run_bg(tmp_path, ORBITAL_SESSION_ID=CN_STEM,
                             ORBITAL_PROJECT_ID="proj_abc", ORBITAL_AGENT="codex")
    assert record["session_id"] == CN_STEM
    assert "Orbital will send you a message when it finishes" in stdout


def test_bg_still_strips_json_breaking_bytes(tmp_path):
    record, _ = _run_bg(tmp_path, ORBITAL_SESSION_ID=CN_STEM + '"\\x\t',
                        ORBITAL_AGENT='co"dex')
    assert record["session_id"] == CN_STEM + "x"
    assert record["agent"] == "codex"


def test_bg_ps1_keeps_unicode_word_characters():
    # .NET's \w is Unicode; ConvertTo-Json does the escaping.
    assert "-replace '[^\\w.:-]', ''" in job_launcher.BG_PS1
    assert "A-Za-z0-9" not in job_launcher.BG_PS1


# ---------------------------------------------------------------------------
# D: create / import keep the name as typed
# ---------------------------------------------------------------------------

def test_create_keeps_the_typed_name_verbatim(tmp_path):
    store = ProjectStore(data_dir=str(tmp_path))
    pid = store.create_project({"name": "  我的 项目（测试）、v1.0：草稿!  ",
                                "workspace": str(tmp_path / "w")})
    project = store.get_project(pid)
    assert project["name"] == "我的 项目（测试）、v1.0：草稿!"
    assert project["agent_name"] == "我的 项目（测试）、v1.0：草稿!"


def test_import_style_create_keeps_folder_punctuation(tmp_path):
    store = ProjectStore(data_dir=str(tmp_path))
    pid = store.create_project({"name": "导入 测试（一）", "workspace": str(tmp_path / "w"),
                                "auto_unique_name": True})
    assert store.get_project(pid)["name"] == "导入 测试（一）"
    pid2 = store.create_project({"name": "导入 测试（一）", "workspace": str(tmp_path / "w2"),
                                 "auto_unique_name": True})
    assert store.get_project(pid2)["name"] == "导入 测试（一）-2"
