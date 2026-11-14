# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 160's confirm card and prompts, after batch 1114.

Batch 1114 removed the background read of imported projects (its queue,
triggers, signs and switch: ``test_import_no_read_1114.py`` pins that they
stay gone). What these tests still cover is what outlived it: the card that
asks whether drafted goals are right (spec 160 round 4/5, now posted by a
first run's record turn, spec 167) and its answers; the guess-first prompt
text; the first-use seed of a project an older import already gave its
``orbital/`` folder.
"""

from __future__ import annotations

import os
from pathlib import PurePath
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.project_store import ProjectStore
from agent_os.onboarding import import_scan
from agent_os.onboarding.deferred_seed import ORBITAL_SEEDED_FIELD, ensure_project_seeded
from agent_os.onboarding.import_scan import PENDING


@pytest.fixture
def store(tmp_path):
    return ProjectStore(str(tmp_path / "data"))


def _goals(ws) -> str:
    return os.path.join(str(ws), "orbital", "instructions", "project_goals.md")


def _write_goals(ws, text="# Mission\nx\n"):
    path = _goals(ws)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _imported(store, tmp_path, name, *, activity=None, state=PENDING, attempts=None):
    """``attempts``: a pre-167 record's attempt counter (left on old records,
    never read now)."""
    ws = tmp_path / f"ws-{name}"
    ws.mkdir()
    fields = {"name": name, "workspace": str(ws), "deferred_seed": True, "imported": True}
    if state is not None:
        # What a v0.20-v0.22 import stamped.
        fields.update({import_scan.STATE_FIELD: state, import_scan.AT_FIELD: "2026-11-13T00:00:00+00:00",
                       import_scan.ACTIVITY_FIELD: activity})
    if attempts is not None:
        fields["import_scan_attempts"] = attempts
    return store.create_project(fields), ws


# ---------------------------------------------------------------------------
# §5.1 the seed split
# ---------------------------------------------------------------------------


def test_first_use_after_an_older_imports_orbital_part_writes_agents_md_and_skips_the_refold(
        store, tmp_path, monkeypatch):
    """A v0.20-v0.22 import ran the orbital/ part at import and recorded
    ``orbital_seeded``; first use on this build must honour it."""
    from agent_os.onboarding import deferred_seed

    pid, ws = _imported(store, tmp_path, "a")
    (ws / "orbital").mkdir()
    store.update_project(pid, {ORBITAL_SEEDED_FIELD: True})
    calls = []
    monkeypatch.setattr(deferred_seed, "fold_private_notes",
                        lambda *a, **k: calls.append(a) or 0)
    assert ensure_project_seeded(store, pid) is True
    assert (ws / "AGENTS.md").is_file()
    assert store.get_project(pid)["deferred_seed"] is False
    # The fold already ran at import: first use must not copy the notes again
    # over anything the user edited since.
    assert calls == []


# ---------------------------------------------------------------------------
# §5.2 / §5.4 the prompts
# ---------------------------------------------------------------------------


def _prompt_ctx(workspace, **over):
    from agent_os.agent.prompt_builder import PromptContext
    from agent_os.daemon_v2.models import Autonomy

    base = dict(
        workspace=str(workspace), model="m", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[{"handle": "claude-code", "display_name": "Claude Code",
                         "type": "cli"}],
        tool_names=["read", "write", "edit", "shell"], os_type="macos",
        datetime_now="2026-11-06T10:00:00+00:00", project_name="lib",
        project_instructions="", agent_name="lib", project_id="proj_x",
    )
    base.update(over)
    return PromptContext(**base)


def _section(tmp_path, **over):
    from agent_os.agent.prompt_builder import PromptBuilder
    return PromptBuilder(str(tmp_path))._onboarding_or_directive(
        _prompt_ctx(tmp_path, **over))


def _flat(text):
    return " ".join(text.split())


@pytest.fixture
def en_locale(monkeypatch):
    from agent_os.onboarding import deferred_seed
    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")


def test_guess_scan_writes_first_then_asks(tmp_path, en_locale):
    from agent_os.agent import onboarding_prompts as op

    text = _section(tmp_path, cold_start="guess")
    flat = _flat(text)
    assert text.startswith("## COLD-START WORKSPACE SCAN")
    assert "Write NOTHING until the user confirms" not in text
    assert "Do not wait for a confirmation before writing" in flat
    # State + index first, goals LAST (the prompt turns into the directive
    # the moment the goals file exists).
    i_state = flat.index("PROJECT_STATE.md")
    i_index = flat.index("orbital/INDEX.md")
    i_goals = flat.index("project_goals.md")
    assert i_state < i_index < i_goals
    # The path as the prompt renders it (forward slashes on Windows too).
    assert f"{PurePath(str(tmp_path)).as_posix()}/orbital/instructions/project_goals.md" in text
    # Round 4: no closing question — the daemon posts the confirm card.
    # Round 5: the summary ends with 1-3 numbered next steps, the same ones
    # recorded under PROJECT_STATE's "## Next steps" (the card reads them
    # there); still never started.
    assert "is this right?" not in flat and "这样理解对吗" not in flat
    assert "do not suggest a next step" not in flat
    assert "End the reply with the same steps, numbered" in flat
    assert "`## Next steps` heading (exactly that heading" in flat
    assert "Do not ask a question and do not start any step" in flat
    # Live round 4: one scan ended with mark_task_complete and no summary.
    assert "That summary is your final message, as plain text" in flat
    assert "Do not start any work" in flat
    assert op.SHARED_MEMORY_LINE_EN in flat
    assert op.SHARED_MEMORY_LINE_ZH not in flat
    # Spec 167: the card's propose-first flavour is gone — no goals, no section.
    assert _section(tmp_path, cold_start=True) is None


def test_guess_reply_language_follows_the_system(tmp_path, monkeypatch):
    from agent_os.agent import onboarding_prompts as op
    from agent_os.onboarding import deferred_seed

    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "zh")
    flat = _flat(_section(tmp_path, cold_start="guess"))
    assert "Reply to the user in Chinese" in flat
    assert op.SHARED_MEMORY_LINE_ZH in flat
    assert "这样理解对吗" not in flat


def test_guess_files_follow_the_reply_language(tmp_path, en_locale):
    """Pass 2b (E36): the reply followed the language rule but the files did
    not, so a zh card showed a Chinese summary over English next steps
    (spring-budget, trail-notes). The three files are written in the reply's
    language, with or without a LANGUAGE section."""
    for cold in ("guess:zh", "guess"):
        flat = _flat(_section(tmp_path, cold_start=cold))
        assert "in the same language as your reply" in flat


def test_shared_memory_line_only_with_more_than_one_agent(tmp_path, en_locale):
    from agent_os.agent import onboarding_prompts as op

    one = _section(tmp_path, cold_start="guess", enabled_agents=[])
    assert op.SHARED_MEMORY_LINE_EN not in _flat(one)
    many = _flat(_section(tmp_path, cold_start="guess:en"))
    assert op.SHARED_MEMORY_LINE_EN in many
    # Spec 167: no interview greeting carries it any more (no goals, no section).
    assert _section(tmp_path) is None


def test_guess_run_keeps_its_section_after_the_goals_write(tmp_path, en_locale):
    """Round 4: the scan run's prompt does not switch to the directive when
    its goals file appears (the guess-aware directive tail is retired); a
    later turn — cold_start reset — gets the plain directive, byte-identical."""
    from agent_os.agent import onboarding_prompts as op

    _write_goals(tmp_path, "# Mission\nA small parser library.\n")
    flat = _flat(_section(tmp_path, cold_start="guess"))
    assert flat.startswith("## COLD-START WORKSPACE SCAN")
    assert not hasattr(op, "guess_first_directive_tail")
    plain = _flat(_section(tmp_path))
    assert plain.startswith("## PROJECT DIRECTIVE")
    # Spec 167: the interview's wait-for-go tail is gone with the interview.
    assert "propose the first step and who does it, then wait" not in plain
    assert _flat(_section(tmp_path, cold_start=False)) == plain


def test_the_import_language_reaches_the_guess_prompt(tmp_path, monkeypatch):
    from agent_os.agent import onboarding_prompts as op
    from agent_os.onboarding import deferred_seed

    monkeypatch.setattr(deferred_seed, "default_lang", lambda: "en")
    flat = _flat(_section(tmp_path, cold_start="guess:zh"))
    assert "Reply to the user in Chinese" in flat and op.SHARED_MEMORY_LINE_ZH in flat
    assert "Reply to the user in English" in _flat(_section(tmp_path, cold_start="guess"))


def test_skeleton_leaves_out_orbitals_own_folder(tmp_path):
    """orbital/ exists before the scan now (§5.1); its files are not the
    project and must not crowd the inventory the scan plans from."""
    from agent_os.agent.workspace_scan import scan_workspace

    (tmp_path / "README.md").write_text("hi")
    (tmp_path / "orbital" / "history").mkdir(parents=True)
    (tmp_path / "orbital" / "INDEX.md").write_text("x")
    (tmp_path / "orbital" / "history" / "blob").write_text("x" * 50)
    (tmp_path / "orbitals.md").write_text("kept: only the folder named orbital is ours")
    text = scan_workspace(str(tmp_path))
    assert "README.md" in text and "orbitals.md" in text
    assert "orbital/" not in text and "orbital\\" not in text
    assert text.startswith("[WORKSPACE SKELETON] 2 files")


# ---------------------------------------------------------------------------
# Round 3: the scan session's title is final; every other session unchanged
# ---------------------------------------------------------------------------


def _session(tmp_path, origin="cold_start"):
    from agent_os.agent.session import Session
    return Session.new("s1", str(tmp_path), session_id="s1", origin=origin)


def test_scan_title_survives_the_users_first_reply(tmp_path):
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.session_index import derive_disk_entry

    s = _session(tmp_path)
    s.set_name("项目梳理")                      # what the scan's mint does
    s.append({"role": "system", "source": "daemon", "content": "[WORKSPACE SKELETON]"})
    s.append({"role": "assistant", "content": "这是一个解析库。这样理解对吗？"})
    s.append({"role": "user", "content": "对"})
    assert s.name == "项目梳理"
    assert Session.load(s.filepath).name == "项目梳理"
    assert derive_disk_entry(s.filepath, "s1")["name"] == "项目梳理"
    s.set_name("我的解析库")                    # a manual rename still works
    assert Session.load(s.filepath).name == "我的解析库"


def test_every_other_session_still_titles_from_its_first_message(tmp_path):
    """Byte-identical auto-title for non-scan sessions: nothing set at mint,
    the first user message names it, later ones don't."""
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.session_index import derive_disk_entry

    s = _session(tmp_path, origin="chat")
    assert s.name is None
    s.append({"role": "user", "content": "帮我把 README 改短一点"})
    s.append({"role": "user", "content": "second"})
    assert s.name == "帮我把 README 改短一点"
    assert Session.load(s.filepath).name == "帮我把 README 改短一点"
    assert derive_disk_entry(s.filepath, "s1")["name"] == "帮我把 README 改短一点"


# ---------------------------------------------------------------------------
# Round 4: the daemon-posted confirm card
# ---------------------------------------------------------------------------


def _confirm_env(tmp_path, *, keyed, state=None, lang="zh"):
    """A real AgentManager behind agents_v2 (the keyless harness), one
    imported project whose scan wrote goals, and the confirm card posted.
    ``state``: the PROJECT_STATE.md the scan wrote (round 5: its Next steps
    section feeds the card)."""
    import asyncio as _asyncio
    import json as _json

    from agent_os.api.routes import agents_v2
    from agent_os.api.routes import workbench as wb
    from tests.unit.keyless_harness import build_client, create_project

    tc, deps = build_client(tmp_path, keyed=keyed)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("# lib\n")
    pid = create_project(tc, ws)["project_id"]
    deps["project_store"].update_project(pid, {import_scan.LANG_FIELD: lang})
    _write_goals(ws)
    if state is not None:
        (ws / "orbital" / "PROJECT_STATE.md").write_text(state, encoding="utf-8")
    sid = "scan_sess_1"
    # The scan session exists on disk (its first row).
    from agent_os.agent.session import Session
    s = Session.new(sid, str(ws), session_id=sid, origin="cold_start")
    s.append({"role": "assistant", "content": "这是一个解析库。"})
    ask_id = _asyncio.run(agents_v2.post_goals_confirm(pid, sid))
    wb.configure(deps["project_store"], deps["agent_manager"], MagicMock(),
                 ws_manager=deps["ws_manager"])
    rows = [_json.loads(l) for l in open(s.filepath, encoding="utf-8")]
    return tc, deps, pid, sid, ws, ask_id, rows


def test_post_scan_confirm_opens_the_ask_and_its_session_row(tmp_path):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, rows = _confirm_env(tmp_path, keyed=True)
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert ask.text == "这样理解对吗？" and ask.options == ("对", "需要改") and ask.is_open
    card = rows[-1]
    assert card["_kind"] == "ask_card" and card["_meta"]["event"] == "ask_opened"
    assert card["_meta"]["ask_id"] == ask_id and card["_meta"]["import_scan_confirm"] is True
    assert deps["project_store"].get_project(pid)[import_scan.ASK_FIELD] == {
        "id": ask_id, "session_id": sid, "yes": "对", "change": "需要改"}


def _answer(tc, pid, sid, ask_id, words, target=None):
    body = {"content": f'[Answer to ask {ask_id}, already closed: "这样理解对吗？"]\n{words}',
            "session_id": sid, "ask_id": ask_id, "ask_answer": words, "nonce": "n1"}
    if target:
        body["target"] = target
    return tc.post(f"/api/v2/agents/{pid}/inject", json=body)


def test_yes_closes_the_ask_records_the_row_and_dispatches_nothing(tmp_path, monkeypatch):
    import json as _json

    from agent_os.agent import asks
    from agent_os.agent.session import Session

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    mgr = deps["agent_manager"]
    calls = []

    async def spy(*a, **k):
        calls.append(a)
        return "delivered"

    monkeypatch.setattr(mgr, "inject_message", spy)
    monkeypatch.setattr(mgr, "start_agent", spy)
    s = Session.load(str(ws / "orbital" / "sessions" / f"{sid}.jsonl"))
    r = _answer(tc, pid, sid, ask_id, "对")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "recorded"
    assert calls == []                                 # no run, no LLM call
    deps["sub_agent_manager"].send.assert_not_awaited()
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert not ask.is_open
    rows = [_json.loads(l) for l in open(s.filepath, encoding="utf-8")]
    user = [r for r in rows if r.get("role") == "user"]
    assert user[-1]["content"].endswith("\n对") and "<orbital_turn_note>" not in user[-1]["content"]


def test_yes_on_a_worker_session_dispatches_nothing_either(tmp_path):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=False)
    r = _answer(tc, pid, sid, ask_id, "对", target="claude-code")
    assert r.status_code == 200 and r.json()["status"] == "recorded"
    deps["sub_agent_manager"].send.assert_not_awaited()


def test_needs_changes_runs_one_turn_with_the_note_in_that_message_only(tmp_path, monkeypatch):
    """The 需要改 option runs one guarded turn (round 4). Round 5: only that
    option carries the note — a typed reply is a normal message (below)."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    sent = []

    async def spy(project_id, content, **k):
        sent.append(content)
        return "delivered"

    monkeypatch.setattr(deps["agent_manager"], "inject_message", spy)
    r = _answer(tc, pid, sid, ask_id, "需要改")
    assert r.status_code == 200, r.text
    assert len(sent) == 1
    assert sent[0].startswith(f'[Answer to ask {ask_id}, already closed:')
    assert "需要改\n\n<orbital_turn_note>" in sent[0]
    assert "只改 orbital/ 下的文件" in sent[0] and sent[0].endswith("</orbital_turn_note>")


def test_a_typed_reply_is_a_normal_message(tmp_path, monkeypatch):
    """Round 5: the daemon cannot tell a typed correction from a typed "做第
    2 步", so typed text goes through as the user's own message (no note)."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    sent = []

    async def spy(project_id, content, **k):
        sent.append(content)
        return "delivered"

    monkeypatch.setattr(deps["agent_manager"], "inject_message", spy)
    r = _answer(tc, pid, sid, ask_id, "不对，这是给团队用的配置工具")
    assert r.status_code == 200, r.text
    assert len(sent) == 1
    assert sent[0].endswith("\n不对，这是给团队用的配置工具")
    assert "<orbital_turn_note>" not in sent[0] and "<orbital_picked_step>" not in sent[0]


def test_a_worker_needs_changes_sends_the_note_but_stores_only_the_words(tmp_path):
    import json as _json

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=False)
    r = _answer(tc, pid, sid, ask_id, "需要改", target="claude-code")
    assert r.status_code == 200, r.text
    args, _k = deps["sub_agent_manager"].send.await_args
    assert args[2].endswith("</orbital_turn_note>") and "需要改" in args[2]
    rows = [_json.loads(l) for l in open(ws / "orbital" / "sessions" / f"{sid}.jsonl", encoding="utf-8")]
    user = [r for r in rows if r.get("role") == "user"]
    assert user[-1]["content"].endswith("需要改")
    assert "<orbital_turn_note>" not in user[-1]["content"]


def test_the_note_reaches_its_turn_only(tmp_path):
    from agent_os.agent.context import ContextManager, strip_turn_notes
    from agent_os.agent.prompt_builder import PromptContext
    from agent_os.agent.session import Session
    from agent_os.daemon_v2.models import Autonomy

    class _B:
        def build(self, context):
            return ("sys", "semi", "dyn")

    ctx = PromptContext(workspace=str(tmp_path), model="m", autonomy=Autonomy.HANDS_OFF,
                        enabled_agents=[], tool_names=["read"], os_type="macos",
                        datetime_now="2026-11-07T00:00:00", project_name="p",
                        project_instructions="", agent_name="p", project_id="x")
    s = Session.new("g1", str(tmp_path), session_id="g1", origin="cold_start")
    s.append({"role": "assistant", "content": "summary"})
    note = import_scan.turn_note_block("zh")
    s.append({"role": "user", "source": "user", "content": "不对，改成连载\n\n" + note})
    mgr = ContextManager(s, _B(), ctx, model_context_limit=200_000)
    sent = mgr.prepare()
    assert any("<orbital_turn_note>" in str(m.get("content")) for m in sent)
    s.append({"role": "assistant", "content": "已改好。"})
    s.append({"role": "user", "source": "user", "content": "好，接下来写第三篇"})
    later = mgr.prepare()
    assert not any("<orbital_turn_note>" in str(m.get("content")) for m in later)
    assert "不对，改成连载" in " ".join(str(m.get("content")) for m in later)
    # The session row itself keeps the note (restart-safe), only the copy
    # bound for the provider drops it.
    assert any("<orbital_turn_note>" in str(m.get("content")) for m in s.get_messages())
    assert strip_turn_notes([{"role": "user", "content": "a" + note}])[0]["content"].endswith(note)


def test_an_open_scan_ask_is_still_handled_after_a_restart(tmp_path, monkeypatch):
    """The record lives on the project and the ask in ASKS.md: a fresh daemon
    (new routes, new manager) still answers 对 without a run."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    from agent_os.daemon_v2.project_store import ProjectStore as _PS
    from tests.unit.keyless_harness import build_client
    # "Restart": a new client over the same data dir and workspace.
    tc2, deps2 = build_client(tmp_path, keyed=True)
    assert _PS(str(tmp_path / "orbital-data")).get_project(pid)[import_scan.ASK_FIELD]["id"] == ask_id
    called = []

    async def spy(*a, **k):
        called.append(a)
        return "delivered"

    monkeypatch.setattr(deps2["agent_manager"], "inject_message", spy)
    r = _answer(tc2, pid, sid, ask_id, "对")
    assert r.status_code == 200 and r.json()["status"] == "recorded" and called == []


# ---------------------------------------------------------------------------
# Round 5: the scan proposes next steps; the confirm card offers step 1
# ---------------------------------------------------------------------------

_STATE_WITH_STEPS = (
    "<!--format PROJECT_STATE is what is true NOW-->\n"
    "# Project State\n\n"
    "## Current focus\n- tinyparse v0.3.0\n\n"
    "## Next steps\n"
    "- Implement `dumps()` so it round-trips with `loads()`\n"
    "<!--mem id=a1 created=2026-11-07-->\n"
    "  detail line under the bullet, not a step\n"
    "- Add line numbers to `ParseError`\n"
    "- Publish 0.4.0 to PyPI\n"
    "- A fourth one past the cap\n\n"
    "## Notes\n- not a step\n"
)


def _rows(ws, sid):
    import json as _json
    return [_json.loads(l) for l in open(ws / "orbital" / "sessions" / f"{sid}.jsonl",
                                         encoding="utf-8")]


def test_next_steps_reads_the_state_files_section(tmp_path):
    from agent_os.onboarding.import_scan import next_steps

    orb = tmp_path / "orbital"
    orb.mkdir()
    assert next_steps(str(tmp_path)) == []                     # no state file
    (orb / "PROJECT_STATE.md").write_text(_STATE_WITH_STEPS, encoding="utf-8")
    assert next_steps(str(tmp_path)) == [
        "Implement `dumps()` so it round-trips with `loads()`",
        "Add line numbers to `ParseError`",
        "Publish 0.4.0 to PyPI",
    ]
    # The heading as earlier scans wrote it, in either language; numbered items.
    for heading in ("## Likely next steps (guess)", "## Next steps (best guess)",
                    "## 下一步", "### Next Steps"):
        (orb / "PROJECT_STATE.md").write_text(
            f"# S\n\n{heading}\n1. 续写第二篇\n2) 动笔《散步》\n", encoding="utf-8")
        assert next_steps(str(tmp_path)) == ["续写第二篇", "动笔《散步》"], heading
    # Nothing clearly next: no section, or only a placeholder.
    for body in ("# S\n\n## Current focus\n- x\n", "# S\n\n## Next steps\n- (none yet)\n",
                 "# S\n\n## Next steps\n- 无\n",
                 # Spec 167: the record turn's own "nothing is next" lines.
                 "# S\n\n## Next steps\n- None pending; awaiting the user's next instruction.\n",
                 "# S\n\n## Next steps\n- Nothing yet\n",
                 "# S\n\n## Next steps\n- 暂无。\n"):
        (orb / "PROJECT_STATE.md").write_text(body, encoding="utf-8")
        assert next_steps(str(tmp_path)) == [], body
    # A real step that merely starts with those letters is kept.
    (orb / "PROJECT_STATE.md").write_text(
        "# S\n\n## Next steps\n- Nonetheless publish 0.4.0\n- 无锁队列改为有锁\n", encoding="utf-8")
    assert next_steps(str(tmp_path)) == ["Nonetheless publish 0.4.0", "无锁队列改为有锁"]
    # "Wait for the user" is not a step either (dsh wrote it, batch 1110).
    for body in ("# S\n\n## Next steps\n- Wait for the user's actual request; ask what the project is for.\n",
                 "# S\n\n## Next steps\n- Waiting for user input\n",
                 "# S\n\n## Next steps\n- 等待用户提出具体需求\n"):
        (orb / "PROJECT_STATE.md").write_text(body, encoding="utf-8")
        assert next_steps(str(tmp_path)) == [], body
    (orb / "PROJECT_STATE.md").write_text(
        "# S\n\n## Next steps\n- Wait for CI to finish, then tag 0.4.0\n", encoding="utf-8")
    assert next_steps(str(tmp_path)) == ["Wait for CI to finish, then tag 0.4.0"]


def test_the_card_with_steps_offers_step_one(tmp_path):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, rows = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert ask.text == "这样理解对吗？" and ask.is_open
    assert ask.options == ("对，做第 1 步", "对，先不做", "需要改")
    # Spec 169 D: the goals file leads the detail, the steps follow unchanged.
    assert ask.detail == (
        "[project_goals.md](orbital/instructions/project_goals.md)\n\n"
        "1. Implement `dumps()` so it round-trips with `loads()`\n"
        "2. Add line numbers to `ParseError`\n"
        "3. Publish 0.4.0 to PyPI")
    assert rows[-1]["_meta"]["import_scan_confirm"] is True
    rec = deps["project_store"].get_project(pid)[import_scan.ASK_FIELD]
    assert rec["id"] == ask_id and rec["yes"] == "对，先不做"
    assert rec["go"] == "对，做第 1 步" and rec["change"] == "需要改"
    assert rec["steps"][0] == "Implement `dumps()` so it round-trips with `loads()`"
    # English import.
    tc2, deps2, pid2, sid2, ws2, ask2, _ = _confirm_env(
        tmp_path / "en", keyed=True, state=_STATE_WITH_STEPS, lang="en")
    (a2,) = [a for a in asks.read_asks(str(ws2 / "orbital")) if a.id == ask2]
    assert a2.text == "Did I get this right?"
    assert a2.options == ("Yes, do step 1", "Yes, not now", "Needs changes")


def test_the_card_without_steps_is_unchanged(tmp_path):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state="# S\n\n## Current focus\n- x\n")
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    # Spec 169 D: no steps, so the detail is just the goals file's link.
    assert ask.options == ("对", "需要改")
    assert ask.detail == "[project_goals.md](orbital/instructions/project_goals.md)"
    rec = deps["project_store"].get_project(pid)[import_scan.ASK_FIELD]
    assert rec["yes"] == "对" and "go" not in rec and "steps" not in rec


def _spy_inject(monkeypatch, mgr):
    sent = []

    async def spy(project_id, content, **k):
        sent.append(content)
        return "delivered"

    monkeypatch.setattr(mgr, "inject_message", spy)
    monkeypatch.setattr(mgr, "start_agent", spy)
    return sent


def test_do_step_one_runs_once_with_an_explicit_approval(tmp_path, monkeypatch):
    from agent_os.agent import asks

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    r = _answer(tc, pid, sid, ask_id, "对，做第 1 步")
    assert r.status_code == 200, r.text
    assert len(sent) == 1                                   # exactly one run
    msg = sent[0]
    # The user's tap text stays the visible answer; the approval names step 1.
    assert msg.startswith(f'[Answer to ask {ask_id}, already closed: "这样理解对吗？"]\n对，做第 1 步')
    assert ("<orbital_picked_step>对，按第 1 步开始：Implement `dumps()` so it "
            "round-trips with `loads()`</orbital_picked_step>") in msg
    assert "<orbital_turn_note>" not in msg                 # not a guarded turn
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert not ask.is_open


def test_do_step_one_approval_stays_in_context(tmp_path):
    from agent_os.agent.context import strip_turn_notes
    from agent_os.agent.session import Session

    s = Session.new("g2", str(tmp_path), session_id="g2", origin="cold_start")
    s.append({"role": "assistant", "content": "summary\n1. a\n2. b"})
    block = import_scan.picked_step_block("zh", "a")
    s.append({"role": "user", "source": "user", "content": "对，做第 1 步\n\n" + block})
    later = strip_turn_notes(s.get_messages() + [{"role": "user", "content": "next"}])
    assert block in later[-2]["content"]                    # kept on later turns


def test_do_step_one_on_a_worker_sends_the_approval_stores_the_tap(tmp_path):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=False, state=_STATE_WITH_STEPS)
    r = _answer(tc, pid, sid, ask_id, "对，做第 1 步", target="claude-code")
    assert r.status_code == 200, r.text
    assert deps["sub_agent_manager"].send.await_count == 1
    args, _k = deps["sub_agent_manager"].send.await_args
    assert "<orbital_picked_step>对，按第 1 步开始：Implement" in args[2]
    user = [r for r in _rows(ws, sid) if r.get("role") == "user"]
    assert user[-1]["content"].endswith("\n对，做第 1 步")


def test_not_now_is_zero_call(tmp_path, monkeypatch):
    from agent_os.agent import asks
    from agent_os.agent.session import Session

    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    r = _answer(tc, pid, sid, ask_id, "对，先不做")
    assert r.status_code == 200 and r.json()["status"] == "recorded"
    assert sent == []
    (ask,) = [a for a in asks.read_asks(str(ws / "orbital")) if a.id == ask_id]
    assert not ask.is_open


@pytest.mark.parametrize("typed", ["对", "对。", " 对！", "对的", "Yes", "yes."])
def test_a_typed_yes_is_still_zero_call(tmp_path, monkeypatch, typed):
    """The round-3 bug: a bare 对 must never become the go-ahead, on either
    card, typed or tapped."""
    for state in (_STATE_WITH_STEPS, None):
        root = tmp_path / ("steps" if state else "plain")
        tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(root, keyed=True, state=state)
        sent = _spy_inject(monkeypatch, deps["agent_manager"])
        r = _answer(tc, pid, sid, ask_id, typed)
        assert r.status_code == 200 and r.json()["status"] == "recorded", (typed, state)
        assert sent == []


def test_typed_do_step_two_is_a_normal_run(tmp_path, monkeypatch):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    r = _answer(tc, pid, sid, ask_id, "做第 2 步")
    assert r.status_code == 200, r.text
    assert len(sent) == 1 and sent[0].endswith("\n做第 2 步")
    assert "<orbital_turn_note>" not in sent[0] and "<orbital_picked_step>" not in sent[0]


def test_needs_changes_on_the_steps_card_is_the_guarded_turn(tmp_path, monkeypatch):
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(
        tmp_path, keyed=True, state=_STATE_WITH_STEPS)
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    _answer(tc, pid, sid, ask_id, "需要改")
    assert len(sent) == 1 and "<orbital_turn_note>" in sent[0]
    assert "<orbital_picked_step>" not in sent[0]


def test_a_round_four_record_still_answers(tmp_path, monkeypatch):
    """An ask posted by the round-4 daemon (record without go/change): 对 is
    zero-call, 需要改 the guarded turn."""
    tc, deps, pid, sid, ws, ask_id, _ = _confirm_env(tmp_path, keyed=True)
    deps["project_store"].update_project(pid, {import_scan.ASK_FIELD: {
        "id": ask_id, "session_id": sid, "yes": "对"}})
    sent = _spy_inject(monkeypatch, deps["agent_manager"])
    _answer(tc, pid, sid, ask_id, "需要改")
    assert len(sent) == 1 and "<orbital_turn_note>" in sent[0]


def test_a_prefilled_change_answer_is_still_the_guarded_change_turn():
    """Batch 1110 item 2: the card prefills 「需要改：」 / "Needs changes: " and
    the user types the correction after it; the whole line classifies as
    change (one guarded turn), not as a free message."""
    from agent_os.onboarding.import_scan import classify_answer
    rec = {"yes": "对", "change": "需要改"}
    assert classify_answer(rec, "需要改", "zh") == "change"
    assert classify_answer(rec, "需要改：目标应该说这是练手项目", "zh") == "change"
    assert classify_answer(rec, "需要改: goals should say kata", "zh") == "change"
    assert classify_answer(rec, "需要改进一下文风", "zh") == "message"  # not the option + colon
    rec_en = {"yes": "Yes", "change": "Needs changes"}
    assert classify_answer(rec_en, "Needs changes: it is a shop, not a blog", "en") == "change"
    assert classify_answer(rec_en, "Needs changes to the README", "en") == "message"
