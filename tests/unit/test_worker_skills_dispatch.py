# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Workers learn the skills: the listing rides the dispatched message.

It goes in front of the worker's message (after Orbital's first-turn note,
before the recap), never into a system prompt, config or file. Once per
worker session; again only when the set changed since it was last sent to
that session, or when the worker's thread started over. With nothing to
tell, the dispatched text is byte-identical to before.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_os.agent import skills as skills_mod
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager, _QueuedPrompt

SID = "s1"


def _skill(folder, name, description="Does a thing."):
    os.makedirs(os.path.join(folder, name), exist_ok=True)
    with open(os.path.join(folder, name, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n")
    return os.path.join(folder, name, "SKILL.md")


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    home.mkdir()
    monkeypatch.setenv("ORBITAL_SKILLS_HOME", str(home))
    glob = tmp_path / "data" / "skills"
    skills_mod.configure_global_skills_dir(str(glob))
    yield SimpleNamespace(home=str(home), ws=str(ws), glob=str(glob))
    skills_mod.configure_global_skills_dir(None)


def _manager(ws):
    mgr = SubAgentManager(process_manager=MagicMock(),
                          lifecycle_observer=MagicMock(on_message_routed=AsyncMock()))
    mgr._dispatch_async = AsyncMock()
    store = MagicMock()
    store.get_project.return_value = {"workspace": ws}
    mgr._project_store = store
    return mgr


def _prompt(message):
    return _QueuedPrompt(message=message, dispatch_id="d1",
                         transcript_path="/tmp/t.jsonl", initiator="user_pinned")


def _adapter(status="resumed"):
    return SimpleNamespace(_resume_status=(status, None))


async def _send(mgr, adapter, message, handle="codex", session_id=SID):
    await mgr._dispatch_prompt_locked(adapter, _prompt(message), "p1", session_id, handle)
    return mgr._dispatch_async.await_args.args[3]


async def test_no_skills_means_the_message_is_byte_identical(env):
    mgr = _manager(env.ws)
    assert await _send(mgr, _adapter(), "fix it") == "fix it"


async def test_first_message_carries_the_listing_and_the_second_does_not(env):
    path = _skill(env.glob, "pdf-tidy", "Clean scanned PDFs.")
    mgr = _manager(env.ws)
    adapter = _adapter()
    first = await _send(mgr, adapter, "fix it")
    assert first.startswith("Skills you can use (from Orbital)")
    assert f"- pdf-tidy: Clean scanned PDFs. ({path})" in first
    assert first.endswith("\n\nfix it")
    assert await _send(mgr, adapter, "again") == "again"


async def test_the_listing_sits_after_the_first_turn_note(env):
    _skill(env.glob, "pdf-tidy")
    mgr = _manager(env.ws)
    adapter = _adapter()
    adapter._orbital_first_turn_note = "SEAT NOTE"
    sent = await _send(mgr, adapter, "go")
    assert sent.startswith("SEAT NOTE\n\nSkills you can use")
    assert sent.endswith("\n\ngo")


async def test_a_changed_set_is_sent_again(env):
    _skill(env.glob, "pdf-tidy")
    _skill(os.path.join(env.home, ".claude", "skills"), "workctl")
    mgr = _manager(env.ws)
    adapter = _adapter()
    assert "workctl" in await _send(mgr, adapter, "one")
    assert await _send(mgr, adapter, "two") == "two"
    skills_mod.set_skill_enabled("claude:workctl", False)
    third = await _send(mgr, adapter, "three")
    assert third.startswith("Skills you can use") and "workctl" not in third
    assert "pdf-tidy" in third


async def test_a_rebuilt_worker_that_resumed_its_thread_is_not_told_again(env):
    _skill(env.glob, "pdf-tidy")
    mgr = _manager(env.ws)
    await _send(mgr, _adapter(), "one")
    # A rebuild (connector change, takeover) resumes the same thread.
    assert await _send(mgr, _adapter("resumed"), "two") == "two"


async def test_a_worker_whose_thread_started_over_is_told_again(env):
    _skill(env.glob, "pdf-tidy")
    mgr = _manager(env.ws)
    await _send(mgr, _adapter(), "one")
    fresh = _adapter("fresh")
    assert (await _send(mgr, fresh, "two")).startswith("Skills you can use")
    assert await _send(mgr, fresh, "three") == "three"


async def test_each_session_and_each_worker_is_told_once(env):
    _skill(env.glob, "pdf-tidy")
    mgr = _manager(env.ws)
    assert "pdf-tidy" in await _send(mgr, _adapter(), "a", session_id="s1")
    assert "pdf-tidy" in await _send(mgr, _adapter(), "b", session_id="s2")
    assert "pdf-tidy" in await _send(mgr, _adapter(), "c", handle="claude-code")
    assert await _send(mgr, _adapter(), "d", session_id="s1") == "d"


async def test_the_worker_is_not_told_what_it_loads_itself(env):
    _skill(os.path.join(env.home, ".claude", "skills"), "claude-only")
    _skill(os.path.join(env.home, ".codex", "skills"), "codex-only")
    mgr = _manager(env.ws)
    to_claude = await _send(mgr, _adapter(), "x", handle="claude-code")
    to_codex = await _send(mgr, _adapter(), "x", handle="codex")
    assert "codex-only" in to_claude and "claude-only" not in to_claude
    assert "claude-only" in to_codex and "codex-only" not in to_codex


async def test_project_skills_are_listed_with_absolute_paths(env):
    path = _skill(os.path.join(env.ws, "orbital", "skills"), "weekly-report")
    mgr = _manager(env.ws)
    assert f"- weekly-report: Does a thing. ({path})" in await _send(mgr, _adapter(), "x")


async def test_a_listing_failure_never_blocks_the_dispatch(env, monkeypatch):
    _skill(env.glob, "pdf-tidy")

    def boom(*a, **kw):
        raise RuntimeError("disk gone")
    monkeypatch.setattr(skills_mod, "worker_skills", boom)
    mgr = _manager(env.ws)
    assert await _send(mgr, _adapter(), "still goes") == "still goes"


async def test_fanout_workers_never_get_it(env):
    """Fanout builds NativeWorkerAdapter directly and never dispatches
    through ``_dispatch_prompt_locked``: there is nothing for it to carry."""
    import inspect
    from agent_os.daemon_v2 import sub_agent_manager as sam
    src = inspect.getsource(sam.SubAgentManager.dispatch_fanout)
    assert "_dispatch_prompt_locked" not in src
