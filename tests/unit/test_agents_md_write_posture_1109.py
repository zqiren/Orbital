# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""1109 W4: the seeded AGENTS.md no longer grants write access to the memory
files that the sub-agent prompt forbids.

Codex gets no Orbital system prompt, so AGENTS.md is all it reads. The v0.21.0
seed said "Full read/write on everything listed above … no asking permission
first", which let a manager-dispatched Codex rewrite PROJECT_STATE / DECISIONS
/ LESSONS / INDEX mid-run, while every other worker is told not to and the
manager records at the end of the run (spec 112). New projects get a seed that
leaves those files to Orbital's record turns. Existing projects keep the file
they have: an unedited v0.21.0 seed is still recognised as Orbital's own (the
empty-workspace check) but is not rewritten.
"""

import os

from agent_os.daemon_v2 import agent_md_seeder as seeder
from agent_os.daemon_v2.project_store import ProjectStore


def _flat(text: str) -> str:
    return " ".join(text.split())


def _project(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    store = ProjectStore(data_dir=str(tmp_path / "data"))
    pid = store.create_project({"name": "P", "workspace": str(ws)})
    return store, pid, ws


def test_new_seed_leaves_memory_files_to_orbital():
    text = _flat(seeder.AGENT_MD_TEMPLATE.format(project_name="P", agent_name="A"))
    assert "Full read/write on everything" not in text
    assert "[Orbital memory]" in text
    assert "change them only when Orbital asks you to record" in text
    # The append-only ASKS rule and the hands-off runtime zone stay.
    assert "append-only" in text
    assert "orbital/sessions/" in text
    # Still Orbital's own signpost for spec 138's detection.
    assert seeder.SEED_MARKER in seeder.AGENT_MD_TEMPLATE


def test_seed_does_not_tell_agents_to_update_memory_as_they_go():
    text = _flat(seeder.AGENT_MD_TEMPLATE)
    for stale in ("update it when you learn something worth keeping",
                  "Append whenever you recover",
                  "update it when files move"):
        assert stale not in text


def test_new_project_gets_the_new_seed(tmp_path):
    store, pid, ws = _project(tmp_path)
    assert seeder.seed_project_agent_md(store, pid)["status"] == "ok"
    body = (ws / "AGENTS.md").read_text(encoding="utf-8")
    assert "[Orbital memory]" in body


def test_v0_21_seed_is_still_recognised_as_orbitals_own(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text(seeder._TEMPLATE_V0_21.format(project_name="Old", agent_name="Old"),
                    encoding="utf-8")
    assert seeder.is_seeded_agent_md(str(path))


def test_existing_v0_21_seed_is_not_rewritten_at_pin(tmp_path):
    store, pid, ws = _project(tmp_path)
    old = seeder._TEMPLATE_V0_21.format(project_name="P", agent_name="P")
    dest = ws / "AGENTS.md"
    dest.write_text(old, encoding="utf-8")
    before = os.path.getmtime(dest)

    result = seeder.reseed_project_agent_md(store, pid)

    assert result["status"] == "kept_previous_seed"
    assert dest.read_text(encoding="utf-8") == old
    assert os.path.getmtime(dest) == before
