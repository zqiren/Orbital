# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""PROJECT_STATE edits match the view the agent sees (spec 089 v2).

Every PROJECT_STATE bullet now carries a hidden ``<!--mem id … -->`` line,
which the injected prompt view strips. An edit whose old_text spans several
bullets — built from that view — used to miss and cost the agent a re-read.
The edit tool now retries such a miss against the comment-stripped file, and
the write chokepoint re-attaches every unchanged bullet's id.
"""

from __future__ import annotations

import json
import os

from agent_os.agent import user_flags
from agent_os.agent.tools.edit import EditTool
from agent_os.agent.tools.write import WriteTool

STATE = "# Current state\n\n- Webinar date: 2026-10-02.\n- Webinar language: undecided.\n- Venue: TBD.\n"


def _setup(tmp_path):
    ws = str(tmp_path)
    os.makedirs(os.path.join(ws, "orbital"), exist_ok=True)
    WriteTool(workspace=ws).execute(path="orbital/PROJECT_STATE.md", content=STATE)
    path = os.path.join(ws, "orbital", "PROJECT_STATE.md")
    with open(path, encoding="utf-8") as f:
        stamped = f.read()
    return ws, path, stamped


def _ids(content):
    return {e.text.strip(): e.id for e in user_flags.parse_entries(content) if e.id}


def test_a_multi_bullet_edit_built_from_the_stripped_view_applies(tmp_path):
    ws, path, stamped = _setup(tmp_path)
    assert "<!--mem" in stamped  # every bullet is stamped
    before = _ids(stamped)

    old = "- Webinar date: 2026-10-02.\n- Webinar language: undecided.\n"
    new = "- Webinar date: 2026-10-02.\n- Webinar language: English (decided with cofounder).\n"
    out = EditTool(workspace=ws).execute(path="orbital/PROJECT_STATE.md", old_text=old, new_text=new)
    assert json.loads(out.content.split("\n\n")[0])["status"] == "success"

    with open(path, encoding="utf-8") as f:
        after = f.read()
    visible = user_flags.strip_mem_comments(after)
    assert "- Webinar language: English (decided with cofounder).\n" in visible
    assert "undecided" not in visible
    ids = _ids(after)
    # Unchanged bullets keep their ids; the edited one gets a fresh one.
    assert ids["Webinar date: 2026-10-02."] == before["Webinar date: 2026-10-02."]
    assert ids["Venue: TBD."] == before["Venue: TBD."]
    assert ids["Webinar language: English (decided with cofounder)."] not in before.values()


def test_a_miss_in_both_views_still_reports_not_found(tmp_path):
    ws, path, stamped = _setup(tmp_path)
    out = EditTool(workspace=ws).execute(
        path="orbital/PROJECT_STATE.md", old_text="- Nothing like this\n", new_text="x")
    assert out.content.startswith("Error: old_text not found")
    with open(path, encoding="utf-8") as f:
        assert f.read() == stamped


def test_other_files_keep_exact_matching(tmp_path):
    ws = str(tmp_path)
    with open(os.path.join(ws, "notes.md"), "w", encoding="utf-8") as f:
        f.write("a\n<!--mem id:abc123-->\nb\n")
    out = EditTool(workspace=ws).execute(path="notes.md", old_text="a\nb\n", new_text="c\n")
    assert out.content.startswith("Error: old_text not found")
