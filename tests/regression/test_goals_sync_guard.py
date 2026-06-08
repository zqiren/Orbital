import os

from agent_os.agent.project_paths import ProjectPaths
from agent_os.api.routes.agents_v2 import _maybe_sync_instructions_to_goals


def test_sync_writes_goals_when_absent(tmp_path):
    ws = str(tmp_path)
    _maybe_sync_instructions_to_goals(ws, goals_content=None, instructions="do X")
    assert "do X" in open(ProjectPaths(ws).project_goals).read()


def test_sync_does_not_clobber_existing_goals(tmp_path):
    ws = str(tmp_path)
    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    open(pp.project_goals, "w").write("SCAN-AUTHORED GOALS")
    _maybe_sync_instructions_to_goals(ws, goals_content=None, instructions="stale field")
    assert open(pp.project_goals).read() == "SCAN-AUTHORED GOALS"


def test_explicit_goals_content_always_wins(tmp_path):
    ws = str(tmp_path)
    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    open(pp.project_goals, "w").write("old")
    _maybe_sync_instructions_to_goals(ws, goals_content="explicit new", instructions="ignored")
    assert open(pp.project_goals).read() == "explicit new"
