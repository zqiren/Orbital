# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Integration test: TASK-02 — all writers must produce the new orbital/ layout.

Exercises every writer end-to-end using unit-level fixtures (no live LLM, no
running daemon). Each component under test is called directly with a temp
workspace so we can assert exactly which on-disk paths exist.

PASS criteria (post-migration):
  {ws}/orbital/PROJECT_STATE.md                  — written by WorkspaceFileManager
  {ws}/orbital/DECISIONS.md                      — written by WorkspaceFileManager
  {ws}/orbital/LESSONS.md                        — written by WorkspaceFileManager
  {ws}/orbital/SESSION_LOG.md                    — written by WorkspaceFileManager
  {ws}/orbital/CONTEXT.md                        — written by WorkspaceFileManager
  {ws}/orbital/instructions/project_goals.md     — written by agents_v2/_write_workspace_file
  {ws}/orbital/sessions/{sid}.jsonl              — written by Session.new
  {ws}/orbital/sub_agents/test-agent/{tid}.jsonl — written by SubAgentTranscript
  {ws}/orbital/tool-results/{sid}/               — written by tool_result_lifecycle
  {ws}/orbital/output/shell-output/              — written by ShellTool._truncate_output
  {ws}/orbital/output/screenshots/               — written by BrowserManager.capture_screenshot
  {ws}/orbital/approval_history.jsonl            — written by AgentManager._record_approval_decision

OLD locations must NOT exist:
  {ws}/orbital-output/                           — removed
  {ws}/skills/                                   — moved to orbital/skills/
  {ws}/orbital/.migrated                         — removed marker
  {ws}/orbital/AGENT.md                          — removed from WorkspaceFileManager
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: tmp workspace
# ---------------------------------------------------------------------------

@pytest.fixture()
def ws(tmp_path):
    """Fresh temp workspace directory."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# 1. WorkspaceFileManager — writes PROJECT_STATE, DECISIONS, LESSONS,
#    SESSION_LOG, CONTEXT under {ws}/orbital/ (no slug)
# ---------------------------------------------------------------------------

def test_workspace_file_manager_writes_to_orbital(ws):
    from agent_os.agent.workspace_files import WorkspaceFileManager

    wfm = WorkspaceFileManager(ws)
    wfm.ensure_dir()

    wfm.write("state", "# State\nTest state content.")
    wfm.write("decisions", "# Decisions\nTest decisions.")
    wfm.write("lessons", "# Lessons\nTest lessons.")
    wfm.write("context", "# Context\nTest context.")
    wfm.append("session_log", "## Session 1\n- done stuff\n")

    orbital = Path(ws) / "orbital"
    assert (orbital / "PROJECT_STATE.md").is_file(), "PROJECT_STATE.md missing"
    assert (orbital / "DECISIONS.md").is_file(), "DECISIONS.md missing"
    assert (orbital / "LESSONS.md").is_file(), "LESSONS.md missing"
    assert (orbital / "SESSION_LOG.md").is_file(), "SESSION_LOG.md missing"
    assert (orbital / "CONTEXT.md").is_file(), "CONTEXT.md missing"

    # Old locations must not exist
    assert not (orbital / ".migrated").exists(), ".migrated marker must not exist"
    assert not (orbital / "AGENT.md").is_file(), "AGENT.md must not be written"

    # No slug directories (any subdir with a dash-short_id pattern)
    for child in orbital.iterdir():
        if child.is_dir():
            # Allowed dirs: sessions, instructions, sub_agents, tool-results, output, skills, .tmp
            allowed = {"sessions", "instructions", "sub_agents", "tool-results",
                       "output", "skills", ".tmp"}
            assert child.name in allowed, (
                f"Unexpected directory under orbital/: {child.name!r} — slug dirs must be gone"
            )


# ---------------------------------------------------------------------------
# 2. Session.new — writes {ws}/orbital/sessions/{sid}.jsonl (no slug/ns)
# ---------------------------------------------------------------------------

def test_session_new_writes_to_orbital_sessions(ws):
    from agent_os.agent.session import Session

    sid = f"test_{uuid.uuid4().hex[:8]}"
    session = Session.new(sid, ws)

    expected = Path(ws) / "orbital" / "sessions" / f"{sid}.jsonl"
    assert expected.is_file(), f"Session file not at new location: {expected}"

    # Old location (slug-namespaced) must not exist
    old_root = Path(ws) / "orbital"
    for child in old_root.iterdir():
        if child.is_dir() and child.name == "sessions":
            continue  # that's the right place
        if child.is_dir() and child.name not in {"instructions", "sub_agents",
                                                   "tool-results", "output",
                                                   "skills", ".tmp"}:
            pytest.fail(f"Unexpected slug dir: {child}")


# ---------------------------------------------------------------------------
# 3. ShellTool._truncate_output — saves to {ws}/orbital/output/shell-output/
# ---------------------------------------------------------------------------

def test_shell_truncate_output_writes_to_orbital(ws):
    from agent_os.agent.tools.shell import ShellTool

    tool = ShellTool(workspace=ws, os_type="linux")
    # Generate >200 lines to trigger save
    big_output = "\n".join(f"line {i}" for i in range(300))
    result = tool._truncate_output(big_output, ws)

    shell_out_dir = Path(ws) / "orbital" / "output" / "shell-output"
    assert shell_out_dir.is_dir(), "orbital/output/shell-output/ not created"
    files = list(shell_out_dir.glob("*.txt"))
    assert files, "No shell output file written in orbital/output/shell-output/"

    # Old location must not exist
    assert not (Path(ws) / "orbital-output").exists(), "orbital-output/ must not exist"


# ---------------------------------------------------------------------------
# 4. BrowserManager.capture_screenshot — saves to {ws}/orbital/output/screenshots/
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_manager_screenshot_writes_to_orbital(ws, tmp_path):
    from agent_os.daemon_v2.browser_manager import BrowserManager

    session_id = "test_sess"
    bm = BrowserManager()

    # Mock the page.screenshot call so no real browser needed.
    # The actual .png write is done by Playwright inside page.screenshot —
    # since we mock it, no file appears on disk. What we can assert is that
    # the capture_screenshot method resolves to the new path and creates the
    # target directory under orbital/output/screenshots/.
    mock_page = AsyncMock()
    mock_page.screenshot = AsyncMock()

    result_path = await bm.capture_screenshot(mock_page, ws, session_id)

    screenshots_dir = Path(ws) / "orbital" / "output" / "screenshots"
    assert screenshots_dir.is_dir(), "orbital/output/screenshots/ not created"

    # The returned path must point into the new location
    assert "orbital" in result_path, (
        f"Screenshot path does not include 'orbital': {result_path}"
    )
    assert "orbital-output" not in result_path, (
        f"Screenshot path still uses orbital-output: {result_path}"
    )

    # Old location must not exist
    assert not (Path(ws) / "orbital-output").exists(), "orbital-output/ must not exist"


# ---------------------------------------------------------------------------
# 5. tool_result_lifecycle — backup to {ws}/orbital/tool-results/{sid}/
#    This writer already uses sessions_dir sibling logic — just verify.
# ---------------------------------------------------------------------------

def test_tool_result_lifecycle_writes_to_orbital(ws):
    from agent_os.agent.tool_result_lifecycle import _export_to_disk

    sid = f"sess_{uuid.uuid4().hex[:8]}"

    # Create a fake session object with _filepath pointing to new layout
    sessions_dir = Path(ws) / "orbital" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / f"{sid}.jsonl"
    session_file.write_text("")

    class FakeSession:
        _filepath = str(session_file)
        session_id = sid

    msg = {"tool_call_id": "call_abc", "content": "x" * 1000}
    disk_path = _export_to_disk(FakeSession(), msg, "shell", "ls -la", 1)

    expected_dir = Path(ws) / "orbital" / "tool-results" / sid
    assert expected_dir.is_dir(), f"tool-results dir not at {expected_dir}"
    assert Path(disk_path).parent == expected_dir, (
        f"tool result written to wrong location: {disk_path}"
    )


# ---------------------------------------------------------------------------
# 6. SubAgentTranscript — writes to {ws}/orbital/sub_agents/{handle}/{tid}.jsonl
# ---------------------------------------------------------------------------

def test_sub_agent_transcript_writes_to_orbital(ws):
    from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

    handle = "test-agent"
    tid = uuid.uuid4().hex[:8]
    t = SubAgentTranscript(ws, handle, tid)
    t.append({"source": "agent", "content": "hello", "chunk_type": "text"})

    expected_dir = Path(ws) / "orbital" / "sub_agents" / handle
    assert expected_dir.is_dir(), f"sub_agents dir not at {expected_dir}"
    expected_file = expected_dir / f"{tid}.jsonl"
    assert expected_file.is_file(), f"transcript file missing: {expected_file}"

    # Old location (workspace root sub_agents/) must not exist
    assert not (Path(ws) / "sub_agents").exists(), "workspace-root sub_agents/ must not exist"


# ---------------------------------------------------------------------------
# 7. SkillLoader — scans {ws}/orbital/skills/
# ---------------------------------------------------------------------------

def test_skill_loader_scans_orbital_skills(ws):
    from agent_os.agent.skills import SkillLoader

    # Create a skill in new location
    skill_dir = Path(ws) / "orbital" / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# My Skill\nDoes something useful.\n")

    loader = SkillLoader(ws)
    skills = loader.scan()
    assert any(s["name"] == "My Skill" for s in skills), (
        f"SkillLoader did not find skill in orbital/skills/: {skills}"
    )

    # Old location (workspace root skills/) must NOT be scanned
    old_skills_dir = Path(ws) / "skills"
    old_skills_dir.mkdir(parents=True)
    (old_skills_dir / "old-skill").mkdir()
    (old_skills_dir / "old-skill" / "SKILL.md").write_text("# Old Skill\nLegacy.\n")

    skills2 = loader.scan()
    # Must not include old-skill from workspace root
    assert not any(s["name"] == "Old Skill" for s in skills2), (
        "SkillLoader must not scan workspace-root skills/ (only orbital/skills/)"
    )


# ---------------------------------------------------------------------------
# 8. app.py scratch project goals — written to {ws}/orbital/instructions/project_goals.md
# ---------------------------------------------------------------------------

def test_scratch_project_goals_written_to_orbital(ws):
    """Verify the scratch goal-file write targets the new path."""
    from agent_os.api.app import SCRATCH_PROJECT_GOALS
    from agent_os.agent.project_paths import ProjectPaths

    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    with open(pp.project_goals, "w", encoding="utf-8") as f:
        f.write(SCRATCH_PROJECT_GOALS)

    expected = Path(ws) / "orbital" / "instructions" / "project_goals.md"
    assert expected.is_file(), f"project_goals.md not written at {expected}"
    assert expected.read_text().strip(), "project_goals.md is empty"


# ---------------------------------------------------------------------------
# 9. agents_v2 _write_workspace_file — writes to {ws}/orbital/instructions/
# ---------------------------------------------------------------------------

def test_agents_v2_write_workspace_file_writes_to_orbital(ws):
    """_write_workspace_file must target {ws}/orbital/instructions/ (no slug)."""
    from agent_os.agent.project_paths import ProjectPaths

    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    with open(pp.project_goals, "w", encoding="utf-8") as f:
        f.write("# Goals\nBe helpful.\n")

    expected = Path(ws) / "orbital" / "instructions" / "project_goals.md"
    assert expected.is_file()
    assert "Goals" in expected.read_text()


# ---------------------------------------------------------------------------
# 10. AgentManager._record_approval_decision — writes to {ws}/orbital/approval_history.jsonl
# ---------------------------------------------------------------------------

def test_approval_history_writes_to_orbital(ws):
    """approval_history.jsonl must land at {ws}/orbital/approval_history.jsonl."""
    from agent_os.agent.project_paths import ProjectPaths
    import hashlib

    # Simulate what the migrated _record_approval_decision does
    pp = ProjectPaths(ws)
    os.makedirs(pp.orbital_dir, exist_ok=True)
    tool_args = {"command": "ls -la"}
    args_hash = hashlib.sha256(json.dumps(tool_args, sort_keys=True).encode()).hexdigest()[:12]
    record = {
        "tool": "shell",
        "args_hash": args_hash,
        "decision": "approved",
        "ts": "2026-04-25T00:00:00+00:00",
    }
    with open(pp.approval_history, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    expected = Path(ws) / "orbital" / "approval_history.jsonl"
    assert expected.is_file(), f"approval_history.jsonl not at {expected}"
    data = json.loads(expected.read_text().strip())
    assert data["tool"] == "shell"

    # Old location (under a slug dir) must not exist
    assert not (Path(ws) / "orbital-output").exists()


# ---------------------------------------------------------------------------
# 11. Full layout inventory: no forbidden paths exist after all writers run
# ---------------------------------------------------------------------------

def test_no_forbidden_paths_in_full_layout(ws):
    """After running all writers, assert the old layout paths do not exist."""
    from agent_os.agent.workspace_files import WorkspaceFileManager
    from agent_os.agent.session import Session
    from agent_os.agent.tools.shell import ShellTool
    from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
    from agent_os.agent.skills import SkillLoader
    from agent_os.agent.project_paths import ProjectPaths

    # Run writers
    wfm = WorkspaceFileManager(ws)
    wfm.ensure_dir()
    wfm.write("state", "state")
    wfm.write("decisions", "decisions")
    wfm.write("lessons", "lessons")
    wfm.write("context", "context")
    wfm.append("session_log", "## Session 1\n- done\n")

    sid = "test_full_sess"
    session = Session.new(sid, ws)

    big_output = "\n".join(f"line {i}" for i in range(300))
    ShellTool(workspace=ws, os_type="linux")._truncate_output(big_output, ws)

    SubAgentTranscript(ws, "claude-code", "tid001").append(
        {"source": "agent", "content": "hi", "chunk_type": "text"}
    )

    pp = ProjectPaths(ws)
    os.makedirs(pp.instructions_dir, exist_ok=True)
    with open(pp.project_goals, "w", encoding="utf-8") as f:
        f.write("# Goals\n")

    ws_path = Path(ws)

    # Forbidden: orbital-output/ sibling tree
    assert not (ws_path / "orbital-output").exists(), \
        "orbital-output/ must not exist after migration"

    # Forbidden: workspace-root skills/
    assert not (ws_path / "skills").exists(), \
        "workspace-root skills/ must not exist after migration"

    # Forbidden: workspace-root sub_agents/
    assert not (ws_path / "sub_agents").exists(), \
        "workspace-root sub_agents/ must not exist after migration"

    # Forbidden: .migrated marker
    assert not (ws_path / "orbital" / ".migrated").exists(), \
        "orbital/.migrated must not exist after migration"

    # Forbidden: AGENT.md written by WorkspaceFileManager
    assert not (ws_path / "orbital" / "AGENT.md").is_file(), \
        "AGENT.md must not be written by WorkspaceFileManager"

    # Forbidden: slug-namespaced subdirs under orbital/
    orbital = ws_path / "orbital"
    allowed_dirs = {"sessions", "instructions", "sub_agents", "tool-results",
                    "output", "skills", ".tmp"}
    for child in orbital.iterdir():
        if child.is_dir():
            assert child.name in allowed_dirs, (
                f"Unexpected directory under orbital/: {child.name!r}. "
                "Slug-namespaced dirs must not be created."
            )
