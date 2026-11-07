# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Quick Tasks can read the other projects' memory files (2026-11-07).

A Quick Tasks session reads the projects inside its access scope (the scope
chip; "all" = every project) as secondary roots. Their ``orbital/`` stays
excluded except exactly four memory files, so "what are my other projects
doing?" can be answered from each project's goals and state:

    orbital/instructions/project_goals.md, orbital/PROJECT_STATE.md,
    orbital/INDEX.md, orbital/DECISIONS.md

Only in-scope project roots (``memory_roots``) get the allowlist: a granted
folder (portal) and a normal project are unchanged. Everything else under
``orbital/`` (sessions, history, ledger, skills, …) and ``.git/`` stays out,
and grep/glob never walk into the rest of ``orbital/``.
"""

from __future__ import annotations

import os
import subprocess
from unittest.mock import MagicMock

import pytest

from agent_os.agent.tools._path_utils import (
    SECONDARY_MEMORY_FILES,
    resolve_safe_read,
)
from agent_os.agent.tools.glob_tool import GlobTool
from agent_os.agent.tools.grep_tool import GrepTool, find_ripgrep
from agent_os.agent.tools.read import ReadTool
from agent_os.daemon_v2.agent_manager import AgentManager

MEMORY = {
    "orbital/instructions/project_goals.md": "goals: record app issues into BACKLOG.md needle\n",
    "orbital/PROJECT_STATE.md": "state: triaging the sidebar needle\n",
    "orbital/INDEX.md": "index needle\n",
    "orbital/DECISIONS.md": "decisions needle\n",
}
INTERNAL = {
    "orbital/sessions/abc.jsonl": "needle session row\n",
    "orbital/sessions/notes.md": "needle session notes\n",
    "orbital/HISTORY.md": "needle history\n",
    "orbital/skills/x/SKILL.md": "needle skill\n",
    "orbital/instructions/user_directives.md": "needle directives\n",
    ".git/config": "needle git\n",
}


def _write(root, rel, text):
    path = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _project_tree(root):
    os.makedirs(root, exist_ok=True)
    for rel, text in {**MEMORY, **INTERNAL}.items():
        _write(root, rel, text)
    _write(root, "docs/plan.md", "needle work file\n")
    return root


@pytest.fixture
def roots(tmp_path):
    primary = str(tmp_path / "scratch")
    os.makedirs(primary)
    other = _project_tree(str(tmp_path / "backlog"))
    return primary, other


def _abs(root, rel):
    return os.path.join(root, *rel.split("/"))


# ---------------------------------------------------------------------------
# resolve_safe_read
# ---------------------------------------------------------------------------

def test_allowlist_is_exactly_the_four_memory_files():
    assert set(SECONDARY_MEMORY_FILES) == set(MEMORY)


def test_memory_files_readable_in_a_memory_root(roots):
    primary, other = roots
    for rel in MEMORY:
        got = resolve_safe_read([primary, other], _abs(other, rel), memory_roots=[other])
        assert got == os.path.realpath(_abs(other, rel)), rel


def test_other_orbital_and_git_files_stay_refused(roots):
    primary, other = roots
    for rel in INTERNAL:
        assert resolve_safe_read([primary, other], _abs(other, rel),
                                 memory_roots=[other]) is None, rel
    # The orbital/ directory itself (a listing) is not on the allowlist.
    assert resolve_safe_read([primary, other], _abs(other, "orbital"),
                             memory_roots=[other]) is None
    assert resolve_safe_read([primary, other], _abs(other, "orbital/instructions"),
                             memory_roots=[other]) is None


def test_memory_files_refused_without_memory_root(roots):
    """A secondary root that is not an in-scope project (a granted folder)
    keeps today's rule: nothing under orbital/."""
    primary, other = roots
    for rel in MEMORY:
        assert resolve_safe_read([primary, other], _abs(other, rel)) is None
        assert resolve_safe_read([primary, other], _abs(other, rel), memory_roots=[]) is None


def test_case_variant_of_orbital_is_still_excluded(roots):
    """On a case-insensitive disk ORBITAL/ is orbital/: never a way around."""
    primary, other = roots
    variant = os.path.join(other, "ORBITAL", "sessions", "abc.jsonl")
    assert resolve_safe_read([primary, other], variant, memory_roots=[other]) is None


def test_symlinked_memory_file_into_internals_is_refused(tmp_path):
    primary = str(tmp_path / "scratch")
    os.makedirs(primary)
    other = str(tmp_path / "proj")
    secret = _write(other, "orbital/sessions/secret.jsonl", "secret\n")
    os.makedirs(os.path.join(other, "orbital", "instructions"), exist_ok=True)
    # PROJECT_STATE.md is a symlink to a session log: the real file is not
    # at an allowlisted location, so it stays refused.
    os.symlink(secret, os.path.join(other, "orbital", "PROJECT_STATE.md"))
    outside = _write(str(tmp_path / "outside"), "x.md", "outside\n")
    os.symlink(outside, os.path.join(other, "orbital", "INDEX.md"))
    assert resolve_safe_read([primary, other], _abs(other, "orbital/PROJECT_STATE.md"),
                             memory_roots=[other]) is None
    assert resolve_safe_read([primary, other], _abs(other, "orbital/INDEX.md"),
                             memory_roots=[other]) is None


def test_symlinked_orbital_dir_escape_is_refused(tmp_path):
    primary = str(tmp_path / "scratch")
    os.makedirs(primary)
    other = str(tmp_path / "proj")
    os.makedirs(other)
    elsewhere = _project_tree(str(tmp_path / "elsewhere"))
    os.symlink(os.path.join(elsewhere, "orbital"), os.path.join(other, "orbital"))
    assert resolve_safe_read([primary, other], _abs(other, "orbital/PROJECT_STATE.md"),
                             memory_roots=[other]) is None


def test_sibling_prefix_root_does_not_grant_memory(tmp_path):
    """``/x/proj`` as a memory root must not cover ``/x/proj-b``."""
    primary = str(tmp_path / "scratch")
    os.makedirs(primary)
    a = _project_tree(str(tmp_path / "proj"))
    b = _project_tree(str(tmp_path / "proj-b"))
    path = _abs(b, "orbital/PROJECT_STATE.md")
    assert resolve_safe_read([primary, a, b], path, memory_roots=[a]) is None
    assert resolve_safe_read([primary, a, b], _abs(a, "orbital/PROJECT_STATE.md"),
                             memory_roots=[a]) is not None


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------

def test_read_tool_reads_memory_of_in_scope_project(roots):
    primary, other = roots
    tool = ReadTool(workspace=primary, read_roots=lambda: [primary, other],
                    memory_roots=lambda: [other])
    out = tool.execute(path=_abs(other, "orbital/instructions/project_goals.md")).content
    assert "record app issues into BACKLOG.md" in out
    out = tool.execute(path=_abs(other, "orbital/sessions/abc.jsonl")).content
    assert "outside workspace" in out


def test_read_tool_without_memory_roots_is_unchanged(roots):
    primary, other = roots
    tool = ReadTool(workspace=primary, read_roots=lambda: [primary, other])
    out = tool.execute(path=_abs(other, "orbital/PROJECT_STATE.md")).content
    assert "outside workspace" in out


def test_cross_project_read_description_names_the_memory_files(roots):
    primary, other = roots
    tool = ReadTool(workspace=primary, read_roots=lambda: [primary, other],
                    memory_roots=lambda: [other], cross_project=True)
    desc = tool.parameters["properties"]["path"]["description"]
    for rel in MEMORY:
        assert rel in desc


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------

def _glob(primary, other, **kw):
    return GlobTool(workspace=primary, read_roots=lambda: [primary, other],
                    root_labels=lambda: {os.path.realpath(other): "Backlog"},
                    memory_roots=lambda: [other]).execute(**kw).content


def test_glob_lists_only_memory_files_from_orbital(roots):
    primary, other = roots
    out = _glob(primary, other, pattern="**/*.md")
    real = os.path.realpath(other)
    for rel in MEMORY:
        assert _abs(real, rel) in out, rel
    assert _abs(real, "docs/plan.md") in out
    for rel in INTERNAL:
        assert _abs(real, rel) not in out, rel


def test_glob_explicit_patterns_reach_memory_files(roots):
    primary, other = roots
    real = os.path.realpath(other)
    out = _glob(primary, other, pattern="orbital/*.md", path=other)
    assert _abs(real, "orbital/PROJECT_STATE.md") in out
    assert _abs(real, "orbital/INDEX.md") in out
    assert _abs(real, "orbital/HISTORY.md") not in out
    assert "project_goals.md" not in out          # one level deeper
    out = _glob(primary, other, pattern="*")
    assert _abs(real, "docs") in out
    assert _abs(real, "orbital") not in out       # the dir itself is not listed


def test_glob_never_walks_into_the_rest_of_orbital(roots, monkeypatch):
    primary, other = roots
    seen: list[str] = []
    real_scandir = os.scandir

    def spy(path="."):
        seen.append(os.path.realpath(os.fspath(path)))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", spy)
    _glob(primary, other, pattern="**/*")
    real = os.path.realpath(other)
    walked_internal = [p for p in seen
                       if p.startswith(os.path.join(real, "orbital"))
                       or p.startswith(os.path.join(real, ".git"))]
    assert walked_internal == []


def test_glob_without_memory_roots_is_unchanged(roots):
    primary, other = roots
    out = GlobTool(workspace=primary, read_roots=lambda: [primary, other]).execute(
        pattern="**/*.md").content
    assert "PROJECT_STATE.md" not in out and "plan.md" in out


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------

needs_rg = pytest.mark.skipif(find_ripgrep() is None, reason="ripgrep not available")


def _grep(primary, other, **kw):
    return GrepTool(workspace=primary, read_roots=lambda: [primary, other],
                    root_labels=lambda: {os.path.realpath(other): "Backlog"},
                    memory_roots=lambda: [other]).execute(**kw).content


@needs_rg
def test_grep_hits_only_memory_files_from_orbital(roots):
    primary, other = roots
    out = _grep(primary, other, pattern="needle")
    real = os.path.realpath(other)
    assert "[project: Backlog]" in out
    for rel in MEMORY:
        assert f"{_abs(real, rel)}:1:" in out, rel
    assert f"{_abs(real, 'docs/plan.md')}:1:" in out
    for rel in INTERNAL:
        assert _abs(real, rel) not in out, rel


@needs_rg
def test_grep_glob_filter_applies_to_memory_files(roots):
    primary, other = roots
    out = _grep(primary, other, pattern="needle", glob_filter="*.py")
    assert out == "No matches found"
    out = _grep(primary, other, pattern="needle", glob_filter="PROJECT_STATE.md")
    assert "PROJECT_STATE.md:1:" in out and "INDEX.md" not in out


@needs_rg
def test_grep_on_one_memory_file(roots):
    primary, other = roots
    target = _abs(other, "orbital/PROJECT_STATE.md")
    out = _grep(primary, other, pattern="sidebar", path=target)
    assert f"{os.path.realpath(target)}:1:" in out
    out = _grep(primary, other, pattern="needle", path=_abs(other, "orbital/HISTORY.md"))
    assert "outside workspace" in out


@needs_rg
def test_grep_never_targets_the_rest_of_orbital(roots, monkeypatch):
    """rg walks the root with orbital/ and .git/ pruned; the memory files
    are passed by name. No rg call is aimed at orbital/ as a directory."""
    primary, other = roots
    calls: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd, *a, **kw):
        calls.append(list(cmd))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", spy)
    _grep(primary, other, pattern="needle")
    real = os.path.realpath(other)
    allowed = {os.path.realpath(_abs(other, rel)) for rel in MEMORY}
    for cmd in calls:
        targets = cmd[cmd.index("--") + 2:]
        for t in targets:
            t_real = os.path.realpath(t)
            if t_real.startswith(os.path.join(real, "orbital")):
                assert t_real in allowed, t
        if real in [os.path.realpath(t) for t in targets]:
            assert "!orbital" in cmd and "!.git" in cmd


@needs_rg
def test_grep_without_memory_roots_is_unchanged(roots):
    primary, other = roots
    out = GrepTool(workspace=primary, read_roots=lambda: [primary, other]).execute(
        pattern="needle").content
    assert "PROJECT_STATE.md" not in out and "plan.md" in out


# ---------------------------------------------------------------------------
# Wiring: which roots are memory roots
# ---------------------------------------------------------------------------

def _store(projects):
    store = MagicMock()
    by_id = {p["project_id"]: p for p in projects}
    store.get_project.side_effect = lambda pid: by_id.get(pid)
    store.list_projects.side_effect = lambda: list(projects)
    return store


def _manager(projects):
    return AgentManager(
        project_store=_store(projects), ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(), activity_translator=MagicMock(),
        process_manager=MagicMock(), platform_provider=None,
    )


def _proj(tmp_path, pid, **extra):
    ws = str(tmp_path / pid)
    os.makedirs(ws, exist_ok=True)
    return {"project_id": pid, "name": pid, "workspace": ws, "is_scratch": False,
            "safe_mode": True, **extra}


def test_memory_roots_follow_the_scope_chip(tmp_path):
    scratch = _proj(tmp_path, "scratch", is_scratch=True)
    a, b = _proj(tmp_path, "p_a"), _proj(tmp_path, "p_b")
    mgr = _manager([scratch, a, b])
    assert sorted(mgr._scope_memory_roots("scratch", "s1")) == sorted(
        [a["workspace"], b["workspace"]])
    mgr.set_session_scope("scratch", "s1", "selected", ["p_a"])
    assert mgr._scope_memory_roots("scratch", "s1") == [a["workspace"]]
    mgr.set_session_scope("scratch", "s1", "off", [])
    assert mgr._scope_memory_roots("scratch", "s1") == []


def test_normal_project_has_no_memory_roots(tmp_path):
    scratch = _proj(tmp_path, "scratch", is_scratch=True)
    a, b = _proj(tmp_path, "p_a"), _proj(tmp_path, "p_b")
    mgr = _manager([scratch, a, b])
    assert mgr._scope_memory_roots("p_a", "s1") == []


def test_chip_on_one_project_refuses_the_other_projects_goals(tmp_path):
    """End to end through the manager's callables: chip on p_a → p_b's goals
    are refused, p_a's are read."""
    scratch = _proj(tmp_path, "scratch", is_scratch=True)
    a = _proj(tmp_path, "p_a")
    b = _proj(tmp_path, "p_b")
    _project_tree(a["workspace"])
    _project_tree(b["workspace"])
    mgr = _manager([scratch, a, b])
    mgr.set_session_scope("scratch", "s1", "selected", ["p_a"])
    tool = ReadTool(
        workspace=scratch["workspace"],
        read_roots=lambda: mgr._compute_scope_roots("scratch", "s1", include_portals=True)[0],
        memory_roots=lambda: mgr._scope_memory_roots("scratch", "s1"),
        cross_project=True,
    )
    goals = "orbital/instructions/project_goals.md"
    assert "record app issues" in tool.execute(path=_abs(a["workspace"], goals)).content
    assert "outside workspace" in tool.execute(path=_abs(b["workspace"], goals)).content


@pytest.mark.parametrize("pattern", [
    "*", "*.md", "**/*.md", "**", "docs/*", "**/plan.md", "d?cs/**/*.md",
    "[dx]*/*.md", ".*", "**/.hidden/*", "src/**", "nope/*.md",
])
def test_pruned_glob_matches_pathlib_outside_orbital(tmp_path, pattern):
    """Over a tree with no orbital/ or .git/, the secondary-root walk finds
    exactly what Path.glob finds."""
    from pathlib import Path
    from agent_os.agent.tools.glob_tool import _pattern_segments, _secondary_glob
    root = str(tmp_path / "proj")
    for rel in ("docs/plan.md", "docs/deep/a.md", "src/x.py", "src/y/z.md",
                "top.md", ".hidden/h.md", ".dot"):
        _write(root, rel, "x\n")
    real = os.path.realpath(root)
    expected = {os.path.realpath(str(p)) for p in Path(real).glob(pattern)}
    got = {os.path.realpath(p) for p in _secondary_glob(
        real, real, _pattern_segments(pattern), memory_ok=True)}
    assert got == expected
