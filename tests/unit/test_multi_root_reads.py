# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-root read resolution (Spec 12 §2a — the reference lens).

Quick Tasks can read across every in-scope project workspace while writes stay
in its own workspace. Reads resolve against a *list* of roots:

  * the primary root (roots[0], the scratch workspace) behaves exactly like a
    normal single-root workspace — relative paths, no exclusions;
  * secondary roots (other projects) are read-only reference and exclude each
    project's ``orbital/`` runtime dir and ``.git/`` (agent internals, not work
    product);
  * relative paths ALWAYS resolve against the primary root only.

Glob/Grep iterate all roots and label out-of-primary hits with the owning
project so the agent can cite its source.
"""

import os

import pytest

from agent_os.agent.tools._path_utils import resolve_safe, resolve_safe_read


# ---------------------------------------------------------------------------
# resolve_safe_read containment
# ---------------------------------------------------------------------------

def _mkroots(tmp_path):
    primary = tmp_path / "scratch"
    secondary = tmp_path / "clientB"
    (primary).mkdir()
    (secondary).mkdir()
    return str(primary), str(secondary)


def test_absolute_inside_primary(tmp_path):
    primary, secondary = _mkroots(tmp_path)
    f = os.path.join(primary, "note.md")
    open(f, "w").close()
    assert resolve_safe_read([primary, secondary], f) == os.path.realpath(f)


def test_absolute_inside_secondary(tmp_path):
    primary, secondary = _mkroots(tmp_path)
    f = os.path.join(secondary, "deck.md")
    open(f, "w").close()
    assert resolve_safe_read([primary, secondary], f) == os.path.realpath(f)


def test_relative_resolves_against_primary_only(tmp_path):
    primary, secondary = _mkroots(tmp_path)
    open(os.path.join(primary, "note.md"), "w").close()
    # A relative path lands in the primary root...
    assert resolve_safe_read([primary, secondary], "note.md") == os.path.realpath(
        os.path.join(primary, "note.md")
    )
    # ...and a bare name always resolves under the primary, never a secondary
    # (even when the file only exists in the secondary root).
    open(os.path.join(secondary, "deck.md"), "w").close()
    assert resolve_safe_read([primary, secondary], "deck.md") == os.path.realpath(
        os.path.join(primary, "deck.md")
    )


def test_symlink_out_of_secondary_rejected(tmp_path):
    primary, secondary = _mkroots(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("x")
    link = os.path.join(secondary, "escape")
    os.symlink(str(outside), link)
    # Absolute path via the symlink resolves outside every root → rejected.
    assert resolve_safe_read([primary, secondary], os.path.join(link, "secret.txt")) is None


def test_sibling_prefix_rejected(tmp_path):
    primary = tmp_path / "proj"
    sibling = tmp_path / "projB"
    primary.mkdir()
    sibling.mkdir()
    open(os.path.join(str(sibling), "x.txt"), "w").close()
    assert resolve_safe_read([str(primary)], os.path.join(str(sibling), "x.txt")) is None


def test_secondary_orbital_and_git_excluded(tmp_path):
    primary, secondary = _mkroots(tmp_path)
    for sub in ("orbital", ".git"):
        d = os.path.join(secondary, sub)
        os.makedirs(d)
        open(os.path.join(d, "inner.txt"), "w").close()
    assert resolve_safe_read([primary, secondary], os.path.join(secondary, "orbital", "inner.txt")) is None
    assert resolve_safe_read([primary, secondary], os.path.join(secondary, ".git", "inner.txt")) is None


def test_primary_orbital_allowed(tmp_path):
    """The primary (own) workspace's orbital/ is NOT excluded — only others'."""
    primary, secondary = _mkroots(tmp_path)
    d = os.path.join(primary, "orbital")
    os.makedirs(d)
    f = os.path.join(d, "state.md")
    open(f, "w").close()
    assert resolve_safe_read([primary, secondary], f) == os.path.realpath(f)


def test_single_root_matches_resolve_safe(tmp_path):
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    open(os.path.join(ws, "a.txt"), "w").close()
    for p in ("a.txt", os.path.join(ws, "a.txt"), ".", "../etc/passwd"):
        assert resolve_safe_read([ws], p) == resolve_safe(ws, p)


# ---------------------------------------------------------------------------
# ReadTool
# ---------------------------------------------------------------------------

def test_read_tool_default_single_root_unchanged(tmp_path):
    from agent_os.agent.tools.read import ReadTool
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    with open(os.path.join(ws, "hello.txt"), "w") as f:
        f.write("hi there")
    tool = ReadTool(workspace=ws)
    assert "hi there" in tool.execute(path="hello.txt").content
    # outside workspace rejected
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    assert "outside workspace" in tool.execute(path=str(outside)).content


def test_read_tool_multi_root_reads_secondary(tmp_path):
    from agent_os.agent.tools.read import ReadTool
    primary, secondary = _mkroots(tmp_path)
    target = os.path.join(secondary, "ref.md")
    with open(target, "w") as f:
        f.write("previous style")
    tool = ReadTool(workspace=primary, read_roots=lambda: [primary, secondary])
    assert "previous style" in tool.execute(path=target).content
    # its orbital/ is still off-limits
    od = os.path.join(secondary, "orbital")
    os.makedirs(od)
    with open(os.path.join(od, "s.md"), "w") as f:
        f.write("internal")
    assert "outside workspace" in tool.execute(path=os.path.join(od, "s.md")).content


# ---------------------------------------------------------------------------
# GlobTool
# ---------------------------------------------------------------------------

def test_glob_tool_default_single_root_unchanged(tmp_path):
    from agent_os.agent.tools.glob_tool import GlobTool
    ws = str(tmp_path / "ws")
    os.makedirs(os.path.join(ws, "src"))
    open(os.path.join(ws, "src", "main.py"), "w").close()
    out = GlobTool(workspace=ws).execute(pattern="**/*.py").content
    assert out == "src/main.py"


def test_glob_tool_labels_secondary_matches(tmp_path):
    from agent_os.agent.tools.glob_tool import GlobTool
    primary, secondary = _mkroots(tmp_path)
    open(os.path.join(primary, "own.py"), "w").close()
    open(os.path.join(secondary, "ref.py"), "w").close()
    tool = GlobTool(
        workspace=primary,
        read_roots=lambda: [primary, secondary],
        root_labels=lambda: {os.path.realpath(secondary): "ClientB"},
    )
    out = tool.execute(pattern="**/*.py").content
    assert "own.py" in out
    assert "[project: ClientB]" in out
    assert os.path.realpath(os.path.join(secondary, "ref.py")) in out


# ---------------------------------------------------------------------------
# GrepTool
# ---------------------------------------------------------------------------

def test_grep_tool_default_single_root_unchanged(tmp_path):
    from agent_os.agent.tools.grep_tool import GrepTool, find_ripgrep
    if find_ripgrep() is None:
        pytest.skip("ripgrep not available")
    ws = str(tmp_path / "ws")
    os.makedirs(ws)
    with open(os.path.join(ws, "a.md"), "w") as f:
        f.write("needle here\n")
    out = GrepTool(workspace=ws).execute(pattern="needle").content
    assert out.startswith("a.md:1:")


def test_grep_tool_labels_secondary_matches(tmp_path):
    from agent_os.agent.tools.grep_tool import GrepTool, find_ripgrep
    if find_ripgrep() is None:
        pytest.skip("ripgrep not available")
    primary, secondary = _mkroots(tmp_path)
    with open(os.path.join(primary, "own.md"), "w") as f:
        f.write("shared_token in mine\n")
    with open(os.path.join(secondary, "ref.md"), "w") as f:
        f.write("shared_token in clientB\n")
    # a hit inside the secondary's orbital/ must NOT surface
    od = os.path.join(secondary, "orbital")
    os.makedirs(od)
    with open(os.path.join(od, "internal.md"), "w") as f:
        f.write("shared_token internal\n")
    tool = GrepTool(
        workspace=primary,
        read_roots=lambda: [primary, secondary],
        root_labels=lambda: {os.path.realpath(secondary): "ClientB"},
    )
    out = tool.execute(pattern="shared_token").content
    assert "own.md:1:" in out
    assert "[project: ClientB]" in out
    assert os.path.realpath(os.path.join(secondary, "ref.md")) in out
    assert "internal" not in out
