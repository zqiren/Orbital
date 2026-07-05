# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

# Spec 12 awareness plane: when a tool is constructed multi-root, its
# LLM-facing text must say cross-project reads are allowed via absolute
# paths. When single-root (normal projects), the text must stay
# byte-identical to the pre-change strings.
from agent_os.agent.tools.read import ReadTool
from agent_os.agent.tools.glob_tool import GlobTool
from agent_os.agent.tools.grep_tool import GrepTool

ROOTS = lambda: ["/tmp/a", "/tmp/b"]


def test_read_single_root_text_unchanged():
    t = ReadTool(workspace="/tmp/a")
    assert t.description == "Read a file or list a directory within the workspace."
    assert t.parameters["properties"]["path"]["description"] == (
        "Path within your workspace. Use a relative path like 'src/main.py' "
        "or 'docs/notes.md'. Do NOT start with '/' and do NOT pass an absolute path."
    )


def test_read_multi_root_text_mentions_absolute_cross_project():
    t = ReadTool(workspace="/tmp/a", read_roots=ROOTS)
    assert "in-scope project" in t.description
    path_desc = t.parameters["properties"]["path"]["description"]
    assert "ABSOLUTE path" in path_desc
    assert "Do NOT pass an absolute path" not in path_desc


def test_glob_and_grep_multi_root_param_text():
    # NOTE: GlobTool's directory-ish parameter is named "path" in the actual
    # JSON schema (confirmed against agent_os/agent/tools/glob_tool.py), not
    # "dir" as the task brief's draft test assumed. Renaming the schema key
    # would change what argument name the LLM must pass while execute() still
    # reads arguments.get("path", ...) — that's a real behavior change to the
    # tool-calling contract, which is out of scope here. Using the file's
    # actual key ("path") keeps this a pure text-content change.
    g = GlobTool(workspace="/tmp/a", read_roots=ROOTS, root_labels=lambda: {})
    r = GrepTool(workspace="/tmp/a", read_roots=ROOTS, root_labels=lambda: {})
    assert "[project:" in g.parameters["properties"]["path"]["description"]
    assert "[project:" in r.parameters["properties"]["path"]["description"]


def test_glob_and_grep_single_root_param_text_unchanged():
    g = GlobTool(workspace="/tmp/a")
    r = GrepTool(workspace="/tmp/a")
    assert g.parameters["properties"]["path"]["description"] == (
        "Directory within your workspace, relative to workspace root "
        "(e.g. 'src' or 'docs/notes'). Defaults to workspace root. Do NOT start with '/'."
    )
    assert r.parameters["properties"]["path"]["description"] == (
        "Directory or file within your workspace, relative to workspace root "
        "(e.g. 'src' or 'docs/notes.md'). Defaults to workspace root. Do NOT start with '/'."
    )
