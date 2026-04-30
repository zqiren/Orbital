# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the shared `resolve_safe` path helper.

These cases pin down behaviour that was previously broken across the five
path-taking tools (read/write/edit/glob/grep), which each carried a private
`_resolve_safe` doing `path.lstrip("/")` + `os.path.join`. That pattern doubled
the workspace prefix for absolute paths pointing inside the workspace
(silently corrupting writes/reads) and accepted sibling-prefix directories as
"inside" because the realpath check used a bare `startswith` without the
`os.sep` guard. The shared helper at `agent_os.agent.tools._path_utils`
must:

  * Pass through absolute paths that resolve inside the workspace as-is.
  * Reject absolute paths that resolve outside the workspace.
  * Resolve relative paths against the workspace, including `.`.
  * Reject `..`-traversal escapes that leave the workspace.
  * Reject sibling-prefix directories like `<ws>-sibling` that share a string
    prefix with the workspace but are not inside it.
"""

import os

from agent_os.agent.tools._path_utils import resolve_safe


def test_relative_inside(tmp_path):
    # "subdir/file.md" → workspace/subdir/file.md
    expected = os.path.realpath(os.path.join(str(tmp_path), "subdir/file.md"))
    assert resolve_safe(str(tmp_path), "subdir/file.md") == expected


def test_absolute_inside_no_doubling(tmp_path):
    # MUST NOT double the path
    abs_path = os.path.realpath(os.path.join(str(tmp_path), "subdir/file.md"))
    resolved = resolve_safe(str(tmp_path), abs_path)
    assert resolved == abs_path
    # Negative: result must NOT contain the workspace path twice.
    # Strip the leading "/" so the count works on POSIX absolute paths
    # (the workspace string itself starts with "/", so it would count once
    # against itself otherwise).
    needle = str(tmp_path).lstrip("/")
    assert resolved.count(needle) == 1


def test_absolute_outside_rejected(tmp_path):
    assert resolve_safe(str(tmp_path), "/etc/passwd") is None


def test_traversal_rejected(tmp_path):
    assert resolve_safe(str(tmp_path), "../../etc/passwd") is None


def test_workspace_root(tmp_path):
    # "." resolves to workspace root
    assert resolve_safe(str(tmp_path), ".") == os.path.realpath(str(tmp_path))


def test_sibling_prefix_not_inside(tmp_path):
    # /home/user-foo must not be considered inside /home/user
    sibling = str(tmp_path) + "-sibling"
    os.makedirs(sibling, exist_ok=True)
    assert resolve_safe(str(tmp_path), sibling) is None
