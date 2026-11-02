# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which child of the worker is its background shell.

Claude Code starts a shell-snapshot bootstrap (``bash -lc echo "$PATH"``,
``bash -c env``) the first time a worker process uses its Bash tool, and those
children appear before the real background shell. Adopting the first new child
tracked the bootstrap, which exits within a second, so the worker read idle for
the whole life of its shell (Windows, v0.20.0 RC, every fresh worker).
"""

from __future__ import annotations

import asyncio

from agent_os.daemon_v2.background_work import BackgroundWorkRegistry

BOOTSTRAP = [r"C:\Git\bin\bash.exe", "-lc", 'echo "$PATH"']
SNAPSHOT_WRITER = [r"C:\Git\bin\bash.exe", "-c", "-l",
                   r"SNAPSHOT_FILE='C:\Users\me\.claude\shell-snapshots\snapshot-bash-1'"]
LAUNCHER = [r"C:\Git\bin\bash.exe", "-c",
            "source /c/Users/me/.claude/shell-snapshots/snapshot-bash-1.sh && ping -n 60 x"]
POWERSHELL = ["powershell.exe", "-NoProfile", "-Command", "ping -n 60 127.0.0.1"]


class _Child:
    def __init__(self, pid, cmdline, born, dies=None):
        self.pid, self._cmdline, self.born, self.dies = pid, cmdline, born, dies
        self.root = None

    def _alive(self):
        return self.dies is None or self.root.elapsed < self.dies

    def cmdline(self):
        return self._cmdline

    def is_running(self):
        return self._alive()


class _Root:
    """The worker process. Time advances one poll at a time, so a child's
    birth and death are stated in polls rather than in real seconds."""

    def __init__(self, *children):
        self._all = list(children)
        for child in self._all:
            child.root = self
        self.elapsed = -1

    def children(self):
        self.elapsed += 1
        return [c for c in self._all if c.born <= self.elapsed and c._alive()]


async def _capture(root):
    registry = BackgroundWorkRegistry(capture_window_s=0.1, capture_poll_s=0.005)
    rec = registry.register("p", "s", "claude-code", command="ping", root_proc=root)
    await rec.capture_task
    return rec


def test_the_bootstrap_shell_is_not_the_background_shell():
    bootstrap = _Child(1, BOOTSTRAP, born=1, dies=4)
    writer = _Child(2, SNAPSHOT_WRITER, born=3, dies=7)
    launcher = _Child(3, LAUNCHER, born=6)
    rec = asyncio.run(_capture(_Root(bootstrap, writer, launcher)))
    assert rec.anchor is launcher
    assert rec.is_live()


def test_a_shell_without_the_snapshot_launcher_is_adopted_if_it_lasts():
    # The PowerShell tool has no snapshot launcher to recognise.
    shell = _Child(1, POWERSHELL, born=1)
    rec = asyncio.run(_capture(_Root(shell)))
    assert rec.anchor is shell


def test_a_child_that_lasted_is_preferred_over_one_that_already_exited():
    bootstrap = _Child(1, BOOTSTRAP, born=1, dies=3)
    shell = _Child(2, POWERSHELL, born=2)
    rec = asyncio.run(_capture(_Root(bootstrap, shell)))
    assert rec.anchor is shell


def test_nothing_is_adopted_when_every_new_child_has_exited():
    bootstrap = _Child(1, BOOTSTRAP, born=1, dies=3)
    rec = asyncio.run(_capture(_Root(bootstrap)))
    assert rec.anchor is None
    assert not rec.is_live()


def test_a_baseline_child_is_never_adopted():
    # Present at registration: warm infrastructure such as an MCP server.
    mcp = _Child(1, ["cmd.exe", "/c", "npx", "server"], born=0)
    rec = asyncio.run(_capture(_Root(mcp)))
    assert rec.anchor is None
