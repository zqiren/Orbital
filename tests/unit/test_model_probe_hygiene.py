# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Live model-list probes: truthful Claude labels, and no leaked processes.

Found on the installed v0.16.0 for Windows (2026-10-11):

* **Labels.** Claude Code 2.1.235 and 2.1.293 answer ``initialize`` with
  unversioned ``displayName``s ("Opus", "Sonnet") while the version lives in
  ``description`` ("Opus 5.5 · Best for ...", "Opus 5 with 1M context · ...").
  On an old CLI the "Opus" choice really runs Opus 5, so a bare "Opus" in the
  dropdown hid which model a dispatch would use.
* **Leaks.** On Windows ``claude``/``codex`` resolve to npm's ``.CMD`` shims,
  so ``proc.kill()`` ended only ``cmd.exe``; the real ``claude.exe`` /
  ``node`` + ``codex.exe`` outlived every settings-page probe (two of each
  were found orphaned after two settings loads).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import psutil
import pytest

from agent_os.agent.transports import claude_models, codex_models, codex_transport


class _Writer:
    def write(self, data: bytes) -> None:
        pass

    async def drain(self) -> None:
        return None


def _init_response(models: list[dict]) -> dict:
    return {"type": "control_response", "response": {
        "subtype": "success", "request_id": "orbital-models",
        "response": {"models": models}}}


def _labels(models: list[dict]) -> dict[str, str]:
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data((json.dumps(_init_response(models)) + "\n").encode())
        return await claude_models.read_models(reader, _Writer(), timeout=2)
    return {m["value"]: m["label"] for m in asyncio.run(run())}


# Recorded from Claude Code 2.1.293 (logged out), fields trimmed.
CLI_2_1_293 = [
    {"value": "default", "resolvedModel": "claude-opus-5-5",
     "displayName": "Default (recommended)",
     "description": "Opus 5.5 · Best for everyday, complex tasks"},
    {"value": "opus", "resolvedModel": "claude-opus-5-5", "displayName": "Opus",
     "description": "Opus 5.5 · Best for everyday, complex tasks"},
    {"value": "fable[1m]", "resolvedModel": "claude-fable-5-1", "displayName": "Fable",
     "description": "Fable 5.1 · Most capable for your hardest and longest-running tasks"},
    {"value": "sonnet", "resolvedModel": "claude-sonnet-5-5", "displayName": "Sonnet",
     "description": "Sonnet 5.5 · Efficient for routine tasks"},
    {"value": "haiku", "resolvedModel": "claude-haiku-5-5", "displayName": "Haiku",
     "description": "Haiku 5.5 · Fastest for quick answers"},
]

# Recorded from Claude Code 2.1.235 (npm, the Windows box that found this).
CLI_2_1_235 = [
    {"value": "opus[1m]", "resolvedModel": "claude-opus-5[1m]",
     "displayName": "Opus (1M context)",
     "description": "Opus 5 with 1M context · Best for everyday, complex tasks"},
    {"value": "haiku", "resolvedModel": "claude-haiku-4-5-20251001", "displayName": "Haiku",
     "description": "Haiku 4.5 · Fastest for quick answers"},
]


class TestVersionedLabels:

    def test_takes_the_version_from_the_description(self):
        assert _labels(CLI_2_1_293) == {
            "opus": "Opus 5.5", "fable[1m]": "Fable 5.1",
            "sonnet": "Sonnet 5.5", "haiku": "Haiku 5.5",
        }

    def test_an_old_cli_shows_the_older_model_it_really_runs(self):
        assert _labels(CLI_2_1_235) == {
            "opus[1m]": "Opus 5 with 1M context", "haiku": "Haiku 4.5",
        }

    def test_a_display_name_that_already_has_a_version_is_kept(self):
        assert _labels([{"value": "opus", "displayName": "Opus 5.5",
                         "description": "Opus 5.5 · Best for everyday tasks"}]) == {
            "opus": "Opus 5.5"}

    @pytest.mark.parametrize("description", [
        None, "", "Best for everyday tasks",          # no version anywhere
        "Sonnet 5.5 · Efficient",                     # a different family
        "Opus 5.5 " + "x" * 60 + " · too long",       # not a label-sized head
    ])
    def test_falls_back_to_the_display_name(self, description):
        entry = {"value": "opus", "displayName": "Opus"}
        if description is not None:
            entry["description"] = description
        assert _labels([entry]) == {"opus": "Opus"}

    def test_no_display_name_falls_back_to_the_value(self):
        assert _labels([{"value": "claude-opus-5"}]) == {"claude-opus-5": "claude-opus-5"}


# ---------------------------------------------------------------------------
# Leaks: a real wrapper -> grandchild tree, like npm's claude.CMD -> claude.exe
# ---------------------------------------------------------------------------


def _wrapper_with_grandchild(tmp_path) -> tuple[str, str]:
    """A CLI stand-in that, like npm's shim, is a wrapper whose real work runs
    in a child process. The child records its pid and never answers."""
    pidfile = tmp_path / "grandchild.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, sys, time\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n")
    if sys.platform == "win32":
        wrapper = tmp_path / "fake-cli.cmd"
        wrapper.write_text(f'@"{sys.executable}" "{child}" "{pidfile}"\r\n')
    else:
        wrapper = tmp_path / "fake-cli"
        # No `exec`: the shell stays the parent, as cmd.exe does for a .CMD.
        wrapper.write_text(f'#!/bin/sh\n"{sys.executable}" "{child}" "{pidfile}"\n')
        wrapper.chmod(0o755)
    return str(wrapper), str(pidfile)


def _grandchild_pid(pidfile: str) -> int:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if os.path.exists(pidfile) and open(pidfile).read().strip():
            return int(open(pidfile).read())
        time.sleep(0.05)
    raise AssertionError("the fake CLI never started its child")


def _assert_gone(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.05)
    try:
        psutil.Process(pid).kill()
    finally:
        pytest.fail(f"probe left its CLI's child pid={pid} running")


@pytest.mark.parametrize("fetch", [
    lambda b: claude_models.fetch_claude_models(b, timeout=3),
    lambda b: codex_models.fetch_codex_models(b, timeout=3),
    lambda b: codex_transport.fetch_codex_rate_limits(b, timeout=3),
], ids=["claude", "codex", "codex-usage"])
def test_a_timed_out_probe_kills_the_whole_cli_tree(tmp_path, fetch):
    wrapper, pidfile = _wrapper_with_grandchild(tmp_path)

    async def run():
        task = asyncio.ensure_future(fetch(wrapper))
        pid = await asyncio.to_thread(_grandchild_pid, pidfile)
        return await task, pid

    result, pid = asyncio.run(run())
    assert result is None
    _assert_gone(pid)
