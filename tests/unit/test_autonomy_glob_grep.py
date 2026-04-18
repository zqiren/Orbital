# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests that glob/grep tools auto-approve under ALL autonomy presets."""

from unittest.mock import MagicMock

import pytest

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.autonomy import AutonomyInterceptor


def _make(preset):
    return AutonomyInterceptor(
        preset=preset,
        ws_manager=MagicMock(),
        project_id="test_proj",
    )


@pytest.mark.parametrize("preset", [Autonomy.HANDS_OFF, Autonomy.CHECK_IN, Autonomy.SUPERVISED])
def test_glob_never_intercepted(preset):
    interceptor = _make(preset)
    assert interceptor.should_intercept({
        "id": "tc_1",
        "name": "glob",
        "arguments": {"pattern": "**/*.py"},
    }) is False


@pytest.mark.parametrize("preset", [Autonomy.HANDS_OFF, Autonomy.CHECK_IN, Autonomy.SUPERVISED])
def test_grep_never_intercepted(preset):
    interceptor = _make(preset)
    assert interceptor.should_intercept({
        "id": "tc_1",
        "name": "grep",
        "arguments": {"pattern": "def foo"},
    }) is False


@pytest.mark.parametrize("preset", [Autonomy.HANDS_OFF, Autonomy.CHECK_IN, Autonomy.SUPERVISED])
def test_no_approval_broadcast_for_glob(preset):
    """If should_intercept is False, the loop never calls on_intercept.
    Assert the ws broadcast was never triggered when we follow the protocol."""
    ws = MagicMock()
    interceptor = AutonomyInterceptor(preset=preset, ws_manager=ws, project_id="p1")
    tool_call = {"id": "1", "name": "glob", "arguments": {"pattern": "*.py"}}
    if interceptor.should_intercept(tool_call):
        interceptor.on_intercept(tool_call, [])
    ws.broadcast.assert_not_called()


@pytest.mark.parametrize("preset", [Autonomy.HANDS_OFF, Autonomy.CHECK_IN, Autonomy.SUPERVISED])
def test_no_approval_broadcast_for_grep(preset):
    ws = MagicMock()
    interceptor = AutonomyInterceptor(preset=preset, ws_manager=ws, project_id="p1")
    tool_call = {"id": "1", "name": "grep", "arguments": {"pattern": "x"}}
    if interceptor.should_intercept(tool_call):
        interceptor.on_intercept(tool_call, [])
    ws.broadcast.assert_not_called()


def test_supervised_still_intercepts_shell_write_edit():
    """Sanity check: supervised still gates everything else."""
    interceptor = _make(Autonomy.SUPERVISED)
    assert interceptor.should_intercept({"id": "1", "name": "shell", "arguments": {}}) is True
    assert interceptor.should_intercept({"id": "2", "name": "write", "arguments": {}}) is True
    assert interceptor.should_intercept({"id": "3", "name": "edit", "arguments": {}}) is True
    # Read, glob, grep pass through
    assert interceptor.should_intercept({"id": "4", "name": "read", "arguments": {}}) is False
    assert interceptor.should_intercept({"id": "5", "name": "glob", "arguments": {}}) is False
    assert interceptor.should_intercept({"id": "6", "name": "grep", "arguments": {}}) is False
