# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for the path-contract tightening (TASK-tighten-path-contract).

Two surfaces teach the model how to spell paths:

  * The `path` parameter description on each path-taking tool
    (read/write/edit/glob/grep).
  * The safety block in `PromptBuilder._safety()`.

After TASK-fix-path-resolution made absolute-path mishandling non-fatal, this
task removes the *ambiguity* that produced those bug reports in the first
place. The tool descriptions and the safety block must explicitly forbid
leading slashes / absolute paths so every model receives the same rule.
"""


def test_tool_descriptions_forbid_leading_slash():
    """All path-taking tools must explicitly forbid leading slash in path param."""
    from agent_os.agent.tools.read import ReadTool
    from agent_os.agent.tools.write import WriteTool
    from agent_os.agent.tools.edit import EditTool
    from agent_os.agent.tools.glob_tool import GlobTool
    from agent_os.agent.tools.grep_tool import GrepTool

    for tool_cls in [ReadTool, WriteTool, EditTool, GlobTool, GrepTool]:
        tool = tool_cls("/tmp/test")
        path_desc = tool.parameters["properties"]["path"]["description"]
        assert "Do NOT start with '/'" in path_desc or "do NOT start with '/'" in path_desc, \
            f"{tool_cls.__name__} path description must forbid leading slash. Got: {path_desc}"


def test_safety_block_states_path_convention():
    """The safety block must explicitly tell the model not to pass absolute paths."""
    from agent_os.agent.prompt_builder import PromptBuilder, PromptContext, Autonomy
    builder = PromptBuilder(workspace="/tmp/ws")
    ctx = PromptContext(
        workspace="/tmp/ws", model="x", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=[], tool_names=["read", "write"], os_type="linux",
        datetime_now="2026-01-01T00:00:00", context_usage_pct=0.0,
        project_name="t", is_scratch=False,
    )
    safety = builder._safety(ctx)
    assert "PATH CONVENTION" in safety
    assert "Do NOT pass absolute paths" in safety or "do NOT pass absolute paths" in safety
    assert "Your workspace is" in safety  # must still orient the model
