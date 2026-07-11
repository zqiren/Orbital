# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""ShellTool appends a [network-policy] hint when output shows a proxy block."""

from types import SimpleNamespace

from agent_os.agent.tools.shell import ShellTool
from agent_os.platform.types import CommandResult


class _StubProvider:
    def __init__(self, stderr: str):
        self._stderr = stderr

    def get_capabilities(self):
        return SimpleNamespace(setup_complete=True)

    async def run_command(self, **kwargs):
        return CommandResult(exit_code=56, stdout="", stderr=self._stderr)


def _make_tool(tmp_path, stderr):
    return ShellTool(
        workspace=str(tmp_path),
        os_type="darwin",
        platform_provider=_StubProvider(stderr),
        project_id="p1",
    )


def test_hint_appended_on_proxy_403(tmp_path):
    tool = _make_tool(
        tmp_path,
        "curl: (56) Received HTTP code 403 from proxy after CONNECT",
    )
    result = tool.execute(command="curl https://x.com/somepage")
    assert "[network-policy]" in result.content
    assert "browser tool" in result.content


def test_hint_appended_on_policy_body(tmp_path):
    tool = _make_tool(
        tmp_path,
        "Blocked by Orbital network policy: 'x.com' is not on this project's allowlist.",
    )
    result = tool.execute(command="curl http://x.com/")
    assert "[network-policy]" in result.content


def test_no_hint_without_block_marker(tmp_path):
    tool = _make_tool(tmp_path, "curl: (6) Could not resolve host: nosuch.example")
    result = tool.execute(command="curl https://nosuch.example/")
    assert "[network-policy]" not in result.content


def test_no_hint_for_non_network_command(tmp_path):
    tool = _make_tool(tmp_path, "Received HTTP code 403 from proxy after CONNECT")
    result = tool.execute(command="echo hello")
    assert "[network-policy]" not in result.content
