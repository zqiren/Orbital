# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 139 — the prompt's front stays byte-stable; what changes rides the tail.

Two measured cache leaks on the manager (DeepSeek via OpenCode Go):

A. ``ContextManager.prepare()`` pasted ASKS.md's open + recently dropped asks
   into the per-call runtime block on every call. That block moves off the
   prefix on the next call, so those ~1.5–2k tokens never hit the cache. Asks
   are now best-effort: the prompt says how to use ASKS.md and the agent reads
   it itself when it needs to.
B. ``_sub_agent_awareness`` lived in the system region and appeared,
   disappeared and changed state at dispatch / report-back — exactly the run
   boundary where the manager wakes — so the first call after a wake fell to
   the system-region plateau. The coordination rules are now gated on the
   project's enabled agents (fixed for the session) and the live state is one
   line in the tail.

A0 tests below drive a real ``PromptBuilder`` through ``prepare()`` and diff
the wire bytes between calls.
"""

from __future__ import annotations

import re

import pytest

from agent_os.agent import asks
from agent_os.agent.context import _RUNTIME_MARKER, ContextManager
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.agent.session import Session
from agent_os.agent.workspace_files import WorkspaceFileManager

ENABLED = [{"handle": "claude-code", "display_name": "Claude Code", "type": "cli"}]


def _ctx(workspace: str, **over) -> PromptContext:
    base = dict(
        workspace=workspace, model="deepseek-v4.1-flash", autonomy=Autonomy.HANDS_OFF,
        enabled_agents=ENABLED, tool_names=["read", "write", "agent_message"],
        os_type="macos", datetime_now="2026-10-27T10:00", context_usage_pct=0.0,
        project_name="p", project_id="p1",
    )
    base.update(over)
    return PromptContext(**base)


def _manager(tmp_path, provider=None, **ctx_over):
    ws = str(tmp_path)
    wfm = WorkspaceFileManager(ws)
    wfm.ensure_dir()
    session = Session.new("cache-tail-139", ws)
    session.append({"role": "user", "content": "hello", "source": "user"})
    session.append({"role": "assistant", "content": "hi — what next?", "source": "management"})
    session.append({"role": "user", "content": "carry on", "source": "user"})
    cm = ContextManager(
        session, PromptBuilder(ws), _ctx(ws, **ctx_over),
        workspace_files=wfm, sub_agent_provider=provider,
    )
    return cm


def _split(prepared: list[dict]):
    """(everything before the final message, the final message)."""
    return prepared[:-1], prepared[-1]


def _system_rows(prepared: list[dict]) -> list[str]:
    return [m["content"] for m in prepared if m.get("role") == "system"]


_TIME_LINE = re.compile(r"^Current time: .*$", re.M)


def _no_clock(prepared: list[dict]) -> list[dict]:
    """``prepare()`` stamps the wall clock (minute granularity) into the tail;
    drop that line so two calls can be compared across a minute boundary."""
    out = [dict(m) for m in prepared]
    if isinstance(out[-1].get("content"), str):
        out[-1]["content"] = _TIME_LINE.sub("Current time: -", out[-1]["content"])
    return out


def _runtime_part(tail: dict) -> str:
    content = tail["content"]
    assert isinstance(content, str) and _RUNTIME_MARKER in content
    return content.split(_RUNTIME_MARKER, 1)[1]


# ---------------------------------------------------------------------------
# A0 (a): the sub-agent state no longer moves the system region
# ---------------------------------------------------------------------------

class TestSubAgentStateStaysOutOfTheSystemRegion:
    def _three_calls(self, tmp_path):
        state = {"subs": []}
        cm = _manager(tmp_path, provider=lambda: list(state["subs"]))
        calls = [cm.prepare()]
        state["subs"] = [{"handle": "claude-code", "status": "running"}]
        calls.append(cm.prepare())
        state["subs"] = [{"handle": "claude-code", "status": "idle"}]
        calls.append(cm.prepare())
        return calls

    def test_system_region_is_byte_identical_across_dispatch_and_report_back(self, tmp_path):
        none, running, idle = self._three_calls(tmp_path)
        assert _system_rows(none) == _system_rows(running) == _system_rows(idle)

    def test_only_the_final_message_differs(self, tmp_path):
        none, running, idle = self._three_calls(tmp_path)
        assert _split(none)[0] == _split(running)[0] == _split(idle)[0]
        assert _split(none)[1] != _split(running)[1] != _split(idle)[1]

    def test_state_is_one_tail_line_and_absent_when_nothing_runs(self, tmp_path):
        none, running, idle = self._three_calls(tmp_path)
        assert "Sub-agents:" not in _runtime_part(none[-1])
        assert "\nSub-agents: claude-code running" in "\n" + _runtime_part(running[-1])
        assert "\nSub-agents: claude-code idle" in "\n" + _runtime_part(idle[-1])
        for prepared in (none, running, idle):
            for m in prepared[:-1]:
                assert "Sub-agents: claude-code" not in str(m.get("content"))

    def test_coordination_rules_are_present_from_the_first_call(self, tmp_path):
        none, _, _ = self._three_calls(tmp_path)
        system = "\n".join(_system_rows(none))
        assert "### Dispatching" in system  # 1109 R5
        assert "Current Sub-Agent States" not in system


# ---------------------------------------------------------------------------
# A0 (b): asks no longer ride the per-call tail
# ---------------------------------------------------------------------------

ASKS_V1 = "\n".join([
    asks.FORMAT_HEADER,
    "- open a1b2c3 2026-10-01 Choose the venue for the meetup",
    "- open d4e5f6 2026-10-02 Sponsor the GOAI booth",
    "- dropped d4e5f6 2026-10-03 by:user",
    "",
])
ASKS_V2 = ASKS_V1 + "\n".join([
    "- open 0a0b0c 2026-10-20 due:2026-11-01 Renew the domain before it lapses",
    "- dropped a1b2c3 2026-10-21 by:user",
    "",
])


class TestAsksStayOutOfThePrompt:
    def _write(self, tmp_path, text):
        (tmp_path / "orbital").mkdir(exist_ok=True)
        (tmp_path / "orbital" / "ASKS.md").write_text(text, encoding="utf-8")

    def test_changing_asks_md_leaves_every_byte_identical(self, tmp_path):
        self._write(tmp_path, ASKS_V1)
        cm = _manager(tmp_path)
        # The first call shows "Context usage: ~0%" (nothing measured yet);
        # later calls show the measured figure. Warm it so the comparison is
        # about ASKS.md, not about the prompt's size crossing a 5% step.
        cm.prepare()
        first = cm.prepare()
        self._write(tmp_path, ASKS_V2)
        second = cm.prepare()
        assert _no_clock(first) == _no_clock(second)

    @pytest.mark.parametrize("text", ["", ASKS_V1, ASKS_V2], ids=["no-asks", "v1", "v2"])
    def test_tail_carries_no_asks_text(self, tmp_path, text):
        if text:
            self._write(tmp_path, text)
        else:
            (tmp_path / "orbital").mkdir(exist_ok=True)
        tail = _runtime_part(_manager(tmp_path).prepare()[-1])
        for needle in ("Open asks", "Dropped by the user", "never re-propose",
                       "Choose the venue", "GOAI", "Renew the domain"):
            assert needle not in tail

    def test_asks_section_is_byte_identical_across_two_asks_files(self, tmp_path):
        ws = str(tmp_path)
        builder = PromptBuilder(ws)
        self._write(tmp_path, ASKS_V1)
        one = builder._asks(_ctx(ws))
        _, semi_one, _ = builder.build(_ctx(ws))
        self._write(tmp_path, ASKS_V2)
        two = builder._asks(_ctx(ws))
        _, semi_two, _ = builder.build(_ctx(ws))
        assert one == two
        assert semi_one == semi_two
        for needle in ("Choose the venue", "GOAI", "a1b2c3", "Renew the domain"):
            assert needle not in one

    def test_asks_section_tells_the_agent_to_read_the_file_itself(self, tmp_path):
        text = " ".join(PromptBuilder()._asks(_ctx(str(tmp_path))).split())
        assert "Before opening one, read ASKS.md: don't duplicate an open ask." in text
        assert ("When a user message answers something you raised, check for an "
                "open ask to close.") in text
        assert "runtime note" not in text
