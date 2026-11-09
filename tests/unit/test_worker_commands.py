# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spec 166: the pinned worker's own commands in the `/` picker.

The parsers turn each protocol's report into one normal form (or None when
the report is malformed), the store keeps the last good list per
(worker, workspace) and persists it, the cold probe fills an empty key,
and the merge folds the list into Orbital's catalog.
"""

import asyncio
import json
import os

import pytest

from agent_os.daemon_v2 import worker_commands as wc


# ---------------------------------------------------------------- parsers

class TestClaudeParser:
    def test_commands_become_slash_invocations_with_hints(self):
        info = {"commands": [
            {"name": "review", "description": "Review a PR (project)", "argumentHint": "<pr>"},
            {"name": "compact", "description": "Compact the conversation", "argumentHint": ""},
        ]}
        assert wc.claude_commands(info) == [
            {"name": "review", "description": "Review a PR (project)",
             "invoke": "/review ", "hint": "<pr>", "enabled": True},
            {"name": "compact", "description": "Compact the conversation",
             "invoke": "/compact ", "hint": "", "enabled": True},
        ]

    def test_malformed_is_none_and_empty_is_empty(self):
        assert wc.claude_commands(None) is None
        assert wc.claude_commands({}) is None
        assert wc.claude_commands({"commands": "x"}) is None
        assert wc.claude_commands({"commands": []}) == []

    def test_bad_rows_dropped_duplicates_collapsed(self):
        info = {"commands": [{"name": ""}, "x", {"description": "no name"},
                             {"name": "a"}, {"name": "A", "description": "dup"}]}
        assert [c["name"] for c in wc.claude_commands(info)] == ["a"]


class TestCodexParser:
    def test_skills_across_groups_dollar_invocation_and_disabled_kept(self):
        result = {"data": [
            {"cwd": "/ws", "skills": [
                {"name": "imagegen", "description": "Make images", "scope": "system", "enabled": True},
                {"name": "old", "description": "Off", "scope": "user", "enabled": False},
            ]},
            {"cwd": "/ws", "skills": [
                {"name": "imagegen", "description": "again", "enabled": True},
                {"name": "repo-skill", "description": "", "scope": "repo"},
            ]},
        ]}
        assert wc.codex_commands(result) == [
            {"name": "imagegen", "description": "Make images",
             "invoke": "$imagegen ", "hint": "", "enabled": True},
            {"name": "old", "description": "Off",
             "invoke": "$old ", "hint": "", "enabled": False},
            {"name": "repo-skill", "description": "",
             "invoke": "$repo-skill ", "hint": "", "enabled": True},
        ]

    def test_malformed_is_none_and_empty_is_empty(self):
        assert wc.codex_commands(None) is None
        assert wc.codex_commands({}) is None
        assert wc.codex_commands({"data": {}}) is None
        assert wc.codex_commands({"data": [{"cwd": "/ws", "skills": None}]}) is None
        assert wc.codex_commands({"data": []}) == []
        assert wc.codex_commands({"data": [{"cwd": "/ws", "skills": []}]}) == []


class TestAcpParser:
    def test_available_commands_with_input_hint(self):
        from acp.schema import AvailableCommand
        cmds = [
            AvailableCommand.model_validate({"name": "compact", "description": "Compact",
                                             "input": {"hint": "instructions"}}),
            {"name": "status", "description": "Status"},
        ]
        assert wc.acp_commands(cmds) == [
            {"name": "compact", "description": "Compact",
             "invoke": "/compact ", "hint": "instructions", "enabled": True},
            {"name": "status", "description": "Status",
             "invoke": "/status ", "hint": "", "enabled": True},
        ]

    def test_malformed_is_none(self):
        assert wc.acp_commands(None) is None
        assert wc.acp_commands("x") is None
        assert wc.acp_commands([]) == []


# ------------------------------------------------------------------ store

CMDS = [{"name": "review", "description": "R", "invoke": "/review ", "hint": "", "enabled": True}]


class TestStore:
    def test_publish_get_and_workspace_is_realpathed(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        link = tmp_path / "link"
        os.symlink(ws, link)
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"))
        assert s.get("claude-code", str(ws)) is None
        s.publish("claude-code", str(link), CMDS)
        assert s.get("claude-code", str(ws)) == CMDS
        assert s.get("codex", str(ws)) is None

    def test_none_never_overwrites(self, tmp_path):
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"))
        s.publish("kimi-code", "/ws", CMDS)
        s.publish("kimi-code", "/ws", None)
        assert s.get("kimi-code", "/ws") == CMDS
        s.publish("kimi-code", "/ws", [])
        assert s.get("kimi-code", "/ws") == []

    def test_persistence_round_trip(self, tmp_path):
        path = str(tmp_path / "wc.json")
        wc.WorkerCommandStore(path).publish("kimi-code", "/ws", CMDS)
        assert wc.WorkerCommandStore(path).get("kimi-code", "/ws") == CMDS

    def test_unchanged_publish_does_not_rewrite(self, tmp_path):
        path = tmp_path / "wc.json"
        s = wc.WorkerCommandStore(str(path))
        s.publish("kimi-code", "/ws", CMDS)
        path.write_text("sentinel")
        s.publish("kimi-code", "/ws", list(CMDS))
        assert path.read_text() == "sentinel"

    def test_corrupt_file_loads_empty(self, tmp_path):
        path = tmp_path / "wc.json"
        path.write_text("{not json")
        s = wc.WorkerCommandStore(str(path))
        assert s.get("kimi-code", "/ws") is None
        s.publish("kimi-code", "/ws", CMDS)
        assert json.loads(path.read_text())["entries"][0]["slug"] == "kimi-code"


class TestColdProbe:
    @pytest.mark.asyncio
    async def test_cached_entry_is_returned_without_probing(self, tmp_path):
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"))
        s.publish("claude-code", "/ws", CMDS)

        async def probe():
            raise AssertionError("must not probe a warm key")
        assert await s.commands_for("claude-code", "/ws", probe) == CMDS

    @pytest.mark.asyncio
    async def test_cold_key_probes_once_for_concurrent_callers_and_stores(self, tmp_path):
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"))
        calls = []

        async def probe():
            calls.append(1)
            await asyncio.sleep(0.02)
            return CMDS
        a, b = await asyncio.gather(s.commands_for("claude-code", "/ws", probe),
                                    s.commands_for("claude-code", "/ws", probe))
        assert a == b == CMDS
        assert len(calls) == 1
        assert s.get("claude-code", "/ws") == CMDS

    @pytest.mark.asyncio
    async def test_failure_backs_off(self, tmp_path):
        now = [100.0]
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"), clock=lambda: now[0])
        calls = []

        async def probe():
            calls.append(1)
            return None
        assert await s.commands_for("codex", "/ws", probe) == []
        assert await s.commands_for("codex", "/ws", probe) == []
        assert len(calls) == 1
        now[0] += wc.PROBE_FAILURE_BACKOFF_S + 1
        assert await s.commands_for("codex", "/ws", probe) == []
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_probe_exception_is_a_failure_not_a_raise(self, tmp_path):
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"))

        async def probe():
            raise RuntimeError("boom")
        assert await s.commands_for("codex", "/ws", probe) == []
        assert s.get("codex", "/ws") is None


class TestModulePublish:
    def test_publish_without_store_is_a_noop(self):
        wc.install(None)
        wc.publish("claude-code", "/ws", CMDS)  # never raises

    def test_publish_reaches_installed_store(self, tmp_path):
        s = wc.WorkerCommandStore(str(tmp_path / "wc.json"))
        wc.install(s)
        try:
            wc.publish("claude-code", "/ws", CMDS)
            assert s.get("claude-code", "/ws") == CMDS
        finally:
            wc.install(None)


# ------------------------------------------------------------------ merge

def _entry(key, name=None, source="project"):
    name = name or key
    return {"key": key, "name": name, "description": "", "source": source,
            "path": f"/x/{name}/SKILL.md", "dir_name": name}


class TestMerge:
    def test_worker_entry_replaces_same_key_others_kept(self):
        catalog = [_entry("task-planning"), _entry("wecomcli-msg", source="agents")]
        cmds = [{"name": "wecomcli-msg", "description": "Msg (user)",
                 "invoke": "/wecomcli-msg ", "hint": "", "enabled": True},
                {"name": "review", "description": "R", "invoke": "/review ",
                 "hint": "<pr>", "enabled": True}]
        merged = wc.merge_catalog(catalog, cmds)
        by_key = {e["key"]: e for e in merged}
        assert set(by_key) == {"task-planning", "wecomcli-msg", "review"}
        assert by_key["task-planning"]["path"] == "/x/task-planning/SKILL.md"
        assert by_key["wecomcli-msg"] == {
            "key": "wecomcli-msg", "name": "wecomcli-msg", "description": "Msg (user)",
            "source": "worker", "path": "", "dir_name": "",
            "invoke": "/wecomcli-msg ", "hint": ""}
        assert by_key["review"]["hint"] == "<pr>"

    def test_disabled_worker_skill_hides_the_catalog_copy(self):
        catalog = [_entry("old", source="codex"), _entry("keep", source="codex")]
        cmds = [{"name": "Old", "description": "", "invoke": "$Old ", "hint": "", "enabled": False}]
        assert [e["key"] for e in wc.merge_catalog(catalog, cmds)] == ["keep"]

    def test_no_commands_is_the_catalog_unchanged(self):
        catalog = [_entry("a"), _entry("b")]
        assert wc.merge_catalog(catalog, []) == catalog
