# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for management-agent sibling-session tools (spec 019)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_os.agent.tools.sessions_tool import ListSessionsTool, ReadSessionTool
from agent_os.daemon_v2.agent_manager import AgentManager
from agent_os.daemon_v2.models import AgentConfig, Autonomy, make_session_key


def _result(tool, **arguments):
    return json.loads(tool.execute(**arguments).content)


def _rows():
    return [
        {
            "session_id": "current",
            "session_uuid": "current",
            "name": "Current",
            "origin": "chat",
            "status": "running",
            "last_activity_at": "2026-07-14T12:00:00+00:00",
        },
        {
            "session_id": "legacy-facing-id",
            "session_uuid": "legacy_uuid",
            "name": "beta",
            "origin": "cold_start",
            "status": "idle",
            "last_activity_at": "2026-07-13T12:00:00Z",
        },
        {
            "session_id": "alpha_uuid",
            "session_uuid": "alpha_uuid",
            "name": "Alpha",
            "origin": "queue",
            "status": "waiting",
            "last_activity_at": "2026-07-12T12:00:00+00:00",
        },
        {
            "session_id": "undated",
            "session_uuid": "undated",
            "name": None,
            "origin": "chat",
            "status": "idle",
            "last_activity_at": None,
        },
    ]


class TestListSessionsTool:
    def test_excludes_self_by_default_and_can_include_it(self):
        tool = ListSessionsTool(list_sessions=_rows, current_session_id="current")

        default = _result(tool)
        included = _result(tool, include_self=True)

        assert [row["session_uuid"] for row in default["sessions"]] == [
            "legacy_uuid", "alpha_uuid", "undated",
        ]
        assert default["total_available"] == default["returned"] == 3
        assert included["sessions"][0]["session_uuid"] == "current"
        assert included["total_available"] == included["returned"] == 4

    def test_limit_since_and_null_timestamp_are_deterministic(self):
        tool = ListSessionsTool(list_sessions=_rows, current_session_id="current")

        result = _result(
            tool,
            since="2026-07-12T12:00:00Z",
            limit=1,
            include_self=True,
        )

        assert [row["session_uuid"] for row in result["sessions"]] == ["current"]
        assert result["total_available"] == 3
        assert result["returned"] == 1

    def test_name_sort_is_case_insensitive_and_none_last(self):
        tool = ListSessionsTool(list_sessions=_rows, current_session_id="current")

        result = _result(tool, sort="name")

        assert [row["session_uuid"] for row in result["sessions"]] == [
            "alpha_uuid", "legacy_uuid", "undated",
        ]

    def test_invalid_since_is_returned_as_tool_error(self):
        tool = ListSessionsTool(list_sessions=_rows, current_session_id="current")

        assert tool.execute(since="yesterday").content.startswith("Error:")


@pytest.fixture
def session_workspace(tmp_path):
    sessions = tmp_path / "orbital" / "sessions"
    sessions.mkdir(parents=True)

    records = [
        {"role": "meta", "event": "session_start", "name": "Sibling"},
        {"role": "user", "source": "user", "timestamp": "t1", "content": "first"},
        {"role": "assistant", "source": "management", "timestamp": "t2", "content": "Needle one"},
        {
            "role": "user",
            "source": "user",
            "timestamp": "t3",
            "content": [
                {"type": "text", "text": "A MULTIMODAL needle"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        },
        {
            "role": "assistant",
            "source": "management",
            "timestamp": "t4",
            "content": "x" * 4100,
            "tool_call_id": "call-1",
            "tool_calls": [{"id": "call-2", "function": {"name": "read"}}],
        },
        {"role": "tool", "source": "tool", "timestamp": "t5", "content": "last"},
    ]
    path = sessions / "legacy_uuid.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in records[:2]:
            stream.write(json.dumps(record) + "\n")
        stream.write("{broken json\n\n")
        for record in records[2:]:
            stream.write(json.dumps(record) + "\n")

    # Present on disk but absent from the authoritative management listing.
    (sessions / "worker_fanout-0.jsonl").write_text(
        json.dumps({"role": "user", "content": "worker secret"}) + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _read_tool(workspace):
    return ReadSessionTool(
        workspace=str(workspace),
        list_sessions=lambda: [
            {
                "session_id": "legacy-facing-id",
                "session_uuid": "legacy_uuid",
                "name": "Sibling",
                "origin": "chat",
                "status": "idle",
                "last_activity_at": "2026-07-14T12:00:00Z",
            }
        ],
    )


class TestReadSessionTool:
    @pytest.mark.parametrize(
        "stem",
        [
            "../legacy_uuid",
            "..",
            "legacy..uuid",
            "/tmp/legacy_uuid",
            "nested/session",
            "bad stem",
        ],
    )
    def test_rejects_traversal_and_unsafe_stems(self, session_workspace, stem):
        result = _read_tool(session_workspace).execute(session_uuid=stem)
        assert result.content.startswith("Error:")

    @pytest.mark.parametrize("stem", ["unknown", "worker_fanout-0"])
    def test_rejects_unknown_or_worker_stems(self, session_workspace, stem):
        result = _read_tool(session_workspace).execute(session_uuid=stem)
        assert "not available" in result.content

    def test_rejects_negative_offset(self, session_workspace):
        result = _read_tool(session_workspace).execute(
            session_uuid="legacy_uuid", offset=-1
        )
        assert "non-negative integer" in result.content

    def test_excludes_meta_and_malformed_records(self, session_workspace):
        result = _result(_read_tool(session_workspace), session_uuid="legacy_uuid")

        assert result["total_matches"] == 5
        assert [message["role"] for message in result["messages"]] == [
            "user", "assistant", "user", "assistant", "tool",
        ]
        assert all("event" not in message for message in result["messages"])

    def test_literal_filter_is_case_insensitive_and_searches_multimodal_cells(
        self, session_workspace
    ):
        result = _result(
            _read_tool(session_workspace),
            session_uuid="legacy_uuid",
            grep="NeEdLe",
        )

        assert result["total_matches"] == 2
        assert [message["timestamp"] for message in result["messages"]] == ["t2", "t3"]

    def test_tail_pagination_returns_chronological_pages(self, session_workspace):
        tool = _read_tool(session_workspace)

        newest = _result(tool, session_uuid="legacy_uuid", limit=2)
        older = _result(tool, session_uuid="legacy_uuid", limit=2, offset=2)
        oldest = _result(tool, session_uuid="legacy_uuid", limit=2, offset=4)

        assert [message["timestamp"] for message in newest["messages"]] == ["t4", "t5"]
        assert newest["next_offset"] == 2
        assert [message["timestamp"] for message in older["messages"]] == ["t2", "t3"]
        assert older["next_offset"] == 4
        assert [message["timestamp"] for message in oldest["messages"]] == ["t1"]
        assert "next_offset" not in oldest

    def test_long_content_is_explicitly_truncated_and_tool_fields_preserved(
        self, session_workspace
    ):
        result = _result(
            _read_tool(session_workspace),
            session_uuid="legacy_uuid",
            grep="xxxx",
        )
        message = result["messages"][0]

        assert len(message["content"]) == 4000
        assert message["content"].endswith("chars]")
        assert "TRUNCATED" in message["content"]
        assert message["tool_call_id"] == "call-1"
        assert message["tool_calls"][0]["id"] == "call-2"


def _write_materialized_session(path, content="session content"):
    path.write_text(
        json.dumps({
            "role": "user",
            "source": "user",
            "timestamp": "2026-07-14T12:00:00Z",
            "content": content,
        }) + "\n",
        encoding="utf-8",
    )


def _authorized_read_tool(workspace, stem):
    return ReadSessionTool(
        workspace=str(workspace),
        list_sessions=lambda: [{
            "session_id": stem,
            "session_uuid": stem,
            "name": None,
            "origin": "chat",
            "status": "idle",
            "last_activity_at": "2026-07-14T12:00:00Z",
        }],
    )


class TestSessionSymlinkSecurity:
    def test_external_sessions_root_is_never_authorized_or_read(self, tmp_path):
        workspace = tmp_path / "workspace"
        orbital = workspace / "orbital"
        external = tmp_path / "other-project-sessions"
        orbital.mkdir(parents=True)
        external.mkdir()
        _write_materialized_session(external / "foreign.jsonl", "foreign secret")
        (orbital / "sessions").symlink_to(external, target_is_directory=True)

        manager = _manager(workspace)
        assert manager.list_sessions("project-1") == []

        result = _authorized_read_tool(workspace, "foreign").execute(
            session_uuid="foreign"
        )
        assert result.content.startswith("Error:")
        assert "foreign secret" not in result.content

    def test_entry_symlink_escaping_sessions_root_is_excluded_and_unreadable(self, tmp_path):
        workspace = tmp_path / "workspace"
        sessions = workspace / "orbital" / "sessions"
        sessions.mkdir(parents=True)
        external = tmp_path / "outside.jsonl"
        _write_materialized_session(external, "outside secret")
        (sessions / "alias.jsonl").symlink_to(external)

        manager = _manager(workspace)
        assert manager.list_sessions("project-1") == []

        result = _authorized_read_tool(workspace, "alias").execute(
            session_uuid="alias"
        )
        assert result.content.startswith("Error:")
        assert "outside secret" not in result.content

    def test_worker_alias_symlink_cannot_hide_resolved_worker_identity(self, tmp_path):
        workspace = tmp_path / "workspace"
        sessions = workspace / "orbital" / "sessions"
        sessions.mkdir(parents=True)
        worker_stem = "worker_worker_deadbeef_0_cafebabe"
        worker = sessions / f"{worker_stem}.jsonl"
        _write_materialized_session(worker, "worker secret")
        (sessions / "friendly-session.jsonl").symlink_to(worker.name)

        manager = _manager(workspace)
        assert manager.list_sessions("project-1") == []

        result = _authorized_read_tool(workspace, "friendly-session").execute(
            session_uuid="friendly-session"
        )
        assert result.content.startswith("Error:")
        assert "worker secret" not in result.content


def _manager(workspace):
    project_store = MagicMock()
    project_store.get_project.return_value = {
        "workspace": str(workspace),
        "model": "gpt-4o",
        "api_key": "test",
    }
    manager = AgentManager(
        project_store=project_store,
        ws_manager=MagicMock(),
        sub_agent_manager=MagicMock(),
        activity_translator=MagicMock(),
        process_manager=MagicMock(),
        provider_registry=MagicMock(),
    )
    manager._platform_provider = None
    manager._user_credential_store = None
    manager._browser_manager = None
    manager._trigger_manager = None
    return manager


class TestRegistration:
    def test_management_registry_has_both_session_tools(self, tmp_path):
        from agent_os.agent.tools.registry import ToolRegistry

        manager = _manager(tmp_path)
        manager.list_sessions = MagicMock(return_value=[])
        registry = ToolRegistry()
        config = AgentConfig(
            workspace=str(tmp_path),
            model="gpt-4o",
            api_key="test",
            autonomy=Autonomy.HANDS_OFF,
        )

        manager._register_tools(
            registry, config, project_id="project-1", session_id="current"
        )

        assert {"list_sessions", "read_session"} <= set(registry.tool_names())
        assert registry._tools["list_sessions"]._current_session_id == "current"

    def test_native_worker_registry_excludes_session_tools(self, tmp_path):
        manager = _manager(tmp_path)
        handle = SimpleNamespace(
            loop=SimpleNamespace(_utility_provider=MagicMock()),
        )
        manager._handles[make_session_key("project-1", "current")] = handle

        deps = manager.build_worker_deps("project-1", "current")
        registry = deps.make_tool_registry(None, None, "worker:fanout-0")

        assert "list_sessions" not in registry.tool_names()
        assert "read_session" not in registry.tool_names()
