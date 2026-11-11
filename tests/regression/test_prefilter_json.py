# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: JSON pre-filter compacts API responses, never file content.

Pretty-printed JSON from API-style tools (connector shims, triggers,
calendar, sessions, browser) is compacted to save tokens. Content that came
from a file or from another agent (``read``, ``grep``, ``glob``,
``agent_message``, ``fanout``…) is passed through byte-identical: ``edit``
matches text exactly, so a compacted ``read`` made the model copy an
``old_text`` that does not exist on disk (spec 171 A4).
"""

import json

import pytest

from agent_os.agent.tool_result_filters import dispatch_prefilter, _prefilter_json


PRETTY_JSON = json.dumps(
    {
        "status": "success",
        "data": {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com", "roles": ["admin", "user"]},
                {"id": 2, "name": "Bob", "email": "bob@example.com", "roles": ["user"]},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com", "roles": ["user", "moderator"]},
            ],
            "pagination": {
                "page": 1,
                "per_page": 10,
                "total": 3,
                "total_pages": 1,
            },
        },
        "meta": {
            "request_id": "abc-123",
            "timestamp": "2026-03-03T08:45:00Z",
        },
    },
    indent=4,
)


class TestPrefilterJSON:
    """JSON pre-filter compacts whitespace from pretty-printed JSON."""

    def test_whitespace_removed(self):
        """Filtered JSON has no unnecessary whitespace."""
        filtered = _prefilter_json(PRETTY_JSON)
        assert "\n" not in filtered
        assert "    " not in filtered

    def test_semantically_equivalent(self):
        """Filtered JSON parses to the same data structure."""
        filtered = _prefilter_json(PRETTY_JSON)
        original = json.loads(PRETTY_JSON)
        compacted = json.loads(filtered)
        assert original == compacted

    def test_output_smaller_than_input(self):
        """Compacted JSON is smaller than pretty-printed JSON."""
        filtered = _prefilter_json(PRETTY_JSON)
        assert len(filtered) < len(PRETTY_JSON)

    def test_non_json_returned_unchanged(self):
        """Non-JSON input is returned as-is."""
        plain = "This is not JSON at all."
        assert _prefilter_json(plain) == plain

    def test_invalid_json_returned_unchanged(self):
        """Invalid JSON-like input is returned as-is."""
        invalid = '{"key": "value", missing_quotes: true}'
        assert _prefilter_json(invalid) == invalid

    def test_deeply_nested_json(self):
        """Deeply nested JSON compacts correctly."""
        nested = {"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}
        pretty = json.dumps(nested, indent=2)
        filtered = _prefilter_json(pretty)
        assert json.loads(filtered) == nested
        assert len(filtered) < len(pretty)

    def test_json_array(self):
        """JSON arrays are compacted correctly."""
        arr = [{"id": i, "value": f"item_{i}"} for i in range(10)]
        pretty = json.dumps(arr, indent=4)
        filtered = _prefilter_json(pretty)
        assert json.loads(filtered) == arr

    def test_dispatch_compacts_connector_api_response(self):
        """A connector shim's JSON response is compacted."""
        filtered = dispatch_prefilter("google-calendar.list_events", {}, PRETTY_JSON)
        assert "\n" not in filtered
        assert json.loads(filtered) == json.loads(PRETTY_JSON)

    @pytest.mark.parametrize("tool", [
        "list_triggers", "create_trigger", "update_trigger", "delete_trigger",
        "calendar_read", "list_sessions", "read_session",
    ])
    def test_dispatch_compacts_builtin_api_tools(self, tool):
        filtered = dispatch_prefilter(tool, {}, PRETTY_JSON)
        assert "\n" not in filtered
        assert json.loads(filtered) == json.loads(PRETTY_JSON)

    def test_read_of_pretty_json_is_byte_identical(self):
        """read returns the file exactly as on disk, so edit can match it."""
        package_json = '{\n  "name": "demo",\n  "version": "1.0.0",\n  "private": true\n}\n'
        assert dispatch_prefilter("read", {"path": "package.json"}, package_json) == package_json

    @pytest.mark.parametrize("tool", [
        "read", "grep", "glob", "agent_message", "fanout", "edit", "write",
        "notify", "checkpoint_state",
    ])
    def test_file_and_agent_content_never_compacted(self, tool):
        """Tools that return file bytes or another agent's words keep them."""
        assert dispatch_prefilter(tool, {}, PRETTY_JSON) == PRETTY_JSON
        line = '[1, 2, 3]'
        assert dispatch_prefilter(tool, {}, line) == line

    def test_dispatch_does_not_compact_non_json(self):
        """dispatch_prefilter does not apply JSON filter to non-JSON content."""
        plain = "Exit code: 0\nSome output here"
        assert dispatch_prefilter("list_triggers", {}, plain) == plain
        assert dispatch_prefilter("read", {"path": "file.txt"}, plain) == plain
