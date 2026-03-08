# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: JSON pre-filter compacts pretty-printed JSON responses.

Removes whitespace from pretty-printed JSON API responses, reducing
token count by 30-50%.
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

    def test_dispatch_routes_json_content(self):
        """dispatch_prefilter auto-detects JSON content and compacts it."""
        # JSON content from a non-browser, non-shell tool
        filtered = dispatch_prefilter("web_fetch", {}, PRETTY_JSON)
        assert "\n" not in filtered
        assert json.loads(filtered) == json.loads(PRETTY_JSON)

    def test_dispatch_does_not_compact_non_json(self):
        """dispatch_prefilter does not apply JSON filter to non-JSON content."""
        plain = "Exit code: 0\nSome output here"
        filtered = dispatch_prefilter("read", {"path": "file.txt"}, plain)
        assert filtered == plain
