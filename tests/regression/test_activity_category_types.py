# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression: browser_automation and credential_request ActivityCategory.

Root cause: backend emits 'browser_automation' and 'credential_request' as
activity categories via _TOOL_CATEGORY_MAP, but frontend ActivityCategory
union type didn't include them, causing type mismatches.

Fix: Added both values to ActivityCategory in types.ts and icon mappings
in ActivityBlock.tsx.
"""

from agent_os.daemon_v2.activity_translator import _TOOL_CATEGORY_MAP


class TestActivityCategoryMap:
    """Verify backend emits the expected category strings."""

    def test_browser_maps_to_browser_automation(self):
        """'browser' tool maps to 'browser_automation' category."""
        assert _TOOL_CATEGORY_MAP["browser"] == "browser_automation"

    def test_request_credential_maps_to_credential_request(self):
        """'request_credential' tool maps to 'credential_request' category."""
        assert _TOOL_CATEGORY_MAP["request_credential"] == "credential_request"

    def test_known_categories_complete(self):
        """All tool category values are documented strings."""
        expected_categories = {
            "file_read", "file_write", "file_edit",
            "command_exec", "request_access", "agent_message",
            "browser_automation", "credential_request",
            "file_search", "content_search",
        }
        actual = set(_TOOL_CATEGORY_MAP.values())
        # All mapped values should be recognized
        assert actual.issubset(expected_categories), (
            f"Unknown categories: {actual - expected_categories}"
        )
