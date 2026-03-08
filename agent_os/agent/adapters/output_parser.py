# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output classification for CLI adapter streams."""

import re
from datetime import datetime, timezone

from agent_os.agent.adapters.base import OutputChunk


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OutputParser:
    def __init__(self, approval_patterns: list[str]):
        self._approval_regexes = [re.compile(p) for p in approval_patterns]
        self._activity_patterns = [
            re.compile(r'(?i)^(reading|writing|editing|creating|deleting|running|executing|searching)\b'),
            re.compile(r'(?i)^[\u2502\u2514\u251c]\s'),  # tree output
            re.compile(r'(?i)\b(wrote|created|updated|deleted|found)\s+\d+'),
        ]
        self._status_patterns = [
            re.compile(r'(?i)^\s*(thinking|analyzing|processing|loading)'),
            re.compile(r'(?i)\[\d+/\d+\]'),  # progress indicators [3/10]
            re.compile(r'(?i)\.{3,}$'),       # trailing ellipsis
        ]

    def parse(self, raw_text: str) -> OutputChunk:
        text = raw_text.strip()
        if not text:
            return OutputChunk(text="", chunk_type="response", timestamp=_now())

        # Check approval patterns first (highest priority)
        for regex in self._approval_regexes:
            if regex.search(text):
                return OutputChunk(text=text, chunk_type="approval_request", timestamp=_now())

        # Check activity patterns
        for regex in self._activity_patterns:
            if regex.search(text):
                return OutputChunk(text=text, chunk_type="tool_activity", timestamp=_now())

        # Check status patterns
        for regex in self._status_patterns:
            if regex.search(text):
                return OutputChunk(text=text, chunk_type="status", timestamp=_now())

        # Default: response
        return OutputChunk(text=text, chunk_type="response", timestamp=_now())
