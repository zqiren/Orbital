# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""@mention routing for user messages.

Parses @handle prefix at start of message. Case-insensitive match against known handles.
"""

import re


class MessageRouter:
    def __init__(self, known_handles: set[str]):
        self._handles = {h.lower() for h in known_handles}

    def route(self, raw_message: str) -> tuple[str | None, str]:
        """Parse @mention prefix.

        '@claudecode analyse this' -> ('claudecode', 'analyse this')
        'do something' -> (None, 'do something')

        Only matches handles at the start of message. Case-insensitive match.
        """
        if not raw_message:
            return (None, "")

        match = re.match(r"^@(\S+)\s*(.*)", raw_message, re.DOTALL)
        if match:
            handle = match.group(1).lower()
            rest = match.group(2).strip()
            if handle in self._handles:
                return (handle, rest)

        return (None, raw_message)
