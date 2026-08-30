# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""RememberAboutUserTool — file an explicit user-stated fact into the
user-level memory file (spec 073).

Daemon-side tool, same escape-hatch pattern as CreateTriggerTool/NotifyTool:
the file lives OUTSIDE the workspace (~/orbital/user_memory.md by default),
where write/edit and the sandbox can never reach. The origin stamp
(``<!--from:<project> <date>-->``) is a plain comment for the user's benefit
when pruning in Settings — it is NOT the ``<!--mem-->`` grammar and this file
must never be run through ``memory_entries.process_on_write`` (that pipeline
is workspace-scoped by design).
"""

import os
import re
import time
from datetime import date

from agent_os.utils.file_lock import FileLockError, session_lock

from .base import Tool, ToolResult

# Caps enforced at write time (spec 073 D4): the file never enters the Layer-1
# budget arithmetic, so the size guard lives here. Whichever cap hits first
# wins; at the cap the tool REFUSES — never silently drops a user fact.
MAX_LINES = 60
MAX_CHARS = 6000

# acquire() is non-blocking and raises FileLockError; appends are rare, so a
# short bounded retry is all the concurrency machinery this needs.
_LOCK_ATTEMPTS = 3
_LOCK_RETRY_DELAY_S = 0.1

_STAMP_RE = re.compile(r"\s*<!--from:.*?-->\s*$")


def _normalize(text: str) -> str:
    """Collapse internal whitespace runs to single spaces and strip ends."""
    return " ".join(str(text).split())


def _fact_of_line(line: str) -> str:
    """The comparable fact text of a stored line: bullet and stamp removed."""
    line = _STAMP_RE.sub("", line.strip())
    if line.startswith("- "):
        line = line[2:]
    return _normalize(line)


class RememberAboutUserTool(Tool):
    """Append one user-stated durable fact to the user memory file."""

    def __init__(self, user_memory_path: str, project_name: str = ""):
        self._path = user_memory_path
        self._project_name = project_name or "unknown"
        self.name = "remember_about_user"
        self.description = (
            "Save a durable fact the user EXPLICITLY stated about themselves, "
            "in first person, to their user-level memory (shared across all "
            "their projects, editable in Global Settings). Use ONLY for: "
            "identity/role/employer, a stable preference or habit, a recurring "
            "person named by relationship, or a status that persists past "
            "today. NEVER file an inference, a conclusion drawn from the work, "
            "or transient session state. NEVER file government IDs, financial "
            "account numbers, health details, or anything about children. One "
            "short single-line fact per call, recorded at the stated level "
            "(one mention earns \"mentioned X once\", not \"X enthusiast\"). "
            "After a successful save, acknowledge it to the user in one line."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": (
                        "The fact to remember, as one short single line, "
                        "e.g. \"Works as a PM at Tencent\"."
                    ),
                },
            },
            "required": ["fact"],
        }

    def execute(self, **arguments) -> ToolResult:
        raw = arguments.get("fact", "")
        stripped = str(raw).strip()
        if not stripped:
            return ToolResult(content="Error: fact must be a non-empty string.")
        if "\n" in stripped or "\r" in stripped:
            return ToolResult(content=(
                "Error: fact must be a single line. File one fact per call."
            ))
        fact = _normalize(stripped)
        if fact.startswith("- "):
            fact = _normalize(fact[2:])
        if not fact:
            return ToolResult(content="Error: fact must be a non-empty string.")

        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        except OSError as exc:
            return ToolResult(content=f"Error: cannot create user memory directory: {exc}")

        lock = session_lock(self._path)
        acquired = False
        for attempt in range(_LOCK_ATTEMPTS):
            try:
                lock.acquire()
                acquired = True
                break
            except FileLockError:
                if attempt < _LOCK_ATTEMPTS - 1:
                    time.sleep(_LOCK_RETRY_DELAY_S)
            except OSError as exc:
                return ToolResult(content=f"Error: cannot lock user memory file: {exc}")
        if not acquired:
            return ToolResult(content=(
                "Error: the user memory file is busy (another session is "
                "writing). Try again in a moment."
            ))

        # Dedup and the caps are checked under the lock so concurrent appends
        # can neither interleave nor double-file the same fact.
        try:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                content = ""

            if any(_fact_of_line(line) == fact
                   for line in content.splitlines() if line.strip()):
                return ToolResult(content=(
                    f"Already on file: \"{fact}\" — no change made."
                ))

            stamp = f"<!--from:{self._project_name} {date.today().isoformat()}-->"
            entry = f"- {fact} {stamp}\n"
            line_count = sum(1 for line in content.splitlines() if line.strip())
            if line_count + 1 > MAX_LINES or len(content) + len(entry) > MAX_CHARS:
                return ToolResult(content=(
                    "Error: the user memory file is at capacity "
                    f"({MAX_LINES} lines / {MAX_CHARS} chars). Nothing was "
                    "saved. Ask the user to prune it in Global Settings "
                    "(About the User) before filing new facts."
                ))

            try:
                with open(self._path, "a", encoding="utf-8") as f:
                    if content and not content.endswith("\n"):
                        f.write("\n")
                    f.write(entry)
            except OSError as exc:
                return ToolResult(content=f"Error: failed to write user memory: {exc}")
        finally:
            lock.release()

        return ToolResult(content=(
            f"Noted: \"{fact}\" — saved to the user's memory (applies to all "
            "their projects; editable in Global Settings). Acknowledge this "
            "to the user in one line."
        ))
