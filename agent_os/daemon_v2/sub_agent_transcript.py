# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent transcript — append-only JSONL log for one sub-agent's output."""

import json
import os
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SubAgentTranscript:
    """Append-only JSONL log for one sub-agent's output within a project."""

    def __init__(self, workspace: str, handle: str, transcript_id: str):
        self._dir = os.path.join(workspace, ".agent-os", "sub_agents", handle)
        os.makedirs(self._dir, exist_ok=True)
        self._filename = f"{transcript_id}.jsonl"
        self._filepath = os.path.join(self._dir, self._filename)
        # Create the file
        open(self._filepath, "a", encoding="utf-8").close()
        # Update latest pointer (use .latest text file on Windows)
        self._update_latest()

    def _update_latest(self) -> None:
        """Point latest to current transcript."""
        latest_path = os.path.join(self._dir, ".latest")
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                f.write(self._filename)
        except OSError:
            pass

    def append(self, entry: dict) -> None:
        """Append one JSON line. Entry should have: source, content, timestamp, chunk_type."""
        if "timestamp" not in entry:
            entry["timestamp"] = _now()
        line = json.dumps(entry, ensure_ascii=False)
        with open(self._filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    @property
    def filepath(self) -> str:
        """Return the absolute path to the transcript file."""
        return self._filepath

    @classmethod
    def read(cls, filepath: str) -> list[dict]:
        """Read all entries from a transcript file."""
        entries = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries
