# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Producer↔renderer parity guard for [Sub-agent] system markers (backlog
#23 D2).

``web/src/utils/subAgentMarkerFixtures.json`` is the one source of truth for
every sub-agent system-marker shape the backend writes AND the chat renderer
must whitelist. It is read by two independent tests:

- This file: drives the REAL ``LifecycleObserver`` producer methods with
  fixed sample args and asserts the rendered text (``_meta.display_content``
  when present, else the raw content — mirroring chatTransform.ts's own
  selection) exactly matches the fixture. A producer shape that drifts from
  its fixture entry fails HERE.
- ``web/src/utils/chatTransform.test.ts``: feeds every fixture entry's
  ``content`` through ``transformChatHistory`` and asserts it renders to a
  ``sub_agent_activity`` item (never silently dropped) with the expected
  ``action``. A renderer whitelist gap (e.g. on_error's "stopped with
  error:" shape before this fix) fails THERE.

Together: adding or changing a producer marker shape without updating the
fixture — or without a matching renderer rule — fails one of the two tests.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "web" / "src" / "utils"
    / "subAgentMarkerFixtures.json"
)

PROJECT_ID = "fixture-proj"
SESSION_ID = "fixture-sess"
HANDLE = "claude-code"
TRANSCRIPT_PATH = "/tmp/fixture-transcript.jsonl"


def _load_fixtures() -> dict[str, dict]:
    return {f["shape"]: f for f in json.loads(FIXTURE_PATH.read_text())}


class _AgentManager:
    """Records every injection exactly as ``inject_system_message`` receives
    it — content plus whatever ``meta``/``session_id`` kwargs came along."""

    def __init__(self):
        self.injections: list[tuple[str, dict]] = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((content, kwargs))


def _rendered(content: str, kwargs: dict) -> str:
    """What the chat renderer actually shows for one injected marker:
    ``_meta.display_content`` when present, else the raw content — the same
    selection chatTransform.ts's ``activityContent`` makes."""
    meta = kwargs.get("meta") or {}
    return meta.get("display_content", content)


@pytest.mark.asyncio
async def test_every_fixture_shape_matches_its_producer_exactly():
    fixtures = _load_fixtures()
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, MagicMock())

    await observer.on_started(
        PROJECT_ID, HANDLE, initiator="management_agent",
        transcript_path=TRANSCRIPT_PATH, session_id=SESSION_ID)
    await observer.on_message_routed(
        PROJECT_ID, HANDLE, initiator="management_agent",
        message_preview="run the tests", transcript_path=TRANSCRIPT_PATH,
        session_id=SESSION_ID, dispatch_id="fixture-sess:aaaa1111")
    # user_mention (backlog #23 D3): same dispatch shape as "sent" above, but
    # the LLM-facing content carries a guidance line — the RENDERED text
    # (meta.display_content) must still be the same clean "Message sent to
    # …" form, which is why this fixture row's content is byte-identical to
    # "sent"'s.
    await observer.on_message_routed(
        PROJECT_ID, HANDLE, initiator="user_mention",
        message_preview="run the tests", transcript_path=TRANSCRIPT_PATH,
        session_id=SESSION_ID, dispatch_id="fixture-sess:bbbb2222")
    await observer.on_completed(
        PROJECT_ID, HANDLE, "All tests passing", TRANSCRIPT_PATH,
        session_id=SESSION_ID)
    await observer.on_failed(
        PROJECT_ID, HANDLE, "adapter crashed", session_id=SESSION_ID)
    await observer.on_error(
        PROJECT_ID, HANDLE, "model timed out", TRANSCRIPT_PATH,
        session_id=SESSION_ID)

    produced = {
        "started": agent_manager.injections[0],
        "sent": agent_manager.injections[1],
        "sent_user_mention": agent_manager.injections[2],
        "completed": agent_manager.injections[3],
        "failed": agent_manager.injections[4],
        "stopped_with_error": agent_manager.injections[5],
    }

    # Every producer call above landed one marker, no more, no less.
    assert len(agent_manager.injections) == len(produced)
    # Producer set and fixture set must not have drifted apart in either
    # direction — a new producer shape with no fixture entry, or a stale
    # fixture entry with no producer, both fail here.
    assert set(produced) == set(fixtures)

    for shape, (content, kwargs) in produced.items():
        rendered = _rendered(content, kwargs)
        assert rendered == fixtures[shape]["content"], (
            f"{shape!r} marker's rendered text no longer matches "
            f"web/src/utils/subAgentMarkerFixtures.json — update the "
            f"fixture (and re-run the paired Vitest parity test) if this "
            f"drift is intentional"
        )
