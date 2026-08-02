# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Producer↔renderer parity guard for [Sub-agent] system markers (backlog
#23 D2, extended by #27).

``web/src/utils/subAgentMarkerFixtures.json`` is the one source of truth for
the sub-agent system-marker shapes: started / sent / sent_user_mention /
completed / failed / stopped-with-error / interaction_required /
stopped_by_user (± terminated background work) / turn_interrupted /
background_work_lost. It is read by two independent tests:

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

Backlog #27 closes the hole in that "together": both tests only ever saw the
shapes someone remembered to list, so a BRAND-NEW producer was caught by
neither — which is how four markers (on_user_stopped, on_turn_interrupted,
on_background_work_lost, on_interaction_required) shipped rendering as
nothing. ``test_every_marker_producer_has_a_fixture_row`` below now
discovers the producer set from the observer's own AST instead of trusting
this file's memory.
"""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2 import lifecycle_observer as lifecycle_observer_module
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "web" / "src" / "utils"
    / "subAgentMarkerFixtures.json"
)
OBSERVER_SOURCE_PATH = Path(lifecycle_observer_module.__file__)

PROJECT_ID = "fixture-proj"
SESSION_ID = "fixture-sess"
HANDLE = "claude-code"
TRANSCRIPT_PATH = "/tmp/fixture-transcript.jsonl"
MARKER_PREFIX = "[Sub-agent]"

# Producer method -> the fixture shape(s) it is expected to cover. This is the
# ONLY hand-maintained half left, and the AST guard below pins it to reality
# from both sides: a method that starts writing a marker without an entry
# here fails, and an entry naming a shape the fixture file doesn't have fails
# too. (One method can own several shapes — a branch that changes the marker
# text, like on_message_routed's user_mention split or on_user_stopped's
# optional background-work tail, is its own shape.)
PRODUCER_SHAPES = {
    "on_started": {"started"},
    "on_message_routed": {"sent", "sent_user_mention"},
    "on_interaction_required": {"interaction_required"},
    "on_completed": {"completed"},
    "on_error": {"stopped_with_error"},
    "on_failed": {"failed"},
    "on_user_stopped": {"stopped_by_user", "stopped_by_user_with_background"},
    "on_turn_interrupted": {"turn_interrupted"},
    "on_background_work_lost": {"background_work_lost"},
}


def _load_fixtures() -> dict[str, dict]:
    return {f["shape"]: f for f in json.loads(FIXTURE_PATH.read_text())}


def _is_marker_literal(node: ast.AST) -> bool:
    """True for a string expression whose literal text opens with the
    ``[Sub-agent]`` marker prefix.

    Works off the AST rather than the raw text so formatting is irrelevant:
    a plain string, an f-string, and a run of implicitly-concatenated
    (f-)string pieces split across lines all parse to a single ``Constant``
    or ``JoinedStr`` whose first literal chunk carries the prefix.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value.startswith(MARKER_PREFIX)
    if isinstance(node, ast.JoinedStr):
        first = node.values[0] if node.values else None
        return (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and first.value.startswith(MARKER_PREFIX)
        )
    return False


def _discover_marker_producers() -> set[str]:
    """Every ``LifecycleObserver`` method that builds a ``[Sub-agent] …``
    string, read straight out of the module's AST.

    Docstrings are excluded — this module's prose quotes marker text, and a
    docstring mentioning one does not make its method a producer.
    """
    tree = ast.parse(OBSERVER_SOURCE_PATH.read_text())
    docstring_nodes = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    producers = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if id(inner) in docstring_nodes:
                continue
            if _is_marker_literal(inner):
                producers.add(node.name)
                break
    return producers


def test_every_marker_producer_has_a_fixture_row():
    """Auto-discovery guard (backlog #27).

    The parity pair above only covers shapes someone hand-listed, so a NEW
    producer method was caught by neither test and rendered as nothing —
    exactly how four markers silently vanished from the timeline. Discover
    the producers from source instead, and fail if any lacks a fixture row.
    """
    discovered = _discover_marker_producers()
    assert discovered == set(PRODUCER_SHAPES), (
        "lifecycle_observer.py's [Sub-agent] marker producers no longer match "
        "PRODUCER_SHAPES.\n"
        f"  producers with no fixture mapping: {sorted(discovered - set(PRODUCER_SHAPES))}\n"
        f"  mapped names that produce nothing: {sorted(set(PRODUCER_SHAPES) - discovered)}\n"
        "A new producer needs: a fixture row in "
        "web/src/utils/subAgentMarkerFixtures.json, a matching parse rule in "
        "web/src/utils/chatTransform.ts (without one the marker renders as "
        "NOTHING), a drive call in the exact-match test below, and an entry "
        "here."
    )

    mapped_shapes = {s for shapes in PRODUCER_SHAPES.values() for s in shapes}
    assert mapped_shapes == set(_load_fixtures()), (
        "PRODUCER_SHAPES and subAgentMarkerFixtures.json disagree about which "
        "shapes exist.\n"
        f"  mapped but not in the fixture file: {sorted(mapped_shapes - set(_load_fixtures()))}\n"
        f"  in the fixture file but unmapped: {sorted(set(_load_fixtures()) - mapped_shapes)}"
    )


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
    await observer.on_interaction_required(
        PROJECT_ID, HANDLE, interaction_id="fixture-int-1", kind="question",
        prompt="Which file should I edit?", session_id=SESSION_ID)
    # backlog #27: the two on_user_stopped shapes. A bare stop is neutral —
    # the user asked for it; the second is the same event carrying the
    # background work the kill destroyed, which is the part that must never
    # be silent.
    await observer.on_user_stopped(
        PROJECT_ID, HANDLE, session_id=SESSION_ID)
    await observer.on_user_stopped(
        PROJECT_ID, HANDLE,
        terminated=["npm run dev", "python server.py"],
        session_id=SESSION_ID)
    await observer.on_turn_interrupted(
        PROJECT_ID, HANDLE, TRANSCRIPT_PATH, session_id=SESSION_ID)
    await observer.on_background_work_lost(
        PROJECT_ID, HANDLE,
        commands=["npm run dev", "python server.py"],
        session_id=SESSION_ID)

    produced = {
        "started": agent_manager.injections[0],
        "sent": agent_manager.injections[1],
        "sent_user_mention": agent_manager.injections[2],
        "completed": agent_manager.injections[3],
        "failed": agent_manager.injections[4],
        "stopped_with_error": agent_manager.injections[5],
        "interaction_required": agent_manager.injections[6],
        "stopped_by_user": agent_manager.injections[7],
        "stopped_by_user_with_background": agent_manager.injections[8],
        "turn_interrupted": agent_manager.injections[9],
        "background_work_lost": agent_manager.injections[10],
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
