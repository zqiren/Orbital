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

Backlog #35 closes the hole in THAT: #27's scan read one file. A marker's
rendered text is not always written in one file — ``sub_agent_manager.py``
composes the *reason* half of the dropped-queue marker at each of its drop
sites and hands it to the observer, so #27's tripwire covered the sentence
frame and nothing that fills it. Two scans now run over BOTH modules:

- ``test_every_marker_producer_has_a_fixture_row`` walks every source in
  ``MARKER_SOURCE_MODULES`` and keys producers by ``module.function``, so a
  raw ``[Sub-agent] …`` string written anywhere in either file needs a
  fixture.
- ``test_every_queue_drop_site_is_marked_and_pinned`` discovers the drop
  sites structurally — every ``self._prompt_queues.pop(...)`` — and fails on
  any pop that neither marks its dropped prompts nor sits under an
  emptiness guard, then drives the REAL ``SubAgentManager`` helper with each
  discovered reason so the cross-module composed text is pinned to a
  fixture. A sixth drop site cannot be added silently, and a reworded reason
  cannot drift away from what the renderer was tested against.
"""

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_os.daemon_v2 import lifecycle_observer as lifecycle_observer_module
from agent_os.daemon_v2 import sub_agent_manager as sub_agent_manager_module
from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "web" / "src" / "utils"
    / "subAgentMarkerFixtures.json"
)

# Every module that contributes text to a ``[Sub-agent]`` marker. Adding one
# here is all it takes to bring a new marker-producing module inside the
# tripwire (backlog #35b).
MARKER_SOURCE_MODULES = {
    "lifecycle_observer": Path(lifecycle_observer_module.__file__),
    "sub_agent_manager": Path(sub_agent_manager_module.__file__),
}
MANAGER_SOURCE_PATH = MARKER_SOURCE_MODULES["sub_agent_manager"]

PROJECT_ID = "fixture-proj"
SESSION_ID = "fixture-sess"
HANDLE = "claude-code"
TRANSCRIPT_PATH = "/tmp/fixture-transcript.jsonl"
MARKER_PREFIX = "[Sub-agent]"

# The FIFO whose pops are drop sites, and the helper that turns a dropped
# prompt into a durable timeline row.
PROMPT_QUEUE_ATTR = "_prompt_queues"
DROP_MARKER_HELPER = "_mark_queued_prompts_dropped"
QUEUE_DROPPED_ACTION = "queue_dropped"

# ``module.method`` -> the fixture shape(s) it is expected to cover. This is
# the ONLY hand-maintained half left, and the AST guard below pins it to
# reality from both sides: a method that starts writing a marker without an
# entry here fails, and an entry naming a shape the fixture file doesn't have
# fails too. (One method can own several shapes — a branch that changes the
# marker text, like on_message_routed's user_mention split or
# on_user_stopped's optional background-work tail, is its own shape. So is
# each distinct drop reason on_queue_dropped is handed, and THOSE are
# discovered from sub_agent_manager.py rather than listed — see
# ``test_every_queue_drop_site_is_marked_and_pinned``.)
PRODUCER_SHAPES = {
    "lifecycle_observer.on_started": {"started"},
    "lifecycle_observer.on_message_routed": {"sent", "sent_user_mention"},
    "lifecycle_observer.on_interaction_required": {"interaction_required"},
    "lifecycle_observer.on_completed": {"completed"},
    "lifecycle_observer.on_error": {"stopped_with_error"},
    "lifecycle_observer.on_failed": {"failed"},
    "lifecycle_observer.on_user_stopped": {
        "stopped_by_user", "stopped_by_user_with_background"},
    "lifecycle_observer.on_turn_interrupted": {"turn_interrupted"},
    "lifecycle_observer.on_background_work_lost": {"background_work_lost"},
    "lifecycle_observer.on_queue_dropped": {
        "queue_dropped_transport_ended",
        "queue_dropped_agent_unavailable",
        "queue_dropped_prior_send_failed",
        "queue_dropped_prior_dispatch_failed",
        "queue_dropped_agent_stopped",
    },
}


def _load_fixtures() -> dict[str, dict]:
    return {f["shape"]: f for f in json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))}


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


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """ids of every docstring expression in ``tree``.

    Excluded from the marker scan — both modules' prose quotes marker text,
    and a docstring mentioning one does not make its method a producer.
    """
    return {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }


def _discover_marker_producers() -> set[str]:
    """Every function in ``MARKER_SOURCE_MODULES`` that builds a
    ``[Sub-agent] …`` string, read straight out of each module's AST and
    returned as ``module.function``.

    Cross-module by construction (backlog #35b): #27's single-file scan meant
    a marker written outside lifecycle_observer.py — the very thing the
    dropped-queue shape was about to become — sat outside the tripwire.
    """
    producers = set()
    for module_name, source_path in MARKER_SOURCE_MODULES.items():
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        docstring_nodes = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if id(inner) in docstring_nodes:
                    continue
                if _is_marker_literal(inner):
                    producers.add(f"{module_name}.{node.name}")
                    break
    return producers


def _innermost_function_scopes(tree: ast.AST) -> dict[int, str | None]:
    """node id -> name of the innermost function enclosing it.

    Needed because ``ast.walk`` over a function also descends into functions
    nested inside it: ``send``'s body contains ``_background_send``, and a
    drop site in the inner one must be reported against the inner one.
    """
    scopes: dict[int, str | None] = {}

    def visit(node: ast.AST, current: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = f"{current}.{child.name}" if current else child.name
                scopes[id(child)] = inner
                visit(child, inner)
            else:
                scopes[id(child)] = current
                visit(child, current)

    visit(tree, None)
    return scopes


def _is_prompt_queue_pop(node: ast.AST) -> bool:
    """True for ``self._prompt_queues.pop(...)`` — a queue drop site."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == PROMPT_QUEUE_ATTR
    )


def _emptiness_guarded_node_ids(tree: ast.AST) -> set[int]:
    """ids of every node lexically inside an ``if not <x>:`` body.

    A pop that throws its return value away is only honest when the queue is
    already known empty, and in this module that knowledge always takes the
    form of an ``if not queue:`` guard immediately above it.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
        ):
            for stmt in node.body:
                for inner in ast.walk(stmt):
                    guarded.add(id(inner))
    return guarded


def _discover_queue_drop_sites() -> tuple[list[tuple[str, int, str]], list[str]]:
    """Scan sub_agent_manager.py for every queue drop site.

    Returns ``(marked, untracked)`` where ``marked`` is
    ``(function, lineno, why)`` for each pop whose dropped prompts reach
    ``_mark_queued_prompts_dropped``, and ``untracked`` describes every pop
    that neither marks nor sits under an emptiness guard — i.e. every place a
    queued prompt can still vanish from the timeline without a trace.
    """
    tree = ast.parse(MANAGER_SOURCE_PATH.read_text(encoding="utf-8"))
    scopes = _innermost_function_scopes(tree)
    guarded = _emptiness_guarded_node_ids(tree)

    # Every marking call as (function, variable, lineno, reason). A list, not
    # a dict: one function marks several drops and they all name the popped
    # queue ``dropped``, so a pop is paired with the FIRST marking call below
    # it — the one a reader pairs it with — rather than with whichever
    # same-named call happens to come last.
    mark_calls: list[tuple[str | None, str, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == DROP_MARKER_HELPER
        ):
            continue
        first = node.args[0] if node.args else None
        why = next(
            (kw.value for kw in node.keywords if kw.arg == "why"), None)
        assert isinstance(first, ast.Name), (
            f"{DROP_MARKER_HELPER} at line {node.lineno} must be handed the "
            "popped queue by name so the guard can pair it with its pop"
        )
        assert isinstance(why, ast.Constant) and isinstance(why.value, str), (
            f"{DROP_MARKER_HELPER} at line {node.lineno} must pass a literal "
            "why= — the reason is half the rendered marker text, and a "
            "computed one cannot be pinned to a fixture"
        )
        mark_calls.append(
            (scopes.get(id(node)), first.id, node.lineno, why.value))
    mark_calls.sort(key=lambda c: c[2])

    marked: list[tuple[str, int, str]] = []
    untracked: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.Expr)):
            continue
        pop = next(
            (n for n in ast.walk(node) if _is_prompt_queue_pop(n)), None)
        if pop is None:
            continue
        scope = scopes.get(id(node))
        where = f"{MANAGER_SOURCE_PATH.name}:{pop.lineno} in {scope}()"
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            why = next(
                (
                    reason for call_scope, name, lineno, reason in mark_calls
                    if call_scope == scope
                    and isinstance(target, ast.Name) and name == target.id
                    and lineno > pop.lineno
                ),
                None,
            )
            if why is None:
                untracked.append(f"{where} — popped but never marked")
            else:
                marked.append((scope or "?", pop.lineno, why))
        elif id(node) not in guarded:
            untracked.append(f"{where} — return value discarded")
    return marked, untracked


def test_every_marker_producer_has_a_fixture_row():
    """Auto-discovery guard (backlog #27, cross-module since #35).

    The parity pair above only covers shapes someone hand-listed, so a NEW
    producer method was caught by neither test and rendered as nothing —
    exactly how four markers silently vanished from the timeline. Discover
    the producers from source instead, and fail if any lacks a fixture row.
    """
    discovered = _discover_marker_producers()
    assert discovered == set(PRODUCER_SHAPES), (
        f"the [Sub-agent] marker producers in {sorted(MARKER_SOURCE_MODULES)} "
        "no longer match PRODUCER_SHAPES.\n"
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


@pytest.mark.asyncio
async def test_every_queue_drop_site_is_marked_and_pinned():
    """Cross-module drop-site guard (backlog #35b).

    The dropped-queue marker's rendered text is written in two files: the
    sentence frame in lifecycle_observer.py, the reason in whichever
    sub_agent_manager.py drop site produced it. #27's scan saw only the
    frame, so the reasons — the half a reader actually reads — were pinned by
    nothing. Discover the sites structurally instead:

    1. every ``self._prompt_queues.pop(...)`` must either mark its dropped
       prompts or sit under an ``if not queue:`` emptiness guard, so a sixth
       drop site cannot be added silently the way the fifth one was;
    2. driving the REAL ``SubAgentManager`` helper with each discovered
       reason must reproduce exactly the fixture rows the renderer is tested
       against, so a reworded reason cannot drift away from them.
    """
    marked, untracked = _discover_queue_drop_sites()
    assert not untracked, (
        "sub_agent_manager.py drops queued prompts with no timeline row.\n  "
        + "\n  ".join(untracked)
        + f"\nEach queued prompt has its own orphaned 'You -> handle' bubble "
        f"upstream; a pop that does not reach {DROP_MARKER_HELPER}() leaves "
        "it unexplained forever. Mark the drop (and add a fixture row for a "
        "new reason), or — if the queue is provably empty there — pop it "
        "under an `if not queue:` guard."
    )

    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, MagicMock())
    manager = SubAgentManager(MagicMock(), lifecycle_observer=observer)
    for _function, _lineno, why in marked:
        await manager._mark_queued_prompts_dropped(
            [object()], PROJECT_ID, HANDLE, session_id=SESSION_ID, why=why)

    produced = {_rendered(content, kwargs)
                for content, kwargs in agent_manager.injections}
    expected = {f["content"] for f in _load_fixtures().values()
                if f["action"] == QUEUE_DROPPED_ACTION}
    assert produced == expected, (
        "the text sub_agent_manager.py's drop sites actually produce no "
        "longer matches the "
        f"{QUEUE_DROPPED_ACTION} rows in subAgentMarkerFixtures.json.\n"
        f"  produced with no fixture row: {sorted(produced - expected)}\n"
        f"  fixture rows nothing produces: {sorted(expected - produced)}"
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
    # The queue_dropped rows are deliberately absent here: their reason text
    # is composed at sub_agent_manager.py's drop sites, so driving the
    # observer with a made-up reason would pin nothing real. They are driven
    # through the manager instead, in
    # test_every_queue_drop_site_is_marked_and_pinned above.
    fixtures = {shape: f for shape, f in _load_fixtures().items()
                if f["action"] != QUEUE_DROPPED_ACTION}
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
