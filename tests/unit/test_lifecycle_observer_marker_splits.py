# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for LifecycleObserver marker-content fixes (backlog #24, Task 1: D2 + D3).

D2: ``on_completed``'s injected content mixes a user-relevant fact ("[Sub-agent]
{handle} completed. Summary: ...") with agent-facing steering guidance ("do NOT
repeat or re-summarize it..."). The guidance must stay in the LLM-facing
content but never render in the chat timeline. Fix mirrors commit 967237d's
fanout join-summary ``_meta.display_content`` split.

D3 (backlog #23, supersedes an intermediate backlog #24 D3 no-op): the
@mention API route (``agent_os/api/routes/agents_v2.py``) used to fire its
own ``on_message_routed(initiator="user_mention", ...)`` notification in
ADDITION to the one ``SubAgentManager.send()`` already fires internally (via
``_dispatch_prompt_locked``) for the very same ``dispatch_id`` — one physical
dispatch, two markers. Current contract: ``send()`` now threads the caller's
``initiator`` through ``_QueuedPrompt`` to that one internal notification
(fired immediately or, for a queued prompt, when ``_on_prompt_turn_closed``
drains it), and the @mention route passes ``initiator="user_mention"`` into
``send()`` instead of firing a direct call of its own. Since this is the
ONLY marker a mention dispatch ever gets, ``on_message_routed`` injects
exactly one marker whose agent-facing ``content`` carries a supervise/relay
guidance line (the user addressed the sub-agent directly; don't answer on
its behalf), while ``_meta.display_content`` holds the clean "Message sent
to …" text the renderer actually shows — the guidance never reaches the
chat timeline.
"""

from types import SimpleNamespace

import pytest

from agent_os.daemon_v2.lifecycle_observer import LifecycleObserver
from agent_os.daemon_v2.models import make_session_key
from agent_os.daemon_v2.sub_agent_manager import SubAgentManager


class _AgentManager:
    def __init__(self):
        self.injections = []

    async def inject_system_message(self, project_id, content, **kwargs):
        self.injections.append((project_id, content, kwargs))


class _WS:
    def __init__(self):
        self.events = []

    def broadcast(self, project_id, payload):
        self.events.append(payload)


class _ProcessManager:
    """Minimal double matching test_sub_agent_prompt_queue_fifo.py's — just
    enough surface for SubAgentManager.send() to dispatch without a real
    provider."""

    def set_turn_closed_callback(self, callback):
        pass

    def set_permission_request_callback(self, callback):
        pass

    def set_active_dispatch(self, *args, **kwargs):
        pass

    def clear_dispatch(self, *args, **kwargs):
        pass


class _Transport:
    def __init__(self):
        self.messages = []

    async def dispatch(self, message):
        self.messages.append(message)


def _manager_with_adapter(observer, project_id="p1", session_id="s1", handle="cursor"):
    manager = SubAgentManager(_ProcessManager(), lifecycle_observer=observer)
    transport = _Transport()
    manager._adapters[make_session_key(project_id, session_id)] = {
        handle: SimpleNamespace(_transport=transport, _broken=False),
    }
    return manager, transport


# ---------------------------------------------------------------------------
# D2 — completion marker display split
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_completed_display_content_excludes_guidance_with_summary():
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())

    await observer.on_completed(
        "p1", "claude-code", "2, 3, 5, 7", "/x/y.jsonl", session_id="s1")

    _, content, kwargs = agent_manager.injections[0]
    display = kwargs["meta"]["display_content"]
    assert display == (
        "[Sub-agent] claude-code completed. Summary: 2, 3, 5, 7. "
        "Transcript: /x/y.jsonl."
    )
    assert "do NOT repeat" not in display
    assert "Verify the work" not in display
    # The split is real, not a deletion — the full LLM-facing content still
    # carries the steering guidance, and starts with the same display text.
    assert "do NOT repeat" in content
    assert content.startswith(display)


@pytest.mark.asyncio
async def test_on_completed_display_content_excludes_guidance_no_output_variant():
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())

    await observer.on_completed(
        "p1", "claude-code", "", "/x/y.jsonl", session_id="s1")

    _, content, kwargs = agent_manager.injections[0]
    display = kwargs["meta"]["display_content"]
    assert display == (
        "[Sub-agent] claude-code completed. Summary: (no output). "
        "Transcript: /x/y.jsonl."
    )
    assert "nothing was shown to" not in display
    assert "nothing was shown to" in content
    assert content.startswith(display)


@pytest.mark.asyncio
async def test_on_completed_absorbed_by_fanout_skips_injection_entirely():
    """When a fanout registry absorbs the terminal event, on_completed must
    not ALSO inject its own per-worker marker — the fanout join summary
    (already _meta-split as of commit 967237d) is the one user-visible
    marker for the whole group."""
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())

    class _AbsorbingRegistry:
        def absorb_terminal(self, *args, **kwargs):
            return True

    observer.fanout_registry = _AbsorbingRegistry()

    await observer.on_completed(
        "p1", "worker:f-0", "ok", "/x/y.jsonl", session_id="s1")

    assert agent_manager.injections == []


# ---------------------------------------------------------------------------
# D3 — @mention double dispatch marker (backlog #24), superseded by backlog
# #23 D3's initiator-aware guidance (send() now threads the caller's
# initiator through to its one internal on_message_routed call; the
# @mention route no longer fires a second, direct call of its own).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_mention_initiator_adds_guidance_but_keeps_display_clean():
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())

    await observer.on_message_routed(
        "p1", "cursor", initiator="user_mention",
        message_preview="hi", transcript_path="/x/y.jsonl",
        session_id="s1", dispatch_id="s1:abc123")

    assert len(agent_manager.injections) == 1
    _, content, kwargs = agent_manager.injections[0]
    display = kwargs["meta"]["display_content"]
    assert display == '[Sub-agent] Message sent to cursor: "hi". Transcript: /x/y.jsonl'
    assert content.startswith(display)
    assert "do not answer on its behalf" in content
    assert "do not answer on its behalf" not in display


@pytest.mark.asyncio
async def test_management_agent_initiator_still_writes_message_sent_marker():
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())

    await observer.on_message_routed(
        "p1", "cursor", initiator="management_agent",
        message_preview="hi", transcript_path="/x/y.jsonl",
        session_id="s1", dispatch_id="s1:abc123")

    assert len(agent_manager.injections) == 1
    _, content, kwargs = agent_manager.injections[0]
    assert content.startswith('[Sub-agent] Message sent to cursor: "hi"')
    assert kwargs["meta"]["dispatch_id"] == "s1:abc123"
    # No guidance for an ordinary management-agent dispatch — no split needed.
    assert "display_content" not in kwargs["meta"]


@pytest.mark.asyncio
async def test_mention_dispatch_writes_exactly_one_marker_end_to_end():
    """Mirrors agent_os/api/routes/agents_v2.py's @mention handler post
    backlog #23 D3: mint a dispatch_id and call SubAgentManager.send() with
    initiator="user_mention" — the route fires no notification of its own
    anymore. Exactly one marker lands, carrying the guidance line, with a
    clean display_content for the renderer."""
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())
    manager, transport = _manager_with_adapter(observer)

    dispatch_id = "s1:deadbeef"
    result = await manager.send(
        "p1", "cursor", "hi @cursor", session_id="s1", dispatch_id=dispatch_id,
        initiator="user_mention")

    assert result.startswith("Message sent")
    assert transport.messages == ["hi @cursor"]
    assert len(agent_manager.injections) == 1
    _, content, kwargs = agent_manager.injections[0]
    assert content.startswith("[Sub-agent] Message sent to cursor")
    assert "do not answer on its behalf" in content
    assert kwargs["meta"]["display_content"].startswith("[Sub-agent] Message sent to cursor")
    assert "do not answer on its behalf" not in kwargs["meta"]["display_content"]


@pytest.mark.asyncio
async def test_mention_dispatch_writes_exactly_one_marker_when_queued():
    """Same @mention dispatch, but the target is already busy so send()
    defers to the FIFO instead of dispatching immediately — the internal
    on_message_routed notification only fires once the queue drains. Still
    exactly one marker, and the drained call still carries the
    "user_mention" initiator threaded through _QueuedPrompt."""
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())
    manager, transport = _manager_with_adapter(observer)
    manager._prompt_active.add(("p1", "s1", "cursor"))

    dispatch_id = "s1:queued01"
    result = await manager.send(
        "p1", "cursor", "hi @cursor", session_id="s1", dispatch_id=dispatch_id,
        initiator="user_mention")
    assert "queued" in result.lower()
    assert agent_manager.injections == []

    # Draining the FIFO now fires send()'s own internal notification.
    manager._prompt_active.discard(("p1", "s1", "cursor"))
    await manager._on_prompt_turn_closed("p1", "cursor", session_id="s1", cause="success")

    assert transport.messages == ["hi @cursor"]
    assert len(agent_manager.injections) == 1
    _, content, kwargs = agent_manager.injections[0]
    assert content.startswith("[Sub-agent] Message sent to cursor")
    assert "do not answer on its behalf" in content
    assert "do not answer on its behalf" not in kwargs["meta"]["display_content"]


@pytest.mark.asyncio
async def test_two_management_agent_dispatches_each_write_their_own_marker():
    """Regression guard: the D3 fix must not affect markers for ordinary
    (non-@mention) dispatches — each gets a fresh dispatch_id, its own
    marker, and no guidance line, unaffected by the user_mention handling
    above."""
    agent_manager = _AgentManager()
    observer = LifecycleObserver(agent_manager, _WS())
    manager, transport = _manager_with_adapter(observer)

    first = await manager.send("p1", "cursor", "first", session_id="s1")
    manager._prompt_active.discard(("p1", "s1", "cursor"))
    second = await manager.send("p1", "cursor", "second", session_id="s1")

    assert first.startswith("Message sent")
    assert second.startswith("Message sent")
    assert transport.messages == ["first", "second"]
    assert len(agent_manager.injections) == 2
    for _, content, _kwargs in agent_manager.injections:
        assert "do not answer on its behalf" not in content


# ---------------------------------------------------------------------------
# Spec 079 — a queue dispatch's marker must not wake the manager
# ---------------------------------------------------------------------------
#
# Found in the live daemon smoke, not by a unit test: an item assigned to a
# worker dispatched correctly, and then the manager woke on THIS marker,
# read the ``[QUEUE ITEM | …]`` row and HEADER_CONTRACT already sitting in
# that session, and did the task itself — racing the worker it had just
# dispatched, and handing the dispatcher a stray turn to classify as the
# item's verdict. The initiator ``queue_item`` keeps the mention funnel but
# stamps suppress_wake here, leaving exactly one management turn: the one
# the worker's terminal event starts.


@pytest.mark.asyncio
async def test_queue_item_marker_is_wake_suppressed():
    mgr = _AgentManager()
    obs = LifecycleObserver(mgr, _WS())

    await obs.on_message_routed(
        "proj", "codex", initiator="queue_item",
        message_preview="build the thing", transcript_path="/t/x.jsonl",
        session_id="sess1", dispatch_id="sess1:abcd1234",
    )

    assert len(mgr.injections) == 1
    _, content, kwargs = mgr.injections[0]
    meta = kwargs["meta"]
    assert meta["suppress_wake"] is True, (
        "a queue dispatch must not start a management turn — the manager "
        "would race its own worker against the queue contract in history"
    )
    # Still a normal marker in every other respect: joinable to the
    # transcript, and rendered as the clean one-liner.
    assert meta["dispatch_id"] == "sess1:abcd1234"
    assert meta["handle"] == "codex"
    assert meta["display_content"] == (
        '[Sub-agent] Message sent to codex: "build the thing". '
        'Transcript: /t/x.jsonl'
    )
    # The guidance is agent-facing only, and is written for the LATER wake
    # turn, which reads this row as history.
    assert "do NOT do the task yourself" in content
    assert "mark_task_complete" in content
    assert "do NOT do the task yourself" not in meta["display_content"]


@pytest.mark.asyncio
async def test_queue_item_dispatch_is_not_marked_pinned():
    """The terminal event must still wake: only ``user_pinned`` is pinned.

    This is the other half of the fix. Suppressing the dispatch marker is
    only correct because the worker's terminal event is NOT suppressed —
    that is the turn that verifies the result and declares the verdict. The
    terminal hooks stamp suppress_wake from ``_is_pinned_dispatch``, so a
    queue dispatch must leave the key unpinned.
    """
    obs = LifecycleObserver(_AgentManager(), _WS())

    obs.set_dispatch_initiator("proj", "codex", "user_pinned", session_id="s1")
    assert obs._is_pinned_dispatch("proj", "codex", "s1") is True

    obs.set_dispatch_initiator("proj", "codex", "queue_item", session_id="s1")
    assert obs._is_pinned_dispatch("proj", "codex", "s1") is False, (
        "a queue dispatch is not pinned — its terminal event must wake the "
        "manager for the verdict"
    )


@pytest.mark.asyncio
async def test_user_mention_marker_still_wakes():
    """Regression guard on the chat path: unchanged by spec 079."""
    mgr = _AgentManager()
    obs = LifecycleObserver(mgr, _WS())

    await obs.on_message_routed(
        "proj", "codex", initiator="user_mention",
        message_preview="hi", transcript_path="/t/x.jsonl",
        session_id="sess1", dispatch_id="sess1:abcd1234",
    )

    _, content, kwargs = mgr.injections[0]
    assert "suppress_wake" not in (kwargs["meta"] or {})
    assert "do not answer on its behalf" in content
