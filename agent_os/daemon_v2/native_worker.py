# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native worker adapter: an in-process ``AgentLoop`` wrapped in the duck-typed
adapter interface ``SubAgentManager``'s dispatch machinery already expects from
CLI adapters (spec 009, W1).

A worker is a one-shot, anonymous session: the fanout spawner (Task 2) mints
one ``NativeWorkerAdapter`` per dispatched sub-task, calls ``send()`` exactly
once with the task brief, and reads the result back off ``_last_response``
once the turn completes. Keeping ``_transport`` at ``None`` routes every
``send()`` through ``SubAgentManager._dispatch_async``'s ``_background_send``
fallback (sub_agent_manager.py:939-1036) — the same non-blocking-dispatch path
PTY/Pipe/ACP CLI adapters use — with zero changes to that machinery.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Protocol

from agent_os.agent.context import ContextManager
from agent_os.agent.loop import AgentLoop
from agent_os.agent.project_paths import ProjectPaths
from agent_os.agent.prompt_builder import Autonomy, PromptBuilder, PromptContext
from agent_os.agent.providers.think_splitter import InlineThinkSplitter
from agent_os.agent.session import Session, persist_user_row
from agent_os.daemon_v2.models import detect_os

logger = logging.getLogger(__name__)


def make_worker_handle(fanout_id: str, index: int) -> str:
    """Return the per-task worker handle: ``worker:<fanout_id>-<index>``."""
    return f"worker:{fanout_id}-{index}"


# Filename/uuid stem prefix of every fanout worker session JSONL: the adapter
# mints f"worker_{_sanitize_for_filename(handle)}_{uuid8}" and the handle is
# itself "worker:<fanout_id>-<i>", so sanitization yields a double prefix.
# Kept adjacent to the minting so the two cannot drift. Session listers use
# this as the legacy fallback for worker files whose session_kind meta was
# destroyed by pre-fix JSONL rewrites (see Session._collect_meta_lines).
WORKER_SESSION_STEM_PREFIX = "worker_worker_"

_WORKER_SESSION_STEM_RE = re.compile(r"worker_worker_[0-9a-f]{8}_\d+_[0-9a-f]{8}\Z")


def is_worker_session_stem(stem: str) -> bool:
    """True when a session JSONL filename stem belongs to a fanout worker.

    Full-shape match, not a prefix test: a project the user named
    "Worker Worker" mints chat stems like ``worker_worker_<hex8>`` which a
    prefix check would silently hide from the sidebar and merged chat.
    """
    return _WORKER_SESSION_STEM_RE.fullmatch(stem) is not None


class ToolRegistryLike(Protocol):
    """Shape of a tool registry as consumed by ``AgentLoop`` (mirrors
    ``agent_os.agent.tools.registry.ToolRegistry``). Documentation-only —
    Task 2's ``make_tool_registry`` factory returns the real, path-restricted
    registry; tests here use a plain stub satisfying this shape."""

    def schemas(self) -> list[dict]: ...
    def is_async(self, name: str) -> bool: ...
    def execute(self, name: str, arguments: dict): ...
    async def execute_async(self, name: str, arguments: dict): ...
    def reset_run_state(self) -> None: ...
    def tool_names(self) -> list[str]: ...


@dataclass
class WorkerDeps:
    """Bundle of project-level dependencies the fanout spawner (Task 2)
    resolves ONCE per fanout batch and hands to each ``NativeWorkerAdapter``.

    ``provider`` is already the correct model for workers (the project's
    utility model when configured, else the main model — resolved by the
    spawner, never here). ``make_tool_registry`` is the restricted-registry
    factory the spawner provides; this task only defines its call shape.

    ``on_activity`` (Task 2 activity plumbing, spec 009 §0.5-5): an optional
    zero-arg callback the adapter fires so the fanout watchdog's
    ``last_activity`` clock advances. ``WorkerDeps`` is one shared
    instance per fanout batch (not per task), so a spawner dispatching N
    tasks must rebind ``deps.on_activity`` to a per-handle closure
    immediately before constructing EACH adapter — the adapter reads it once
    at ``__init__`` time, so mutating the shared ``deps`` between
    constructions is safe and does not race.

    Granularity: fires at ``send()`` start/end (turn boundaries) AND before
    every tool execution, via ``_ActivityTrackingRegistry`` wrapping
    whatever ``make_tool_registry`` returns. ``AgentLoop`` itself has no
    per-tool-call hook to wire (checked at the loop's constructor,
    ``agent_os/agent/loop.py:81-97``) — wrapping the registry gets true
    per-tool-call granularity without touching ``loop.py``, since the loop
    calls the registry for every tool call regardless (``loop.py:1087-
    1097``). This matters because a worker is single-turn by design:
    without the registry wrap, a long multi-tool-call turn would never
    advance ``last_activity`` until the whole turn ends, and the default
    600s stall threshold would kill any task running past 10 minutes —
    precisely the long investigative work the activity design exists to
    protect. A long gap with NO tool calls (one slow LLM generation) still
    counts toward stall; accepted as pathological past 10 minutes of
    uninterrupted streaming.
    """

    provider: object
    workspace: str
    project_id: str
    parent_session_id: str
    make_tool_registry: Callable[
        [list[str] | None, list[str] | None, str], ToolRegistryLike
    ]
    on_activity: Callable[[], None] | None = None
    # WS fan-out for live drill-in streaming: ``broadcast(project_id, payload)``
    # (WebSocketManager.broadcast's shape). When set, each adapter wires its
    # session's ``on_stream`` to emit ``chat.stream_delta`` events addressed to
    # the WORKER's own session_uuid — ChatView filters strictly by viewed
    # session id, so these reach only a drill-in subscribed to that worker.
    # ``None`` → no streaming observer, byte-identical to the pre-streaming path.
    broadcast: Callable[[str, dict], None] | None = None
    # Closes the worker's isolated browser context (BrowserManager.
    # close_worker_scope). None → no browser cleanup (no browser manager).
    close_browser_scope: Callable[[str], Awaitable[None]] | None = None


_HANDLE_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")


def _sanitize_for_filename(text: str) -> str:
    """Replace anything not alnum/underscore (e.g. ``:``, ``-``) so a handle
    is safe to embed in a session filename cross-platform — ``:`` is a
    reserved character on Windows filesystems."""
    return _HANDLE_SANITIZE_RE.sub("_", text)


def _strip_inline_think(text: str) -> str:
    """Strip a leading/embedded inline ``<think>...</think>`` reasoning
    block from a worker's final response text, reusing the same
    ``InlineThinkSplitter`` the streaming provider layer uses for live chat
    display (agent_os/agent/providers/think_splitter.py).

    Defensive, not the primary fix: for a correctly-registered provider the
    split already happens upstream in ``openai_compat.py`` before text ever
    reaches ``AgentLoop`` / the session JSONL. But a worker's provider can
    still persist raw ``<think>`` text into its assistant message (e.g. a
    reasoning-model config gap), and that raw text is exactly what
    ``_read_final_response`` reads back — surfacing as reasoning leaking
    into transcripts and fanout join summaries. Text with no think tag
    passes through UNCHANGED (the cheap membership check below short-
    circuits before constructing a splitter), so this never reformats a
    normal response. Falls back to the original text if stripping would
    otherwise produce an empty string (never surface a blank response)."""
    if "<think>" not in text:
        return text
    splitter = InlineThinkSplitter()
    visible, _ = splitter.feed(text)
    tail, _ = splitter.flush()
    stripped = (visible + tail).strip()
    return stripped or text


def _fanout_id_from_handle(handle: str) -> str:
    """Best-effort recover the fanout id from a ``worker:<fanout_id>-<index>``
    handle, for the session's ``session_kind`` meta tag. Falls back to the
    whole handle body for a handle that doesn't match the convention —
    forensic metadata only, never used for routing."""
    body = handle.split(":", 1)[-1]
    fanout_id, _, _ = body.rpartition("-")
    return fanout_id or body


class _ActivityTrackingRegistry:
    """Wraps a ``ToolRegistryLike`` so every tool execution fires the
    worker's activity hook BEFORE delegating — true per-tool-call
    granularity for the fanout stall watchdog (see ``WorkerDeps.on_activity``
    docstring for why turn-boundary-only bumps are insufficient). Delegates
    every other member unchanged; only ``execute``/``execute_async`` fire
    the hook, since those are the only members ``AgentLoop`` calls per tool
    call (``loop.py:1087-1097``)."""

    def __init__(self, inner, fire_activity) -> None:
        self._inner = inner
        self._fire_activity = fire_activity

    def schemas(self) -> list[dict]:
        return self._inner.schemas()

    def is_async(self, name: str) -> bool:
        return self._inner.is_async(name)

    def execute(self, name: str, arguments: dict):
        self._fire_activity()
        return self._inner.execute(name, arguments)

    async def execute_async(self, name: str, arguments: dict):
        self._fire_activity()
        return await self._inner.execute_async(name, arguments)

    def reset_run_state(self) -> None:
        return self._inner.reset_run_state()

    def tool_names(self) -> list[str]:
        return self._inner.tool_names()


class NativeWorkerAdapter:
    """In-process ``AgentLoop`` wrapped in the sub-agent adapter duck-type.

    Attributes/methods read by the existing dispatch/status machinery
    (duck-type contract, not inheritance — a worker has no process to
    ``start``/``read_stream``, so those two ``AgentAdapter`` ABC members are
    not implemented; ``is_idle``/``is_alive`` ARE implemented below, since
    every caller that scans ``SubAgentManager._adapters`` unconditionally —
    ``list_active``/``status`` and, notably, ``QueueDispatcher.
    _continuation_pending`` (outside this package) — calls them on every
    registered adapter regardless of type):
      - ``_transport``: always ``None`` (forces the background-send fallback).
      - ``_last_response``: set by ``send()`` for ``_background_send`` to read.
      - ``_idle`` / ``_broken``: backing state for ``is_idle()``/``is_alive()``.
      - ``display_name``: task label, shown in the fanout progress card.
    """

    agent_type = "native-worker"

    def __init__(self, *, deps: WorkerDeps, handle: str, display_name: str,
                 allowed_paths: list[str] | None,
                 forbidden_paths: list[str] | None) -> None:
        self.handle = handle
        self.display_name = display_name
        self._deps = deps
        # Read once at construction time (not a live `deps` reference) — see
        # WorkerDeps.on_activity docstring: the spawner rebinds
        # `deps.on_activity` to a per-handle closure between constructing
        # each adapter in a batch, so capturing it now is what makes sharing
        # one WorkerDeps instance across N adapters safe.
        self._on_activity = deps.on_activity
        self._close_browser_scope = deps.close_browser_scope

        self._transport = None
        self._last_response: str | None = None
        self._idle = True
        self._broken = False
        self._running = False
        # Round-3 review, IMPORTANT 3: set by stop() so _background_send
        # (sub_agent_manager.py) can tell "the turn was cancelled because
        # something deliberately stopped this worker" apart from "the turn
        # genuinely failed" — both produce the same "Error: task was
        # cancelled..." _last_response via _read_final_response()'s
        # exit_reason=="cancelled" fallback, but only the former should
        # route to on_turn_interrupted instead of on_error.
        self._stop_requested = False
        # Strong-ref slot for a future background-task wrapper, mirroring
        # CLIAdapter's field of the same name (sub_agent_manager.py owns the
        # actual task; this adapter never assigns it itself).
        self._background_send_task: asyncio.Task | None = None

        registry = deps.make_tool_registry(allowed_paths, forbidden_paths, handle)
        if self._on_activity is not None:
            # Per-tool-call activity granularity (see WorkerDeps.on_activity
            # docstring) — wrap AFTER the factory call so allowed/forbidden
            # scoping (Task 3's ScopedToolRegistry) is unaffected; this layer
            # only observes execute()/execute_async(), never alters args or
            # results.
            registry = _ActivityTrackingRegistry(registry, self._fire_activity)

        # Requirement 1: mint a real session JSONL in the project workspace,
        # same directory as any other session (Session.new / ProjectPaths).
        session_uuid = (
            f"worker_{_sanitize_for_filename(handle)}_{uuid.uuid4().hex[:8]}"
        )
        # Exposed (not just internal) so SubAgentManager.dispatch_fanout can
        # thread the real worker session id onto FanoutTask/fanout.started —
        # issues 2+3 (chat-shaped drill-in) need it to resolve the worker's
        # chat transcript via the /chat endpoint while the batch is mid-flight.
        self.session_uuid = session_uuid
        self._session_path = ProjectPaths(deps.workspace).session_file(session_uuid)
        self._session = Session.new(
            session_uuid, deps.workspace, project_id=deps.project_id,
            provider=getattr(deps.provider, "provider", "unknown"),
            model=getattr(deps.provider, "model", "unknown"),
            sdk=getattr(deps.provider, "sdk", "unknown"),
        )
        # Tag immediately (not lazily on first send()) so a worker that is
        # stopped before ever being messaged is still discoverable as a
        # worker thread on disk. list_sessions filters kind="worker" out of
        # the default sidebar listing (spec 009 §3a).
        self._session.append_meta(
            "session_kind",
            kind="worker",
            parent_session_id=deps.parent_session_id,
            fanout_id=_fanout_id_from_handle(handle),
            task_label=display_name,
        )

        # Live drill-in streaming: mirror ActivityTranslator.on_stream_chunk's
        # chat.stream_delta payload, but addressed to the WORKER's session_uuid
        # and with a PER-WORKER seq counter — sharing the translator's
        # per-project counter would interleave with (and reset under) the
        # management stream. The loop drives this via Session.notify_stream on
        # every provider chunk. Broadcast failures must never break the turn.
        if deps.broadcast is not None:
            self._stream_seq = 0

            def _on_stream(chunk, _broadcast=deps.broadcast,
                           _pid=deps.project_id, _sid=session_uuid,
                           _handle=handle):
                try:
                    is_final = getattr(chunk, "is_final", False)
                    self._stream_seq += 1
                    _broadcast(_pid, {
                        "type": "chat.stream_delta",
                        "project_id": _pid,
                        "session_id": _sid,
                        "text": getattr(chunk, "text", ""),
                        "reasoning_content": getattr(chunk, "reasoning_content", ""),
                        "source": _handle,
                        "is_final": is_final,
                        "seq": self._stream_seq,
                    })
                    if is_final:
                        self._stream_seq = 0
                except Exception:
                    logger.debug("worker stream broadcast failed for %s",
                                 _handle, exc_info=True)

            self._session.on_stream = _on_stream

        # Requirement 2: mirror the AgentLoop construction path
        # agent_manager.start_agent uses (~agent_manager.py:775-851), through
        # WorkerDeps rather than reading config/keychain here. Deliberately
        # OMITTED relative to that path:
        #   - interceptor: none. HANDS_OFF for workers — the fanout tool call
        #     itself is the approval boundary (spec 009 §0.5-7), not a
        #     per-worker one.
        #   - on_session_end / on_session_end_refresh: none. Workers are
        #     one-shot, fresh sessions with no resume/summarization lifecycle.
        #   - project_dir / get_budget_config / on_budget_event: none.
        #     loop._emit_ledger_event hardcodes source=SOURCE_MANAGEMENT, so
        #     wiring project_dir here would misattribute worker LLM spend as
        #     management spend. Leaving it unset means workers keep the loop's
        #     token-budget safety net but skip ledger/budget-guard wiring;
        #     proper sub-agent spend attribution is a follow-up if needed.
        # Iteration cap / repetition / ping-pong guards stay on: repetition
        # and ping-pong detection are unconditional in AgentLoop.run(); the
        # iteration cap uses the loop's own default (unbounded) since
        # WorkerDeps carries no config.max_iterations to forward — the fanout
        # stall watchdog (spec 009 §0.5-5) is a separate activity-timeout
        # applied externally by the spawner/join layer, not this loop's cap.
        prompt_builder = PromptBuilder(workspace=deps.workspace)
        base_ctx = PromptContext(
            workspace=deps.workspace,
            model=getattr(deps.provider, "model", "unknown"),
            autonomy=Autonomy.HANDS_OFF,
            enabled_agents=[],
            tool_names=list(getattr(registry, "tool_names", lambda: [])()),
            os_type=detect_os(),
            datetime_now=datetime.now().isoformat(),
            project_id=deps.project_id,
            agent_name=display_name,
        )
        context_manager = ContextManager(self._session, prompt_builder, base_ctx)

        self._loop = AgentLoop(self._session, deps.provider, registry, context_manager)

    async def send(self, message: str) -> None:
        """Run ONE full ``AgentLoop`` turn with ``message`` as the user
        message. Blocks until the turn completes. Never raises for
        task-level failures — the loop's own exception surface (an arbitrary
        provider/tool exception escaping ``AgentLoop.run``) is caught here and
        encoded into ``_last_response`` as ``"Error: ..."``, matching the
        adapter contract ``_background_send`` relies on to route completion
        vs. error (sub_agent_manager.py:986).

        Re-entrancy guard: a worker is one-shot by construction (the fanout
        spawner calls ``send()`` exactly once per adapter), so a second
        concurrent call while a turn is in flight is a caller bug, not a
        legitimate retry. Fail fast rather than run two ``loop.run()``
        invocations on one ``Session`` (``AgentLoop.run()`` itself would raise
        on true re-entry, but only after both calls have already raced to
        persist their brief onto the same session)."""
        if self._running:
            self._last_response = "Error: worker is already running a task"
            return
        self._idle = False
        self._running = True
        self._fire_activity()  # turn start
        try:
            # Bug #59 — whoever injects, persists. AgentLoop.run() no longer
            # appends the initial message, so the brief is written into the
            # worker's JSONL here, before the turn can fail. A worker that
            # dies on its first provider call still leaves the task it was
            # given on disk instead of an empty session. Kept inside the same
            # try as run() so a write failure still encodes as "Error: ..."
            # exactly as it did when the append lived inside run().
            try:
                persist_user_row(self._session, message)
                await self._loop.run()
            except Exception as e:  # noqa: BLE001 — task-level failure, never raise
                logger.exception(
                    "NativeWorkerAdapter %s: turn raised inside AgentLoop.run",
                    self.handle,
                )
                self._broken = True
                self._last_response = f"Error: {e}"
                return
            self._last_response = self._read_final_response()
        finally:
            self._running = False
            self._idle = True
            self._fire_activity()  # turn end
            await self._cleanup_browser_scope()

    def _fire_activity(self) -> None:
        """Best-effort ``on_activity`` invocation — a broken caller-supplied
        hook must never break the worker's turn."""
        if self._on_activity is None:
            return
        try:
            self._on_activity()
        except Exception:
            logger.exception(
                "NativeWorkerAdapter %s: on_activity callback raised",
                self.handle,
            )

    async def _cleanup_browser_scope(self) -> None:
        """Best-effort teardown of this worker's isolated browser context.
        Idempotent (close_worker_scope pops); must never break the turn."""
        if self._close_browser_scope is None:
            return
        try:
            await self._close_browser_scope(self.handle)
        except Exception:
            logger.exception(
                "NativeWorkerAdapter %s: browser scope cleanup failed", self.handle
            )

    async def stop(self) -> None:
        """Cancel the in-flight turn and mark the adapter stopped. Safe to
        call with no turn in flight (``cancel_turn`` is a no-op then).

        Sets ``_stop_requested`` BEFORE cancelling — a deliberate teardown
        (project stop_all, user stop) races against ``send()``'s own
        ``AgentLoop.run()`` returning with a cancelled-turn result; setting
        the flag first guarantees ``_background_send`` observes it as True
        by the time it inspects the response, regardless of exactly when
        ``cancel_turn()``'s effects propagate.
        """
        self._stop_requested = True
        await self._loop.cancel_turn()
        self._running = False
        self._idle = True
        await self._cleanup_browser_scope()

    def is_running(self) -> bool:
        return self._running

    def is_alive(self) -> bool:
        """Mirrors ``CLIAdapter.is_alive()``'s call shape (plain method
        returning ``bool``) for the shared duck-type contract every
        ``_adapters[sk]`` scanner relies on (Task 2 hardening — see class
        docstring). A worker has no process to be alive/dead; "alive" here
        means "still a valid, usable slot" — true until something marks it
        broken. A cleanly stopped worker is popped from ``_adapters``
        entirely (``SubAgentManager.stop``'s ``_kill_confirm_and_release``),
        so it is never observed here as not-alive; only an in-place failure
        (e.g. an unhandled ``_background_send`` exception) sets ``_broken``
        on a still-registered adapter.
        """
        return not self._broken

    def is_idle(self) -> bool:
        """Mirrors ``CLIAdapter.is_idle()``'s call shape — see ``is_alive``.
        A worker's only state is "turn in flight or not", so idle is simply
        the negation of ``is_running()``."""
        return not self.is_running()

    @property
    def session_path(self) -> str:
        """Path of the worker's session JSONL (for drill-in links)."""
        return self._session_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_final_response(self) -> str:
        """Return the worker's final assistant text, or an ``Error: ...``
        description derived from the loop's completion state when the turn
        ended without a usable trailing assistant message (e.g.
        ``mark_task_blocked``, a mid-turn stop, or a guard exit with no
        final text). Never returns an empty string — ``_background_send``
        treats a falsy response as "no output happened" and skips its
        completion/error routing entirely (sub_agent_manager.py:963)."""
        for msg in reversed(self._session.get_messages()):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return _strip_inline_think(content)
            # Newest assistant row carries no usable text — the turn did not
            # end on a text-only response. No earlier assistant row (from a
            # prior tool-calling iteration of this SAME turn) is a better
            # answer, so stop looking and fall through to completion state.
            break

        exit_reason, exit_summary, exit_block_reason = self._loop.get_completion_state()
        if exit_reason == "complete":
            return exit_summary or "Task completed."
        if exit_reason == "blocked":
            return f"Error: task blocked ({exit_block_reason or 'no reason given'})"
        if exit_reason == "cancelled":
            return "Error: task was cancelled before producing a result."
        if exit_reason == "budget_blocked":
            return "Error: stopped — project budget limit reached."
        return "Error: worker turn ended with no output."
