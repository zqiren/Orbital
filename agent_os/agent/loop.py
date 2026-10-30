# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Main agent execution loop.

Owned by Component A. Core while loop: prepare context -> call LLM ->
process response -> execute tools -> repeat.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from collections import deque
from uuid import uuid4

from agent_os.agent.providers.types import (
    StreamAccumulator,
    StreamChunk,
    LLMResponse,
    ContextOverflowError,
    LLMError,
    ErrorCategory,
)
from agent_os.agent.tools.base import ToolResult
from agent_os.utils.text_io import read_text_lenient
from agent_os.agent.tool_result_filters import dispatch_prefilter
from agent_os.agent.tool_result_lifecycle import (
    archive_and_supersede_tool_results,
)
from agent_os.budget.ledger import (
    LedgerEvent,
    SOURCE_MANAGEMENT,
    append_event,
)

logger = logging.getLogger(__name__)

# Permanent, always-on per-iteration diagnostics. Kept on a dedicated logger so
# the loop's disposition/exit path is greppable and routable independently of
# the noisy module logger. Records carry a structured `diag` dict on the
# LogRecord (via extra=) plus a greppable "agent.diag.<event>" message. See
# TASK-loop-diagnostic-instrumentation. Low-cost: one log call per iteration.
_diag_logger = logging.getLogger("agent.diag")

# Per-session cache-hit aggregation (Part 4D). Shares the provider's cache
# audit channel so per-call and per-session lines land together in daemon.log.
_cache_logger = logging.getLogger("orbital.cache_audit")

# Spec 134 A2: sync tools (shell above all) run on their own bounded pool, not
# the event loop's default one. The default pool is ``min(32, cpu + 4)``
# threads — 8 on a 4-core laptop — and every request handler that uses
# ``asyncio.to_thread`` shares it; a few stuck shell calls used to fill it and
# freeze the whole app. With the spec 134 runner no tool call outlives its
# timeout, so this pool always drains.
TOOL_POOL_SIZE = 16
_tool_pool: ThreadPoolExecutor | None = None
_tool_pool_lock = threading.Lock()


def tool_executor() -> ThreadPoolExecutor:
    """The process-wide pool sync tools run on (created on first use)."""
    global _tool_pool
    with _tool_pool_lock:
        if _tool_pool is None:
            _tool_pool = ThreadPoolExecutor(
                max_workers=TOOL_POOL_SIZE, thread_name_prefix="orbital-tool")
            from agent_os.utils import stall_watch

            stall_watch.register_executor("tools", _tool_pool)
        return _tool_pool


# Spec 134 Part B: one id per tool batch (one assistant message's calls), so
# every ``shell(background=true)`` call of that message joins one job group.
# Sync tools see it through ``run_sync_tool``'s copied context.
_tool_batch: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "orbital_tool_batch", default=None)


def current_tool_batch() -> str | None:
    """The id of the tool batch the calling tool runs in, or None."""
    return _tool_batch.get()


async def run_sync_tool(func, /, *args):
    """``asyncio.to_thread`` on the tool pool (context vars carried along)."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        tool_executor(), functools.partial(ctx.run, func, *args))


# Spec 013: minimum seconds between STARTED background memory passes.
# Applies to every scheduler-routed trigger (agent_decided, over_budget,
# agent_decided_coalesced) — NOT to token_pressure or the session boundary,
# which stay synchronous and drain the gate instead. (The every-50-iterations
# turn-count trigger was removed by spec 089: the memory editor now runs when
# a file is over budget, noticed in ContextManager.prepare().)
REFRESH_DEBOUNCE_S = 300

# Spec 013's single-flight gate and debounce originally lived on the AgentLoop,
# but the files a pass rewrites ({workspace}/orbital/*.md) belong to the
# PROJECT, not the session. Two sessions on one project each got a fresh loop
# (_refresh_task None, _last_merge_at -inf), so both guards read open and both
# spawned a pass over the same four files. The loser's OCC baselines — captured
# before a 13-15 min LLM merge — were invalidated by the winner's own write, so
# its merge was discarded and logged as "user mid-edit detected" with no user
# involved (orbital-marketing, 2026-07-28 and 2026-07-29).
#
# Keyed by workspace path because that IS the contended resource. Entries are
# small and bounded by project count; last_merge_at must outlive the task, so
# entries are not evicted when a pass finishes. Loops with no project_dir
# (lightweight/test loops) fall back to the per-loop fields, unchanged.
_PROJECT_REFRESH: dict[str, dict] = {}

# Max nudge-retries per run when the model returns a no-tool-call response
# with no visible text (e.g. generation died inside its thinking block and
# the provider finalized the stream anyway — MiniMax-M3, 2026-07-13). Bounded
# so a persistently misbehaving model becomes a declared error, not a spend
# loop.
_EMPTY_RESPONSE_NUDGE_LIMIT = 2

# ---------------------------------------------------------------------------
# Spec 112 A6 — the end-of-run memory record turn.
#
# Every run that ends normally (final text, or mark_task_complete) and every
# run about to compact gets ONE daemon row asking the model whether anything
# durable happened, with the current PROJECT_STATE text and the three append
# anchors in hand, and then a short, bounded continuation. The reply to the
# user is already persisted and broadcast by then; every row the record turn
# produces carries ``_kind: MEMORY_RECORD_KIND`` so it never replaces it.
# ---------------------------------------------------------------------------
from agent_os.agent.context import MEMORY_RECORD_KIND  # noqa: E402 — re-exported
MEMORY_RECORD_MAX_LLM_CALLS = 3
MEMORY_RECORD_MAX_TOOL_CALLS = 6
# The four Layer-1 files, by on-disk name (agent_os.agent.context._LAYER1_FILES
# carries the same list keyed for WorkspaceFileManager).
_MEMORY_FILE_NAMES = ("PROJECT_STATE.md", "DECISIONS.md", "LESSONS.md", "INDEX.md")
# The only tools the record turn may run, on the four files only. Scoped to
# memory persistence, so it runs without the approval interceptor (FO-1) —
# the precedent set by the (since removed) pre-compaction flush turn.
_MEMORY_RECORD_TOOLS = frozenset({"read", "edit", "write"})

_MEMORY_RECORD_BODY = (
    "If so, update orbital/PROJECT_STATE.md so the next session starts from what is\n"
    "true after this run: current focus, what just finished, what is next, blockers.\n"
    "Overwrite stale lines in place; one fact per line; no dated history.\n"
    "Only if this run settled a decision, add one line to DECISIONS.md. Only if it\n"
    "taught a non-obvious lesson, one line to LESSONS.md. Only if a new file matters\n"
    "later, one line to INDEX.md.\n"
    "\n"
    "Write only what you already know from this run. Do not read sessions, do not\n"
    "tidy or trim anything, do not reply to the user. Edit with the exact text\n"
    "below as old_text; to add a line to DECISIONS / LESSONS / INDEX, anchor on the\n"
    "last line shown. One or two edits, then stop.\n"
    "If nothing durable happened, reply exactly: nothing to record\n"
)

_MEMORY_RECORD_OPENING = {
    "run_end": "[MEMORY] This run is ending. Decide whether anything durable happened in it.",
    "compaction": (
        "[MEMORY] This run is about to compact its history. Decide whether "
        "anything durable happened in it so far."
    ),
}

_WORKER_MEMORY_RECORD_TEXT = (
    "[Orbital memory] Your last turn is ending. Decide whether anything durable\n"
    "happened in it. Files this turn changed: {files}\n"
    "If so, update orbital/PROJECT_STATE.md so the next session starts from what is\n"
    "true after your work: overwrite stale lines in place, one fact per line, no\n"
    "dated history. Add one line to orbital/DECISIONS.md or orbital/LESSONS.md only\n"
    "if this turn settled a decision or taught a non-obvious lesson. Edit only those\n"
    "files, write only what you already know, then stop without further reply.\n"
    "If nothing durable happened, reply exactly: nothing to record"
)

_MISSING_FILE_NOTE = "(the file does not exist yet — write it if there is something to record)"


def classify_written_path(workspace: str, path) -> str | None:
    """``"memory"`` for one of the four Layer-1 files, ``"workspace"`` for a
    file outside ``orbital/``, None for anything else (other orbital/ files,
    empty or non-string paths). Relative paths resolve against the workspace
    the way the edit/write tools resolve them."""
    if not workspace or not isinstance(path, str) or not path.strip():
        return None
    raw = path.strip()
    full = raw if os.path.isabs(raw) else os.path.join(workspace, raw)
    full = os.path.normpath(full)
    orbital = os.path.normpath(os.path.join(workspace, "orbital"))
    if os.path.dirname(full) == orbital and os.path.basename(full) in _MEMORY_FILE_NAMES:
        return "memory"
    if full == orbital or full.startswith(orbital + os.sep):
        return None
    return "workspace"


def _read_memory_file(workspace: str, name: str) -> str | None:
    return read_text_lenient(os.path.join(workspace, "orbital", name))


def _last_line(text: str | None) -> str:
    if text is None:
        return _MISSING_FILE_NOTE
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return "(the file is empty)"


def build_memory_record_text(workspace: str, changed_files: list[str],
                             sub_agent_results: list[str], *,
                             trigger: str = "run_end") -> str:
    """The ``[MEMORY]`` row (spec 112 A6). Fixed wording; only the changed-file
    line (omitted when none), the sub-agent line and the file excerpts are
    filled from facts. ``trigger`` picks the opening sentence (run end vs
    before a compaction)."""
    lines = [_MEMORY_RECORD_OPENING.get(trigger, _MEMORY_RECORD_OPENING["run_end"])]
    if changed_files:
        lines.append("Files this run changed: " + ", ".join(changed_files))
    lines.append("Sub-agent results received: "
                 + (", ".join(sub_agent_results) if sub_agent_results else "none"))
    head = "\n".join(lines) + "\n\n" + _MEMORY_RECORD_BODY
    state = _read_memory_file(workspace, "PROJECT_STATE.md")
    state_text = state.rstrip("\n") if state is not None else _MISSING_FILE_NOTE
    if state is not None and not state.strip():
        state_text = "(the file is empty)"
    parts = [
        head,
        "--- orbital/PROJECT_STATE.md as on disk now ---",
        state_text,
    ]
    for name in ("DECISIONS.md", "LESSONS.md", "INDEX.md"):
        parts.append(f"--- last line of orbital/{name} ---")
        parts.append(_last_line(_read_memory_file(workspace, name)))
    return "\n".join(parts)


def build_worker_memory_record_text(changed_files: list[str]) -> str:
    """The pinned direct-chat worker's variant of the record row (spec 112
    A6): sent as the worker's next turn on its resumed session."""
    return _WORKER_MEMORY_RECORD_TEXT.format(
        files=", ".join(changed_files) if changed_files else "none")


def maybe_raise_unsandboxed_run(interceptor, tool_call: dict, result,
                                recent_activity, *,
                                queue_state: str = "chat") -> str | None:
    """Spec 117 D4 — after a tool result, raise the ``unsandboxed_run`` card
    if the sandbox blocked it. Returns the new pending id, or ``None``.

    The ONE post-execution check shared by the loop's tool-result path and
    ``AgentManager.approve()`` (which executes an approved call itself, off
    the loop) — keep both on this helper so they cannot diverge again (they
    did: an approved shell call under check_in never raised the card).

    ``result`` is a ``ToolResult``; a ``meta["sandbox_denial"]`` (the daemon's
    own denial detection, safe mode only) triggers ``interceptor.on_sandbox_
    denial``. ``recent_activity`` is a zero-arg callable, evaluated only when
    a card is actually raised. Never under a queue run (``queue_state ==
    "running"``) — there is no card to answer there. A hook that raises is
    logged and treated as "no card" so the caller's normal path continues.
    The caller pauses the session on a returned id.
    """
    meta = getattr(result, "meta", None) or {}
    denial = meta.get("sandbox_denial") if isinstance(meta, dict) else None
    if (not denial or interceptor is None or queue_state == "running"
            or not hasattr(interceptor, "on_sandbox_denial")):
        return None
    try:
        return interceptor.on_sandbox_denial(tool_call, denial, recent_activity())
    except Exception:
        logger.exception("on_sandbox_denial raised for tool_call_id=%s",
                         (tool_call or {}).get("id"))
        return None


def normalize_tool_call(tc_raw: dict) -> dict:
    """Normalize tool call from nested/flat/string-args formats."""
    if "function" in tc_raw:
        func = tc_raw["function"]
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        return {"id": tc_raw.get("id", ""), "name": func.get("name", ""), "arguments": args}
    args = tc_raw.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}
    return {"id": tc_raw.get("id", ""), "name": tc_raw.get("name", ""), "arguments": args}


class AgentLoop:
    """Core agent execution loop."""

    def __init__(
        self,
        session,
        provider,
        tool_registry,
        context_manager,
        interceptor=None,
        utility_provider=None,
        fallback_providers=None,
        auth_fallback_provider=None,
        max_iterations: int = 0,
        token_budget: int = 100_000_000,
        get_budget_config=None,
        on_session_end=None,
        on_session_end_refresh=None,
        project_dir: str | None = None,
        on_budget_event=None,
        on_credential_error=None,
        memory_record_enabled: bool = True,
        tool_guard=None,
        tool_call_cap: int | None = None,
        routing_project_id: str | None = None,
    ):
        self._session = session
        self._provider = provider
        self._tool_registry = tool_registry
        self._context_manager = context_manager
        # Terminal LLM failure of the most recent run (see run() reset).
        self.last_llm_error: Exception | None = None
        self._interceptor = interceptor
        self._utility_provider = utility_provider
        self._fallback_providers = fallback_providers or []
        # Spec 072 — the global-default rung, built wholesale from GLOBAL
        # settings (None when the project inherits the global key or global is
        # unusable). Held OUTSIDE the rotation list: it joins _all_providers
        # lazily, only after a 401/403 proves the project key dead, so
        # transient-rotation semantics stay byte-identical until then.
        self._auth_fallback_provider = auth_fallback_provider
        # Spec 140: set for trigger-started manager sessions only. Fixed for
        # the life of this loop (later messages and hot resumes keep it); a
        # project-scoped provider then routes by project, not by session.
        self._routing_project_id = routing_project_id or None
        self.bind_session_headers()
        self._max_iterations = max_iterations
        # PURE token safety-net cap (NOT budget-derived). Budget Piece 2 cut the
        # dollar→token derivation that used to feed this; the gate now only
        # protects against a runaway non-terminating loop. The single
        # dollar-budget enforcement point is the pre-flight guard (below).
        self._token_budget = token_budget
        # LIVE budget config resolver (Budget Piece 2). A zero-arg callable
        # returning the CURRENT project budget config dict (or None). Resolved
        # fresh on EVERY guard evaluation — never snapshotted (the B5 fix). When
        # None, the pre-flight guard is a no-op (e.g. lightweight test loops).
        self._get_budget_config = get_budget_config
        # Budget threshold/trip emitter sink (Budget Piece 2 — Task E). An
        # optional callable ``on_budget_event(payload: dict)`` that routes a
        # codes-only budget event through the relay/WS broadcast path. When None
        # (lightweight test loops, no relay wiring), the emitter is a no-op.
        self._on_budget_event = on_budget_event
        # Spec 082 §3.5 — credential-health sink. An optional
        # ``on_credential_error(status: int, message: str)`` invoked ONLY for
        # 401/402/403 on the PRIMARY provider: the "this credential is bad"
        # set (spec 072 D1), plus 402 because the 2026-09-03 incident was
        # exactly an OpenRouter credit pre-flight. Transient errors never
        # reach it, and nothing is acted on — the manager writes the card's
        # last_error so the card list can show it (spec 072 D2 stands: no
        # auto-disable, no auto-switch).
        self._on_credential_error = on_credential_error
        # Per-project token ledger root: the workspace, resolved to
        # {workspace}/orbital/ledger/usage.jsonl via ProjectPaths. When None,
        # ledger emission is skipped (loop runs unchanged). Also the project_dir
        # the pre-flight guard reads the ledger spend from.
        self._project_dir = project_dir
        # Debounced ``budget.spend_updated`` WS broadcaster (Budget Piece 3 —
        # Task P3-C). Fed the post-append window spend after each ledger append
        # (reusing the notify check's already-computed numbers — zero extra
        # spend() query when a limit is set). Routes through the SAME
        # ``on_budget_event`` sink as the notify events. ≥1s leading+trailing
        # debounce per project so a multi-turn burst collapses to at most two
        # frames, the trailing carrying the LAST append's numbers. A None sink
        # makes ``submit`` a silent no-op (lightweight test loops / no relay).
        from agent_os.budget.spend_broadcast import SpendBroadcaster
        self._spend_broadcaster = SpendBroadcaster(emit=on_budget_event)
        # Install the post-append hook at the LEDGER chokepoint rather than at
        # this loop's own append site. Sub-agent transports append to the same
        # ledger without going through the loop, so a loop-side notification
        # left their spend invisible to the UI until something else refreshed
        # it. Registered per workspace and removed in terminate().
        if project_dir is not None:
            from agent_os.budget import ledger as _ledger_mod
            _ledger_mod.register_append_hook(project_dir, self._on_ledger_append)
        self._running = False
        self._llm_failed = False
        self._on_session_end = on_session_end

        # Per-session cache-hit telemetry (Part 4D). Accumulated across every
        # run() of this loop (one loop == one session), logged at each run end.
        # The CONTEXT.md on-judgment-update decision is an empirical bet on the
        # prefix cache; this is the instrument that measures its impact.
        self._cache_read_tokens_total = 0
        self._prompt_input_tokens_total = 0
        self._llm_call_count = 0

        # Periodic state refresh callback (async callable(trigger_name: str))
        # Injected by agent_manager so loop.py stays decoupled from workspace_files.
        self._on_session_end_refresh = on_session_end_refresh
        # prepare() reports over-budget memory files here (spec 089 §4.2).
        try:
            context_manager._on_memory_over_budget = self._schedule_over_budget_edit
        except AttributeError:
            pass  # a context manager that cannot take the hook just never asks

        # Cancellation state (cancel_turn / terminate)
        self._task: asyncio.Task | None = None
        self._inflight_stream: asyncio.Task | None = None
        self._stream_accumulator: StreamAccumulator | None = None
        self._turn_cancelled: bool = False
        self._cancellation_marker_appended_this_turn: bool = False

        # In-flight refresh task (registered with cancel pathway)
        self._refresh_task: asyncio.Task | None = None

        # Spec 013 async checkpoint scheduler state. _refresh_task doubles as
        # the single-flight gate for background passes (the spec's
        # `_refresh_inflight`). Dirty = a trigger arrived while a pass was in
        # flight or debounced; consumed at the turn boundary or session end.
        #
        # These two stay per-loop, but they are only AUTHORITATIVE when this
        # loop has no project_dir. When it does, `_PROJECT_REFRESH` holds the
        # real gate — see `_refresh_gate_state`. The dirty bit is deliberately
        # per-loop either way: it records that THIS session still wants a pass,
        # and its own turn boundary picks it up once the project gate opens.
        self._refresh_dirty: bool = False
        self._last_merge_at: float = float("-inf")   # monotonic; -inf → first schedule always passes debounce

        # State refresh tracking
        self._turns_since_last_update: int = 0  # resets after each refresh
        self._last_refresh_iteration: int = 0   # loop iteration# of last refresh
        self._current_iteration: int = 0        # current loop iteration (updated at top of loop)
        # Outcome of the last background pass (run_session_end_routine's
        # vocabulary: llm_merged / backstop_only / no_delta /
        # skipped_idempotent, plus "failed" when the callback raised). Pushed
        # to the context manager so the hygiene flag + status line can tell
        # the agent whether the async consolidation actually worked.
        self._last_refresh_outcome: str | None = None

        # Spec 112 A6 — end-of-run memory record turn. ``memory_record_enabled``
        # is the construction-time switch (spec 124's Memory session passes
        # False; a session whose origin is "memory" is skipped regardless).
        # The per-run fields are reset at every run() entry and after each
        # compaction (which begins a new snapshot segment).
        self._memory_record_enabled = memory_record_enabled
        self._run_changed_files: list[str] = []
        self._run_memory_written: bool = False
        self._memory_record_done: bool = False
        self._memory_record_active: bool = False

        # Spec 124 §3.7 — the Memory session's loop. ``tool_guard(tc)`` is a
        # deterministic check run before any interceptor: an error string is
        # handed back as the tool result and the tool never executes (the
        # Memory session confines edit/write to orbital/*.md with it).
        # ``tool_call_cap`` bounds the executed tool calls of ONE run; when
        # it is reached the rest of the batch is skipped and the run ends.
        # ``last_run_stats`` is what the pass reports (calls, tokens, exit).
        self._tool_guard = tool_guard
        self._tool_call_cap = tool_call_cap
        self._run_tool_calls: int = 0
        self.last_run_stats: dict = self._fresh_run_stats()
        # Spec 124 §3.10 — the PROJECT_STATE ids this run has seen, bound
        # per snapshot segment for the write path's conservation guard.
        self._seen_ids_token = None

        # Queue integration (Phase 2 — exit_reason; Phase 3 — queue_state)
        # Reset to defaults at the start of every run().
        self._exit_reason: str = "text"
        self._exit_summary: str | None = None
        self._exit_block_reason: str | None = None
        # "chat" (default) preserves legacy approval-pause; "running" makes
        # the dispatcher treat approval-required tool calls as blocks. The
        # dispatcher sets this before each run via agent_manager helpers.
        self._queue_state: str = "chat"

    @property
    def is_running(self) -> bool:
        return self._running

    def get_completion_state(self) -> "tuple[str, str | None, str | None]":
        """Return how the last run ended: (exit_reason, summary, block_reason).

        ``exit_reason`` is one of ``complete`` | ``blocked`` | ``cancelled`` |
        ``budget_blocked`` | ``text`` (text-only — no completion tool called).
        ``budget_blocked`` (Budget Piece 2) is a first-class exit distinct from
        the others: the pre-flight budget guard tripped, so the loop stopped
        BEFORE spending again. The queue dispatcher routes it explicitly (no
        corrective turn, no contract violation). This accessor formalizes what
        was previously a raw read of the private ``_exit_*`` attributes.
        """
        return (self._exit_reason, self._exit_summary, self._exit_block_reason)

    # ------------------------------------------------------------------
    # Diagnostics (permanent, always-on; see TASK-loop-diagnostic-instrumentation)
    # ------------------------------------------------------------------
    def _diag(self, event: str, **fields) -> None:
        """Emit one structured diagnostic record to the ``agent.diag`` logger.

        Observation only — never mutates session/prompt/cache state. The
        structured payload rides on the LogRecord as ``record.diag`` (via
        ``extra=``) so consumers assert on fields, not free text. project_id is
        not available at the loop layer (it lives in agent_manager); session_id
        / session_uuid identify the run.
        """
        payload = {
            "event": event,
            "session_id": getattr(self._session, "session_id", None),
            "session_uuid": getattr(self._session, "session_uuid", None),
            **fields,
        }
        # Render the payload into the message so it is greppable as a single
        # line in plain daemon.log text (`grep 'agent.diag'`), and ALSO attach
        # it structured on the record (`record.diag`) for programmatic/test use.
        _diag_logger.info(
            "agent.diag %s", json.dumps(payload, default=str, ensure_ascii=False),
            extra={"diag": payload},
        )

    def _drain_queued_user_messages(self) -> list:
        """Pop queued user messages, append each to the session, and notify the
        manager of the drained nonces so it can clear the FE "Waiting…" lines.

        Returns the drained items (empty list when the queue was empty), so the
        caller can gate its per-turn resets on ``if queued:`` exactly as before.

        The notification goes through the optional ``session.on_queue_drained``
        observer (mirrors ``on_append``/``on_stream``; wired by the manager).
        It is fault-isolated: a raising observer must NEVER break the drain, and
        an unset observer is a no-op so existing behavior is unchanged. Only
        non-empty nonces are reported (no FE bubble exists for a None nonce).
        """
        queued = self._session.pop_queued_messages()
        if not queued:
            return queued
        drained_nonces: list[str] = []
        for q_item in queued:
            if isinstance(q_item, tuple):
                q_content, q_nonce = q_item
            else:
                q_content, q_nonce = q_item, None
            q_msg: dict = {
                "role": "user",
                "content": q_content,
                "source": "user",
            }
            if q_nonce:
                q_msg["nonce"] = q_nonce
                drained_nonces.append(q_nonce)
            self._session.append(q_msg)
        if drained_nonces:
            cb = getattr(self._session, "on_queue_drained", None)
            if cb is not None:
                try:
                    cb(drained_nonces)
                except Exception:
                    logger.exception("on_queue_drained callback failed")
        return queued

    def _note_exit(self, name: str, iteration: int) -> None:
        """Record the named loop-exit/continue path taken this iteration."""
        self._loop_exit_path = name
        self._diag("loop_exit", iteration=iteration, exit=name)

    def _budget_tripped(self):
        """Evaluate the pre-flight budget guard. Returns a tripped
        ``BudgetDecision`` (truthy → trip) or None.

        THE single dollar-budget enforcement point lives in
        ``agent_os/budget/guard.py``; this method only WIRES it into the loop's
        between-LLM-call chokepoint with LIVE config (resolved on every call via
        ``self._get_budget_config`` — never a session-start snapshot). No-op when
        no resolver or no ledger root was plumbed.

        Convertibility: if some spend can't be converted to the limit's currency
        (unknown FX pair), the convertible total is the spend (see the guard
        module docstring). Spend-QUERY failures (ledger I/O) fail OPEN in the
        guard with a logged warning; a malformed spend() return is a contract
        bug and RAISES out of the guard (deliberately loud — never silently
        masked as $0).
        """
        if self._get_budget_config is None or self._project_dir is None:
            return None
        try:
            cfg = self._get_budget_config()
        except Exception:  # noqa: BLE001 — a config read must not crash the loop
            logger.warning("budget config resolver raised; skipping guard",
                           exc_info=True)
            return None
        from agent_os.budget.guard import evaluate_budget
        fx_rates = (cfg or {}).get("fx_rates") if isinstance(cfg, dict) else None
        decision = evaluate_budget(
            self._project_dir, cfg, fx_rates=fx_rates,
        )
        return decision if decision.tripped else None

    def _trip_budget(self, decision, iteration: int) -> None:
        """Record a budget trip: set the first-class ``budget_blocked`` exit
        reason and append a structured timeline event (codes/numbers only).

        Uses the session's typed-event affordance (a structured ``append`` row
        with ``source="budget"`` + a ``payload`` dict) — the same shape the
        queue signals use — NOT an English ``append_system`` sentence, per the
        i18n rule (no display strings; the client renders the message). The
        ``budget_action`` ("pause" | "stop") is normalized by the guard. It is
        carried on ``_exit_block_reason`` so the dispatcher can route pause vs
        stop without re-reading config (``get_completion_state`` surfaces it).

        Also the SECOND trip-emission site (coordinator ruling, Task E): emits
        the ``budget.exhausted`` relay event through the same notify module and
        same marker as the crossing-append site, so the trip push fires at
        whichever happens first — crossing append or guard trip — exactly once
        per (project, window). This covers trips with NO new append (limit
        lowered below current spend; fresh run on an already-over project with
        an absent/stale marker).
        """
        self._exit_reason = "budget_blocked"
        self._exit_block_reason = decision.action
        self._session.append({
            "role": "system",
            "source": "budget",
            "event": "budget_blocked",
            "payload": {
                "action": decision.action,
                "window": decision.window,
                "spend": decision.spend,
                "limit": decision.limit,
                "currency": decision.currency,
            },
        })
        self._emit_trip_notify(decision)
        self._note_exit("budget_blocked", iteration)

    def _emit_trip_notify(self, decision) -> None:
        """Emit the trip relay event from the guard-trip site (dedup'd).

        Delegates to ``budget.notify.maybe_emit_trip`` with the tripped
        ``BudgetDecision``'s own numbers (no spend re-query). The shared marker
        key makes this a no-op when the crossing-append site already fired this
        window. Never raises. No-op when the emitter sink or project_dir was
        not plumbed (lightweight test loops).
        """
        if self._on_budget_event is None or self._project_dir is None:
            return
        try:
            # anchor_ts only matters for the "total" window's identity; read it
            # from the live config (same resolver the guard used).
            anchor_ts = None
            if self._get_budget_config is not None:
                try:
                    cfg = self._get_budget_config()
                    if isinstance(cfg, dict):
                        anchor_ts = cfg.get("budget_anchor_ts")
                except Exception as e:  # noqa: BLE001 — config read must not block emit
                    # Degrades the "total" window identity to epoch-keyed, which
                    # can re-fire one duplicate trip push — observable, not silent.
                    logger.warning(
                        "budget notify: anchor_ts read failed; using None: %s",
                        e, exc_info=True,
                    )
            from agent_os.budget.notify import maybe_emit_trip
            maybe_emit_trip(
                self._project_dir,
                window=decision.window,
                spend=decision.spend,
                limit=decision.limit,
                currency=decision.currency,
                anchor_ts=anchor_ts,
                emit=self._on_budget_event,
            )
        except Exception:  # noqa: BLE001 — never break the loop on notify errors
            logger.warning(
                "budget trip notify failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )

    def _emit_ledger_event(self, active_provider, usage) -> None:
        """Append one token-ledger event for a management LLM response.

        ``active_provider`` is the provider that ACTUALLY served this response
        (post-rotation), so its ``provider`` / ``model`` / ``sdk`` give correct
        attribution. Usage is normalized into the four disjoint token fields via
        the ONLY normalization entry point (``ProviderSemantics.from_sdk`` +
        ``normalize_usage``).

        No-ops when ``project_dir`` was not plumbed. Never raises: this is wrapped
        in its own guard AND ``append_event`` is itself resilient, so a ledger
        failure cannot crash or alter the agent loop. The legacy budget path is
        completely independent of this method.
        """
        if self._project_dir is None:
            return
        try:
            # Imported lazily: budget.normalize -> providers.types pulls in the
            # agent package __init__ (which eagerly imports this module), so a
            # module-level import here would be a circular import.
            from agent_os.budget.normalize import (
                ProviderSemantics,
                normalize_usage,
            )

            sdk = getattr(active_provider, "sdk", None)
            semantics = ProviderSemantics.from_sdk(sdk)
            normalized = normalize_usage(usage, semantics)
            append_event(
                self._project_dir,
                LedgerEvent(
                    session_id=getattr(self._session, "session_id", "") or "",
                    source=SOURCE_MANAGEMENT,
                    provider=getattr(active_provider, "provider", "unknown"),
                    model=getattr(active_provider, "model", "unknown"),
                    usage=normalized,
                ),
            )
        except Exception:  # noqa: BLE001 — never break the loop on ledger errors
            logger.warning(
                "Token-ledger emission failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )
            return
        # NOTE: the threshold/trip notify + spend broadcast are NOT invoked
        # here. Ledger-append is still the CAUSAL moment a spend crosses
        # 80%/100% (Budget Piece 2 — Task E), but the trigger now lives at the
        # ledger chokepoint (``budget.ledger.append_event`` → the hook this loop
        # registered in __init__ → ``_on_ledger_append``), so the sub-agent
        # transports' appends fire it too. The append above lands in that same
        # chokepoint, so this path is unchanged in effect.

    def _record_prompt_tokens(self, active_provider, usage, tool_schemas=None) -> None:
        """Hand the provider-reported prompt size to the context manager.

        should_compact() then measures the prompt by what the provider billed
        rather than by the ``len/4`` estimate (spec 086 §8.1). The prompt is
        the three disjoint input fields of the normalized usage — the same sum
        the composer's context meter reads from the ledger
        (``budget.ledger.last_context_usage``) — so the compaction mark the UI
        draws and the trigger it predicts share one measure.

        No usage, a zero count, or an sdk without known usage semantics leaves
        prepare()'s estimate in place. Stand-in context managers without the
        hook are skipped. ``tool_schemas`` rode the request outside the
        messages; the provider counted them, so their estimate goes along for
        the context manager's estimate/real ratio.
        """
        record = getattr(self._context_manager, "record_prompt_tokens", None)
        if record is None or not usage:
            return
        # Lazy for the same circular-import reason as _emit_ledger_event.
        from agent_os.budget.normalize import ProviderSemantics, normalize_usage

        try:
            n = normalize_usage(
                usage,
                ProviderSemantics.from_sdk(getattr(active_provider, "sdk", None)),
            )
        except (TypeError, ValueError):
            return
        prompt = n.uncached_input + n.cache_read + n.cache_write
        if prompt > 0:
            try:
                overhead = len(json.dumps(tool_schemas)) / 4 if tool_schemas else 0.0
            except (TypeError, ValueError):
                overhead = 0.0
            record(prompt, overhead_estimate=overhead)

    def _on_ledger_append(self, project_dir: str, event) -> None:
        """Post-append hook registered in ``budget.ledger`` for this workspace.

        Fires for EVERY appender — this loop's management responses AND both
        sub-agent transports — which is what makes sub-agent spend refresh the
        settings breakdown instead of silently going stale until the next
        management append.

        Threshold/trip notification stays MANAGEMENT-ONLY: the limit governs
        management spend exclusively (guard isolation, the P3-A pinned
        regression passes ``sources=["management"]``), so a sub-agent append can
        never move a crossing — running the detector on it would only buy an
        extra spend() query. The spend BROADCAST runs for every source.

        Never raises (``_check_budget_notify`` is fully guarded) and never
        blocks (``submit`` is a synchronous debounce).
        """
        if project_dir != self._project_dir:
            return
        self._check_budget_notify(
            notify=getattr(event, "source", None) == SOURCE_MANAGEMENT,
        )

    def _check_budget_notify(self, *, notify: bool = True) -> None:
        """Run the budget threshold/trip emitter after a ledger append, then
        feed the debounced ``budget.spend_updated`` broadcast (Budget Piece 3).

        Reads the LIVE budget config (same resolver the pre-flight guard uses)
        and delegates to ``budget.notify.maybe_emit_budget_event``, which emits
        AT MOST ONCE per (project, window) for the 80% threshold and the trip.
        It now also RETURNS the post-append management-spend snapshot it computed
        (``window``/``spend``/``limit``/``currency`` in ``budget_currency``) so
        the spend_updated broadcast reuses it — ZERO additional spend() query per
        append when a limit is set. When no limit is configured the notify path
        runs no query (returns None); the corner still shows spend, so we run ONE
        management-only query here for the broadcast. A spend/window-math failure
        skips the broadcast silently (already logged in the notify module).

        ``notify=False`` (a SUB-AGENT append) skips the threshold/trip detector
        entirely — sub-agent spend cannot cross a management-only limit — and
        goes straight to the single management-only spend query the broadcast
        needs, so the settings breakdown still refreshes.

        Never raises. No-op when ``on_budget_event``/``get_budget_config``/
        ``project_dir`` were not plumbed (e.g. lightweight test loops).
        """
        if (self._on_budget_event is None
                or self._get_budget_config is None
                or self._project_dir is None):
            return
        try:
            cfg = self._get_budget_config()
            fx_rates = (cfg or {}).get("fx_rates") if isinstance(cfg, dict) else None
            snapshot = None
            if notify:
                from agent_os.budget.notify import maybe_emit_budget_event
                snapshot = maybe_emit_budget_event(
                    self._project_dir,
                    cfg,
                    emit=self._on_budget_event,
                    fx_rates=fx_rates,
                )
        except Exception:  # noqa: BLE001 — never break the loop on notify errors
            logger.warning(
                "budget threshold/trip notify failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )
            return
        # Feed the debounced spend_updated broadcast. Wrapped in its own guard so
        # a broadcast failure can never affect the notify check above nor the
        # ledger append nor the loop.
        try:
            if snapshot is None:
                # No limit configured, a sub-agent append (detector skipped) or
                # a notify-side failure → run one management-only spend query so
                # the corner/breakdown still refresh.
                snapshot = self._compute_spend_snapshot(cfg, fx_rates)
            if snapshot is not None:
                self._spend_broadcaster.submit(
                    window=snapshot["window"],
                    spend=snapshot["spend"],
                    limit=snapshot["limit"],
                    currency=snapshot["currency"],
                )
        except Exception:  # noqa: BLE001 — never break the loop on broadcast errors
            logger.warning(
                "budget spend_updated broadcast failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )

    def _compute_spend_snapshot(self, cfg, fx_rates) -> "dict | None":
        """Compute the post-append management-spend snapshot for the
        ``budget.spend_updated`` broadcast when the notify check ran no query.

        That is the case when no limit is configured (``maybe_emit_budget_event``
        returns None early) and when the append came from a SUB-AGENT (the
        detector is skipped — see ``_check_budget_notify``). The corner shows
        spend either way, so we run exactly ONE management-only query here,
        converted into ``budget_currency`` (consistent with the meter). Returns
        ``{"window", "spend", "limit", "currency"}`` or None when spend cannot be
        computed (guard-style failure → skip the broadcast silently with a
        warning).
        """
        cfg = cfg or {}
        window = cfg.get("budget_period") or "daily"
        currency = cfg.get("budget_currency") or "USD"
        anchor_ts = cfg.get("budget_anchor_ts")
        # The limit is reported in ``budget_currency`` (same convention as the
        # notify snapshot). None/0/negative → no limit, exactly as before.
        limit = cfg.get("budget_limit_usd")
        try:
            limit_val = float(limit) if limit is not None else None
        except (TypeError, ValueError):
            limit_val = None
        if limit_val is not None and limit_val <= 0:
            limit_val = None
        try:
            from agent_os.budget.ledger import spend as _spend
            result = _spend(
                self._project_dir,
                window,
                target_currency=currency,
                fx_rates=fx_rates or {},
                anchor_ts=anchor_ts,
                # Management-only, consistent with the limit-governed meter
                # (the limit only governs management spend; sub-agent display
                # is a separate block).
                sources=["management"],
            )
            spend_amount = float(result["converted_total"]["amount"])
        except Exception:  # noqa: BLE001 — guard-style: skip the broadcast
            logger.warning(
                "budget spend_updated: spend query failed (session=%s); "
                "skipping broadcast.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )
            return None
        return {
            "window": window,
            "spend": spend_amount,
            "limit": limit_val,
            "currency": currency,
        }

    def _flush_spend_broadcast(self) -> None:
        """Emit the debouncer's pending spend payload at an idle boundary.

        Called at the end of every run (turn over → the agent is idle) and from
        ``terminate()``. Synchronous and self-guarded: it runs inside run()'s
        ``finally``, which may execute while a CancelledError is propagating, so
        it must never raise and never await.
        """
        try:
            self._spend_broadcaster.flush()
        except Exception:  # noqa: BLE001 — an idle-settle failure must not
            # disturb loop teardown; the next append re-broadcasts anyway.
            logger.warning(
                "budget spend_updated idle flush failed (session=%s); continuing.",
                getattr(self._session, "session_id", "?"),
                exc_info=True,
            )

    def _current_session_id(self):
        """The conversation id gateways with a registry ``session_header``
        receive (OpenCode Go's ``x-opencode-session``). The session uuid is
        unique across projects; the F1 chat id is the fallback. Read live
        from ``self._session`` so a session swap on this loop follows."""
        session = self._session
        return (getattr(session, "session_uuid", None)
                or getattr(session, "session_id", None))

    def bind_session_headers(self) -> None:
        """Point every provider this loop owns at its live session id.
        Called from __init__ and again whenever the manager rebuilds the
        provider set on a hot resume (the new clients start unbound).
        A trigger loop also binds its routing project (spec 140); a loop
        without one never touches it, so a worker sharing its parent's
        client cannot clear the parent's project key."""
        providers = [self._provider, self._utility_provider,
                     self._auth_fallback_provider, *self._fallback_providers]
        project_id = getattr(self, "_routing_project_id", None)
        for prov in providers:
            bind = getattr(prov, "bind_session_id", None)
            if callable(bind):
                bind(self._current_session_id)
            if project_id:
                bind_project = getattr(prov, "bind_routing_project", None)
                if callable(bind_project):
                    bind_project(project_id)

    async def _stream_response(self, context, tool_schemas) -> LLMResponse:
        """Stream LLM response from the primary provider.

        Convenience wrapper preserved for tests that monkey-patch this
        method. Delegates to the unified internal helper.
        """
        return await self._stream_response_internal(
            self._provider, context, tool_schemas,
        )

    async def _stream_response_with(self, provider, context, tool_schemas) -> LLMResponse:
        """Stream LLM response using a specific provider (rotation).

        Routes the primary provider through _stream_response so any test
        monkey-patches on that method continue to fire; otherwise goes
        directly to the unified internal helper. Both paths share the
        same accumulator-publishing semantics so cancel_turn() can rely
        on a consistent contract regardless of which provider is active.
        """
        if provider is self._provider:
            return await self._stream_response(context, tool_schemas)
        return await self._stream_response_internal(
            provider, context, tool_schemas,
        )

    async def _stream_response_internal(self, provider, context, tool_schemas) -> LLMResponse:
        """Single source-of-truth streaming implementation.

        Publishes the accumulator on self._stream_accumulator before any
        await so cancel_turn() can inspect partial state. The async
        iteration is wrapped in a try/finally so CancelledError reaches
        the underlying generator's cleanup path (its httpx context).
        """
        accumulator = StreamAccumulator()
        self._stream_accumulator = accumulator
        try:
            async for chunk in provider.stream(context, tools=tool_schemas):
                accumulator.add(chunk)
                self._session.notify_stream(chunk)
            return accumulator.finalize()
        finally:
            # Do not clear self._stream_accumulator here — cancel_turn()
            # may still need to read it on its way out. The next turn
            # will reassign it at the top of the next call.
            pass

    @staticmethod
    def _rotate_provider(providers, current_idx, cooldowns):
        """Find next available provider not in cooldown. Returns index or None."""
        now = time.monotonic()
        n = len(providers)
        for offset in range(1, n):
            candidate = (current_idx + offset) % n
            if now >= cooldowns.get(candidate, 0):
                return candidate
        return None

    async def run(self) -> None:
        """Main loop entry point.

        Takes no initial message. Bug #59: the user's row is persisted by
        whoever injects it, through ``agent_os.agent.session.persist_user_row``,
        *before* the caller enters the start window — so ``run()`` always
        resumes from a session that already contains the message rather than
        carrying it in an argument that dies with an un-executed coroutine.
        There is exactly one writer, and it is not here.
        """
        if self._running:
            # Re-entrancy guard: a second concurrent run() on the same loop would
            # write two interleaved turns into one session. Reject rather than
            # silently start a duplicate loop. (Defense-in-depth alongside the
            # single-active-slot discipline.)
            raise RuntimeError(
                f"AgentLoop.run() re-entered while already running "
                f"(session {getattr(self._session, 'session_id', '?')})"
            )
        self._running = True
        # Re-arm the ledger chokepoint hook: a prior terminate() released it, and
        # this same loop object is what a hot resume (_start_loop) runs again.
        # Registering at every run start also makes the RUNNING loop the one that
        # broadcasts when several loops share a workspace. Idempotent.
        if self._project_dir is not None:
            from agent_os.budget import ledger as _ledger_mod
            _ledger_mod.register_append_hook(
                self._project_dir, self._on_ledger_append,
            )
        # Reset queue exit-reason at the start of every run so the dispatcher
        # always reads a fresh value scoped to this run.
        self._exit_reason = "text"
        self._exit_summary = None
        self._exit_block_reason = None
        # Terminal LLM failure of THIS run (ABORT or retries-exhausted). The
        # loop handles those internally (session system row + normal exit), so
        # _on_loop_done reads this to broadcast a classified error the UI can
        # show — without it the failure is invisible outside the JSONL.
        self.last_llm_error = None
        # Capture our own task handle so terminate() can cancel us.
        self._task = asyncio.current_task()
        # Every run() invocation begins a fresh turn boundary: clear the
        # sticky cancel-marker flag so a Stop click on this run writes a
        # new marker rather than no-opping on the previous run's flag.
        self._cancellation_marker_appended_this_turn = False
        # Diagnostics: track exit path + appended-row delta for this run.
        self._loop_exit_path = None
        self.last_run_stats = self._fresh_run_stats()
        self._run_tool_calls = 0
        rows_at_run_start = None  # set before the while; guarded in finally
        iteration = 0  # bound here so the except-backstop can always report it
        # Spec 112: a run is one snapshot segment. The Layer-1 rendering and
        # the media-pruning horizon are taken here, before the first build,
        # and again right after a compaction.
        self._begin_snapshot_segment()
        try:
            # Resolve pending if not resuming from approval pause
            if self._session._paused_for_approval:
                self._session._paused_for_approval = False  # reset here, not in resume()
            else:
                self._session.resolve_pending_tool_calls()

            # NOTE (bug #59): no initial-message append here. The injector
            # already persisted the user row via persist_user_row() before it
            # entered the start window. Re-adding an append here re-introduces
            # the double-write this fix removed — see
            # tests/unit/test_message_write_ahead.py.

            # Reset per-run state on all tools (e.g. send counters)
            self._tool_registry.reset_run_state()

            # Reset loop guards
            iteration = 0
            cumulative_tokens = 0
            action_hashes: deque = deque(maxlen=15)
            consecutive_overflows = 0
            llm_retries = 0
            empty_response_nudges = 0

            # Fallback provider rotation state
            _all_providers = [self._provider] + list(self._fallback_providers)
            _current_idx = 0
            _cooldowns: dict[int, float] = {}
            _COOLDOWN_SECONDS = 60.0
            _retries_on_current = 0
            # Provider indices that have answered at least once in THIS run.
            # Read by the 400 handler below to tell a malformed request from a
            # gateway hiccup. Per-index, so rotating to a fresh fallback does
            # not inherit the primary's proven-good standing.
            _answered: set[int] = set()

            # Ping-pong detection state
            _pair_hashes: deque = deque(maxlen=20)
            _prev_action_hash: str | None = None

            # Circuit breaker state (reset per run / new user message)
            error_tracker: dict[str, dict] = {}  # tool_name -> {"error": str, "count": int}
            blocked_tools: set[str] = set()

            # Diagnostics: row count at run start (the injector's user row is
            # already in the session — see the #59 note above) — the
            # run_terminal disposition compares against this to detect a turn
            # that appended nothing.
            rows_at_run_start = len(self._session._messages)

            while True:
                # Drain queued user messages before preparing context
                queued = self._drain_queued_user_messages()
                if queued:
                    # New user input ⇒ a brand new "turn" begins. Reset the
                    # cancellation marker flag so subsequent cancel_turn()
                    # calls write a fresh marker. (Safe to reset after the
                    # synchronous drain above — no cancel_turn can interleave
                    # without an await, and the appends never read this flag.)
                    self._cancellation_marker_appended_this_turn = False
                    # Reset circuit breaker on new user messages
                    error_tracker.clear()
                    blocked_tools.clear()

                # Check iteration cap (0 = unlimited)
                if self._max_iterations and iteration >= self._max_iterations:
                    self._session.append_system(
                        "Maximum iteration limit reached. Save your current state and stop."
                    )
                    self._note_exit("max_iter", iteration)
                    break

                # Check token budget (PURE non-budget safety-net cap — the
                # dollar→token derivation was cut in Budget Piece 2).
                if cumulative_tokens >= self._token_budget:
                    self._session.append_system(
                        "Token budget exceeded. Save your current state and stop."
                    )
                    self._note_exit("token_budget", iteration)
                    break

                # --- Pre-flight budget guard (Budget Piece 2) ---------------
                # THE single dollar-budget enforcement point. Evaluated BETWEEN
                # LLM calls (here at the top of every tool-loop iteration), with
                # LIVE config. On a trip the loop exits with the first-class
                # ``budget_blocked`` reason; the dispatcher routes it explicitly.
                # Sits before the LLM stream so a tripped budget spends nothing
                # more. Never interrupts an in-flight tool execution.
                _budget_decision = self._budget_tripped()
                if _budget_decision is not None:
                    self._trip_budget(_budget_decision, iteration)
                    break

                iteration += 1

                # --- Memory pass bookkeeping ---
                # Track current iteration for the checkpoint status line.
                self._current_iteration = iteration
                # Increment the per-refresh turn counter and expose it to the
                # context_manager so the agent sees accurate metadata.
                self._turns_since_last_update += 1
                if hasattr(self._context_manager, "_turns_since_last_update"):
                    self._context_manager._turns_since_last_update = self._turns_since_last_update

                # Pick up a coalesced/deferred checkpoint_state request. The
                # automatic trigger is not here: prepare() reports over-budget
                # files to _schedule_over_budget_edit on every call.
                self._maybe_consume_dirty()

                # Reset per-iteration cancellation flags. The
                # cancellation-marker flag is sticky across cancelled
                # iterations: it's only reset when a new user message has
                # been drained (in the queue-drain block above) — so a
                # cancel_turn() called twice with no intervening user
                # message is idempotent.
                self._turn_cancelled = False

                # Diagnostics: row count at the start of this iteration's work,
                # to classify the bottom-of-loop continue as tool_continue
                # (rows grew) vs fell_through (nothing appended).
                rows_at_iter_start = len(self._session._messages)

                # Prepare context
                context = self._context_manager.prepare()

                # Get tool schemas
                tool_schemas = self._tool_registry.schemas()

                # Stream LLM response (using active provider from rotation).
                # Wrap in asyncio.Task so cancel_turn() can interrupt it.
                active_provider = _all_providers[_current_idx]
                self._inflight_stream = asyncio.create_task(
                    self._stream_response_with(
                        active_provider, context, tool_schemas,
                    ),
                    name=f"stream-{self._session.session_uuid}-{iteration}",
                )
                try:
                    response = await self._inflight_stream
                    consecutive_overflows = 0
                    llm_retries = 0
                    _retries_on_current = 0
                    _answered.add(_current_idx)
                except asyncio.CancelledError:
                    # cancel_turn() invoked while streaming. Marker + cost
                    # debit handled by cancel_turn() before propagating.
                    self._inflight_stream = None
                    if self._session.is_stopped():
                        # terminate() also fired — let outer cleanup run.
                        raise
                    # Bare cancel_turn: exit the while loop. The finally
                    # block + _on_loop_done drain any queued messages and
                    # hot-resume a fresh loop task if needed, preserving
                    # sub-agents / browser pages / sandbox state.
                    self._note_exit("cancel", iteration)
                    break
                except ContextOverflowError:
                    self._inflight_stream = None
                    consecutive_overflows += 1
                    if consecutive_overflows >= 3:
                        self._session.append_system(
                            "Context overflow cannot be resolved. Save state and stop."
                        )
                        self._note_exit("context_overflow", iteration)
                        break
                    self._context_manager.reduce_window(0.5)
                    continue
                except LLMError as e:
                    self._inflight_stream = None
                    category = e.category

                    # Card health (spec 082 §3.5). Scoped to the PRIMARY rung:
                    # a fallback's rejection says nothing about the card this
                    # project points at, and stamping it there would paint a
                    # working card red. Fault-isolated — a health write must
                    # never turn an LLM error into a crashed loop.
                    if (self._on_credential_error is not None
                            and _current_idx == 0
                            and e.status_code in (401, 402, 403)):
                        try:
                            self._on_credential_error(e.status_code, e.message)
                        except Exception:
                            logger.warning(
                                "on_credential_error callback failed",
                                exc_info=True,
                            )

                    # A 400 from a provider that has ALREADY answered in this
                    # run is not a malformed request — it is an upstream
                    # hiccup the gateway mislabelled as a client error.
                    # Observed 2026-08-19: OpenCode Go answered one mid-session
                    # call with "[unsupported_tool_schema] ...
                    # (tool_count_limit)" for a tool array that had just worked
                    # on 30 consecutive calls; the identical payload replayed
                    # 200 seven times afterwards. Aborting there threw away
                    # eight iterations of work a single retry would have saved.
                    # Structural 400s (bad tool schema, unsupported param,
                    # wrong model) are deterministic and fail a provider's
                    # FIRST call, so a cold 400 stays fatal and surfaces
                    # instantly. Scoped to 400: a 401/403 never self-heals.
                    if (
                        category == ErrorCategory.ABORT
                        and e.status_code == 400
                        and _current_idx in _answered
                    ):
                        category = ErrorCategory.RETRY

                    # Auth fallback (Spec 072): a 401/403 means THIS key is
                    # dead, not the run. The "never self-heals" lesson above
                    # is about retrying the SAME key — rotating to a
                    # different key is exactly the self-heal that works.
                    # Infinite-cooldown every rung sharing the rejected key,
                    # lazily append the global-default rung (None when the
                    # project inherits the global key or global is unusable),
                    # and rotate. From that moment the rung is a uniform
                    # rotation member (transient cooldowns, the per-index
                    # _answered 400-guard). Scoped strictly to 401/403 —
                    # 400 stays cold-ABORT exactly as above. Nothing is
                    # persisted: the next run tries the project key fresh.
                    _auth_abort_note = None
                    if (
                        category == ErrorCategory.ABORT
                        and e.status_code in (401, 403)
                    ):
                        dead_key = getattr(
                            _all_providers[_current_idx], "api_key", None,
                        )
                        for i, p in enumerate(_all_providers):
                            if getattr(p, "api_key", None) == dead_key:
                                _cooldowns[i] = float("inf")
                        if (
                            self._auth_fallback_provider is not None
                            and self._auth_fallback_provider not in _all_providers
                        ):
                            _all_providers.append(self._auth_fallback_provider)
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            new_provider = _all_providers[rotated]
                            model_name = new_provider.model
                            provider_name = getattr(
                                new_provider, "provider", "unknown",
                            )
                            logger.info(
                                "API key rejected (HTTP %s); rotating to %s/%s",
                                e.status_code, provider_name, model_name,
                            )
                            self._session.append_meta(
                                "model_swap",
                                provider=provider_name,
                                model=model_name,
                                sdk=getattr(new_provider, "sdk", "unknown"),
                                reason=(
                                    "auth: project API key rejected "
                                    f"(HTTP {e.status_code})"
                                ),
                            )
                            self._session.append_system(
                                f"This project's API key was rejected "
                                f"(HTTP {e.status_code}). Using the global "
                                f"default for this run: {provider_name} · "
                                f"{model_name}. The project key was not "
                                f"changed — fix or replace it in Project "
                                f"Settings."
                            )
                            continue
                        # No different-key rung left: fall through to ABORT
                        # with a message naming the global-default status.
                        if self._auth_fallback_provider is None:
                            _auth_abort_note = (
                                "The API key was rejected and the global "
                                "default is not available as a separate "
                                "fallback (not configured, or the same key)."
                            )
                        else:
                            _auth_abort_note = (
                                "The API key was rejected and the global "
                                "default was also rejected."
                            )

                    # Abort: non-recoverable errors (401, 403, 400)
                    if category == ErrorCategory.ABORT:
                        self._llm_failed = True
                        self.last_llm_error = e
                        abort_msg = f"LLM error (non-recoverable): {e.message}."
                        if _auth_abort_note:
                            abort_msg += f" {_auth_abort_note}"
                        self._session.append_system(abort_msg + " Stopping.")
                        self._note_exit("llm_error", iteration)
                        break

                    # Transient: rotate to next fallback provider
                    if category == ErrorCategory.TRANSIENT:
                        _cooldowns[_current_idx] = time.monotonic() + _COOLDOWN_SECONDS
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            new_provider = _all_providers[rotated]
                            model_name = new_provider.model
                            logger.info("Rotating to fallback model: %s", model_name)
                            self._session.append_meta(
                                "model_swap",
                                provider=getattr(new_provider, "provider", "unknown"),
                                model=model_name,
                                sdk=getattr(new_provider, "sdk", "unknown"),
                                reason=f"transient: {e.message}",
                            )
                            self._session.append_system(
                                f"Primary model unavailable ({e.message}). "
                                f"Switching to {model_name}."
                            )
                            continue

                    # Retry same provider (500, or transient with no fallback)
                    _retries_on_current += 1
                    llm_retries += 1
                    if _retries_on_current >= 3:
                        # Exhausted retries on current — try rotating
                        rotated = self._rotate_provider(
                            _all_providers, _current_idx, _cooldowns,
                        )
                        if rotated is not None:
                            _current_idx = rotated
                            _retries_on_current = 0
                            new_provider = _all_providers[rotated]
                            self._session.append_meta(
                                "model_swap",
                                provider=getattr(new_provider, "provider", "unknown"),
                                model=new_provider.model,
                                sdk=getattr(new_provider, "sdk", "unknown"),
                                reason=f"retries_exhausted: {e.message}",
                            )
                            continue
                        # All providers exhausted
                        self._llm_failed = True
                        self.last_llm_error = e
                        self._session.append_system(
                            f"LLM error after {llm_retries} retries: "
                            f"{e.message}. Stopping."
                        )
                        self._note_exit("llm_error", iteration)
                        break
                    await asyncio.sleep(2 ** _retries_on_current)
                    continue

                # Successful stream completion — clear inflight task handle.
                self._inflight_stream = None

                # Diagnostics (Point 1): per-iteration finalized response shape,
                # logged BEFORE any branch decision. `final_usage_chunk_seen`
                # is the explicit named signal for the "no CACHE_AUDIT" anomaly.
                self._diag(
                    "response_shape",
                    iteration=iteration,
                    text_len=len(response.text or ""),
                    tool_call_names=[
                        normalize_tool_call(tc)["name"]
                        for tc in (response.tool_calls or [])
                    ],
                    usage_present=bool(
                        response.usage
                        and (response.usage.input_tokens or response.usage.output_tokens)
                    ),
                    final_usage_chunk_seen=getattr(
                        response, "final_usage_chunk_seen", False
                    ),
                    finish_reason=response.finish_reason,
                )

                # Track token usage
                if response.usage:
                    cumulative_tokens += response.usage.input_tokens + response.usage.output_tokens
                    # Per-session cache-hit aggregation (Part 4D)
                    self._prompt_input_tokens_total += response.usage.input_tokens
                    self._cache_read_tokens_total += response.usage.cache_read_tokens
                    self._llm_call_count += 1
                    stats = self.last_run_stats
                    stats["llm_calls"] += 1
                    stats["input_tokens"] += response.usage.input_tokens
                    stats["cached_tokens"] += response.usage.cache_read_tokens
                    stats["output_tokens"] += response.usage.output_tokens

                # Token ledger (Budget Piece 1): append one disjoint-token event
                # per management LLM response. Attribution uses `active_provider`
                # — the post-rotation provider that ACTUALLY served THIS response
                # (re-read at the top of every iteration from _all_providers[
                # _current_idx]) — never the project's configured model. The
                # ledger is the SINGLE source of recorded spend; the pre-flight
                # guard reads it. Budget Piece 2 deleted the legacy in-loop cost
                # accumulator (_budget_spent_usd / on_cost_update / _estimate_cost_usd)
                # and the post-hoc dollar pause/stop branch that used to live here.
                if response.usage:
                    self._emit_ledger_event(active_provider, response.usage)

                # Compaction measures this prompt by what the provider billed.
                self._record_prompt_tokens(
                    active_provider, response.usage, tool_schemas,
                )

                # Text-only response: append and exit — but only when the
                # model actually said something. A turn may end in exactly
                # three states: a user-visible message, a tool handoff, or a
                # declared error. "No tool calls" alone does not prove the
                # first.
                if not response.has_tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response.text,
                        "source": "management",
                    }
                    # Carry reasoning through. finalize() writes
                    # reasoning_content onto raw_message only when reasoning
                    # exists, so mirror the tool-call persist path: include it
                    # when present, never write an empty/None field. Dropping it
                    # here is the silent-vanish bug for reasoned text-only turns.
                    reasoning = (response.raw_message or {}).get("reasoning_content")
                    if reasoning:
                        assistant_msg["reasoning_content"] = reasoning

                    if not (response.text or "").strip():
                        # Whitespace-only, no tool calls: the generation was
                        # cut off (typically inside its thinking block; the
                        # provider finalizes the stream cleanly so nothing
                        # raises). Ending the turn here is the silent-stall
                        # bug — the UI shows a thinking capsule, then nothing.
                        # Persist the partial (its reasoning helps the model
                        # resume), then nudge, bounded per run.
                        self._session.append(assistant_msg)
                        if empty_response_nudges < _EMPTY_RESPONSE_NUDGE_LIMIT:
                            empty_response_nudges += 1
                            logger.warning(
                                "Empty final response (finish_reason=%s, "
                                "reasoning_len=%d) — nudging model to "
                                "continue (%d/%d)",
                                response.finish_reason,
                                len(reasoning or ""),
                                empty_response_nudges,
                                _EMPTY_RESPONSE_NUDGE_LIMIT,
                            )
                            self._diag(
                                "empty_response_nudge",
                                iteration=iteration,
                                finish_reason=response.finish_reason,
                                nudge=empty_response_nudges,
                            )
                            self._session.append_system(
                                "Your previous reply contained no visible text "
                                "and no tool call — the response appears to "
                                "have been cut off (finish_reason="
                                f"{response.finish_reason}). Continue from "
                                "where you stopped: produce the user-facing "
                                "message or the next tool call."
                            )
                            continue
                        self._llm_failed = True
                        self._session.append_system(
                            f"The model returned {empty_response_nudges + 1} "
                            "consecutive empty responses (no text, no tool "
                            "calls). Giving up on this turn — retry, or switch "
                            "the project's model."
                        )
                        self._note_exit("empty_response", iteration)
                        break

                    self._session.append(assistant_msg)
                    # No blanket stubbing of consumed tool results here any
                    # more. Rewriting every tool result on every LLM response
                    # took documents away from the agent while it was still
                    # reasoning about them, and the stub was shaped like a
                    # successful retrieval, so history told the model it had
                    # already looked. Context pressure belongs to compaction
                    # (compaction.py, the token-pressure trigger below, and the
                    # sliding window in context.py) — not to history mutation.
                    # What remains: the disk archive, plus stubbing a result
                    # whose target was fetched again later in this session.
                    archive_and_supersede_tool_results(self._session, iteration)
                    self._note_exit("text_complete", iteration)
                    # Spec 112 A6: the reply is persisted (and broadcast by
                    # the session observer); the record turn runs INSIDE the
                    # run, before anything hooked on "run ended".
                    if self._memory_record_wanted():
                        await self._memory_record_turn(
                            active_provider, iteration, trigger="run_end",
                            rows_at_run_start=rows_at_run_start,
                        )
                    break

                # Extract status
                status = response.status_text
                if not status and response.tool_calls:
                    first_tc = normalize_tool_call(response.tool_calls[0])
                    status = f"Using {first_tc['name']}"

                # Append raw assistant message
                raw_msg = dict(response.raw_message)
                raw_msg["source"] = "management"
                if status:
                    raw_msg["_status"] = status
                self._session.append(raw_msg)

                # Archive large tool results to disk, and stub only those a
                # later fetch of the same target has superseded. The old
                # blanket truncation of every consumed result lived here: it
                # mutated history on every iteration and cost the agent
                # documents it was still using mid-turn. Context pressure is
                # compaction's job (compaction.py, the token-pressure trigger
                # further down this loop, the sliding window in context.py).
                archive_and_supersede_tool_results(self._session, iteration)

                # ── Queue signal detection ────────────────────────────────
                # Normalize tool calls, then scan for every terminal queue
                # signal (mark_task_complete / mark_task_blocked) in the
                # response. The FIRST one by emission order (`_signal_tc`) is
                # deferred and honored below, after its siblings run. Any
                # EXTRA signal calls are excluded from the per-tool loop the
                # same way but never executed — they're left pending for the
                # existing resolve_pending_tool_calls/CANCEL machinery, same
                # as any other orphaned tool call. Every non-signal tool call
                # — sibling calls before OR after the signal — still executes
                # through the normal per-tool path below. This replaces the
                # old short-circuit, which discarded co-emitted tools
                # outright and silently dropped batched work like notify +
                # checkpoint_state alongside mark_task_complete. The signal
                # itself is only honored once every sibling has run without
                # pausing or aborting the batch.
                tool_calls = [normalize_tool_call(tc) for tc in response.tool_calls]
                _signal_tc = None
                _signal_tc_ids: set[str] = set()
                for _tc in tool_calls:
                    if _tc["name"] in ("mark_task_complete", "mark_task_blocked"):
                        if not _tc["id"]:
                            _tc["id"] = uuid4().hex
                        _signal_tc_ids.add(_tc["id"])
                        if _signal_tc is None:
                            _signal_tc = _tc

                exit_outer = False
                intercepted_tc_id = None
                # Spec 134 Part B: a call asking to end the turn AFTER its
                # batch (``yield_after_batch``, background jobs) lets its
                # siblings run first, unlike ``yield_turn``.
                yield_after_batch = False
                _batch_token = _tool_batch.set(uuid4().hex[:12])

                for tc in tool_calls:
                    # Cancellation check: if cancel_turn() fired between or
                    # during tool execution, skip remaining tool calls.
                    if self._turn_cancelled:
                        break

                    # Defer every signal call out of the normal tool path.
                    # The first (`_signal_tc`) is handled after this for-loop
                    # so it can be skipped cleanly if a sibling pauses or
                    # aborts the run first (see queue signal handling below).
                    # Any extras are simply left pending — the CANCEL
                    # machinery picks them up like any orphaned tool call.
                    # Id-based skip assumes ids are unique within a batch
                    # (provider contract); a non-signal call reusing a signal
                    # call's id would be skipped with it.
                    if tc["id"] in _signal_tc_ids:
                        continue

                    # Post-normalization validation
                    if not tc["id"]:
                        tc["id"] = uuid4().hex
                        logger.warning("Tool call missing ID, generated: %s", tc["id"])

                    if not tc["name"]:
                        self._session.append_tool_result(
                            tc["id"],
                            "Error: LLM returned tool call with no function name",
                        )
                        continue

                    # Spec 124 §3.7: the run's tool cap. Reached → this call
                    # and the rest of the batch get a skip result and the
                    # run ends; the daemon, not the model, ends the turn.
                    if (self._tool_call_cap is not None
                            and self._run_tool_calls >= self._tool_call_cap):
                        for rest in tool_calls:
                            if rest["id"] in _signal_tc_ids or not rest["id"]:
                                continue
                            if rest["id"] in self._session.pending_tool_calls:
                                self._session.append_tool_result(
                                    rest["id"],
                                    f"Skipped: this turn's tool budget ({self._tool_call_cap} "
                                    "calls) is used up; the turn ends here.",
                                )
                        self._note_exit("tool_cap", iteration)
                        exit_outer = True
                        break

                    # Spec 124 §3.7: deterministic guard (path confinement
                    # for the Memory session). Before the interceptor so a
                    # refused write never asks for approval.
                    if self._tool_guard is not None:
                        try:
                            refusal = self._tool_guard(tc)
                        except Exception as e:  # noqa: BLE001 — fail closed
                            refusal = f"Error: tool guard failed ({e}); the tool was not executed."
                        if refusal:
                            self._session.append_tool_result(tc["id"], refusal)
                            continue

                    # Interceptor check (fail-closed)
                    if self._interceptor is not None:
                        try:
                            should_intercept = self._interceptor.should_intercept(tc)
                        except Exception:
                            should_intercept = True

                        if should_intercept:
                            # Phase 3: under queue-running state, an
                            # approval-required tool call is a block signal,
                            # not a pause. Skip on_intercept (no UI card —
                            # the dispatcher records this as the block reason)
                            # and exit the loop. The dispatcher rotates the
                            # session and proceeds to the next queued item.
                            if self._queue_state == "running":
                                _block_reason = (
                                    f"approval required for tool '{tc['name']}'"
                                )
                                _args_str = ""
                                try:
                                    _args_str = json.dumps(tc.get("arguments") or {})[:200]
                                except (TypeError, ValueError):
                                    _args_str = ""
                                if _args_str and _args_str not in ("{}", '""'):
                                    _block_reason += f" (args: {_args_str})"
                                self._exit_reason = "blocked"
                                self._exit_block_reason = _block_reason
                                self._session.append({
                                    "role": "system",
                                    "content": f"Task blocked: {_block_reason}",
                                    "source": "queue_signal",
                                    "signal": "blocked",
                                    "payload": {
                                        "reason": _block_reason,
                                        "tool": tc["name"],
                                    },
                                })
                                exit_outer = True
                                break

                            reasoning = response.text if response.text and response.text.strip() else None
                            try:
                                self._interceptor.on_intercept(
                                    tc, self._session.recent_activity(),
                                    reasoning=reasoning,
                                )
                            except Exception as e:
                                # on_intercept failed (e.g. WS broadcast hook
                                # crashed). Without this guard the exception
                                # would escape run(), leave the orphan tool
                                # call as CANCELLED, and end the loop task in
                                # error state — leaving the user with no
                                # approval card and no recovery path.
                                # Treat the failure as a denial: append an
                                # error tool result so the agent can react,
                                # and continue the dispatch loop.
                                logger.exception(
                                    "Interceptor.on_intercept raised for "
                                    "tool_call_id=%s tool=%s",
                                    tc.get("id"), tc.get("name"),
                                )
                                self._session.append_tool_result(
                                    tc["id"],
                                    f"Error requesting approval: {e}. "
                                    "The tool was not executed.",
                                )
                                continue
                            intercepted_tc_id = tc["id"]
                            self._session._paused_for_approval = True
                            self._session.pause()
                            self._note_exit("approval_pause", iteration)
                            exit_outer = True
                            break

                    # Circuit breaker: block tool if it has repeated errors
                    if tc["name"] in blocked_tools:
                        blocked_msg = (
                            f"[SYSTEM] The {tc['name']} tool is currently unavailable "
                            f"(repeated error: {error_tracker[tc['name']]['error']}). "
                            f"Work with the tools that are available or ask the user for help."
                        )
                        self._session.append_tool_result(tc["id"], blocked_msg)
                        continue

                    # Execute tool (async-aware)
                    self._run_tool_calls += 1
                    self.last_run_stats["tool_calls"] += 1
                    try:
                        if self._tool_registry.is_async(tc["name"]):
                            result: ToolResult = await self._tool_registry.execute_async(
                                tc["name"],
                                tc["arguments"],
                            )
                        else:
                            result: ToolResult = await run_sync_tool(
                                self._tool_registry.execute,
                                tc["name"],
                                tc["arguments"],
                            )
                        # Pre-filter tool result before storage
                        filtered_content = dispatch_prefilter(
                            tc["name"], tc["arguments"], result.content,
                        )
                        result = ToolResult(
                            content=filtered_content, meta=result.meta,
                        )

                        self._session.append_tool_result(
                            tc["id"], result.content, meta=result.meta,
                            context_limit=self._context_manager.model_context_limit,
                        )
                        tool_content = result.content
                        # Store result for observation-aware hash
                        _exec_result = result
                        self._note_written_path(tc, tool_content)

                        # yield_turn: a dispatch tool that delivered a task to a
                        # sub-agent ends the management turn immediately so the
                        # LLM cannot busy-poll it (which trips the ping-pong
                        # guard). The result is already appended above
                        # (tool_call_id paired), so resolve_pending_tool_calls
                        # will not CANCEL it. The sub-agent's result is pushed
                        # back later via on_completed and the loop restarts.
                        if result.meta and result.meta.get("yield_turn"):
                            logger.info("yield_turn: dispatch tool ended the management turn")
                            self._note_exit("yield_turn", iteration)
                            exit_outer = True
                            break
                        if result.meta and result.meta.get("yield_after_batch"):
                            yield_after_batch = True

                        # Pause if credential was requested (wait for user input)
                        if result.meta and result.meta.get("credential_request"):
                            self._session.pause()
                            self._note_exit("credential_request", iteration)
                            exit_outer = True
                            break

                        # Spec 117 D4 — a safe-mode shell result the sandbox
                        # blocked: the interceptor raises an ``unsandboxed_run``
                        # approval off the daemon's own denial detection and
                        # the turn pauses on it exactly like an intercepted
                        # call (the rest of the batch is CANCELLED below). Not
                        # under a queue run — there is no card to answer there.
                        # Shared with AgentManager.approve() (approved calls
                        # run off the loop) via maybe_raise_unsandboxed_run.
                        _pending_id = maybe_raise_unsandboxed_run(
                            self._interceptor, tc, result,
                            self._session.recent_activity,
                            queue_state=self._queue_state,
                        )
                        if _pending_id:
                            intercepted_tc_id = tc["id"]
                            self._session._paused_for_approval = True
                            self._session.pause()
                            self._note_exit("approval_pause", iteration)
                            exit_outer = True
                            break
                    except Exception as e:
                        tool_content = f"Error executing tool: {e}"
                        self._session.append_tool_result(
                            tc["id"], tool_content
                        )
                        _exec_result = None

                    # Observation-aware repetition detection (post-execution)
                    # Browser tools: skip hash detection (handled by advisory counter in browser tool)
                    if tc["name"] != "browser":
                        result_prefix = str(tool_content)[:500] if tool_content else ""
                        action_hash = hashlib.md5(
                            (tc["name"] + str(tc["arguments"]) + result_prefix).encode()
                        ).hexdigest()

                        action_hashes.append(action_hash)
                        if list(action_hashes).count(action_hash) >= 5:
                            self._session.resolve_pending_tool_calls()
                            self._session.append_system(
                                "Repetitive action detected. Save your state and try a different approach."
                            )
                            self._note_exit("repetition", iteration)
                            exit_outer = True
                            break

                        # Ping-pong / cycle detection: track consecutive pairs
                        if _prev_action_hash is not None:
                            pair = (_prev_action_hash, action_hash)
                            _pair_hashes.append(pair)
                            if list(_pair_hashes).count(pair) >= 3:
                                self._session.resolve_pending_tool_calls()
                                self._session.append_system(
                                    f"Ping-pong pattern detected: you have been "
                                    f"alternating between the same tool calls for "
                                    f"multiple iterations with no progress. "
                                    f"Stop and explain what you are trying to "
                                    f"accomplish, then try a fundamentally "
                                    f"different approach."
                                )
                                self._note_exit("ping_pong", iteration)
                                exit_outer = True
                                break
                        _prev_action_hash = action_hash

                    # Circuit breaker: trip only on genuine tool/infra failures,
                    # never on recoverable input errors the model can correct
                    # (e.g. edit old_text-not-found — the agent should re-read and
                    # retry, not lose the tool). Recoverability is read from the
                    # structured ToolResult marker, not by string-sniffing; a
                    # raised exception (_exec_result is None) is never recoverable.
                    # Identity is compared on the FULL error string so two
                    # genuinely different failures are not collapsed.
                    is_error = isinstance(tool_content, str) and tool_content.startswith("Error")
                    is_recoverable = bool(
                        _exec_result is not None
                        and _exec_result.meta
                        and _exec_result.meta.get("recoverable")
                    )
                    if is_error and not is_recoverable:
                        prev = error_tracker.get(tc["name"])
                        if prev and prev["error"] == tool_content:
                            prev["count"] += 1
                        else:
                            error_tracker[tc["name"]] = {"error": tool_content, "count": 1}
                        if error_tracker[tc["name"]]["count"] >= 2:
                            blocked_tools.add(tc["name"])
                            self._session.append_system(
                                f"[SYSTEM] The {tc['name']} tool is currently unavailable "
                                f"(repeated error: {tool_content}). "
                                f"Work with the tools that are available or ask the user for help."
                            )
                    else:
                        # Successful execution OR a recoverable input error clears
                        # error tracking for this tool.
                        error_tracker.pop(tc["name"], None)

                _tool_batch.reset(_batch_token)

                # Drain deferred messages (lifecycle notifications)
                for msg in self._session.pop_deferred_messages():
                    self._session.append(msg)

                # Cancellation landed mid-tool-loop (cancel_turn fired between
                # tool calls or during a tool's execution). Append marker if
                # not already done, resolve any unstarted tool calls, then
                # exit the loop. The finally block + _on_loop_done drain any
                # queued messages and hot-resume a fresh loop task if needed.
                # terminate() also sets is_stopped(); the cancel-only path
                # exits via break here without needing the is_stopped check
                # below.
                if self._turn_cancelled:
                    if not self._cancellation_marker_appended_this_turn:
                        self._persist_cancellation_marker()
                    # Resolve any tool calls that never got a result — these
                    # are the ones we skipped via the per-tool break above.
                    self._session.resolve_pending_tool_calls()
                    self._note_exit("cancel", iteration)
                    break

                # Resolve unprocessed tool calls from this batch on intercept-break
                if exit_outer and intercepted_tc_id is not None:
                    for tc in tool_calls:
                        tc_id = tc["id"]
                        if tc_id in self._session.pending_tool_calls and tc_id != intercepted_tc_id:
                            self._session.append_tool_result(
                                tc_id,
                                "CANCELLED: Earlier tool call requires approval. "
                                "This tool call will be retried after approval.",
                            )

                # Check if we need to exit the outer loop
                if exit_outer:
                    break

                # Honor the deferred queue signal now that every sibling in
                # this batch has executed. Reaching here means neither of the
                # two unconditional breaks above fired, so the run neither
                # paused (approval intercept) nor aborted (turn_cancelled) —
                # exactly the "not exit_outer and not self._turn_cancelled"
                # condition. Run the signal through the registry so its
                # tool_call_id gets a real result row (queue_signals.py's
                # execute()) instead of falling through to the CANCELLED-
                # pending path, then append the marker and exit as before.
                if _signal_tc is not None:
                    try:
                        if self._tool_registry.is_async(_signal_tc["name"]):
                            _sig_result: ToolResult = await self._tool_registry.execute_async(
                                _signal_tc["name"], _signal_tc["arguments"],
                            )
                        else:
                            _sig_result: ToolResult = await run_sync_tool(
                                self._tool_registry.execute,
                                _signal_tc["name"], _signal_tc["arguments"],
                            )
                        _sig_content = dispatch_prefilter(
                            _signal_tc["name"], _signal_tc["arguments"],
                            _sig_result.content,
                        )
                        self._session.append_tool_result(
                            _signal_tc["id"], _sig_content, meta=_sig_result.meta,
                            context_limit=self._context_manager.model_context_limit,
                        )
                    except Exception as e:
                        self._session.append_tool_result(
                            _signal_tc["id"], f"Error executing tool: {e}",
                        )

                    if _signal_tc["name"] == "mark_task_complete":
                        summary = (_signal_tc["arguments"] or {}).get("summary", "")
                        summary = summary.strip() if isinstance(summary, str) else ""
                        self._exit_reason = "complete"
                        self._exit_summary = summary
                        self._session.append({
                            "role": "system",
                            "content": f"Task completed: {summary}" if summary else "Task completed.",
                            "source": "queue_signal",
                            "signal": "complete",
                            "payload": {"summary": summary},
                        })
                        self._note_exit("task_complete", iteration)
                        # Spec 112 A6: a declared completion is a normal end.
                        if self._memory_record_wanted():
                            await self._memory_record_turn(
                                active_provider, iteration, trigger="run_end",
                                rows_at_run_start=rows_at_run_start,
                            )
                    else:  # mark_task_blocked
                        reason = (_signal_tc["arguments"] or {}).get("reason", "")
                        reason = reason.strip() if isinstance(reason, str) else ""
                        self._exit_reason = "blocked"
                        self._exit_block_reason = reason
                        self._session.append({
                            "role": "system",
                            "content": f"Task blocked: {reason}" if reason else "Task blocked.",
                            "source": "queue_signal",
                            "signal": "blocked",
                            "payload": {"reason": reason},
                        })
                        self._note_exit("task_blocked", iteration)
                    break

                if yield_after_batch:
                    logger.info("yield_after_batch: background job(s) started; "
                                "the turn ends after its tool batch")
                    self._note_exit("yield_after_batch", iteration)
                    break

                # Check stopped
                if self._session.is_stopped():
                    self._session.resolve_pending_tool_calls()
                    self._note_exit("stopped", iteration)
                    break

                # Check paused (redundant with exit_outer but kept for safety)
                if self._session.is_paused():
                    self._note_exit("paused", iteration)
                    break

                # Check compaction (on the prompt size the provider reported
                # for this iteration's call; see _record_prompt_tokens)
                if self._context_manager.should_compact():
                    # Spec 112 A6: a compaction is a deterministic boundary
                    # in a long autonomous run — ask what to record first
                    # (while the history is still intact), then compact, then
                    # begin a fresh snapshot segment (A1/A5).
                    _fits = getattr(self._context_manager, "flush_fits", None)
                    if self._memory_record_wanted() and (
                            not callable(_fits) or _fits()):
                        await self._memory_record_turn(
                            active_provider, iteration, trigger="compaction",
                            rows_at_run_start=rows_at_run_start,
                        )
                    if await self._compact_now(iteration):
                        break
                    self._begin_snapshot_segment()

                # Bottom of the loop body: no named branch broke out, so this
                # iteration is continuing to the next. Classify as tool_continue
                # when rows grew this iteration, else fell_through — the
                # catch-all for a path that produced no appended row (the
                # original-bug shape), captured even though no named branch fired.
                _appended_this_iter = len(self._session._messages) - rows_at_iter_start
                self._note_exit(
                    "tool_continue" if _appended_this_iter > 0 else "fell_through",
                    iteration,
                )

        except Exception as e:
            # Backstop for every failure the LLMError branch above does not
            # own. Those escape to agent_manager._record_loop_error, which
            # logs them and broadcasts a transient agent.status frame — and
            # that frame is the ONLY surfacing. A run nobody had open at that
            # instant (a 09:00 scheduled trigger) therefore died leaving the
            # session holding the prompt and nothing after it, which reads as
            # "the trigger fired and the agent never answered". Persist the
            # failure where the reader will actually look, then let it
            # propagate unchanged so the manager still logs and broadcasts.
            #
            # CancelledError and GeneratorExit are BaseException, so a
            # cancel_turn()/stop still takes the loop's own cancel path.
            self._llm_failed = True
            try:
                self._session.append_system(
                    f"Run failed ({type(e).__name__}): {e}. Stopping."
                )
            except Exception:  # noqa: BLE001 — never mask the original failure
                logger.exception("Failed to persist the run-failure row")
            self._note_exit("internal_error", iteration)
            raise

        finally:
            if not self._session._paused_for_approval:
                self._session.resolve_pending_tool_calls()
            # Drain any remaining deferred messages
            for msg in self._session.pop_deferred_messages():
                self._session.append(msg)
            # Per-session cache-hit summary (Part 4D). Logged at every run-end
            # boundary; the totals are cumulative across the session, so the
            # last line for a session is its session total. Lets us compare
            # cache hit rate before/after the CONTEXT.md on-judgment change.
            if self._llm_call_count > 0 and self._prompt_input_tokens_total > 0:
                _rate = self._cache_read_tokens_total / self._prompt_input_tokens_total
                _cache_logger.info(
                    "[CACHE_SESSION] session=%s calls=%d prompt_tokens=%d "
                    "cached_tokens=%d cache_rate=%.1f%%",
                    getattr(self._session, "session_uuid", "?"),
                    self._llm_call_count,
                    self._prompt_input_tokens_total,
                    self._cache_read_tokens_total,
                    _rate * 100,
                )
            # NOTE: fire-and-forget asyncio.create_task(self._on_session_end())
            # was previously here. Removed in TASK-cancel-arch-04: the
            # synchronous session-end inside agent_manager.new_session() is
            # now the sole authoritative summarization path. The fire-and-
            # forget had no strong reference and would emit
            # "Task was destroyed but it is pending" on cancel.

            # Diagnostics (Point 3): run-level terminal disposition. This is the
            # authoritative capture of the original-bug fingerprint —
            # `none_appended` means the loop exited without persisting any new
            # row this turn. Runs even while an exception propagates through the
            # finally (detected via sys.exc_info), which is logged as `error`.
            _base = (
                rows_at_run_start
                if rows_at_run_start is not None
                else len(self._session._messages)
            )
            _rows_appended = len(self._session._messages) - _base
            if sys.exc_info()[0] is not None:
                _disposition = "error"
            elif _rows_appended == 0:
                _disposition = "none_appended"
            elif self._cancellation_marker_appended_this_turn or self._turn_cancelled:
                _disposition = "cancel"
            else:
                _disposition = "normal"
            self._diag(
                "run_terminal",
                disposition=_disposition,
                rows_appended=_rows_appended,
                exit_path=self._loop_exit_path,
            )

            self._running = False
            self.last_run_stats["ended"] = self._loop_exit_path
            self._unbind_seen_state_ids()
            # The turn is over: settle the spend corner on the FINAL figure now
            # rather than leaving the debouncer's pending payload to fire (or
            # not) up to a second later. The run's last append almost always
            # lands inside the 1s window, so this pending payload is precisely
            # the number that would otherwise sit stale for the whole idle
            # period. A no-op when nothing is pending (the last append already
            # went out on a leading edge) — no duplicate frame.
            self._flush_spend_broadcast()

    # ------------------------------------------------------------------
    # State refresh helpers
    # ------------------------------------------------------------------

    async def _compact_now(self, iteration: int) -> bool:
        """Compaction + reorientation (moved out of ``run()`` so spec 112's
        record turn can precede it). Returns True when the pre-flight budget
        guard tripped and the run must end.

        There is no pre-compaction "memory flush" turn any more: the record
        turn that runs right before this (``_memory_record_turn`` with
        trigger ``"compaction"``) is the agent's chance to save working
        state, and it does so with the project-state text and anchors in
        hand. The flush used to ask the same question a second time, at the
        utility model when one was configured (a cold, full-window call).
        """
        from agent_os.agent import compaction as compaction_mod

        # Pre-flight budget guard (Budget Piece 2): the compaction summary
        # is a real LLM call that spends (the old rows go to the summarizer
        # as text). Evaluate the SAME single guard before it fires so a
        # tripped budget stops here rather than buying one more response.
        # On trip, exit with ``budget_blocked`` without compacting — the
        # loop is ending anyway; the next resume re-evaluates.
        _summary_budget = self._budget_tripped()
        if _summary_budget is not None:
            self._trip_budget(_summary_budget, iteration)
            return True

        # Append-only: the summary lands as a marker row and the
        # kept tail is sized to the window (compaction.run).
        _keep = getattr(self._context_manager, "compaction_keep_tokens", None)
        await compaction_mod.run(
            self._session,
            self._provider,
            utility_provider=self._utility_provider,
            keep_tokens=_keep() if callable(_keep) else None,
        )
        _note = getattr(self._context_manager, "note_compacted", None)
        if callable(_note):
            _note()

        # Post-compaction reorientation
        workspace = getattr(
            getattr(self._context_manager, "_base_ctx", None),
            "workspace", None,
        )
        if workspace:
            compaction_mod.inject_reorientation(workspace, self._session)
        return False

    # ------------------------------------------------------------------
    # Spec 112 A6 — end-of-run memory record turn
    # ------------------------------------------------------------------

    @staticmethod
    def _fresh_run_stats() -> dict:
        return {"llm_calls": 0, "tool_calls": 0, "input_tokens": 0,
                "cached_tokens": 0, "output_tokens": 0, "ended": None}

    def _begin_snapshot_segment(self) -> None:
        """Run start / post-compaction: refresh the Layer-1 snapshot and the
        media-pruning horizon (spec 112 A1/A5), reset the per-segment
        record-turn bookkeeping, and bind the PROJECT_STATE ids this segment
        starts from (spec 124 §3.10: the write path protects only ids the
        writer never saw)."""
        begin = getattr(self._context_manager, "begin_run", None)
        if callable(begin):
            try:
                begin()
            except Exception:  # noqa: BLE001 — a snapshot bug must not break a turn
                logger.warning("begin_run failed", exc_info=True)
        self._run_changed_files = []
        self._run_memory_written = False
        self._memory_record_done = False
        self._bind_seen_state_ids()

    def _bind_seen_state_ids(self) -> None:
        from agent_os.agent import memory_entries as _mem
        from agent_os.agent import state_blocks as _blocks
        self._unbind_seen_state_ids()
        workspace = self._workspace()
        if not workspace:
            return
        try:
            path = os.path.join(workspace, "orbital", "PROJECT_STATE.md")
            text = read_text_lenient(path)
            ids = ({b.id for b in _blocks.parse(text)[1] if b.id}
                   if text is not None else set())
        except Exception:  # noqa: BLE001 — a parse bug must not break a turn
            logger.warning("seen-set binding failed", exc_info=True)
            return
        self._seen_ids_token = _mem.bind_seen_state_ids(ids)

    def _unbind_seen_state_ids(self) -> None:
        if self._seen_ids_token is None:
            return
        from agent_os.agent import memory_entries as _mem
        try:
            _mem.unbind_seen_state_ids(self._seen_ids_token)
        except ValueError:
            pass  # bound in another context (tests driving run() across tasks)
        self._seen_ids_token = None

    def _workspace(self) -> str | None:
        base = getattr(self._context_manager, "_base_ctx", None)
        return getattr(base, "workspace", None) or None

    def _note_written_path(self, tc: dict, tool_content) -> None:
        """Track what edit/write touched this run: the four memory files make
        the run self-recorded (no record turn); anything outside orbital/ is
        listed on the [MEMORY] row."""
        if tc.get("name") not in ("edit", "write"):
            return
        if isinstance(tool_content, str) and tool_content.startswith("Error"):
            return
        args = tc.get("arguments") or {}
        path = args.get("path") if isinstance(args, dict) else None
        kind = classify_written_path(self._workspace() or "", path)
        if kind == "memory":
            self._run_memory_written = True
        elif kind == "workspace":
            shown = path.strip()
            ws = self._workspace() or ""
            if os.path.isabs(shown) and ws:
                try:
                    shown = os.path.relpath(shown, ws).replace(os.sep, "/")  # POSIX form on every platform (the row text is prompt bytes)
                except ValueError:
                    pass
            if shown not in self._run_changed_files:
                self._run_changed_files.append(shown)

    def _memory_record_wanted(self) -> bool:
        """Deterministic trigger gate (spec 112 A6). Skips only what cannot
        be meaningful: a run that already wrote a memory file, the Memory
        session itself, a scratch project, a loop without a workspace, and a
        second ask in the same segment."""
        if not self._memory_record_enabled:
            return False
        if self._memory_record_done or self._memory_record_active:
            return False
        # Spec 142: a held user message goes first. That run stays unrecorded.
        if self._session.list_queued():
            return False
        if self._run_memory_written:
            return False
        if getattr(self._session, "origin", None) == "memory":
            return False
        if getattr(self._session, "source", None) == "memory":
            return False
        base = getattr(self._context_manager, "_base_ctx", None)
        if base is None or getattr(base, "is_scratch", False):
            return False
        return bool(getattr(base, "workspace", None))

    def yield_memory_record(self) -> bool:
        """Cancel the record turn's in-flight call so a user message just held
        doesn't wait for it (spec 142). True if a call was cancelled; a no-op
        outside a record turn, so a normal turn is never interrupted. The turn
        catches the cancel itself and ends quietly, with no Stop marker."""
        stream = self._inflight_stream
        if not self._memory_record_active or stream is None or stream.done():
            return False
        stream.cancel()
        return True

    def _sub_agent_results_this_run(self, rows_at_run_start) -> list[str]:
        """Terminal rows that reached this run: the rows that triggered it
        (everything after the previous assistant row) plus what was appended
        during it."""
        rows = self._session.get_messages()
        start = rows_at_run_start if rows_at_run_start is not None else len(rows)
        start = min(start, len(rows))
        while start > 0 and rows[start - 1].get("role") != "assistant":
            start -= 1
        found: list[str] = []
        for row in rows[start:]:
            meta = row.get("_meta") or {}
            if not isinstance(meta, dict) or meta.get("event") != "sub_agent_terminal":
                continue
            handle = meta.get("handle")
            if not handle:
                display = meta.get("display_content") or row.get("content") or ""
                if isinstance(display, str) and display.startswith("[Sub-agent] "):
                    handle = display[len("[Sub-agent] "):].split(" ", 1)[0]
            label = f"{handle or 'sub-agent'} ({meta.get('kind') or 'terminal'})"
            if label not in found:
                found.append(label)
        return found

    def _record_turn_tool_allowed(self, tc: dict) -> bool:
        if tc.get("name") not in _MEMORY_RECORD_TOOLS:
            return False
        args = tc.get("arguments") or {}
        path = args.get("path") if isinstance(args, dict) else None
        return classify_written_path(self._workspace() or "", path) == "memory"

    async def _stream_quiet(self, provider, context, tool_schemas) -> LLMResponse:
        """``_stream_response_internal`` without the per-chunk stream
        notification: the record turn's text is never shown live."""
        accumulator = StreamAccumulator()
        self._stream_accumulator = accumulator
        async for chunk in provider.stream(context, tools=tool_schemas):
            accumulator.add(chunk)
        return accumulator.finalize()

    async def _memory_record_turn(self, provider, iteration: int, *,
                                  trigger: str, rows_at_run_start) -> None:
        """Append the ``[MEMORY]`` row and run the bounded record turn.

        Bounds hold by construction: at most ``MEMORY_RECORD_MAX_LLM_CALLS``
        calls and ``MEMORY_RECORD_MAX_TOOL_CALLS`` executed tools; only
        read/edit/write on the four memory files run (the rest get an error
        result); the turn ends on the first text-only reply. Every row is
        tagged ``_kind: memory_record`` through ``session.row_kind``; queued
        user messages and deferred lifecycle rows are NOT drained here — they
        follow the turn, exactly as they followed the run before. A held user
        message ends the turn early (spec 142, ``yield_memory_record``). Never
        raises into the run: a failing record turn is logged and the run
        ends as it would have.
        """
        workspace = self._workspace() or ""
        self._memory_record_active = True
        self._memory_record_done = True
        llm_calls = 0
        tool_calls = 0
        outcome = "nothing"
        try:
            # The record turn's own rows are its conversation: in the window
            # now, dropped from every later call (context.py).
            self._context_manager.include_memory_record_rows = True
            text = build_memory_record_text(
                workspace, list(self._run_changed_files),
                self._sub_agent_results_this_run(rows_at_run_start),
                trigger=trigger,
            )
            self._session.row_kind = MEMORY_RECORD_KIND
            self._session.append({
                "role": "system",
                "content": text,
                "source": "daemon",
                "_meta": {"event": "memory_record", "trigger": trigger},
            })
            while llm_calls < MEMORY_RECORD_MAX_LLM_CALLS:
                if self._session.list_queued():
                    outcome = "yielded"
                    break
                llm_calls += 1
                context = self._context_manager.prepare()
                tool_schemas = self._tool_registry.schemas()
                self._inflight_stream = asyncio.create_task(
                    self._stream_quiet(provider, context, tool_schemas),
                    name=f"memory-record-{self._session.session_uuid}-{llm_calls}",
                )
                try:
                    response = await self._inflight_stream
                except asyncio.CancelledError:
                    self._inflight_stream = None
                    if self._session.is_stopped():
                        raise
                    outcome = "cancelled"
                    break
                except (LLMError, ContextOverflowError) as e:
                    self._inflight_stream = None
                    logger.warning("memory record turn: LLM call failed (%s)", e)
                    outcome = "llm_error"
                    break
                self._inflight_stream = None
                if response.usage:
                    self._prompt_input_tokens_total += response.usage.input_tokens
                    self._cache_read_tokens_total += response.usage.cache_read_tokens
                    self._llm_call_count += 1
                    self._emit_ledger_event(provider, response.usage)
                self._record_prompt_tokens(provider, response.usage, tool_schemas)
                if not response.has_tool_calls:
                    self._session.append({
                        "role": "assistant",
                        "content": response.text,
                        "source": "management",
                        **({"reasoning_content": (response.raw_message or {}).get("reasoning_content")}
                           if (response.raw_message or {}).get("reasoning_content") else {}),
                    })
                    break
                raw_msg = dict(response.raw_message)
                raw_msg["source"] = "management"
                self._session.append(raw_msg)
                for tc_raw in response.tool_calls:
                    tc = normalize_tool_call(tc_raw)
                    if not tc["id"]:
                        tc["id"] = uuid4().hex
                    if tool_calls >= MEMORY_RECORD_MAX_TOOL_CALLS:
                        self._session.append_tool_result(
                            tc["id"],
                            "Skipped: the memory record turn's tool budget is "
                            "used up. Stop now.",
                        )
                        continue
                    tool_calls += 1
                    if not self._record_turn_tool_allowed(tc):
                        self._session.append_tool_result(
                            tc["id"],
                            "Error: only read, edit and write on "
                            "orbital/PROJECT_STATE.md, DECISIONS.md, LESSONS.md "
                            "and INDEX.md are available in the memory record "
                            "turn. Do not do other work here.",
                        )
                        continue
                    try:
                        if self._tool_registry.is_async(tc["name"]):
                            result = await self._tool_registry.execute_async(
                                tc["name"], tc["arguments"])
                        else:
                            result = await run_sync_tool(
                                self._tool_registry.execute,
                                tc["name"], tc["arguments"])
                        content = dispatch_prefilter(
                            tc["name"], tc["arguments"], result.content)
                        self._session.append_tool_result(
                            tc["id"], content, meta=result.meta,
                            context_limit=self._context_manager.model_context_limit,
                        )
                        if tc["name"] in ("edit", "write") and not (
                                isinstance(content, str) and content.startswith("Error")):
                            outcome = "edited"
                    except Exception as e:  # noqa: BLE001 — a tool failure ends nothing here
                        self._session.append_tool_result(
                            tc["id"], f"Error executing tool: {e}")
                if tool_calls >= MEMORY_RECORD_MAX_TOOL_CALLS:
                    if outcome == "nothing":
                        outcome = "tool_cap"
                    break
            else:
                if outcome == "nothing":
                    outcome = "llm_cap"
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the record turn must never fail the run
            logger.warning("memory record turn failed", exc_info=True)
            outcome = "error"
        finally:
            self._session.row_kind = None
            self._memory_record_active = False
            try:
                self._context_manager.include_memory_record_rows = False
            except AttributeError:
                pass
            self._diag(
                "memory_record", iteration=iteration, trigger=trigger,
                llm_calls=llm_calls, tool_calls=tool_calls, outcome=outcome,
                changed_files=len(self._run_changed_files),
            )

    async def _run_refresh(self, trigger_name: str, iteration: int) -> None:
        """Execute a state refresh and update checkpoint metadata.

        Runs the session-end routine with bypass_idempotency=True so it can
        fire mid-session. Updates context_manager checkpoint fields after a
        successful run so the agent sees accurate last-update metadata on
        the next turn.

        Broadcasts state_refresh.lifecycle WS events (in_progress / done /
        failed) via the on_session_end_refresh callback, which is wired by
        agent_manager to include ws.broadcast calls.

        Always resets _turns_since_last_update and _last_refresh_iteration
        regardless of success/failure so cooldown still engages (prevents
        rapid retry storms on persistent LLM failures).
        """
        if self._on_session_end_refresh is None:
            return

        outcome: str | None = None
        try:
            result = await self._on_session_end_refresh(trigger_name)
            if isinstance(result, str):
                outcome = result
        except Exception:
            outcome = "failed"
            logger.exception("State refresh failed (trigger=%s)", trigger_name)
        finally:
            # Always record the refresh so cooldown engages
            self._turns_since_last_update = 0
            self._last_refresh_iteration = iteration
            self._last_refresh_outcome = outcome
            # Push metadata into context_manager so the agent sees it
            ts = datetime.now(timezone.utc).isoformat()
            if hasattr(self._context_manager, "_last_checkpoint_turn"):
                self._context_manager._last_checkpoint_turn = iteration
                self._context_manager._last_checkpoint_ts = ts
                self._context_manager._turns_since_last_update = 0
                self._context_manager._last_checkpoint_outcome = outcome
            # The pass is over (success, failure, or cancellation) — clear the
            # in-flight marker set by _spawn_refresh so the hygiene flag stops
            # rendering "pass in flight".
            if hasattr(self._context_manager, "_refresh_in_flight"):
                self._context_manager._refresh_in_flight = False
                self._context_manager._refresh_in_flight_since_turn = None

    def _spawn_refresh(self, trigger: str) -> None:
        """Start a background consolidation pass. SYNCHRONOUS — no awaits
        between the caller's in-flight check and create_task (cooperative
        event loop ⇒ the single-flight gate is TOCTOU-free; spec 013 race #3).
        """
        if self._session.is_stopped():
            # Belt for the _maybe_consume_dirty path, which bypasses
            # schedule_checkpoint's own is_stopped() gate below. Without
            # this, a pass can spawn during the terminate() window between
            # cancel_turn() waking the loop's turn boundary and
            # session.stop() actually landing (spec 013 race #7).
            logger.debug(
                "State refresh: refusing to spawn (trigger=%s) — session stopped",
                trigger,
            )
            return
        iteration = self._current_iteration
        self._last_merge_at = time.monotonic()
        gate = self._refresh_gate()
        if gate is not None:
            # Project-scoped debounce: a sibling session starting later must
            # see this spawn, or a fresh loop's -inf reopens the window.
            gate["last_merge_at"] = self._last_merge_at
        self._refresh_dirty = False           # fresh pass reads files fresh — covers all prior triggers
        # Reset at SPAWN (and again in _run_refresh's finally): turns counted
        # while a slow pass runs are deliberately discarded — the status line
        # counts from the latest pass.
        self._turns_since_last_update = 0
        # Mark the pass in flight for the context manager BEFORE create_task
        # (synchronous, so the very next prompt build renders the hygiene flag
        # in its "pass in flight — no action needed" state). Cleared in
        # _run_refresh's finally.
        if hasattr(self._context_manager, "_refresh_in_flight"):
            self._context_manager._refresh_in_flight = True
            self._context_manager._refresh_in_flight_since_turn = iteration
        task = asyncio.create_task(
            self._run_refresh(trigger, iteration),
            name=f"refresh-{trigger}-{self._session.session_uuid}-{iteration}",
        )
        self._refresh_task = task
        self._register_refresh_task(task)

        def _on_refresh_done(t: asyncio.Task) -> None:
            if self._refresh_task is t:
                self._refresh_task = None
            self._clear_refresh_task(t)
            # Idempotent with _run_refresh's finally; covers a task cancelled
            # before its first tick (finally never ran → marker would leak).
            if hasattr(self._context_manager, "_refresh_in_flight"):
                self._context_manager._refresh_in_flight = False
                self._context_manager._refresh_in_flight_since_turn = None
            if not t.cancelled() and t.exception() is not None:
                # _run_refresh swallows callback exceptions; this catches
                # crashes in _run_refresh itself.
                logger.error(
                    "Background state refresh crashed (trigger=%s)",
                    trigger, exc_info=t.exception(),
                )

        task.add_done_callback(_on_refresh_done)

    def _refresh_gate(self) -> dict | None:
        """This project's gate entry, or None when the loop isn't project-bound."""
        if not self._project_dir:
            return None
        return _PROJECT_REFRESH.setdefault(
            self._project_dir, {"task": None, "last_merge_at": float("-inf")}
        )

    def _refresh_gate_state(self) -> tuple[asyncio.Task | None, float]:
        """(in-flight pass, last spawn time) for whatever scope owns the files.

        Project scope when this loop has a project_dir — the four Layer-1 files
        are shared by every session on that project, so a sibling session's
        pass must count as in-flight here. Per-loop otherwise.
        """
        gate = self._refresh_gate()
        if gate is None:
            return self._refresh_task, self._last_merge_at
        return gate["task"], gate["last_merge_at"]

    def _register_refresh_task(self, task: asyncio.Task) -> None:
        """Publish a started pass to the project gate (no-op when unbound)."""
        gate = self._refresh_gate()
        if gate is not None:
            gate["task"] = task

    def _clear_refresh_task(self, task: asyncio.Task) -> None:
        """Retract a finished pass, unless a newer one already replaced it."""
        gate = self._refresh_gate()
        if gate is not None and gate["task"] is task:
            gate["task"] = None

    def schedule_checkpoint(self, trigger: str = "agent_decided") -> str:
        """Single-flight, debounced, coalescing scheduler (spec 013).

        Synchronous by design: no await between the in-flight check and the
        spawn, so two callers can never both observe 'no pass in flight'. That
        holds across sessions too — the gate consulted here is project-scoped
        (`_refresh_gate_state`) and every mutation of it is synchronous.
        """
        if self._on_session_end_refresh is None:
            return "State refresh unavailable in this session."
        if self._session.is_stopped():
            # Checked before the in-flight/debounce checks so a terminating
            # session never reports "scheduled" for a pass that _spawn_refresh
            # will then refuse (spec 013 race #7) — keeps the ack truthful.
            return "State refresh unavailable in this session."
        inflight, last_merge_at = self._refresh_gate_state()
        if inflight is not None and not inflight.done():
            self._refresh_dirty = True
            return ("Consolidation already in flight — your request was "
                    "coalesced into it. No further calls needed; the "
                    "[MEMORY HYGIENE] flag may persist until the pass lands.")
        if time.monotonic() - last_merge_at < REFRESH_DEBOUNCE_S:
            self._refresh_dirty = True
            remaining = max(
                1, int(REFRESH_DEBOUNCE_S - (time.monotonic() - last_merge_at)) + 1
            )
            return (f"Consolidation ran recently — deferred; it will fire "
                    f"automatically in ~{remaining}s when the debounce window "
                    "elapses. Your edits are already persisted.")
        self._spawn_refresh(trigger)
        return ("Consolidation scheduled in background. It can take a few "
                "minutes — the [MEMORY HYGIENE] flag may persist for several "
                "turns while it runs; do not re-trigger checkpoint_state or "
                "hand-edit the file in the meantime.")

    def _schedule_over_budget_edit(self, over_keys: list[str]) -> bool:
        """Automatic memory-editor trigger (spec 089 §4.2).

        Called by ``ContextManager.prepare()`` with the Layer-1 files it just
        measured over their soft budget — so it fires on files written by any
        writer, external agents included. Synchronous and cheap: it only
        spawns a background pass through the same project-scoped single-flight
        gate and debounce as ``schedule_checkpoint``, and only when a live file
        changed since the last pass and that pass is old enough
        (``memory_editor.auto_run_due``). Never sets the dirty bit: the next
        prepare() re-evaluates anyway. Returns True when it spawned.
        """
        if self._on_session_end_refresh is None or self._session.is_stopped():
            return False
        inflight, last_merge_at = self._refresh_gate_state()
        if inflight is not None and not inflight.done():
            return False
        if time.monotonic() - last_merge_at < REFRESH_DEBOUNCE_S:
            return False
        if self._project_dir:
            from agent_os.agent import memory_editor
            from agent_os.agent.project_paths import ProjectPaths
            if memory_editor.is_running(self._project_dir):
                return False       # e.g. the pinned-chat coordinator's pass
            if not memory_editor.auto_run_due(ProjectPaths(self._project_dir).orbital_dir):
                return False
        logger.info(
            "Memory editor: over-budget trigger at iteration %d (%s)",
            self._current_iteration, ", ".join(over_keys),
        )
        self._spawn_refresh("over_budget")
        return True

    def _maybe_consume_dirty(self) -> None:
        """Turn-boundary pickup of a coalesced/deferred trigger (spec 013)."""
        inflight, last_merge_at = self._refresh_gate_state()
        if (
            self._refresh_dirty
            and self._on_session_end_refresh is not None
            and (inflight is None or inflight.done())
            and time.monotonic() - last_merge_at >= REFRESH_DEBOUNCE_S
        ):
            self._spawn_refresh("agent_decided_coalesced")

    async def drain_refresh(self) -> None:
        """Await any in-flight background pass for this PROJECT. Called before
        the two passes that stay synchronous (token_pressure, session boundary)
        so two consolidations never overlap (spec 013 race #3).

        Project-scoped, not loop-scoped: draining only our own task would let a
        sibling session's pass keep running straight into the synchronous one,
        which is the collision this gate exists to prevent.
        """
        task, _ = self._refresh_gate_state()
        if task is not None and not task.done():
            mine = task is self._refresh_task
            try:
                await task
            except asyncio.CancelledError:
                # Our own cancellation must propagate (terminate()). A sibling
                # session cancelling ITS pass is not a cancellation of this
                # turn — distinguished by who actually got cancelled, so a
                # cancel delivered to us still re-raises.
                if mine or not task.cancelled():
                    raise
            except Exception:
                pass  # _run_refresh already logged it

    async def trigger_checkpoint(self) -> str:
        """Public entry point for the agent-decided refresh trigger.

        Spec 013: schedules a BACKGROUND pass and returns the scheduler's
        status string immediately — never blocks the loop.
        """
        return self.schedule_checkpoint("agent_decided")

    # ------------------------------------------------------------------
    # Cancellation API (cancel_turn / terminate)
    # ------------------------------------------------------------------

    async def cancel_turn(self) -> None:
        """Interrupt the current turn. Loop exits, agent stays alive.

        Backs the POST /api/v2/agents/{pid}/cancel HTTP verb (the UI
        Stop button). Writes a single cancellation marker to the session
        JSONL, cancels any in-flight LLM stream, and breaks the loop's
        while body so the task completes; _on_loop_done then drains any
        queued messages and either broadcasts idle or hot-resumes for
        queued input.

        Idempotent: safe to call when no turn is in flight, or twice in
        rapid succession. Exactly one cancellation marker is written per
        cancellation event regardless of when the cancel landed (during
        stream, between stream and tools, or mid-tool-loop).

        terminate() — used by stop_agent / new_session — sets the stop
        flag in addition to calling cancel_turn(), which causes the
        loop's CancelledError handler to re-raise instead of break (the
        terminate path also tears down sub-agents, browser pages, etc.).
        cancel_turn() alone preserves all of that ancillary state.
        """
        # Idempotency: if we have already initiated a cancel for the
        # current turn (marker pending or appended), do nothing. The flag
        # is reset unconditionally at run() entry so the hot-resume path
        # (a fresh run() after a cancelled turn) does not inherit the
        # prior turn's True flag.
        if self._cancellation_marker_appended_this_turn:
            return

        if self._inflight_stream is None or self._inflight_stream.done():
            # No turn in flight; possibly idle, possibly between LLM and
            # tool exec, possibly mid-tool-loop. Set the flag so the loop
            # body skips remaining tool calls and the post-tool block
            # appends the marker if appropriate.
            self._turn_cancelled = True
            return

        # Claim the inflight task BEFORE any await.
        inflight = self._inflight_stream
        self._inflight_stream = None

        # Mark cancelled FIRST so any concurrent loop work observes it,
        # then mark "marker pending for this turn" to make further
        # cancel_turn() calls no-ops until a new turn starts.
        self._turn_cancelled = True
        self._cancellation_marker_appended_this_turn = True

        # Persist the marker SYNCHRONOUSLY before the await so its JSONL
        # position is adjacent to the cancelled turn. The loop's break-
        # on-cancel exit means nothing else writes to the session after
        # this synchronous append before the task completes, so marker
        # position is trivially correct under the current design.
        # A Stop during the memory record turn (spec 112) ends it; the
        # marker itself stays a visible row.
        self._session.row_kind = None
        self._persist_cancellation_marker_inner()

        inflight.cancel()
        try:
            await asyncio.wait_for(inflight, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            logger.exception("inflight stream raised during cancel_turn")
        # Budget Piece 2: the legacy partial-output cost debit
        # (_debit_cancelled_turn_cost → _budget_spent_usd / on_cost_update) was
        # deleted. The token ledger is the single source of recorded spend and
        # records whole responses; partial cancelled-stream tokens are not
        # ledgered (the provider's final usage chunk never arrived), which is
        # acceptable — the next guard evaluation reads the ledger total.

    async def terminate(self) -> None:
        """End the loop. Cancels current turn and exits the while True.

        Session JSONL preserved; agent task done; handle cleanup is the
        caller's responsibility (caller awaits self._task or shields it).
        Do NOT await self._task here — this would deadlock if terminate
        was invoked from within the loop's own task.
        """
        # Cancel any in-flight refresh task before cancelling the turn.
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
        await self.cancel_turn()
        # Cooperative flag: loop's between-iterations check picks this up.
        self._session.stop()
        # Settle the spend corner, THEN disarm. This runs AFTER cancel_turn() +
        # session.stop() deliberately: those two are the last moments a ledger
        # append can land, so flushing here catches the final numbers instead of
        # racing them. aclose() now FLUSHES rather than cancels — the pending
        # payload it used to drop was systematically the run's last append, i.e.
        # exactly the figure the user then stared at for the whole idle period.
        # It leaves nothing pending and no armed timer, so nothing can fire
        # later against a torn-down loop.
        await self._spend_broadcaster.aclose()
        # Release the ledger chokepoint hook. Identity-checked inside, so if a
        # newer loop for this workspace already replaced ours, this is a no-op
        # and the successor keeps broadcasting.
        if self._project_dir is not None:
            from agent_os.budget import ledger as _ledger_mod
            _ledger_mod.unregister_append_hook(
                self._project_dir, self._on_ledger_append,
            )
        # Re-cancel: the await in cancel_turn() above can let the loop's own
        # task wake into the turn-boundary block (schedule_checkpoint /
        # _maybe_consume_dirty) BEFORE session.stop() lands above, spawning a
        # new pass that the first cancel (before cancel_turn) never saw.
        # Closes that window (spec 013 race #7 — "terminate() cancels the
        # in-flight pass").
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_task.cancel()
        if self._task is not None and not self._task.done():
            # If we're called from a different task (typical for /stop),
            # cancel the loop task so it exits the inflight stream wait
            # via the CancelledError -> is_stopped() raise branch.
            cur = None
            try:
                cur = asyncio.current_task()
            except RuntimeError:
                pass
            if cur is not self._task:
                self._task.cancel()

    def _persist_cancellation_marker(self) -> None:
        """Append a system message indicating user cancellation.

        Called exactly once per cancellation event. Sets the per-turn
        flag so cancel_turn() and the post-tool-loop block do not double-
        append. The flag is reset at the top of each loop iteration.
        """
        if self._cancellation_marker_appended_this_turn:
            return
        self._cancellation_marker_appended_this_turn = True
        self._persist_cancellation_marker_inner()

    def _persist_cancellation_marker_inner(self) -> None:
        """Write the cancellation marker without re-checking the flag.

        Internal helper used by cancel_turn() once it has already claimed
        the per-turn marker slot via _cancellation_marker_appended_this_turn.

        Sets _exit_reason = "cancelled" synchronously so the queue
        dispatcher's _await_and_handle observes the cancellation as soon as
        the loop task terminates. Setting this here (rather than from the
        dispatcher) makes the signal correct for any cancel path, including
        the /cancel HTTP verb that doesn't go through QueueDispatcher.pause.
        """
        self._exit_reason = "cancelled"
        self._session.append({
            "role": "system",
            "content": "[cancelled by user]",
            "source": "management",
            "cancelled_by_user": True,
        })
