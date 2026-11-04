# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Codex app-server transport: JSON-RPC 2.0 over stdio.

Replaces the PTY screen-scrape path for Codex with a real protocol client
(`codex app-server`), giving a genuine turn boundary. Every wire shape here
is [OBSERVED] against codex-cli 0.125.0 — see
artifacts-2026-06-06-codex-lifecycle/codex_appserver/FINDINGS-codex-appserver-lifecycle-parity.md.
The protocol is version-pinned: on a codex upgrade, re-run
`codex app-server generate-json-schema --out <dir>` and the codex test suite.
Re-verified on 0.144.5 (2026-07-16): schema diff additive on every surface
used here (new server-sent approval fields startedAtMs/environmentId;
AbsolutePathBuf renamed LegacyAppPathString, still a string); every value we
SEND (on-request/never, workspace-write) unchanged; live suite green.

Duck-typed surface mirrors SDKTransport so SubAgentManager / CLIAdapter /
ProcessManager contracts are untouched: dispatch(), _event_queue,
turn_complete {cause, session_id, model, rollout_path}, _pending_approvals,
_pending_approval_data, _proc (psutil), _resume_session_id, update_autonomy().

Auth: on-disk ChatGPT-OAuth reuse, headless. NEVER writes auth state and
NEVER sets CODEX_HOME.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import json
import logging
import os
import re
import uuid
from typing import AsyncIterator

import psutil

from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.models import detect_os
from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.agent.transports.jsonl_stream import (
    LineTooLongError,
    read_jsonl_line,
)
from agent_os.agent.transports.process_kill import kill_spawned_tree
from agent_os.agent.transports.tool_risk import should_auto_approve
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

SUPPORTED_CODEX_VERSION = "0.162.1"

# The app-server's refusal when another app-server (the Codex desktop app
# with the thread loaded, typically) holds the thread's writer lock. Spec 126:
# only this exact refusal is answered with a thread/fork.
_ACTIVE_WRITER_REFUSAL = "already has an active writer"


class CodexRpcError(RuntimeError):
    """A JSON-RPC error reply to one of our requests. ``str()`` is unchanged
    from the plain RuntimeError it replaces; ``code``/``message`` are the
    reply's fields when it had the standard shape."""

    def __init__(self, method: str, error) -> None:
        super().__init__(f"codex {method} failed: {error}")
        self.method = method
        self.error = error
        self.code = error.get("code") if isinstance(error, dict) else None
        self.message = (str(error.get("message") or "") if isinstance(error, dict)
                        else str(error))

    def is_active_writer_refusal(self) -> bool:
        return _ACTIVE_WRITER_REFUSAL in self.message

# Autonomy preset -> (approvalPolicy, thread/start sandbox string enum).
# FINDINGS A4a: `untrusted` is NOT "ask me everything" — it silently
# auto-REJECTS all escalation without asking. It must never appear here.
_POLICY_BY_AUTONOMY: dict[Autonomy, tuple[str, str]] = {
    Autonomy.HANDS_OFF: ("never", "workspace-write"),
    Autonomy.CHECK_IN: ("on-request", "workspace-write"),
    Autonomy.SUPERVISED: ("on-request", "workspace-write"),
}
_DEFAULT_POLICY = ("on-request", "workspace-write")


def policy_for_autonomy(autonomy, os_name: str | None = None) -> tuple[str, str]:
    """``(approvalPolicy, sandbox)`` for a thread under *autonomy*.

    Windows hands-off is ``on-request``, not ``never``: Codex has no sandbox
    there unless the user ran its elevated setup, and with ``never`` it then
    REJECTS every command ("blocked by policy") instead of asking — the
    keyless onboarding worker could not even list the folder (Windows
    batch-1022 run, 2026-10-23). ``on-request`` makes Codex ask, and under
    hands-off this transport answers "accept" itself without surfacing
    anything (``should_auto_approve``), so the preset stays unattended. On a
    machine whose Codex sandbox is set up, Codex runs sandboxed and never
    asks, so this costs nothing there either.
    """
    policy = _POLICY_BY_AUTONOMY.get(autonomy, _DEFAULT_POLICY)
    if autonomy is Autonomy.HANDS_OFF and (os_name or detect_os()) == "windows":
        return ("on-request", policy[1])
    return policy

# Cold-start model resolution (TASK-codex-startup-model): the server default
# (gpt-5.x-codex) 400s on ChatGPT-account auth, and OpenAI's guidance marks
# gpt-5.4-mini as the subagent recommendation. Preference order over the
# account's LIVE model/list — a hardcoded single model would re-plant the
# trap on the next model churn.
_PREFERRED_STARTUP_MODELS: tuple[str, ...] = ("gpt-5.4-mini", "gpt-5.4", "gpt-5.5")

# Codex's own Windows sandbox failing to provision: every command fails
# before it starts, in plain ``codex exec`` too (openai/codex #32855, #42513,
# #52179 — ``[windows] sandbox = "elevated"`` whose elevated setup is broken).
# Orbital inherits Codex's access settings and never rewrites its config, so
# the only honest move is to say what is wrong and where the fix lives. The
# note rides the worker's result once per worker, so the manager tells the
# user in their own words (Windows smoke 2026-11-03, decision 1).
_WINDOWS_SANDBOX_SIGNATURES = ("helper_unknown_error", "setup refresh had errors")
WINDOWS_SANDBOX_NOTE = (
    "[Orbital] Codex's own Windows sandbox is not working on this machine: "
    "Codex reports `helper_unknown_error: setup refresh had errors`, so every "
    "command it runs fails before it starts (the same happens in a plain "
    "`codex exec`). Orbital inherits Codex's access settings and does not "
    "change them. The fix is in Codex: run `codex doctor`, or set "
    "`sandbox = \"unelevated\"` under `[windows]` in ~/.codex/config.toml, "
    "then start a new session."
)

# Server->client approval surfaces we answer with {"decision": ...}.
# item/permissions/requestApproval exists in the schema with a different,
# non-decision response shape and never fired in the probe — it (and any
# other unknown surface) gets a JSON-RPC error: fail-visible, never hang
# (an unanswered server request blocks codex indefinitely).
_APPROVAL_METHODS = {
    "item/commandExecution/requestApproval": "commandExecution",
    "item/fileChange/requestApproval": "fileChange",
}

_VALID_DECISIONS = ("accept", "decline", "cancel")


class CodexTransport(AgentTransport):
    """Transport for `codex app-server` (JSON-RPC 2.0, newline-delimited)."""

    # Honest two-state (FINDINGS A5c/D2): Codex kills the exec process group
    # at turn teardown, so no between-turn background work exists — a
    # "Background" badge would assert a state that never happens. DO NOT set
    # this True for UI parity with Claude Code; the transports display
    # differently because they behave differently.
    supports_background_status = False

    def __init__(self, autonomy: "Autonomy | None" = None,
                 resume_record: dict | None = None,
                 external_sandbox: bool = False):
        record = resume_record or {}
        # Same attribute name as SDKTransport: the manager's honesty
        # downgrade (sub_agent_manager.py ~731) compares it to the record.
        self._resume_session_id: str | None = record.get("session_id")
        # Model is CONFIG, not thread identity (AMENDS piece 2 — see
        # INVESTIGATION-resume-semantics): the record's model is display
        # metadata and is NEVER consulted here. start() resolves the model
        # from the current config-store argv; when none is set the param is
        # omitted and the provider serves the thread's last-used model
        # (wire-verified on both transports).
        self._model: str | None = None
        # True when Orbital picked ``_model`` itself (no user override): only
        # then is a model_not_found answered by re-picking (once) and
        # re-sending the turn. ``_rejected`` holds the error while that runs.
        self._model_auto = False
        self._model_repicked = False
        self._rejected: dict | None = None
        self._last_input: list | None = None
        # ORBITAL_* from the spawn env, re-sent as the thread's
        # shell_environment_policy.set so every command Codex runs sees them.
        self._orbital_env: dict[str, str] = {}
        # Spec 149: reasoning effort from argv, sent on each turn/start.
        self._effort: str | None = None
        self._autonomy: Autonomy | None = autonomy
        self._external_sandbox = external_sandbox
        self._alive = False
        self._workspace = ""
        self._popen: asyncio.subprocess.Process | None = None
        self._proc: psutil.Process | None = None  # manager kill anchor
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        # Notes for the manager about this worker's environment, taken at the
        # turn boundary by ProcessManager (``take_notes``). Each is said once.
        self._pending_notes: list[str] = []
        self._noted_windows_sandbox = False
        # Session identity (thread/start | thread/resume response)
        self._thread_id: str | None = None
        self._rollout_path: str | None = None
        self._effective_model: str | None = None
        # How the thread was opened, in the manager's taxonomy
        # (_provider_confirmed_resume_outcome): ("resumed", None) after a
        # resume or a fork of a held thread (full history either way),
        # ("fresh", "resume_failed") when the fork failed too and a fresh
        # thread had to be started.
        self._resume_outcome: tuple[str, str | None] = ("fresh", None)
        # In-flight turn — REQUIRED for turn/interrupt (FINDINGS R2: both
        # threadId AND turnId; missing turnId -> -32600). Cleared on
        # turn/completed, so interrupt-without-turnId is impossible by
        # construction.
        self._turn_id: str | None = None
        self._turn_open = False
        self._stopping = False
        # JSON-RPC client plumbing
        self._next_id = 0
        self._response_futures: dict[int, asyncio.Future] = {}
        # Pending approvals — the manager scans these two dicts by name
        # (resolve_sub_agent_approval / get_pending_sub_agent_approval).
        self._pending_approvals: dict[str, dict] = {}       # request_id -> {rpc_id, method}
        self._pending_approval_data: dict[str, dict] = {}   # request_id -> card payload
        # Per-turn message accumulation. Keyed by the delta's `itemId`,
        # which IS the completed item's `id` on the wire — _on_item pops by
        # item["id"] intentionally (same identifier, two param names).
        self._message_parts: dict[str, str] = {}
        self._final_texts: list[str] = []          # message texts (send() return)
        # Spec 100: reasoning items whose summary streamed as deltas, so the
        # completed item does not repeat it whole.
        self._reasoning_streamed: set[str] = set()
        self._turn_done: asyncio.Event | None = None
        # Display-only usage capture (P3-B). thread/tokenUsage/updated fires
        # repeatedly within a turn carrying the cumulative `total` and the
        # per-turn `last`; we want exactly ONE ledger event per (threadId,
        # turnId) using the LAST `last` value seen. Dedup mechanism:
        # replace-tracking — store the most recent `last` breakdown keyed by
        # (threadId, turnId), then emit on the turn boundary (turn/completed)
        # and drop the key. The map is bounded (one live turn at a time) and a
        # stuck entry is harmless (display-only, never counts toward
        # enforcement). Keyed by (threadId, turnId) so a resumed/late
        # notification for a finished turn can't double-emit.
        self._codex_last_usage: dict[tuple[str, str], dict] = {}
        # Spec 099: strong ref to the post-start quota read.
        self._rate_limit_task: asyncio.Task | None = None
        self._repick_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # turn bookkeeping
    # ------------------------------------------------------------------

    def _begin_turn(self) -> None:
        self._turn_open = True
        self._turn_id = None
        self._message_parts = {}
        self._final_texts = []
        self._turn_done = asyncio.Event()

    def _turn_meta(self, cause: str) -> dict:
        # ProcessManager persists session_id+model+rollout_path for
        # cause in ("success", "error", "interrupted").
        return {
            "cause": cause,
            "session_id": self._thread_id,
            "model": self._effective_model or self._model,
            "rollout_path": self._rollout_path,
        }

    # ------------------------------------------------------------------
    # message routing (unit-testable without a process)
    # ------------------------------------------------------------------

    async def _route_server_message(self, msg: dict) -> None:
        if "id" in msg and "method" not in msg:
            # Response to one of our requests.
            fut = self._response_futures.get(msg["id"])
            if fut is not None and not fut.done():
                fut.set_result(msg)
            else:
                logger.debug("CodexTransport: orphan response id=%s", msg.get("id"))
            return
        if "id" in msg and "method" in msg:
            await self._on_server_request(msg)
            return
        await self._on_notification(msg)

    async def _on_server_request(self, msg: dict) -> None:
        method = msg["method"]
        rpc_id = msg["id"]
        params = msg.get("params") or {}
        tool_name = _APPROVAL_METHODS.get(method)
        if tool_name is None:
            # Unknown surface — answer with an error; an unanswered server
            # request blocks codex forever (probe pump timeout finding).
            await self._send_raw({"jsonrpc": "2.0", "id": rpc_id, "error": {
                "code": -32601,
                "message": f"orbital: unsupported approval surface {method}"}})
            await self._event_queue.put(TransportEvent(
                event_type="error",
                data={"error": f"unsupported codex approval surface: {method}"},
                raw_text=f"Error: codex requested unsupported approval surface {method}",
            ))
            return
        if self._autonomy is not None and should_auto_approve(tool_name, self._autonomy):
            # Parity with SDKTransport's autonomy filter. With
            # approvalPolicy=never these never fire; defensive belt.
            await self._send_raw({"jsonrpc": "2.0", "id": rpc_id,
                                  "result": {"decision": "accept"}})
            return
        if tool_name == "commandExecution":
            tool_input = {
                "command": params.get("command"),
                "cwd": params.get("cwd"),
                "reason": params.get("reason"),
                # String decisions only; amendment objects (e.g.
                # acceptWithExecpolicyAmendment) are not offered in v1.
                "availableDecisions": [
                    d for d in (params.get("availableDecisions") or [])
                    if isinstance(d, str)],
            }
        else:  # fileChange — params: {grantRoot, itemId, reason, threadId, turnId}
            tool_input = {
                "reason": params.get("reason"),
                "grantRoot": params.get("grantRoot"),
                # Schema-supported subset we implement (FileChangeApprovalDecision)
                "availableDecisions": ["accept", "decline", "cancel"],
            }
        request_id = str(uuid.uuid4())
        self._pending_approvals[request_id] = {"rpc_id": rpc_id, "method": method}
        self._pending_approval_data[request_id] = {
            "request_id": request_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        await self._event_queue.put(TransportEvent(
            event_type="permission_request",
            data=dict(self._pending_approval_data[request_id]),
            raw_text=("Permission requested: "
                      f"{params.get('reason') or params.get('command') or tool_name}"),
        ))

    async def _on_notification(self, msg: dict) -> None:
        method = msg.get("method", "")
        params = msg.get("params") or {}
        if method == "turn/started":
            turn = params.get("turn") or {}
            if turn.get("id"):
                self._turn_id = turn["id"]
            return
        if method == "turn/completed":
            await self._on_turn_completed(params)
            return
        if method == "item/agentMessage/delta":
            item_id = params.get("itemId") or ""
            self._message_parts[item_id] = (
                self._message_parts.get(item_id, "") + params.get("delta", ""))
            return
        if method in ("item/started", "item/completed"):
            await self._on_item(method, params.get("item") or {})
            return
        if method == "item/reasoning/summaryTextDelta":
            # Spec 100: reasoning summaries are display-only thinking. Sent
            # only for some models/accounts; absence renders nothing.
            delta = params.get("delta") or ""
            if delta:
                self._reasoning_streamed.add(params.get("itemId") or "")
                await self._event_queue.put(TransportEvent(
                    event_type="thinking",
                    data={"text": delta, "delta": True}, raw_text=delta))
            return
        if method == "item/reasoning/summaryPartAdded":
            # A new summary part is a paragraph break, once text has streamed.
            if ((params.get("summaryIndex") or 0) > 0
                    and (params.get("itemId") or "") in self._reasoning_streamed):
                await self._event_queue.put(TransportEvent(
                    event_type="thinking",
                    data={"text": "\n\n", "delta": True}, raw_text="\n\n"))
            return
        if method == "item/reasoning/textDelta":
            return  # raw reasoning text stays out (spec 100, out of scope)
        if method == "error":
            # ErrorNotification nests the text: {error: {message, ...}, willRetry}.
            text = ((params.get("error") or {}).get("message")
                    or params.get("message") or json.dumps(params)[:500])
            self._observe_environment_failure(str(text))
            if (self._model_auto and not self._model_repicked and self._turn_open
                    and "model_not_found" in json.dumps(params)):
                self._rejected = params  # held for the re-pick in _on_turn_completed
                return
            await self._event_queue.put(TransportEvent(
                event_type="error", data={"error": params},
                raw_text=f"Error: {text}"))
            return
        if method == "thread/tokenUsage/updated":
            # Display-only usage capture (P3-B). Track the latest per-turn
            # `last` breakdown; the actual ledger emission happens on the turn
            # boundary (turn/completed). This notification NEVER drives idle.
            self._track_token_usage(params)
            return
        if method == "account/rateLimits/updated":
            # Spec 099: a sparse rolling quota update, merged into the quota
            # store. Display-only: no event, NEVER drives idle, no transcript
            # row.
            from agent_os.daemon_v2.quota_store import publish_codex_snapshot
            publish_codex_snapshot(params.get("rateLimits"), push=True)
            return
        # thread/started, thread/status/changed, mcpServer/* — supplementary
        # only. NEVER drives idle (TEST RULE 1) and not worth a transcript row.
        logger.debug("CodexTransport: ignoring notification %s", method)

    def _track_token_usage(self, params: dict) -> None:
        """Record the latest per-turn `last` token breakdown for this turn.

        Replace-tracking dedup: the notification fires repeatedly per turn with
        a cumulative `total` and the per-turn `last`. We keep only the most
        recent `last` keyed by (threadId, turnId); the single ledger event is
        emitted on the turn boundary (see ``_emit_token_usage``). Never raises —
        a malformed notification is logged and skipped, leaving the turn intact.
        """
        try:
            thread_id = params.get("threadId")
            turn_id = params.get("turnId")
            token_usage = params.get("tokenUsage") or {}
            last = token_usage.get("last")
            if not (thread_id and turn_id and isinstance(last, dict)):
                return
            # Last-write-wins: a later notification for the same turn replaces
            # the prior `last` (the final notification carries the turn-final
            # per-turn figures).
            self._codex_last_usage[(thread_id, turn_id)] = last
        except Exception:  # noqa: BLE001 — capture must never break a turn
            logger.warning("CodexTransport: token-usage tracking failed; "
                           "continuing without it.", exc_info=True)

    def _emit_token_usage(self, turn_id: str | None) -> None:
        """Append one ``subagent:codex`` ledger event for a completed turn.

        Display-only (Budget Piece 3 Part B): captured tokens are shown in the
        cost view but NEVER counted toward enforcement (the guard passes
        ``sources=["management"]``; the P3-A pinned regression proves
        isolation). NEVER raises.

        Dedup: exactly one event per (threadId, turnId), using the LAST `last`
        breakdown tracked for that turn. The key is popped on emit so a late
        duplicate notification cannot double-ledger.

        Codex `last` semantics ([OBSERVED] in
        artifacts-2026-06-06-codex-lifecycle/.../traces): the per-turn
        breakdown is ``{totalTokens, inputTokens, cachedInputTokens,
        outputTokens, reasoningOutputTokens}`` where
        ``totalTokens == inputTokens + outputTokens`` and
        ``cachedInputTokens ⊆ inputTokens`` (SUBSET semantics — every trace line
        has cachedInputTokens < inputTokens, and reasoningOutputTokens is a
        subset of outputTokens, never added on top). That matches OpenAI-compat
        subset normalization: uncached_input = inputTokens − cachedInputTokens,
        cache_read = cachedInputTokens, cache_write = 0, output = outputTokens.
        No reported_cost — Codex emits none.
        """
        try:
            if self._thread_id is None or turn_id is None:
                return
            last = self._codex_last_usage.pop((self._thread_id, turn_id), None)
            if not isinstance(last, dict):
                return  # no usage observed for this turn
            from agent_os.agent.providers.types import TokenUsage
            from agent_os.budget.normalize import (
                ProviderSemantics,
                normalize_usage,
            )
            from agent_os.budget.ledger import (
                SOURCE_SUBAGENT_CODEX,
                LedgerEvent,
                append_event,
            )

            token_usage = TokenUsage(
                input_tokens=int(last.get("inputTokens", 0) or 0),
                output_tokens=int(last.get("outputTokens", 0) or 0),
                cache_read_tokens=int(last.get("cachedInputTokens", 0) or 0),
                cache_write_tokens=0,  # Codex reports no cache-write tier
            )
            normalized = normalize_usage(token_usage, ProviderSemantics.OPENAI_COMPAT)
            model = self._effective_model or self._model or "unknown"
            append_event(
                self._workspace,
                LedgerEvent(
                    session_id=self._thread_id or "",
                    source=SOURCE_SUBAGENT_CODEX,
                    provider="openai",
                    model=model,
                    usage=normalized,
                    # Codex emits no cost — tokens only, never a fabricated $.
                ),
            )
        except Exception:  # noqa: BLE001 — capture must never break a turn
            logger.warning(
                "CodexTransport: sub-agent usage capture failed (thread=%s "
                "turn=%s); continuing without it.",
                self._thread_id, turn_id, exc_info=True)

    async def _on_turn_completed(self, params: dict) -> None:
        turn = params.get("turn") or {}
        status = turn.get("status")
        if self._rejected is not None:
            if status == "failed":
                # Not on the read loop: the re-pick awaits RPC answers it reads.
                self._turn_id = None
                self._model_repicked = True
                self._repick_task = asyncio.create_task(
                    self._repick_rejected_model(), name=f"codex-repick-{id(self)}")
                return
            await self._surface_rejected()
        # Flush agentMessages that streamed deltas but never completed
        # (interrupted mid-message): partial text must not be lost.
        for text in self._message_parts.values():
            if text:
                await self._put_message(text)
        self._message_parts = {}
        if status == "completed":
            cause = "success"
        elif status == "interrupted":
            # Two distinct interrupted cases (review correction — the
            # Part-C silent-hang class):
            #   teardown (self._stopping): "stopped" — silence is safe,
            #     stop_for_user's on_user_stopped speaks for the teardown;
            #   agent lives on (cancel approval decision, no teardown):
            #     "interrupted" — ProcessManager routes it to
            #     on_turn_interrupted so an AWAITING management session is
            #     woken honestly instead of hanging forever.
            cause = "stopped" if self._stopping else "interrupted"
        else:  # "failed" (schema TurnStatus) or unrecognized
            cause = "error"
        # Display-only usage capture (P3-B): emit the single ledger event for
        # this turn from the LAST tracked `last` breakdown, BEFORE clearing
        # _turn_id. Prefer the completed turn's own id (the notification carries
        # it) so a turn that completed without _turn_id ever being set on self
        # still attributes correctly; fall back to the in-flight _turn_id.
        completed_turn_id = turn.get("id") or self._turn_id
        self._emit_token_usage(completed_turn_id)
        self._turn_id = None
        self._turn_open = False
        await self._event_queue.put(TransportEvent(
            event_type="turn_complete", data=self._turn_meta(cause)))
        if self._turn_done is not None:
            self._turn_done.set()

    async def _surface_rejected(self) -> None:
        """Emit the held model_not_found as the error it was."""
        params, self._rejected = self._rejected, None
        if params is None:
            return
        text = ((params.get("error") or {}).get("message")
                or params.get("message") or json.dumps(params)[:500])
        await self._event_queue.put(TransportEvent(
            event_type="error", data={"error": params}, raw_text=f"Error: {text}"))

    async def _close_failed_turn(self) -> None:
        await self._surface_rejected()
        self._turn_id = None
        self._turn_open = False
        await self._event_queue.put(TransportEvent(
            event_type="turn_complete", data=self._turn_meta("error")))
        if self._turn_done is not None:
            self._turn_done.set()

    async def _repick_rejected_model(self) -> None:
        """The server refused a model Orbital picked (a stale model/list
        cache can still offer a retired one): pick again from the list
        without it and re-send the same input once, on the same turn
        bookkeeping, so the worker answers as if nothing happened. Any
        failure here surfaces the original error and closes the turn."""
        rejected = self._model
        try:
            model = await self._resolve_startup_model(exclude={rejected} if rejected else None)
            if not model or model == rejected or not self._last_input:
                await self._close_failed_turn()
                return
            logger.info("CodexTransport: %s was rejected (model_not_found); "
                        "re-sending the turn on %s", rejected, model)
            self._model, self._effective_model = model, None
            params = self._turn_start_params(self._last_input)
            params["model"] = model
            # Cleared before sending: the new turn can complete before this
            # task resumes from the await.
            held, self._rejected = self._rejected, None
            try:
                result = await self._request("turn/start", params)
            except Exception:
                self._rejected = held
                raise
        except Exception:  # noqa: BLE001
            logger.warning("CodexTransport: re-pick after model_not_found failed",
                           exc_info=True)
            await self._close_failed_turn()
            return
        turn = result.get("turn") or {}
        if turn.get("id") and self._turn_open:
            self._turn_id = turn["id"]

    async def _on_item(self, phase: str, item: dict) -> None:
        itype = item.get("type")
        if itype == "agentMessage":
            if phase == "item/completed":
                item_id = item.get("id") or ""
                text = item.get("text") or self._message_parts.get(item_id, "")
                self._message_parts.pop(item_id, None)
                if text:
                    self._observe_environment_failure(text)
                    await self._put_message(text, phase_label=item.get("phase"))
            return
        if itype == "commandExecution":
            command = item.get("command", "")
            started = phase == "item/started"
            output = (item.get("aggregatedOutput") or "")[:2000]
            data = {
                "tool_name": "commandExecution",
                "tool_id": item.get("id"),
                # No run_in_background key — Codex has no surviving
                # background work (FINDINGS A5c); the provenance
                # registry stays inert for this transport by design.
                "tool_input": {"command": command, "cwd": item.get("cwd")},
                "status": item.get("status"),
                "exit_code": item.get("exitCode"),
                "aggregated_output": output,
            }
            if not started:
                # Spec 100: the capsule row's result, in the shared key.
                data["result"] = output
                data["is_error"] = item.get("status") == "failed"
                self._observe_environment_failure(output)
            await self._event_queue.put(TransportEvent(
                event_type="tool_use",
                data=data,
                raw_text=(f"[Running command: {command}]" if started else
                          f"[Command finished (exit {item.get('exitCode')}): {command}]"),
            ))
            return
        if itype == "fileChange":
            changes = item.get("changes") or []
            paths = ", ".join(c.get("path", "") for c in changes)
            verb = "Editing" if phase == "item/started" else "Edited"
            data = {
                "tool_name": "fileChange",
                "tool_id": item.get("id"),
                "tool_input": {"changes": changes},
                "status": item.get("status"),
            }
            if phase == "item/completed":
                # The row's result is the diff (FileUpdateChange.diff).
                data["result"] = "\n".join(
                    str(c.get("diff") or c.get("path") or "")
                    for c in changes if isinstance(c, dict))
                data["is_error"] = item.get("status") == "failed"
            await self._event_queue.put(TransportEvent(
                event_type="tool_use",
                data=data,
                raw_text=f"[{verb} files: {paths}]",
            ))
            return
        if itype == "reasoning":
            item_id = item.get("id") or ""
            if phase == "item/completed":
                streamed = item_id in self._reasoning_streamed
                self._reasoning_streamed.discard(item_id)
                summary = [p for p in (item.get("summary") or [])
                           if isinstance(p, str) and p.strip()]
                if summary and not streamed:
                    text = "\n\n".join(summary)
                    await self._event_queue.put(TransportEvent(
                        event_type="thinking", data={"text": text}, raw_text=text))
            return
        tool = _codex_tool_item(itype, item, completed=phase == "item/completed")
        if tool is not None:
            await self._event_queue.put(tool)
            return
        # userMessage echo, todoList, plan, ... — no transcript value.

    async def _put_message(self, text: str, phase_label: str | None = None) -> None:
        self._final_texts.append(text)
        data: dict = {"text": text}
        if phase_label:
            data["phase"] = phase_label
        await self._event_queue.put(TransportEvent(
            event_type="message", data=data, raw_text=text))

    # ------------------------------------------------------------------
    # wire I/O
    # ------------------------------------------------------------------

    async def _send_raw(self, msg: dict) -> None:
        if self._popen is None or self._popen.stdin is None:
            raise RuntimeError("codex app-server not started")
        self._popen.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        await self._popen.stdin.drain()

    async def _request(self, method: str, params: dict | None = None,
                       timeout: float = 30.0) -> dict:
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._response_futures[rid] = fut
        try:
            await self._send_raw({"jsonrpc": "2.0", "id": rid,
                                  "method": method, "params": params or {}})
            msg = await asyncio.wait_for(fut, timeout)
        finally:
            self._response_futures.pop(rid, None)
        if "error" in msg:
            raise CodexRpcError(method, msg["error"])
        return msg.get("result") or {}

    async def read_rate_limits(self, timeout: float = 10.0) -> dict | None:
        """``account/rateLimits/read`` over this live connection (spec 099).

        No thread or turn needed; answers immediately. None when the
        transport is down; raises on an RPC error or timeout (the quota
        store's bounded reader maps both to "no fresh value")."""
        if not self._alive:
            return None
        return await self._request("account/rateLimits/read", None,
                                   timeout=timeout)

    async def _publish_rate_limits(self) -> None:
        from agent_os.daemon_v2.quota_store import (
            pick_codex_snapshot,
            publish_codex_snapshot,
        )
        try:
            result = await self.read_rate_limits()
        except Exception as exc:  # noqa: BLE001 — display data only
            logger.info("CodexTransport: rate-limit read failed (%s)", exc)
            return
        publish_codex_snapshot(pick_codex_snapshot(result))

    async def _notify(self, method: str, params: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send_raw(msg)

    # ------------------------------------------------------------------
    # start() + helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _orbital_env_of(env: dict | None) -> dict[str, str]:
        """The spawn env's ORBITAL_* (session, project, agent attribution)."""
        return {k: str(v) for k, v in (env or {}).items() if k.startswith("ORBITAL_")}

    @staticmethod
    def _argv_model(args: list[str]) -> str | None:
        """Extract `-m <model>` injected by the config store's flag_template
        ("-m {value}", sub_agent_config_store.py). app-server takes model
        per-request, never on argv."""
        for i, a in enumerate(args or []):
            if a == "-m" and i + 1 < len(args):
                return args[i + 1]
        return None

    @staticmethod
    def _argv_effort(args: list[str]) -> str | None:
        """Extract ``-c model_reasoning_effort=<v>`` (the config store's codex
        effort flag_template, spec 149). Sent per turn as ``turn/start``'s
        ``effort`` — the app-server's own knob — never on argv."""
        prefix = "model_reasoning_effort="
        for i, a in enumerate(args or []):
            if a == "-c" and i + 1 < len(args) and args[i + 1].startswith(prefix):
                value = args[i + 1][len(prefix):].strip().strip('"\'')
                return value or None
        return None

    def _observe_environment_failure(self, text: str) -> None:
        """A failure Codex cannot recover from on its own and the user must
        fix in Codex. Queues the matching note, once per worker.

        Read on every command output, error notification and agent message:
        where Codex puts the text depends on its version (the spawn failure
        may never become a command item), and the string is specific enough
        that an ordinary conversation never carries it."""
        if self._noted_windows_sandbox:
            return
        if any(sig in text for sig in _WINDOWS_SANDBOX_SIGNATURES):
            self._noted_windows_sandbox = True
            self._pending_notes.append(WINDOWS_SANDBOX_NOTE)

    def take_notes(self) -> list[str]:
        """Environment notes queued since the last call; the caller appends
        them to the turn's result so the manager sees them with it."""
        notes, self._pending_notes = self._pending_notes, []
        return notes

    def _thread_open_request(self, workspace: str) -> tuple[str, dict]:
        """(method, params) opening the thread — fresh or resume.

        Governance params (cwd/approvalPolicy/sandbox) are ALWAYS present in
        BOTH branches: thread/resume applies its own defaults for omitted
        params (observed drift: approvalPolicy never -> on-request), so
        omission silently loosens governance. Only the MODEL is conditional
        (model-is-config): passed when a current override exists, omitted
        otherwise so the provider serves the thread's last-used model.
        """
        approval_policy, sandbox = policy_for_autonomy(self._autonomy)
        params: dict = {"cwd": workspace, "approvalPolicy": approval_policy,
                        "sandbox": sandbox}
        if self._model:
            params["model"] = self._model
        if self._orbital_env:
            # Per key, so the user's own shell_environment_policy (e.g.
            # inherit = "core") still governs everything else.
            params["config"] = {f"shell_environment_policy.set.{k}": v
                                for k, v in self._orbital_env.items()}
        if self._resume_session_id:
            params["threadId"] = self._resume_session_id
            return "thread/resume", params
        return "thread/start", params

    async def _resolve_startup_model(self, exclude: set[str] | None = None) -> str:
        """Pick a known-good startup model from the account's live
        model/list. Fills ONLY the unset fresh-start case — a user override
        wins upstream, and resume omits the model so the provider serves the
        thread's own last-used (the model-is-config amendment). Never
        returns None: a degraded fallback still beats the unqualified
        server default (the cold-start 400).

        Codex's own ``isDefault`` entry comes first: it tracks the account
        and the catalog (0.162.1 listed only gpt-6.x / gpt-5.6, none of the
        preference names). ``exclude`` drops a model the server refused."""
        exclude = exclude or set()
        try:
            result = await self._request("model/list", {"includeHidden": False},
                                         timeout=15.0)
            entries = [m for m in (result.get("data") or [])
                       if m.get("id") and m.get("id") not in exclude]
            available = [m["id"] for m in entries]
        except Exception as e:
            logger.warning(
                "CodexTransport: model/list failed (%s) — degraded startup "
                "fallback to %s", e, _PREFERRED_STARTUP_MODELS[0])
            return _PREFERRED_STARTUP_MODELS[0]
        for entry in entries:
            if entry.get("isDefault"):
                logger.info("CodexTransport: startup model %s (Codex's default)",
                            entry["id"])
                return entry["id"]
        for preferred in _PREFERRED_STARTUP_MODELS:
            if preferred in available:
                if preferred != _PREFERRED_STARTUP_MODELS[0]:
                    logger.info("CodexTransport: startup model %s (preferred "
                                "%s unavailable)", preferred,
                                _PREFERRED_STARTUP_MODELS[0])
                return preferred
        # Churn fallback: every preference entry retired. Avoid the
        # codex-class ids (the known-rejected class on ChatGPT auth).
        for model_id in available:
            if "codex" not in model_id:
                logger.info("CodexTransport: startup model %s (no preference-"
                            "order entry available)", model_id)
                return model_id
        logger.warning(
            "CodexTransport: model/list offered nothing suitable (%s) — "
            "degraded startup fallback to %s", available,
            _PREFERRED_STARTUP_MODELS[0])
        return _PREFERRED_STARTUP_MODELS[0]

    def _check_version(self, init_result: dict) -> None:
        ua = (init_result or {}).get("userAgent", "")
        m = re.search(r"/(\d+\.\d+\.\d+)", ua)
        found = m.group(1) if m else None
        if found != SUPPORTED_CODEX_VERSION:
            logger.warning(
                "CodexTransport: codex %s != pinned %s — the app-server "
                "protocol is version-pinned; re-run `codex app-server "
                "generate-json-schema` and the codex test suite before "
                "trusting this transport on the new version",
                found, SUPPORTED_CODEX_VERSION,
            )

    async def start(self, command: str, args: list[str], workspace: str,
                    env: dict | None = None) -> None:
        self._workspace = workspace
        self._orbital_env = self._orbital_env_of(env)
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        # NEVER set CODEX_HOME / never write auth state: headless reuse of
        # the on-disk ChatGPT-OAuth login is the auth path (FINDINGS Q7).
        binary = command or "codex"
        self._popen = await asyncio.create_subprocess_exec(
            binary, "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace, env=merged_env,
            # App-server event lines carry full command output; the asyncio
            # default (64 KiB) is routinely exceeded. read_jsonl_line() is
            # tolerant of ANY limit — this just sizes the fast path.
            limit=1024 * 1024,
            creationflags=win_no_window_flags(),
        )
        try:
            self._proc = psutil.Process(self._popen.pid)
        except psutil.Error:
            logger.warning("CodexTransport: could not wrap pid=%s in psutil",
                           self._popen.pid)
        self._alive = True
        self._reader_task = asyncio.create_task(
            self._read_loop(), name=f"codex-read-{id(self)}")
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(), name=f"codex-stderr-{id(self)}")

        self._model = self._argv_model(args)
        self._effort = self._argv_effort(args)

        # clientInfo flows into the rollout `originator` (FINDINGS A6).
        init = await self._request("initialize", {"clientInfo": {
            "name": "orbital", "title": "Orbital", "version": "0.1.0"}})
        self._check_version(init)
        await self._notify("initialized")
        # Spec 099: one quota read per handle start, in the background so a
        # slow answer never delays the thread open.
        self._rate_limit_task = asyncio.create_task(
            self._publish_rate_limits(), name=f"codex-ratelimits-{id(self)}")

        await self._open_thread(workspace)

    async def _open_thread(self, workspace: str) -> None:
        """Open the handle's thread: resume, fork when the resume is refused
        because another app-server holds the thread, else start."""
        if self._model is None and not self._resume_session_id:
            # Never open a FRESH thread on the unqualified server default
            # (cold-start 400 on ChatGPT auth). Resume stays omit-when-no-
            # override: the thread inherits its OWN last-used model, which
            # is qualified — not the server default this guards against.
            self._model = await self._resolve_startup_model()
            self._model_auto = True

        method, params = self._thread_open_request(workspace)
        try:
            result = await self._request(method, params)
        except CodexRpcError as e:
            if method != "thread/resume" or not e.is_active_writer_refusal():
                raise
            result = await self._fork_held_thread(params)
        else:
            self._resume_outcome = (
                ("resumed", None) if method == "thread/resume" else ("fresh", None))
        thread = result.get("thread") or {}
        self._thread_id = thread.get("id")
        self._rollout_path = thread.get("path")
        self._effective_model = result.get("model")
        if not self._thread_id:
            raise RuntimeError(
                f"codex thread/start returned no thread id: {result}")

    async def _fork_held_thread(self, params: dict) -> dict:
        """Spec 126 B: the thread is held by another app-server (the Codex
        desktop app keeps the writer lock while it has the thread loaded,
        not only during a turn), so take a writable copy with the full
        history: ``thread/fork`` with the SAME governance params as the
        resume (fork applies its own defaults for omitted ones). The fork's
        id goes out on turn_complete as the resume record, so later fires
        resume the fork instead of forking again. If the fork fails too, a
        fresh thread keeps the automation running and ``resume_outcome``
        says the history was lost."""
        held = params.get("threadId")
        try:
            # Forking copies the rollout (tens of MB for an old thread).
            result = await self._request("thread/fork", dict(params), timeout=120.0)
        except CodexRpcError as e:
            logger.warning(
                "CodexTransport: codex thread %s is held by another app-server "
                "and thread/fork failed (%s) — starting a fresh thread", held, e)
            fresh = {k: v for k, v in params.items() if k != "threadId"}
            if not fresh.get("model"):
                self._model = await self._resolve_startup_model()
                self._model_auto = True
                fresh["model"] = self._model
            result = await self._request("thread/start", fresh)
            self._resume_outcome = ("fresh", "resume_failed")
            return result
        forked = (result.get("thread") or {}).get("id")
        logger.info("CodexTransport: codex thread %s is held by another "
                    "app-server; forked to %s", held, forked)
        self._resume_outcome = ("resumed", None)
        return result

    async def _drain_stderr(self) -> None:
        try:
            while self._popen is not None and self._popen.stderr is not None:
                try:
                    line = await read_jsonl_line(self._popen.stderr)
                except LineTooLongError as e:
                    logger.warning("CodexTransport: dropped oversized "
                                   "stderr line (%s)", e)
                    continue
                if not line:
                    return
                logger.debug("codex stderr: %s",
                             line.decode("utf-8", errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _read_loop(self) -> None:
        try:
            while self._popen is not None and self._popen.stdout is not None:
                try:
                    line = await read_jsonl_line(self._popen.stdout)
                except LineTooLongError as e:
                    logger.warning("CodexTransport: dropped oversized "
                                   "stdout line (%s)", e)
                    continue
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    logger.warning("CodexTransport: unparseable stdout: %.200s", line)
                    continue
                try:
                    await self._route_server_message(msg)
                except Exception:
                    logger.exception("CodexTransport: routing failed: %.200s", msg)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A dead reader is a dead transport: log NOW (not at GC time as
            # "Task exception was never retrieved") and kill the process —
            # codex would otherwise run the turn on orphaned, then block
            # forever on its next server->client request (the 2026-07-17
            # zombie, pid 75807). The finally block closes the turn honestly.
            logger.exception("CodexTransport: read loop died — killing "
                             "codex pid=%s",
                             getattr(self._popen, "pid", None))
            if self._popen is not None and self._popen.returncode is None:
                try:
                    self._popen.kill()
                except ProcessLookupError:
                    pass
        finally:
            # EOF or cancel. An open turn at stream end (and not stopping)
            # is an abnormal death — close it honestly so the adapter is
            # not stuck busy forever (CLIAdapter flips idle only on the
            # turn_complete chunk; its post-stream idle line is unreachable
            # on the transport path).
            if self._turn_open and not self._stopping:
                await self._event_queue.put(TransportEvent(
                    event_type="turn_complete", data=self._turn_meta("error")))
                self._turn_open = False
                if self._turn_done is not None:
                    self._turn_done.set()
            for fut in self._response_futures.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("codex app-server stream ended"))
            self._alive = False

    # ------------------------------------------------------------------
    # dispatch / send
    # ------------------------------------------------------------------

    async def dispatch(self, message: str) -> None:
        """Fire-and-forget turn start; events flow via read_stream().
        The manager prefers this over send() (hasattr check)."""
        if not self._alive or self._thread_id is None:
            raise RuntimeError("codex transport not started — call start() first")
        self._begin_turn()
        self._last_input = [{"type": "text", "text": message}]
        params = self._turn_start_params(self._last_input)
        logger.info("CodexTransport: turn/start thread=%s model=%s effort=%s",
                    self._thread_id, self._effective_model or self._model,
                    params.get("effort"))
        try:
            result = await self._request("turn/start", params)
        except Exception as e:
            await self._event_queue.put(TransportEvent(
                event_type="error", data={"error": str(e)},
                raw_text=f"Error: {e}"))
            await self._event_queue.put(TransportEvent(
                event_type="turn_complete", data=self._turn_meta("error")))
            self._turn_open = False
            if self._turn_done is not None:
                self._turn_done.set()
            raise
        turn = (result.get("turn") or {})
        # Guard: turn/started notification may set _turn_id BEFORE the
        # turn/start response arrives; only update if the turn is still open
        # (the notification hasn't already closed it via turn/completed).
        if turn.get("id") and self._turn_open:
            self._turn_id = turn["id"]

    def _turn_start_params(self, turn_input: list) -> dict:
        params: dict = {"threadId": self._thread_id, "input": turn_input}
        if self._external_sandbox:
            # Constraint 7: Orbital's own sandbox wraps the process; Codex
            # skips internal enforcement. Tagged-object form exists ONLY on
            # turn/start (param asymmetry).
            params["sandboxPolicy"] = {"type": "externalSandbox"}
        effort = getattr(self, "_effort", None)
        if effort:
            # Spec 149: the thinking level, per turn (protocol: "this turn and
            # subsequent turns"). Omitted = the model's own default.
            params["effort"] = effort
        return params

    async def send(self, message: str) -> str | None:
        """Blocking ABC variant: dispatch + wait for the turn boundary."""
        try:
            await self.dispatch(message)
        except Exception as e:
            return f"Error: codex dispatch failed: {e}"
        done = self._turn_done
        try:
            await asyncio.wait_for(done.wait(), timeout=600.0)
        except asyncio.TimeoutError:
            return "Error: codex turn did not complete within 600s"
        return "\n".join(self._final_texts) or "(no response)"

    # ------------------------------------------------------------------
    # approval responses
    # ------------------------------------------------------------------

    async def respond_to_permission(self, permission_id: str, approved: bool) -> None:
        """Boolean wire (existing /approve + /deny REST): True -> accept;
        False -> decline — the turn CONTINUES and the agent adapts
        (FINDINGS A4d). 'Deny & stop' is the explicit cancel path below."""
        await self.respond_to_permission_decision(
            permission_id, "accept" if approved else "decline")

    async def respond_to_permission_decision(self, permission_id: str,
                                             decision: str) -> None:
        if decision not in _VALID_DECISIONS:
            raise ValueError(f"unsupported codex approval decision: {decision}")
        pending = self._pending_approvals.pop(permission_id, None)
        self._pending_approval_data.pop(permission_id, None)
        if pending is None:
            return
        await self._send_raw({"jsonrpc": "2.0", "id": pending["rpc_id"],
                              "result": {"decision": decision}})

    def update_autonomy(self, preset: Autonomy) -> None:
        """Live preset update. Affects the auto-approve filter immediately;
        thread-level approvalPolicy stays as set at thread/start (sub-agents
        run HANDS_OFF today, so this is future-governance plumbing)."""
        self._autonomy = preset

    # ------------------------------------------------------------------
    # stream / lifecycle
    # ------------------------------------------------------------------

    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        while self._alive:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stopping = True
        # 1. Pending approvals: Stop while a question is open == the
        #    "Deny & stop" semantic -> cancel (turn ends `interrupted`),
        #    not decline (which would let the turn run on into the kill).
        for request_id in list(self._pending_approvals):
            try:
                await self.respond_to_permission_decision(request_id, "cancel")
            except Exception:
                logger.debug("CodexTransport.stop: cancel of %s failed",
                             request_id, exc_info=True)
        # 2. Interrupt the open turn — requires threadId AND turnId. The
        #    ~2 ms ack stops the model loop only; the in-flight exec keeps
        #    running and is killed by the tree-walk below.
        if self._turn_id is not None and self._thread_id is not None:
            try:
                await self._request("turn/interrupt",
                                    {"threadId": self._thread_id,
                                     "turnId": self._turn_id}, timeout=2.0)
            except Exception:
                logger.debug("CodexTransport.stop: interrupt best-effort failed",
                             exc_info=True)
        # 3. Unsubscribe (there is no shutdown RPC).
        if self._thread_id is not None:
            try:
                await self._request("thread/unsubscribe",
                                    {"threadId": self._thread_id}, timeout=2.0)
            except Exception:
                logger.debug("CodexTransport.stop: unsubscribe best-effort failed",
                             exc_info=True)
        # 4. Reader down BEFORE the kill so EOF handling cannot emit a
        #    spurious error-turn_complete mid-teardown.
        for task in (self._reader_task, self._stderr_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._reader_task = None
        self._stderr_task = None
        # 5. Tree-walk kill: SIGTERM alone leaks in-flight execs (reparent
        #    to pid 1). Reuses the Piece-3 reap — NOT a codex-specific one.
        if self._proc is not None:
            try:
                from agent_os.agent.transports.process_kill import kill_process_tree
                outcome = await kill_process_tree(self._proc, label="codex")
                if not outcome.parent_dead:
                    logger.error("CodexTransport.stop: codex app-server "
                                 "survived kill; PID may leak")
            except Exception:
                logger.exception("CodexTransport.stop: kill_process_tree raised")
        elif self._popen is not None and self._popen.returncode is None:
            # psutil wrap failed at start() — no tree handle, but never
            # leave the root process running. Direct kill is the floor.
            try:
                self._popen.kill()
            except ProcessLookupError:
                pass
        # 6. Close stdio, drop refs.
        if self._popen is not None and self._popen.stdin is not None:
            try:
                self._popen.stdin.close()
            except Exception:
                pass
        self._popen = None
        self._proc = None
        self._alive = False

    def is_alive(self) -> bool:
        return (self._alive and self._popen is not None
                and self._popen.returncode is None)

    @property
    def session_id(self) -> str | None:
        return self._thread_id

    @property
    def resume_outcome(self) -> tuple[str, str | None]:
        return self._resume_outcome

    # ------------------------------------------------------------------
    # resume pre-check
    # ------------------------------------------------------------------

    @staticmethod
    def resume_source_path(record: dict) -> str | None:
        """The rollout JSONL a thread record resolves to, or None.

        The saved ``rollout_path`` when it is still there; else the exact
        thread id looked up in the sessions store (``$CODEX_HOME`` or
        ``~/.codex``). Metadata only — the file is never opened. Shared by
        the resume pre-check and the spec 128 dispatch path block."""
        path = record.get("rollout_path")
        if path and os.path.isfile(path):
            return path
        thread_id = record.get("session_id")
        if not thread_id:
            return None
        home = os.environ.get("CODEX_HOME") or os.path.join(
            os.path.expanduser("~"), ".codex")
        pattern = os.path.join(home, "sessions", "*", "*", "*",
                               f"rollout-*-{thread_id}.jsonl")
        matches = sorted(_glob.glob(pattern))
        return matches[-1] if matches else None

    @staticmethod
    def resume_source_exists(record: dict) -> bool:
        """Pre-check the rollout file before resuming. Codex pruning policy
        is unmeasured (R4 open) — never trust the resume call to fail
        loudly; a fresh start must never look like a resume."""
        return CodexTransport.resume_source_path(record) is not None


# ----------------------------------------------------------------------
# Short-lived quota probe (spec 099)
# ----------------------------------------------------------------------

def _codex_text_parts(items) -> str:
    """Text of a codex/MCP content list: the ``text`` of each text-like part."""
    if not isinstance(items, list):
        return ""
    parts = []
    for part in items:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif part.get("type") in ("image", "inputImage"):
                parts.append("[image]")
    return "\n".join(parts)


def _codex_tool_item(itype, item: dict, *, completed: bool) -> "TransportEvent | None":
    """Spec 100: a codex tool-shaped ThreadItem as a ``tool_use`` event.

    Covers the tool items besides commandExecution/fileChange. Keyed by the
    item id, emitted on item/started and again (with ``result``) on
    item/completed. The payloads are not a stable public contract (schema
    pinned at codex-cli 0.144.5), so every field is read defensively: an
    unknown type or a malformed item yields None — skip the row, never fail
    the turn.
    """
    try:
        result = None
        is_error = False
        if itype == "mcpToolCall":
            server, tool = item.get("server") or "", item.get("tool") or "tool"
            name = f"{server}.{tool}" if server else str(tool)
            args = item.get("arguments")
            if completed:
                error = item.get("error")
                if isinstance(error, dict) and error.get("message"):
                    result, is_error = str(error["message"]), True
                else:
                    res = item.get("result")
                    result = _codex_text_parts(
                        res.get("content") if isinstance(res, dict) else None)
        elif itype == "dynamicToolCall":
            namespace, tool = item.get("namespace"), item.get("tool") or "tool"
            name = f"{namespace}.{tool}" if namespace else str(tool)
            args = item.get("arguments")
            if completed:
                result = _codex_text_parts(item.get("contentItems"))
                is_error = item.get("success") is False
        elif itype == "webSearch":
            name = "webSearch"
            args = {"query": item.get("query") or ""}
            if isinstance(item.get("action"), dict):
                args["action"] = item["action"]
            if completed:
                result = ""
        elif itype == "imageView":
            name = "imageView"
            args = {"path": item.get("path") or ""}
            if completed:
                result = ""
        elif itype == "imageGeneration":
            # ``result`` is the image payload itself; never the row text.
            name = "imageGeneration"
            args = {"prompt": item.get("revisedPrompt") or ""}
            if completed:
                result = str(item.get("savedPath") or "")
        elif itype == "collabAgentToolCall":
            name = str(item.get("tool") or "collabAgent")
            args = {"prompt": item.get("prompt") or "", "model": item.get("model")}
            if completed:
                result = ""
        else:
            return None
        status = item.get("status")
        if completed and status == "failed":
            is_error = True
        if not isinstance(args, dict):
            args = {} if args is None else {"arguments": args}
        data: dict = {
            "tool_name": name,
            "tool_id": item.get("id"),
            "tool_input": args,
            "status": status,
        }
        if completed:
            data["result"] = result or ""
            data["is_error"] = is_error
            raw_text = f"[{name} {'failed' if is_error else 'finished'}]"
        else:
            raw_text = f"[Running {name}]"
        return TransportEvent(event_type="tool_use", data=data, raw_text=raw_text)
    except Exception:
        logger.debug("CodexTransport: skipping malformed %s item", itype, exc_info=True)
        return None


async def fetch_codex_rate_limits(binary: str = "codex", *,
                                  timeout: float = 10.0) -> dict | None:
    """Spawn ``<binary> app-server`` just long enough to answer
    ``account/rateLimits/read`` (initialize → initialized → read → kill).

    For when no live codex handle exists. Async, bounded by ``timeout`` in
    total, and never raises: any failure (binary missing, not logged in,
    protocol drift, timeout) returns None. Same handshake as
    ``codex_models.read_model_ids``.
    """
    proc = None

    async def exchange() -> dict:
        def send(obj: dict) -> None:
            proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))

        async def read_response(rpc_id: int) -> dict:
            while True:
                line = await read_jsonl_line(proc.stdout)
                if not line:
                    raise RuntimeError(
                        "codex app-server closed the stream before responding")
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("id") == rpc_id:
                    if "error" in msg:
                        raise RuntimeError(f"codex error: {msg['error']}")
                    return msg.get("result") or {}

        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"clientInfo": {
                  "name": "orbital", "title": "Orbital", "version": "0.1.0"}}})
        await proc.stdin.drain()
        await read_response(1)
        send({"jsonrpc": "2.0", "method": "initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "account/rateLimits/read"})
        await proc.stdin.drain()
        return await read_response(2)

    try:
        proc = await asyncio.create_subprocess_exec(
            binary, "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=1024 * 1024,
            creationflags=win_no_window_flags(),
        )
        return await asyncio.wait_for(exchange(), timeout)
    except Exception as exc:  # noqa: BLE001 — display data, never raise
        logger.info("codex rate-limit probe unavailable (%s: %s)",
                    type(exc).__name__, exc)
        return None
    finally:
        # The whole tree: on Windows `codex` is npm's codex.CMD, and killing
        # only cmd.exe left node + codex.exe running after every usage read.
        await kill_spawned_tree(proc, label="codex rate-limit probe")
