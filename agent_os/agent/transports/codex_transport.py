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
from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.agent.transports.tool_risk import should_auto_approve

logger = logging.getLogger(__name__)

SUPPORTED_CODEX_VERSION = "0.125.0"

# Autonomy preset -> (approvalPolicy, thread/start sandbox string enum).
# FINDINGS A4a: `untrusted` is NOT "ask me everything" — it silently
# auto-REJECTS all escalation without asking. It must never appear here.
_POLICY_BY_AUTONOMY: dict[Autonomy, tuple[str, str]] = {
    Autonomy.HANDS_OFF: ("never", "workspace-write"),
    Autonomy.CHECK_IN: ("on-request", "workspace-write"),
    Autonomy.SUPERVISED: ("on-request", "workspace-write"),
}
_DEFAULT_POLICY = ("on-request", "workspace-write")

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
        # Thread identity: persisted model wins; for fresh starts the
        # config-store `-m <model>` argv is parsed at start() time.
        self._model: str | None = record.get("model")
        self._autonomy: Autonomy | None = autonomy
        self._external_sandbox = external_sandbox
        self._alive = False
        self._workspace = ""
        self._popen: asyncio.subprocess.Process | None = None
        self._proc: psutil.Process | None = None  # manager kill anchor
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        # Session identity (thread/start | thread/resume response)
        self._thread_id: str | None = None
        self._rollout_path: str | None = None
        self._effective_model: str | None = None
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
        self._turn_done: asyncio.Event | None = None

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
        # Approval surfaces — implemented in the lifecycle task.
        pass

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
        if method == "error":
            text = params.get("message") or json.dumps(params)[:500]
            await self._event_queue.put(TransportEvent(
                event_type="error", data={"error": params},
                raw_text=f"Error: {text}"))
            return
        # thread/started, thread/status/changed, thread/tokenUsage/updated,
        # account/rateLimits/updated, mcpServer/* — supplementary only.
        # NEVER drives idle (TEST RULE 1) and not worth a transcript row.
        logger.debug("CodexTransport: ignoring notification %s", method)

    async def _on_turn_completed(self, params: dict) -> None:
        turn = params.get("turn") or {}
        status = turn.get("status")
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
        self._turn_id = None
        self._turn_open = False
        await self._event_queue.put(TransportEvent(
            event_type="turn_complete", data=self._turn_meta(cause)))
        if self._turn_done is not None:
            self._turn_done.set()

    async def _on_item(self, phase: str, item: dict) -> None:
        itype = item.get("type")
        if itype == "agentMessage":
            if phase == "item/completed":
                item_id = item.get("id") or ""
                text = item.get("text") or self._message_parts.get(item_id, "")
                self._message_parts.pop(item_id, None)
                if text:
                    await self._put_message(text, phase_label=item.get("phase"))
            return
        if itype == "commandExecution":
            command = item.get("command", "")
            started = phase == "item/started"
            await self._event_queue.put(TransportEvent(
                event_type="tool_use",
                data={
                    "tool_name": "commandExecution",
                    "tool_id": item.get("id"),
                    # No run_in_background key — Codex has no surviving
                    # background work (FINDINGS A5c); the provenance
                    # registry stays inert for this transport by design.
                    "tool_input": {"command": command, "cwd": item.get("cwd")},
                    "status": item.get("status"),
                    "exit_code": item.get("exitCode"),
                    "aggregated_output": (item.get("aggregatedOutput") or "")[:2000],
                },
                raw_text=(f"[Running command: {command}]" if started else
                          f"[Command finished (exit {item.get('exitCode')}): {command}]"),
            ))
            return
        if itype == "fileChange":
            changes = item.get("changes") or []
            paths = ", ".join(c.get("path", "") for c in changes)
            verb = "Editing" if phase == "item/started" else "Edited"
            await self._event_queue.put(TransportEvent(
                event_type="tool_use",
                data={
                    "tool_name": "fileChange",
                    "tool_id": item.get("id"),
                    "tool_input": {"changes": changes},
                    "status": item.get("status"),
                },
                raw_text=f"[{verb} files: {paths}]",
            ))
            return
        # userMessage echo, todoList, ... — no transcript value.

    async def _put_message(self, text: str, phase_label: str | None = None) -> None:
        self._final_texts.append(text)
        data: dict = {"text": text}
        if phase_label:
            data["phase"] = phase_label
        await self._event_queue.put(TransportEvent(
            event_type="message", data=data, raw_text=text))

    # ------------------------------------------------------------------
    # ABC stubs — completed in the lifecycle task
    # ------------------------------------------------------------------

    async def start(self, command: str, args: list[str], workspace: str,
                    env: dict | None = None) -> None:
        raise NotImplementedError("CodexTransport.start lands in the lifecycle task")

    async def send(self, message: str) -> str | None:
        raise NotImplementedError("CodexTransport.send lands in the lifecycle task")

    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        while self._alive:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        self._stopping = True
        self._alive = False

    def is_alive(self) -> bool:
        return (self._alive and self._popen is not None
                and self._popen.returncode is None)

    @property
    def session_id(self) -> str | None:
        return self._thread_id
