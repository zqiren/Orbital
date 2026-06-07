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
        # Model is CONFIG, not thread identity (AMENDS piece 2 — see
        # INVESTIGATION-resume-semantics): the record's model is display
        # metadata and is NEVER consulted here. start() resolves the model
        # from the current config-store argv; when none is set the param is
        # omitted and the provider serves the thread's last-used model
        # (wire-verified on both transports).
        self._model: str | None = None
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
            raise RuntimeError(f"codex {method} failed: {msg['error']}")
        return msg.get("result") or {}

    async def _notify(self, method: str, params: dict | None = None) -> None:
        msg: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        await self._send_raw(msg)

    # ------------------------------------------------------------------
    # start() + helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _argv_model(args: list[str]) -> str | None:
        """Extract `-m <model>` injected by the config store's flag_template
        ("-m {value}", sub_agent_config_store.py). app-server takes model
        per-request, never on argv."""
        for i, a in enumerate(args or []):
            if a == "-m" and i + 1 < len(args):
                return args[i + 1]
        return None

    def _thread_open_request(self, workspace: str) -> tuple[str, dict]:
        """(method, params) opening the thread — fresh or resume.

        Governance params (cwd/approvalPolicy/sandbox) are ALWAYS present in
        BOTH branches: thread/resume applies its own defaults for omitted
        params (observed drift: approvalPolicy never -> on-request), so
        omission silently loosens governance. Only the MODEL is conditional
        (model-is-config): passed when a current override exists, omitted
        otherwise so the provider serves the thread's last-used model.
        """
        approval_policy, sandbox = _POLICY_BY_AUTONOMY.get(
            self._autonomy, _DEFAULT_POLICY)
        params: dict = {"cwd": workspace, "approvalPolicy": approval_policy,
                        "sandbox": sandbox}
        if self._model:
            params["model"] = self._model
        if self._resume_session_id:
            params["threadId"] = self._resume_session_id
            return "thread/resume", params
        return "thread/start", params

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

        # clientInfo flows into the rollout `originator` (FINDINGS A6).
        init = await self._request("initialize", {"clientInfo": {
            "name": "orbital", "title": "Orbital", "version": "0.1.0"}})
        self._check_version(init)
        await self._notify("initialized")

        method, params = self._thread_open_request(workspace)
        result = await self._request(method, params)
        thread = result.get("thread") or {}
        self._thread_id = thread.get("id")
        self._rollout_path = thread.get("path")
        self._effective_model = result.get("model")
        if not self._thread_id:
            raise RuntimeError(
                f"codex thread/start returned no thread id: {result}")

    async def _drain_stderr(self) -> None:
        try:
            while self._popen is not None and self._popen.stderr is not None:
                line = await self._popen.stderr.readline()
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
                line = await self._popen.stdout.readline()
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
        params: dict = {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": message}],
        }
        if self._external_sandbox:
            # Constraint 7: Orbital's own sandbox wraps the process; Codex
            # skips internal enforcement. Tagged-object form exists ONLY on
            # turn/start (param asymmetry).
            params["sandboxPolicy"] = {"type": "externalSandbox"}
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

    # ------------------------------------------------------------------
    # resume pre-check
    # ------------------------------------------------------------------

    @staticmethod
    def resume_source_exists(record: dict) -> bool:
        """Pre-check the rollout file before resuming. Codex pruning policy
        is unmeasured (R4 open) — never trust the resume call to fail
        loudly; a fresh start must never look like a resume."""
        path = record.get("rollout_path")
        if path and os.path.isfile(path):
            return True
        thread_id = record.get("session_id")
        if not thread_id:
            return False
        home = os.environ.get("CODEX_HOME") or os.path.join(
            os.path.expanduser("~"), ".codex")
        pattern = os.path.join(home, "sessions", "*", "*", "*",
                               f"rollout-*-{thread_id}.jsonl")
        return bool(_glob.glob(pattern))
