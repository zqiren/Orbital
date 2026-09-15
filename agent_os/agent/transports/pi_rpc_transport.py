# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pi RPC transport: Earendil's ``pi --mode rpc`` over stdio JSONL.

A persistent client for Pi's documented RPC protocol instead of its TUI.
Every wire shape here is [OBSERVED] against pi 0.85.1 (probe 2026-09-16; the
redacted records live in tests/fixtures/pi_rpc/, the probe notes in
agent_os/agents/manifests/pi.yaml):

- Records are LF-delimited JSON objects. U+2028/U+2029 are legal inside JSON
  strings, so framing is byte-level on ``\\n`` only. A record that does not
  parse is a protocol failure that fails the turn, never a skipped line.
- Commands carry an ``id``; the matching ``{"type": "response"}`` settles a
  future and never reaches the transcript. Events carry no id.
- ``agent_end`` is not a turn boundary: a timed-out request produced
  ``agent_end`` with ``willRetry: true`` followed by a second run. Only
  ``agent_settled`` closes a turn.
- A provider failure arrives either as a rejected ``prompt`` response (no key
  for the provider) or as an assistant ``message_end`` with
  ``stopReason: "error"``. Both close the turn as ``error``, never as a
  completion.
- ``abort`` stops the running tool, emits ``agent_settled``, and only then
  answers.

Duck-typed like CodexTransport so the SubAgentManager / CLIAdapter /
ProcessManager contracts are untouched: dispatch(), _event_queue,
thread_started + turn_complete {cause, session_id, model, rollout_path},
_proc (psutil), _resume_session_id, resume_outcome.

Pi has no sandbox and no permission prompts of its own. ``--no-approve`` (in
the manifest args) only keeps untrusted project-local Pi resources from
loading; the process runs with the user's permissions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections import deque
from typing import AsyncIterator

import psutil

from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.agent.transports.jsonl_stream import (
    DEFAULT_MAX_LINE_BYTES,
    LineTooLongError,
    read_jsonl_line,
)
from agent_os.agent.transports.process_kill import kill_process_tree
from agent_os.utils.subprocess_flags import win_no_window_flags

logger = logging.getLogger(__name__)

# Orbital's briefing reaches pi as ``--append-system-prompt <file>``: pi reads
# the contents of a path it can stat, and a file survives npm's Windows
# ``pi.cmd`` shim, which cannot carry a multi-line argument.
PROMPT_FILE_PREFIX = ".orbital-system-prompt-"
# pi re-reads the file on /reload, so a live spawn keeps it until stop(). Only
# files abandoned by a daemon that died before stop() are swept.
_STALE_PROMPT_FILE_SECONDS = 7 * 24 * 3600

# Extension UI methods that block pi until the client answers.
_DIALOG_METHODS = frozenset({"select", "confirm", "input", "editor"})
_STDERR_TAIL_LINES = 20
_KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}")
_SEND_TIMEOUT_SECONDS = 600.0


class PiProtocolError(Exception):
    """pi wrote something on stdout that is not a protocol record."""


class _StartupError(RuntimeError):
    """pi exited, or never answered get_state, while starting."""


async def iter_records(reader: asyncio.StreamReader, *,
                       max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
                       ) -> AsyncIterator[dict]:
    """Yield pi RPC records from ``reader`` until EOF.

    Raises :class:`PiProtocolError` on an oversize, unparseable, or non-object
    record: the stream can no longer be trusted past it.
    """
    while True:
        try:
            line = await read_jsonl_line(reader, max_line_bytes=max_line_bytes)
        except LineTooLongError as exc:
            raise PiProtocolError(
                f"pi RPC record exceeded {exc.cap} bytes") from None
        if not line:
            return
        if line.endswith(b"\n"):
            line = line[:-1]
        if line.endswith(b"\r"):
            line = line[:-1]
        if not line.strip():
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise PiProtocolError(
                f"unparseable pi RPC record: {line[:200]!r}") from None
        if not isinstance(record, dict):
            raise PiProtocolError(
                f"pi RPC record is not an object: {line[:200]!r}")
        yield record


def _redact(text: str, secrets: list[str]) -> str:
    """Keep the first 4 characters of every known secret and key-like token."""
    for secret in secrets:
        text = text.replace(secret, secret[:4] + "…")
    return _KEY_LIKE.sub(lambda m: m.group(0)[:4] + "…", text)


def _tool_summary(args) -> str:
    if isinstance(args, dict):
        for field in ("command", "path", "file_path", "pattern"):
            value = args.get(field)
            if isinstance(value, str) and value:
                return value[:200]
    return ""


def _result_text(result) -> str:
    content = result.get("content") if isinstance(result, dict) else None
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "") for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )[:2000]


class PiRPCTransport(AgentTransport):
    """Persistent transport for ``pi --mode rpc``."""

    # pi's tools run inside the turn and abort kills them; there is no
    # between-turn background work for a "Background" badge to describe.
    supports_background_status = False

    def __init__(self, system_prompt: str | None = None,
                 resume_record: dict | None = None,
                 session_dir: str | None = None, *,
                 command_timeout: float = 30.0,
                 abort_timeout: float = 5.0,
                 max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
                 max_pending_commands: int = 32,
                 max_queued_events: int = 10_000):
        record = resume_record or {}
        # Same attribute name as the SDK/Codex transports: the manager's
        # honesty downgrade compares it with the persisted record.
        self._resume_session_id: str | None = record.get("session_id")
        self._resume_outcome: tuple[str, str | None] = ("fresh", None)
        self._system_prompt = system_prompt
        self._session_dir = session_dir
        self._command_timeout = command_timeout
        self._abort_timeout = abort_timeout
        self._max_line_bytes = max_line_bytes
        self._max_pending = max_pending_commands
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue(
            maxsize=max_queued_events)
        self._workspace = ""
        self._popen: asyncio.subprocess.Process | None = None
        self._proc: psutil.Process | None = None  # manager kill anchor
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._next_id = 0
        self._secrets: list[str] = []
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._prompt_file: str | None = None
        self._alive = False
        self._stopping = False
        # Session identity, from get_state.
        self._session_id: str | None = None
        self._session_file: str | None = None
        self._model: str | None = None
        self._provider: str | None = None
        # Turn state.
        self._turn_open = False
        self._turn_done: asyncio.Event | None = None
        self._abort_requested = False
        self._turn_error: str | None = None
        self._last_cause: str | None = None
        # Text blocks of the assistant message in flight, by contentIndex.
        self._text_parts: dict[int, str] = {}
        self._texts_emitted: set[int] = set()
        self._final_texts: list[str] = []

    # ------------------------------------------------------------------
    # turn bookkeeping
    # ------------------------------------------------------------------

    def _begin_turn(self) -> None:
        self._turn_open = True
        self._turn_done = asyncio.Event()
        self._abort_requested = False
        self._turn_error = None
        self._text_parts = {}
        self._texts_emitted = set()
        self._final_texts = []

    def _turn_meta(self, cause: str) -> dict:
        # ProcessManager persists session_id + model + rollout_path for
        # cause in ("success", "error", "interrupted").
        return {
            "cause": cause,
            "session_id": self._session_id,
            "model": self._model,
            "rollout_path": self._session_file,
        }

    async def _put_message(self, text: str) -> None:
        self._final_texts.append(text)
        await self._event_queue.put(TransportEvent(
            event_type="message", data={"text": text}, raw_text=text))

    async def _close_turn(self, cause: str) -> None:
        if not self._turn_open:
            return
        self._turn_open = False
        self._abort_requested = False
        self._last_cause = cause
        await self._event_queue.put(TransportEvent(
            event_type="turn_complete", data=self._turn_meta(cause)))
        if self._turn_done is not None:
            self._turn_done.set()

    async def _fail_turn(self, reason: str) -> None:
        if not self._turn_open:
            return
        text = _redact(reason, self._secrets)
        self._turn_error = text
        await self._event_queue.put(TransportEvent(
            event_type="error", data={"error": text}, raw_text=f"Error: {text}"))
        await self._close_turn("error")

    # ------------------------------------------------------------------
    # record routing (unit-testable without a process)
    # ------------------------------------------------------------------

    async def _route_record(self, record: dict) -> None:
        rtype = record.get("type")
        if rtype == "response":
            future = self._pending.get(record.get("id"))
            if future is not None and not future.done():
                future.set_result(record)
            else:
                logger.debug("PiRPCTransport: uncorrelated %s response",
                             record.get("command"))
            return
        if rtype == "extension_ui_request":
            await self._on_extension_ui_request(record)
            return
        if not self._turn_open:
            logger.debug("PiRPCTransport: %s outside a turn", rtype)
            return
        if rtype == "message_start":
            if (record.get("message") or {}).get("role") == "assistant":
                self._text_parts = {}
                self._texts_emitted = set()
        elif rtype == "message_update":
            await self._on_message_update(record.get("assistantMessageEvent") or {})
        elif rtype == "message_end":
            await self._on_message_end(record.get("message") or {})
        elif rtype == "tool_execution_start":
            await self._on_tool(record, started=True)
        elif rtype == "tool_execution_end":
            await self._on_tool(record, started=False)
        elif rtype == "auto_retry_end":
            if not record.get("success"):
                self._turn_error = str(
                    record.get("finalError") or "pi's automatic retry failed")
        elif rtype == "extension_error":
            logger.warning("PiRPCTransport: pi extension %s failed: %s",
                           record.get("extensionPath"), record.get("error"))
        elif rtype == "agent_settled":
            await self._on_settled()
        # agent_start, agent_end, turn_start/turn_end, tool_execution_update,
        # queue_update, compaction_*, auto_retry_start, summarization_*: progress
        # only — none of them is a turn boundary.

    async def _on_message_update(self, event: dict) -> None:
        kind = event.get("type")
        index = event.get("contentIndex", 0)
        if kind == "text_delta":
            self._text_parts[index] = (
                self._text_parts.get(index, "") + str(event.get("delta") or ""))
        elif kind == "text_end":
            content = event.get("content")
            text = content if isinstance(content, str) else self._text_parts.get(index, "")
            self._text_parts.pop(index, None)
            self._texts_emitted.add(index)
            if text:
                await self._put_message(text)
        # thinking_* and toolcall_* deltas: tool_execution_* carries the
        # activity, and reasoning is not transcript content.

    async def _on_message_end(self, message: dict) -> None:
        if message.get("role") != "assistant":
            return
        # message_end is authoritative: emit any text block that never saw
        # its text_end, and never repeat one that did.
        content = message.get("content")
        if isinstance(content, list):
            for index, block in enumerate(content):
                if (isinstance(block, dict) and block.get("type") == "text"
                        and index not in self._texts_emitted and block.get("text")):
                    self._texts_emitted.add(index)
                    await self._put_message(str(block["text"]))
        self._text_parts = {}
        self._texts_emitted = set()
        stop_reason = message.get("stopReason")
        if stop_reason == "error":
            self._turn_error = str(
                message.get("errorMessage") or "pi reported a provider error")
        elif stop_reason in ("stop", "toolUse", "length"):
            # A later good response supersedes a failure pi retried past.
            self._turn_error = None

    async def _on_tool(self, record: dict, *, started: bool) -> None:
        name = str(record.get("toolName") or "tool")
        args = record.get("args")
        data: dict = {
            "tool_name": name,
            "tool_id": record.get("toolCallId"),
            "tool_input": args if isinstance(args, dict) else {},
        }
        if started:
            summary = _tool_summary(args)
            data["status"] = "running"
            raw_text = f"[Running {name}: {summary}]" if summary else f"[Running {name}]"
        else:
            failed = bool(record.get("isError"))
            data["status"] = "error" if failed else "completed"
            data["result"] = _result_text(record.get("result"))
            raw_text = f"[{name} {'failed' if failed else 'finished'}]"
        await self._event_queue.put(TransportEvent(
            event_type="tool_use", data=data, raw_text=raw_text))

    async def _on_settled(self) -> None:
        # Deltas cut off before text_end/message_end (an abort mid-answer)
        # must not be lost.
        for index in sorted(self._text_parts):
            if self._text_parts[index]:
                await self._put_message(self._text_parts[index])
        self._text_parts = {}
        if self._stopping:
            # Teardown: stop_for_user's own notice speaks for this turn.
            await self._close_turn("stopped")
        elif self._abort_requested:
            await self._close_turn("interrupted")
        elif self._turn_error:
            await self._fail_turn(self._turn_error)
        else:
            await self._close_turn("success")

    async def _on_extension_ui_request(self, record: dict) -> None:
        if record.get("method") not in _DIALOG_METHODS:
            return  # notify/setStatus/...: fire-and-forget, nothing to answer
        # pi blocks on an extension dialog until it is answered, and Orbital
        # has no surface for a Pi extension's own dialogs: cancel it rather
        # than leave the turn hanging.
        logger.warning("PiRPCTransport: cancelling pi extension %s dialog %r",
                       record.get("method"), record.get("title"))
        with contextlib.suppress(Exception):
            await self._send_raw({"type": "extension_ui_response",
                                  "id": record.get("id"), "cancelled": True})

    # ------------------------------------------------------------------
    # wire I/O
    # ------------------------------------------------------------------

    async def _send_raw(self, command: dict) -> None:
        popen = self._popen
        if popen is None or popen.stdin is None or popen.returncode is not None:
            raise RuntimeError("pi is not running")
        popen.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
        await popen.stdin.drain()

    async def _request(self, command: dict, timeout: float | None = None) -> dict:
        if len(self._pending) >= self._max_pending:
            raise RuntimeError("too many unanswered pi commands")
        self._next_id += 1
        request_id = f"orbital-{self._next_id}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send_raw({**command, "id": request_id})
            return await asyncio.wait_for(future, timeout or self._command_timeout)
        finally:
            self._pending.pop(request_id, None)

    def _fail_pending(self, reason: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(RuntimeError(reason))
        self._pending.clear()

    async def _read_loop(self, popen: asyncio.subprocess.Process) -> None:
        failure: str | None = None
        try:
            async for record in iter_records(popen.stdout,
                                             max_line_bytes=self._max_line_bytes):
                try:
                    await self._route_record(record)
                except Exception:
                    logger.exception("PiRPCTransport: routing failed for a %s record",
                                     record.get("type"))
        except asyncio.CancelledError:
            raise
        except PiProtocolError as exc:
            failure = _redact(str(exc), self._secrets)
        except Exception as exc:
            failure = f"pi RPC read loop failed: {exc}"
        await self._on_stream_end(popen, failure)

    async def _on_stream_end(self, popen: asyncio.subprocess.Process,
                             failure: str | None) -> None:
        if popen is not self._popen:
            return  # a process already retired by a resume restart
        if failure is not None:
            # The stream can't be trusted past a bad record: take pi (and any
            # tool it is running) down rather than let it run on unobserved.
            logger.error("PiRPCTransport: %s — killing pi pid=%s", failure, popen.pid)
            await self._kill_tree()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(popen.wait(), 2.0)
        if self._turn_open and not self._stopping:
            reason = failure
            if reason is None:
                reason = f"pi exited unexpectedly (exit code {popen.returncode})"
                tail = await self._stderr_summary()
                if tail:
                    reason = f"{reason}: {tail}"
            await self._fail_turn(reason)
        self._fail_pending(failure or "pi RPC stream ended")
        self._alive = False

    async def _drain_stderr(self, popen: asyncio.subprocess.Process) -> None:
        try:
            while popen.stderr is not None:
                try:
                    line = await read_jsonl_line(popen.stderr, max_line_bytes=64 * 1024)
                except LineTooLongError:
                    continue
                if not line:
                    return
                text = _redact(line.decode("utf-8", errors="replace").rstrip(),
                               self._secrets)
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("pi stderr: %s", text)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _stderr_summary(self) -> str:
        """The redacted stderr tail, once pi's stderr has reached EOF."""
        task = self._stderr_task
        if task is not None and not task.done():
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(task), 1.0)
        return " | ".join(self._stderr_tail)

    # ------------------------------------------------------------------
    # start
    # ------------------------------------------------------------------

    async def start(self, command: str, args: list[str], workspace: str,
                    env: dict | None = None) -> None:
        if self._alive:
            raise RuntimeError("pi transport is already started")
        self._workspace = os.path.abspath(workspace)
        merged_env = dict(os.environ)
        merged_env.pop("CLAUDECODE", None)
        if env:
            merged_env.update(env)
        # The injected env holds the one provider key Orbital hands pi.
        self._secrets = [value for value in (env or {}).values()
                         if isinstance(value, str) and len(value) >= 8]
        argv = [command, *args]
        if self._session_dir:
            argv += ["--session-dir", self._session_dir]
        try:
            self._prompt_file = self._write_prompt_file()
            if self._prompt_file:
                argv += ["--append-system-prompt", self._prompt_file]
            state = await self._open_session(argv, merged_env)
            self._session_id = state.get("sessionId")
            if not self._session_id:
                raise RuntimeError(f"pi get_state returned no session id: {state}")
        except BaseException:
            await self.stop()
            raise
        self._session_file = state.get("sessionFile")
        model = state.get("model") or {}
        self._provider = model.get("provider")
        model_id = model.get("id")
        self._model = f"{self._provider}/{model_id}" if self._provider and model_id else model_id
        # Eager identity: ProcessManager persists the resume record before the
        # first turn finishes.
        await self._event_queue.put(TransportEvent(event_type="thread_started", data={
            "session_id": self._session_id,
            "model": self._model,
            "provider": self._provider,
            "rollout_path": self._session_file,
            "resume_status": self._resume_outcome[0],
            "resume_reason": self._resume_outcome[1],
        }))

    async def _open_session(self, argv: list[str], env: dict) -> dict:
        """Spawn pi, resuming the recorded session when pi confirms it.

        A stored id only proves Orbital once saw it. ``--session`` makes pi
        exit when the id is unknown (``--session-id`` would silently create
        it), and get_state must echo the same id before this reports
        "resumed". Any failure restarts fresh exactly once.
        """
        requested = self._resume_session_id
        if not requested:
            state = await self._spawn(argv, env)
            self._resume_outcome = ("fresh", None)
            return state
        try:
            state = await self._spawn(argv + ["--session", requested], env)
            if state.get("sessionId") != requested:
                raise _StartupError(
                    f"pi opened session {state.get('sessionId')!r}, not {requested!r}")
        except _StartupError as exc:
            logger.warning("PiRPCTransport: session %s could not be resumed; "
                           "starting fresh: %s", requested, exc)
            await self._teardown_process()
            self._resume_session_id = None
            state = await self._spawn(argv, env)
            self._resume_outcome = ("fresh", "resume_failed")
            return state
        self._resume_outcome = ("resumed", None)
        return state

    async def _spawn(self, argv: list[str], env: dict) -> dict:
        self._stderr_tail.clear()
        popen = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._workspace, env=env,
            # message_end/turn_end records repeat whole tool outputs; the
            # asyncio default (64 KiB) is only the fast-path size here.
            limit=1024 * 1024,
            creationflags=win_no_window_flags(),
        )
        self._popen = popen
        try:
            self._proc = psutil.Process(popen.pid)
        except psutil.Error:
            self._proc = None
            logger.warning("PiRPCTransport: could not wrap pid=%s in psutil", popen.pid)
        self._alive = True
        self._reader_task = asyncio.create_task(
            self._read_loop(popen), name=f"pi-read-{popen.pid}")
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(popen), name=f"pi-stderr-{popen.pid}")
        try:
            response = await self._request({"type": "get_state"})
        except Exception as exc:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(popen.wait(), 2.0)
            if popen.returncode is not None:
                reason = f"pi exited during startup (exit code {popen.returncode})"
            else:
                reason = f"pi did not answer get_state: {exc or type(exc).__name__}"
            tail = _redact(await self._stderr_summary(), self._secrets)
            raise _StartupError(f"{reason}: {tail}" if tail else reason) from None
        data = response.get("data")
        if not response.get("success") or not isinstance(data, dict):
            raise _StartupError(f"pi get_state failed: {response.get('error') or response}")
        return data

    def _write_prompt_file(self) -> str | None:
        if not self._system_prompt:
            return None
        directory = self._session_dir or tempfile.gettempdir()
        os.makedirs(directory, exist_ok=True)
        cutoff = time.time() - _STALE_PROMPT_FILE_SECONDS
        with contextlib.suppress(OSError):
            for name in os.listdir(directory):
                if name.startswith(PROMPT_FILE_PREFIX):
                    stale = os.path.join(directory, name)
                    with contextlib.suppress(OSError):
                        if os.path.getmtime(stale) < cutoff:
                            os.unlink(stale)
        path = os.path.join(directory, f"{PROMPT_FILE_PREFIX}{uuid.uuid4().hex}.md")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(self._system_prompt)
        return path

    # ------------------------------------------------------------------
    # dispatch / send
    # ------------------------------------------------------------------

    async def dispatch(self, message: str) -> None:
        """Start a turn; its events flow through read_stream().

        Raises only when no turn could be opened. Once one is open, every
        failure — including pi rejecting the prompt — ends it as an ``error``
        turn on the stream, which is where ProcessManager reports it.
        """
        if not self.is_alive() or self._session_id is None:
            raise RuntimeError("pi transport not started — call start() first")
        if self._turn_open:
            raise RuntimeError("pi is already running a turn")
        self._begin_turn()
        try:
            response = await self._request({"type": "prompt", "message": message})
        except asyncio.TimeoutError:
            await self._fail_turn(
                f"pi did not acknowledge the prompt within {self._command_timeout:.0f}s")
            await self._kill_tree()
            return
        except Exception as exc:
            await self._fail_turn(f"pi did not accept the prompt: {exc}")
            return
        if not response.get("success"):
            await self._fail_turn(str(response.get("error") or "pi rejected the prompt"))

    async def send(self, message: str) -> str | None:
        """Blocking ABC variant: dispatch, then wait for the settled turn."""
        try:
            await self.dispatch(message)
        except Exception as exc:
            return f"Error: pi dispatch failed: {exc}"
        done = self._turn_done
        try:
            await asyncio.wait_for(done.wait(), timeout=_SEND_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return f"Error: pi turn did not settle within {_SEND_TIMEOUT_SECONDS:.0f}s"
        if self._last_cause == "error":
            return f"Error: {self._turn_error}"
        return "\n".join(self._final_texts) or "(no response)"

    # ------------------------------------------------------------------
    # stream / lifecycle
    # ------------------------------------------------------------------

    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        # Drain what is already queued even after pi died: the error and
        # turn_complete of a crashed turn are queued just before _alive flips.
        while self._alive or not self._event_queue.empty():
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            yield event

    async def _kill_tree(self) -> None:
        if self._proc is not None:
            try:
                outcome = await kill_process_tree(self._proc, label="pi")
                if not outcome.parent_dead:
                    logger.error("PiRPCTransport: pi survived kill; PID may leak")
            except Exception:
                logger.exception("PiRPCTransport: kill_process_tree raised")
        elif self._popen is not None and self._popen.returncode is None:
            # psutil wrap failed at spawn — no tree handle, but never leave
            # the root process running.
            with contextlib.suppress(ProcessLookupError):
                self._popen.kill()

    async def _teardown_process(self) -> None:
        # Reader down before the kill so the EOF path cannot report a
        # teardown as a crashed turn.
        tasks = [task for task in (self._reader_task, self._stderr_task)
                 if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader_task = None
        self._stderr_task = None
        await self._kill_tree()
        popen = self._popen
        if popen is not None:
            if popen.stdin is not None:
                with contextlib.suppress(Exception):
                    popen.stdin.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(popen.wait(), 2.0)
        self._fail_pending("pi transport stopped")
        self._popen = None
        self._proc = None
        self._alive = False

    async def stop(self) -> None:
        self._stopping = True
        if self._turn_open and self.is_alive():
            # abort answers only once pi has settled, so the turn closes
            # (as "stopped") before the kill below.
            self._abort_requested = True
            try:
                await self._request({"type": "abort"}, timeout=self._abort_timeout)
            except Exception:
                logger.warning("PiRPCTransport.stop: pi did not acknowledge abort "
                               "within %.1fs; killing it", self._abort_timeout)
        await self._teardown_process()
        path, self._prompt_file = self._prompt_file, None
        if path:
            with contextlib.suppress(OSError):
                os.unlink(path)

    def is_alive(self) -> bool:
        return bool(self._alive and self._popen is not None
                    and self._popen.returncode is None)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def resume_outcome(self) -> tuple[str, str | None]:
        return self._resume_outcome
