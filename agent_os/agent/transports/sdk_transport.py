# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""SDK transport: uses claude-agent-sdk to communicate with Claude Code."""
import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from typing import AsyncIterator, Callable

from agent_os.agent.transports.base import AgentTransport, TransportEvent
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.transports.tool_risk import READ, classify_tool, should_auto_approve
from agent_os.utils.subprocess_flags import CREATE_NO_WINDOW

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        PermissionResultAllow,
        PermissionResultDeny,
    )
    from claude_agent_sdk.types import (
        AssistantMessage,
        ResultMessage,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )
    HAS_SDK = True
except ImportError:
    HAS_SDK = False

try:
    # Spec 099: the CLI's per-turn subscription quota (claude-agent-sdk >=
    # 0.1.58 parses ``rate_limit_event``). Same ``()`` sentinel as below so an
    # older SDK degrades to "no quota" instead of failing the import.
    from claude_agent_sdk.types import RateLimitEvent as _RateLimitEvent
except ImportError:
    _RateLimitEvent = ()

try:
    # Background-task lifecycle messages (claude-agent-sdk >= ~0.1.5x). Used to
    # tell when an AWAITED background task is still running so the dispatch
    # consumer keeps reading until its continuation turn arrives. Imported
    # separately so an older SDK lacking them degrades gracefully (the ``()``
    # sentinel makes ``isinstance(msg, _TaskStarted)`` always False).
    from claude_agent_sdk.types import (
        TaskStartedMessage as _TaskStarted,
        TaskNotificationMessage as _TaskNotification,
    )
except ImportError:
    _TaskStarted = _TaskNotification = ()

try:
    # Not re-exported by the package; same ``()`` sentinel so a move degrades
    # to "every consumer error is checked against the reader".
    from claude_agent_sdk._errors import MessageParseError as _MessageParseError
except ImportError:
    _MessageParseError = ()

logger = logging.getLogger(__name__)


class _NoWindowAnyio:
    """``anyio`` as the SDK's subprocess transport sees it, except that
    ``open_process`` always adds ``CREATE_NO_WINDOW``."""

    _orbital_no_window = True

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    async def open_process(self, command, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | CREATE_NO_WINDOW
        return await self._real.open_process(command, **kwargs)


def install_sdk_no_window_spawn(*, windows: bool | None = None) -> None:
    """Make the SDK start ``claude.exe`` without a console window (Windows).

    The packaged Orbital.exe is a windowed app with no console, so Windows
    gives every console child started without ``CREATE_NO_WINDOW`` a new,
    visible console: an empty "claude" window on every Claude Code dispatch
    (and a flash for the SDK's ``claude -v`` version check). Orbital's own
    spawns pass the flag (``agent_os/utils/subprocess_flags.py``), but the
    SDK starts the CLI itself through ``anyio.open_process`` and offers no
    option for it. Its module-level ``anyio`` reference is swapped for a
    proxy that adds the flag; nothing else in anyio changes, and no other
    module sees the proxy. ``test_sdk_still_spawns_through_anyio_open_process``
    fails if an SDK release stops spawning that way. Idempotent.
    """
    if windows is None:
        windows = sys.platform == "win32"
    if not windows:
        return
    try:
        from claude_agent_sdk._internal.transport import subprocess_cli
    except ImportError:
        return
    current = getattr(subprocess_cli, "anyio", None)
    if current is None or getattr(current, "_orbital_no_window", False):
        return
    subprocess_cli.anyio = _NoWindowAnyio(current)


async def _one_prompt(message: str, prompt_uuid: str):
    """One streamed user prompt with a caller-chosen native uuid: the wire
    shape the SDK writes for a string prompt, plus ``uuid`` (spec 156)."""
    yield {
        "type": "user",
        "message": {"role": "user", "content": message},
        "parent_tool_use_id": None,
        "uuid": prompt_uuid,
    }


def _is_batch_file(path: str) -> bool:
    return path.lower().endswith((".cmd", ".bat"))


# Spec 156 §5.4: every prompt Orbital dispatches to Claude Code carries a
# caller-chosen native uuid, ``uuid5(ORBITAL_PROMPT_NS, dispatch_id)``. The
# dispatch marker already records ``dispatch_id``, so an Orbital dispatch
# maps to its native prompt entry by construction (no content matching), and
# a fork can name the native cut point: the ``parentUuid`` of the first
# discarded prompt. Fixed forever: changing it orphans every stamped prompt.
ORBITAL_PROMPT_NS = uuid.UUID("5b1e6f0c-9a3d-4c2e-8f71-0b17a1f00156")


def prompt_uuid_for(dispatch_id: str) -> str:
    """The native prompt uuid Orbital stamps on ``dispatch_id``'s prompt."""
    return str(uuid.uuid5(ORBITAL_PROMPT_NS, dispatch_id))


# claude-agent-sdk's own message when handed a .cmd/.bat as cli_path
# (``_reject_windows_batch_cli``). Spec 116 D9 maps it to an update hint.
BATCH_REFUSAL_PREFIX = "Refusing to execute batch script"


def native_claude_cli(
    command: str | None,
    *,
    windows: bool | None = None,
    home: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str | None:
    """Swap npm's ``claude.cmd`` shim for a native ``claude.exe`` on Windows.

    claude-agent-sdk 0.2.x refuses to spawn a ``.bat``/``.cmd`` file as the
    CLI (cmd.exe re-parses the argv; no escaping is reliable), and PATH
    resolution of ``claude`` lands on ``%APPDATA%\\npm\\claude.CMD`` for an
    npm install — so every Claude Code dispatch failed at adapter start. The
    shim only runs ``node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe``
    beside it. Order: that exe, then ``claude.exe`` on PATH, then the native
    installer's ``~\\.local\\bin\\claude.exe``. With none of them the shim is
    returned unchanged, so the SDK's own refusal explains what to install.
    """
    if windows is None:
        windows = sys.platform == "win32"
    if not windows or not command or not _is_batch_file(command):
        return command
    candidates = [os.path.join(os.path.dirname(command), "node_modules",
                               "@anthropic-ai", "claude-code", "bin", "claude.exe")]
    on_path = which("claude.exe")
    if on_path and not _is_batch_file(on_path):
        candidates.append(on_path)
    candidates.append(os.path.join(home or os.path.expanduser("~"),
                                   ".local", "bin", "claude.exe"))
    for candidate in candidates:
        if candidate.lower().endswith(".exe") and os.path.isfile(candidate):
            logger.info("Claude Code CLI: using native %s instead of shim %s",
                        candidate, command)
            return candidate
    return command


if HAS_SDK:
    install_sdk_no_window_spawn()


class SDKTransport(AgentTransport):
    """Transport using the Claude Agent SDK (claude-agent-sdk).

    The SDK spawns claude.exe as a subprocess internally. We configure it
    with a can_use_tool callback that routes permission requests through
    AgentOS's approval system.
    """

    # Piece 3 Part A: this transport's event stream carries tool_input
    # (incl. run_in_background) and exposes the subprocess handle, so the
    # provenance-based three-state status applies. Pipe/ACP/PTY discard
    # tool_input (pipe_transport.py / acp_sdk_transport.py) and stay two-state.
    supports_background_status = True

    def __init__(self, autonomy: "Autonomy | None" = None, system_prompt: str | None = None,
                 resume_session_id: str | None = None, model: str | None = None,
                 effort: str | None = None, dispatch_mcp: dict | None = None,
                 fork_session: bool = False, resume_session_at: str | None = None):
        if not HAS_SDK:
            raise ImportError("claude-agent-sdk is not installed")
        self._client: "ClaudeSDKClient | None" = None
        self._session_id: str | None = None
        # Resume identity (TASK-resume-persistence): when set, start() passes
        # ClaudeAgentOptions(resume=..., model=...) so the spawned claude
        # session continues the persisted thread instead of starting fresh.
        # Model is config — the caller passes the current override or None
        # (CLI serves the session's last-used model when omitted;
        # mid-thread switch wire-verified coherent; AMENDS piece 2).
        self._resume_session_id: str | None = resume_session_id
        # Spec 156 §5.4: an edited message's fork resumes the PARENT's native
        # session with ``--fork-session`` (the CLI mints a new id and leaves
        # the parent's file untouched) cut at ``resume_session_at`` (None =
        # keep the whole thread). Only meaningful with a resume id.
        self._fork_session: bool = bool(fork_session and resume_session_id)
        self._resume_session_at: str | None = (
            resume_session_at if self._fork_session else None)
        self._model: str | None = model
        # Thinking level (spec 149): Settings' claude-code effort, or this
        # conversation's override. Config like the model; None = CLI default.
        self._effort: str | None = effort
        # Last model observed on an AssistantMessage this turn — captured so
        # the thread record can persist it (turn_complete carries it).
        self._last_model: str | None = model
        # The CLI process's running ``total_cost_usd`` as of the last result
        # message; each turn records only the increase. Zeroed per client.
        self._session_cost_seen: float = 0.0
        self._alive: bool = False
        self._workspace: str = ""
        # Pending permission requests: request_id -> asyncio.Future
        self._pending_approvals: dict[str, asyncio.Future] = {}
        # Metadata for pending approvals (for REST recovery):
        # request_id -> {"request_id", "tool_name", "tool_input"}
        self._pending_approval_data: dict[str, dict] = {}
        # Event queue for streaming events (tool use, permission requests)
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        # Autonomy preset for filtering permission requests.
        # When set, low-risk tools are auto-approved without surfacing to the user.
        self._autonomy: Autonomy | None = autonomy
        # Spec 129: ONE reader for the life of the process (strongly
        # referenced so the loop cannot GC it; cancelled by stop()). It
        # feeds _event_queue from the SDK's unbounded iterator whether or
        # not a dispatch is pending — the CLI runs turns of its own when a
        # run_in_background shell exits, and the SDK's 100-slot channel
        # must never fill (its reader answers tool permissions inline).
        self._reader_task: asyncio.Task | None = None
        # Dispatch bookkeeping. A ResultMessage closes the pending dispatch
        # when one is pending; otherwise it ends a self-initiated turn.
        self._pending_dispatch: bool = False
        # AWAITED background tasks (local_agent / local_workflow) started by
        # the pending dispatch's turn: while non-empty the dispatch stays
        # pending across results, bounded by CONTINUATION_BUDGET_S.
        self._outstanding: set = set()
        self._last_result_is_error: bool = False
        self._budget_handle: "asyncio.TimerHandle | None" = None
        # The blocking send() variant rides the same reader: it awaits this
        # future (resolved by the dispatch-origin boundary) and returns the
        # message text collected while the dispatch was pending.
        self._dispatch_done: "asyncio.Future | None" = None
        self._dispatch_text: list[str] = []
        # Direction B (own the kill): captured at connect() so stop() can
        # terminate the claude subprocess tree directly, independent of the
        # SDK's cross-task-failing graceful disconnect(). ``_proc`` is a
        # ``psutil.Process`` (PID-reuse-safe kill anchor); ``_raw_proc`` is the
        # SDK's anyio process handle, kept only for best-effort FD close.
        self._proc = None
        self._raw_proc = None
        # Optional system prompt. Forwarded to ClaudeAgentOptions(system_prompt=...)
        # at start() time. Used by sub-agent inheritance flow.
        self._system_prompt: str | None = system_prompt
        # Spec 161: ``{"url", "token"}`` of the daemon's dispatch MCP endpoint
        # when this worker sits at the top of its session, else None. The
        # token travels only as the server's Authorization header.
        self.dispatch_mcp: dict | None = dispatch_mcp

    @staticmethod
    def claude_project_slug(workspace: str) -> str:
        """Derive claude-code's per-cwd session-store directory name.

        Observed layout (confirmed against ~/.claude/projects):
        ``/Users/x/Desktop/orbital-test`` -> ``-Users-x-Desktop-orbital-test``;
        symlinked paths are stored by realpath (``/tmp`` -> ``/private/tmp``
        on macOS). Non ``[A-Za-z0-9-]`` characters become ``-``.
        """
        import re as _re
        real = os.path.realpath(workspace)
        return _re.sub(r"[^A-Za-z0-9-]", "-", real)

    @staticmethod
    def resume_source_path(workspace: str, session_id: str,
                           *, projects_root: str | None = None) -> str | None:
        """The native session JSONL a claude-code session id resolves to,
        or None. Same lookup as ``resume_source_exists`` (cwd slug first,
        then the glob fallback); metadata only, the file is never opened.
        Shared with the spec 128 dispatch path block."""
        import glob as _glob
        root = projects_root or os.path.join(
            os.environ.get("CLAUDE_CONFIG_DIR")
            or os.path.join(os.path.expanduser("~"), ".claude"),
            "projects")
        slug = SDKTransport.claude_project_slug(workspace)
        primary = os.path.join(root, slug, f"{session_id}.jsonl")
        if os.path.isfile(primary):
            return primary
        matches = sorted(_glob.glob(os.path.join(root, "*", f"{session_id}.jsonl")))
        return matches[0] if matches else None

    @staticmethod
    def resume_source_exists(workspace: str, session_id: str,
                             *, projects_root: str | None = None) -> bool:
        """Pre-check that the resumable session state still exists on disk.

        TASK-resume-persistence: do NOT trust the resume call to error on a
        dead id — claude-code prunes its store (~30 days, machine-local).
        Primary check is the derived cwd-slug path; a glob fallback covers
        any slug-derivation drift (session ids are UUIDs, so a cross-project
        match is unambiguous).

        The store is ``$CLAUDE_CONFIG_DIR/projects`` when that is set (Claude
        Code's own override, honoured by the session importer too), else
        ``~/.claude/projects``.
        """
        return SDKTransport.resume_source_path(
            workspace, session_id, projects_root=projects_root) is not None

    @staticmethod
    def fork_point(native_path: str, prompt_uuid: str) -> tuple[bool, str | None]:
        """Spec 156 §5.4: where a native session must be cut so that the
        prompt stamped ``prompt_uuid`` and everything after it are dropped.

        Returns ``(found, parent_uuid)``: the prompt entry's ``parentUuid``
        (the last entry of the turn being kept; None when the prompt opened
        the session). ``found`` is False when no entry carries that uuid (a
        prompt sent before stamping, or a pruned log). Read-only, line by
        line; never raises.
        """
        needle = f'"{prompt_uuid}"'
        try:
            with open(native_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if needle not in line:
                        continue
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(entry, dict) and entry.get("uuid") == prompt_uuid:
                        parent = entry.get("parentUuid")
                        return True, parent if isinstance(parent, str) and parent else None
        except OSError:
            pass
        return False, None

    async def start(self, command: str, args: list[str], workspace: str, env: dict | None = None) -> None:
        self._workspace = workspace
        # Build env for SDK subprocess: override CLAUDECODE to prevent nested session detection
        # The SDK merges options.env on top of os.environ, so we must explicitly blank it
        sdk_env = dict(env) if env else {}
        sdk_env.pop("CLAUDECODE", None)
        sdk_env["CLAUDECODE"] = ""  # Override os.environ value in subprocess
        options_kwargs: dict = dict(
            cwd=workspace,
            permission_mode="default",
            can_use_tool=self._handle_permission,
            cli_path=native_claude_cli(command) or None,
            env=sdk_env,
            # The SDK's 1 MiB default refuses a single stdout line above it
            # and kills its reader (spec 098): a Read of a ~660 KB PNG comes
            # back as a 1.2 MB base64 tool-result line. The CLI already
            # downscales images, so 16 MiB is headroom, not a licence.
            max_buffer_size=16 * 1024 * 1024,
        )
        if self._system_prompt is not None:
            # Re-rendered per dispatch; idempotent on resume — the append is
            # per-connection config, never persisted into the session
            # transcript, so resuming with a fresh render does not duplicate.
            options_kwargs["system_prompt"] = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt,
            }
        if self._resume_session_id is not None:
            options_kwargs["resume"] = self._resume_session_id
        if self._fork_session:
            # An SDK without the fork options would silently RESUME the
            # parent's own id: two Orbital sessions on one native session
            # (spec 156 invariant). Refuse instead; the spawn path's one
            # fresh retry carries the conversation over with the recap.
            fields = getattr(ClaudeAgentOptions, "__dataclass_fields__", {})
            if "fork_session" not in fields or (
                    self._resume_session_at and "resume_session_at" not in fields):
                raise RuntimeError(
                    "claude-agent-sdk has no fork_session/resume_session_at "
                    "option; cannot fork the native session")
            options_kwargs["fork_session"] = True
            if self._resume_session_at:
                options_kwargs["resume_session_at"] = self._resume_session_at
            logger.info("SDKTransport.start: resume=%s fork_session "
                        "resume_session_at=%s", self._resume_session_id,
                        self._resume_session_at)
        if self.dispatch_mcp:
            # Added beside the user's own MCP servers (no strict mode); the
            # CLI names the tool mcp__orbital__dispatch.
            options_kwargs["mcp_servers"] = {"orbital": {
                "type": "http",
                "url": self.dispatch_mcp["url"],
                "headers": {"Authorization": f"Bearer {self.dispatch_mcp['token']}"},
            }}
        if self._model is not None:
            options_kwargs["model"] = self._model
        if self._effort is not None and "effort" in getattr(
                ClaudeAgentOptions, "__dataclass_fields__", {}):
            # The SDK renders this as ``--effort <value>``; an SDK too old to
            # know the field keeps the CLI default rather than failing.
            options_kwargs["effort"] = self._effort
        logger.info("SDKTransport.start: model=%s effort=%s resume=%s",
                    self._model, self._effort, bool(self._resume_session_id))
        options = ClaudeAgentOptions(**options_kwargs)
        self._client = ClaudeSDKClient(options=options)
        self._session_cost_seen = 0.0
        try:
            await self._client.connect()
        except Exception as exc:
            # Spec 116 D9: the SDK's refusal to spawn npm's claude.cmd shim
            # means no native claude.exe exists beside it — an npm layout
            # older than 2.1.114, or a 2.1.114+ install whose postinstall
            # left the stub. Say that, and what to do, instead of the SDK's
            # batch-script paragraph. Everything else re-raises untouched.
            if BATCH_REFUSAL_PREFIX in str(exc):
                raise RuntimeError(
                    f"Claude Code at {options_kwargs['cli_path']} has no native "
                    f"claude.exe (npm layout older than 2.1.114 or an incomplete "
                    f"install); open Settings → Sub-agents and click Update"
                ) from exc
            raise
        self._alive = True
        self._capture_process_handle()
        self._ensure_reader()
        await self._publish_server_commands()

    async def _publish_server_commands(self) -> None:
        """Spec 166: the CLI's ``initialize`` response (stored by the SDK at
        connect, no round trip) lists its slash commands."""
        from agent_os.daemon_v2.worker_commands import claude_commands
        try:
            info = await self._client.get_server_info()
        except Exception as exc:  # noqa: BLE001 — an older SDK; display data only
            logger.info("SDKTransport: no server info (%s)", exc)
            return
        self._publish_commands(claude_commands(info))

    def _capture_process_handle(self) -> None:
        """Capture the claude subprocess handle for own-the-kill teardown.

        Reaches into the SDK's subprocess transport (read-only) for the spawned
        process. Stores a ``psutil.Process`` (identity-checked by pid+create_time,
        so a recycled PID raises ``NoSuchProcess`` rather than signalling an
        innocent process) plus the raw anyio handle for best-effort FD close.
        """
        try:
            raw = self._client._transport._process  # anyio Process
        except Exception:
            logger.warning("SDKTransport: could not reach subprocess handle; "
                           "teardown will rely on disconnect() only")
            return
        self._raw_proc = raw
        pid = getattr(raw, "pid", None)
        if pid is None:
            return
        try:
            import psutil
            self._proc = psutil.Process(pid)
        except Exception:
            logger.warning("SDKTransport: could not wrap subprocess pid=%s in "
                           "psutil.Process", pid, exc_info=True)
        # Windows: the worker (and its shells) die with the daemon.
        from agent_os.utils.windows_job import assign_worker
        assign_worker(pid)

    @property
    def dispatch_pending(self) -> bool:
        """True between ``dispatch()`` and the result that closes it."""
        return self._pending_dispatch

    async def send(self, message: str) -> str | None:
        """Blocking variant: dispatch, then wait for the dispatch-origin
        boundary on the same reader (no second consumer of the SDK stream).
        Events still flow to ``read_stream()``; the collected message text
        is returned."""
        if self._client is None:
            return "Error: SDK client not initialized"
        try:
            await self.dispatch(message)
        except Exception as e:
            return f"Error: SDK query failed: {e}"
        done = self._dispatch_done
        if done is None:
            return "(no response)"
        try:
            await done
        except asyncio.CancelledError:
            raise
        parts, self._dispatch_text = self._dispatch_text, []
        if not parts:
            return "(no response)"
        return "\n".join(parts)

    async def dispatch(self, message: str, *, prompt_uuid: str | None = None) -> None:
        """Send query without waiting for response. Events flow via read_stream().

        ``prompt_uuid`` (spec 156 §5.4) is the native uuid the CLI stores on
        this prompt's transcript entry: the prompt goes out as one streamed
        user message carrying it instead of a bare string. Every daemon
        dispatch passes ``prompt_uuid_for(dispatch_id)``.

        Marks a dispatch pending; the reader's next ``ResultMessage`` closes
        it (``turn_complete`` with ``origin="dispatch"``). A dispatch issued
        while one is still pending closes the prior boundary as ``stopped``
        first, as the per-dispatch consumer's cancel used to, so the
        daemon's dispatch-id FIFO stays paired one boundary per dispatch.
        """
        if self._client is None:
            raise RuntimeError("SDK client not initialized — call start() first")

        if self._pending_dispatch:
            logger.warning("dispatch while a dispatch is pending; closing the "
                           "prior boundary as stopped")
            self._close_dispatch("stopped")

        if prompt_uuid:
            await self._client.query(_one_prompt(message, prompt_uuid))
        else:
            await self._client.query(message)

        self._pending_dispatch = True
        self._outstanding = set()
        self._last_result_is_error = False
        self._dispatch_text = []
        self._dispatch_done = asyncio.get_running_loop().create_future()
        self._ensure_reader()

    def _ensure_reader(self) -> None:
        """Start the persistent reader if it is not running."""
        task = self._reader_task
        if task is not None and not task.done():
            return
        self._reader_task = asyncio.create_task(
            self._reader_loop(), name=f"sdk-reader-{id(self)}")
        self._reader_task.add_done_callback(self._on_reader_done)

    def _on_reader_done(self, task: "asyncio.Task") -> None:
        """Log a reader that died by exception (never silently)."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None or isinstance(exc, asyncio.CancelledError):
            return
        logger.error("SDKTransport: persistent reader raised", exc_info=exc)

    # How long a dispatch stays pending awaiting a background task's
    # continuation turn before giving up (so a stuck/never-finishing task
    # cannot pin the dispatch forever). Generous — real workflows run for
    # minutes. Fire-and-forget ``local_bash`` is excluded from the wait
    # entirely (see _track_background_task), so this only bounds genuinely
    # awaited work.
    CONTINUATION_BUDGET_S: float = 1800.0

    def _track_background_task(self, msg, outstanding: set) -> None:
        """Track AWAITED background tasks by id so the pending dispatch stays
        open until they emit their continuation turn.

        ``TaskStartedMessage`` adds the id; ``TaskNotificationMessage``
        (terminal status: completed/failed/stopped) removes it. Fire-and-forget
        ``run_in_background`` shells (``task_type='local_bash'``) are the
        BackgroundWorkRegistry's domain and are intentionally NOT awaited — the
        foreground turn ends as before so a never-exiting dev server cannot pin
        the dispatch. The shell's exit still reaches the reader: the CLI's own
        follow-up turn is read live and ends as a self-initiated turn.
        """
        if _TaskStarted and isinstance(msg, _TaskStarted):
            if getattr(msg, "task_type", None) != "local_bash":
                tid = getattr(msg, "task_id", None)
                if tid:
                    outstanding.add(tid)
        elif _TaskNotification and isinstance(msg, _TaskNotification):
            tid = getattr(msg, "task_id", None)
            if tid:
                outstanding.discard(tid)

    async def _reader_loop(self) -> None:
        """Read the SDK stream for the life of the client (spec 129 §5A).

        Every message is turned into events exactly as the per-dispatch
        consumer did. Attribution: a ``ResultMessage`` closes the pending
        dispatch if one is pending (the "next result answers the dispatch"
        semantic ``receive_response()`` had — the CLI merges queued user
        messages into an in-flight turn, so no matching by id is attempted);
        otherwise it ends a turn the CLI started on its own, emitted as
        ``turn_complete(origin="self")`` with ``cause`` from ``is_error``.

        The closing ``turn_complete`` of a dispatch carries an honest cause:
        ``success`` (final result, ``is_error`` False), ``stopped`` (a
        deliberate ``stop()``/reap, or a newer dispatch superseding it),
        ``error`` (an error result, a stream exception, or the stream ending
        with the dispatch open — external process death).

        A ``MessageParseError`` is raised in the parse step while the SDK's
        reader carries on: it is reported and reading resumes. Any other
        exception ended the SDK reader (spec 098): emit error +
        turn_complete, mark dead, kill the process so the next dispatch
        rebuilds the client with resume.
        """
        dead_reader_error: Exception | None = None
        try:
            while True:
                try:
                    async for msg in self._client.receive_messages():
                        await self._handle_message(msg)
                    break  # "end": the stream closed (process exit / disconnect)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("SDKTransport: reader error: %s", e)
                    await self._event_queue.put(TransportEvent(
                        event_type="error",
                        data={"error": str(e)},
                        raw_text=f"Error: {e}",
                    ))
                    if isinstance(e, _MessageParseError):
                        continue  # the SDK reader is fine; keep reading
                    # Settled BEFORE turn_complete: the handle's queued-prompt
                    # drain runs on that event and must already see
                    # is_alive() False, or it dispatches into the dead client.
                    if await self._reader_died(e):
                        dead_reader_error = e
                    break
            if self._pending_dispatch:
                logger.warning("SDKTransport: stream ended with a dispatch pending")
                self._close_dispatch("error")
            elif dead_reader_error is not None:
                # Between dispatches: the worker's own turn (if any) is over.
                self._emit_turn_complete(origin="self", cause="error")
        except asyncio.CancelledError:
            if self._pending_dispatch:
                self._close_dispatch("stopped")
            raise
        # After turn_complete, so the turn end reaches the user without
        # waiting on the kill.
        if dead_reader_error is not None:
            await self._kill_after_reader_death(dead_reader_error)

    async def _handle_message(self, msg) -> None:
        self._track_background_task(msg, self._outstanding)
        for event in self._message_to_events(msg):
            if self._pending_dispatch and event.event_type == "message":
                self._dispatch_text.append(event.raw_text)
            await self._event_queue.put(event)
        if isinstance(msg, ResultMessage):
            self._on_result(bool(getattr(msg, "is_error", False)))

    def _on_result(self, is_error: bool) -> None:
        if not self._pending_dispatch:
            self._emit_turn_complete(
                origin="self", cause="error" if is_error else "success")
            return
        self._last_result_is_error = is_error
        if self._outstanding:
            # Awaiting a background task's continuation turn (wire-confirmed:
            # TaskNotification → AssistantMessage → ResultMessage on the same
            # stream, no new query()). Bound the wait so a stuck task cannot
            # pin the dispatch.
            self._arm_budget()
            return
        self._close_dispatch("error" if is_error else "success")

    def _arm_budget(self) -> None:
        if self._budget_handle is not None:
            return
        loop = asyncio.get_running_loop()
        self._budget_handle = loop.call_later(
            self.CONTINUATION_BUDGET_S, self._on_budget_expired)

    def _on_budget_expired(self) -> None:
        self._budget_handle = None
        if not self._pending_dispatch:
            return
        logger.warning(
            "SDKTransport: awaited background task(s) %s did not finish "
            "within %.0fs; ending dispatch without the continuation",
            self._outstanding, self.CONTINUATION_BUDGET_S)
        self._close_dispatch("error" if self._last_result_is_error else "success")

    def _close_dispatch(self, cause: str) -> None:
        """Emit the pending dispatch's boundary and clear it."""
        if self._budget_handle is not None:
            self._budget_handle.cancel()
            self._budget_handle = None
        self._pending_dispatch = False
        self._outstanding = set()
        self._emit_turn_complete(origin="dispatch", cause=cause)
        done = self._dispatch_done
        self._dispatch_done = None
        if done is not None and not done.done():
            done.set_result(cause)

    def _emit_turn_complete(self, *, origin: str, cause: str) -> None:
        """Queue is unbounded, so put_nowait never suspends — safe from the
        cancellation path and from a timer callback. The event carries the
        thread's resume identity (session_id + model) so the daemon can
        persist it (TASK-resume-persistence), and ``origin`` so the consume
        loop can tell a dispatch boundary from a self-initiated turn's end."""
        self._event_queue.put_nowait(TransportEvent(
            event_type="turn_complete",
            data={
                "cause": cause,
                "origin": origin,
                "session_id": self._session_id,
                "model": self._last_model,
            },
        ))

    # The SDK reader's own shutdown (send "error", then "end") can still be in
    # flight when the consumer sees the error; allow it this long to finish.
    READER_EXIT_GRACE_S: float = 2.0

    def _sdk_read_task(self):
        """The SDK's background stdout reader, or None when unreachable.

        claude-agent-sdk 0.1.x stores an ``asyncio.Task``; 0.2.x wraps it in
        its own ``TaskHandle`` (``_internal/_task_compat``). Both expose
        ``done()``, which is all the dead-reader check needs, so duck-type on
        that instead of a class — an ``isinstance(asyncio.Future)`` check
        silently disabled this whole path on 0.2.x.
        """
        task = getattr(getattr(self._client, "_query", None), "_read_task", None)
        return task if callable(getattr(task, "done", None)) else None

    async def _reader_died(self, exc: Exception) -> bool:
        """True when ``exc`` ended the SDK's reader.

        Any exception inside the SDK's reader task (a line over
        ``max_buffer_size``, the CLI exiting mid-turn) ends that task for
        good: nothing reads the CLI's stdout again, so a reused client takes
        the next query and never answers (reproduced on spec 098).
        ``is_alive()`` reports the dead reader so the manager rebuilds the
        client with resume. A ``MessageParseError`` is raised in the
        consumer's parse step while the reader carries on, so it never
        qualifies.
        """
        if isinstance(exc, _MessageParseError):
            return False
        task = self._sdk_read_task()
        if task is None:
            return False
        if not task.done():
            # Poll rather than asyncio.wait(): a 0.2.x TaskHandle is not an
            # awaitable Future, and its wait() re-raises the reader's error.
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.READER_EXIT_GRACE_S
            while not task.done() and loop.time() < deadline:
                await asyncio.sleep(0.02)
        return task.done()

    async def _kill_after_reader_death(self, exc: Exception) -> None:
        """Kill the claude process behind a dead reader.

        Makes the not-alive verdict true for ``list_active()``, which evicts
        a not-alive adapter without ``stop()`` and would otherwise orphan a
        CLI still holding the session.
        """
        proc = self._proc
        logger.warning(
            "SDKTransport: the SDK message reader died (%s); killing claude "
            "pid=%s — the next dispatch rebuilds the client with resume",
            exc, getattr(proc, "pid", None))
        if proc is None:
            return
        try:
            from agent_os.agent.transports.process_kill import kill_process_tree
            await kill_process_tree(proc, label="sdk")
        except Exception:
            logger.exception("SDKTransport: kill after reader death raised")

    async def read_stream(self) -> AsyncIterator[TransportEvent]:
        """Yield events from the queue. Waits for events while alive."""
        while self._alive:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

    async def stop(self) -> None:
        # Deny any pending approvals
        for request_id, future in list(self._pending_approvals.items()):
            if not future.done():
                future.set_result(False)
        self._pending_approvals.clear()
        self._pending_approval_data.clear()

        # Cancel the persistent reader BEFORE disconnecting the client. If
        # the order were reversed, the reader would observe a None/closed
        # client mid-iteration and surface a spurious error. A pending
        # dispatch gets its ``stopped`` boundary from the cancellation path.
        reader = self._reader_task
        if reader is not None and not reader.done():
            reader.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(reader, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "SDKTransport: reader did not cancel within 5s; "
                    "proceeding with disconnect"
                )
        self._reader_task = None
        if self._pending_dispatch:
            # The reader was cancelled before it ever ran (no body to emit
            # from): the boundary still has to be written.
            self._close_dispatch("stopped")
        if self._budget_handle is not None:
            self._budget_handle.cancel()
            self._budget_handle = None

        # 1. Best-effort graceful disconnect. The SDK's Query.close() exits an
        #    anyio task group that was entered in a DIFFERENT task (the one that
        #    connected), so on the daemon's reap/eviction path this raises
        #    "Attempted to exit cancel scope in a different task" — which is
        #    EXPECTED and harmless here. Termination does NOT depend on it
        #    succeeding (FINDINGS-reap-confirmation; Direction B).
        if self._client is not None:
            try:
                await self._client.disconnect()
                logger.debug("SDKTransport.stop: disconnect() returned cleanly")
            except Exception as e:
                logger.debug("SDKTransport.stop: disconnect() best-effort failed "
                             "(expected cross-task): %r", e)

        # 2. Own the kill: terminate the claude process tree ourselves,
        #    independent of the SDK. Snapshots descendants, SIGTERM→SIGKILL,
        #    reaps the parent zombie, never blocks the loop.
        if self._proc is not None:
            try:
                from agent_os.agent.transports.process_kill import kill_process_tree
                outcome = await kill_process_tree(self._proc, label="sdk")
                if not outcome.parent_dead:
                    logger.error("SDKTransport.stop: claude subprocess survived "
                                 "kill; PID may leak")
            except Exception:
                logger.exception("SDKTransport.stop: kill_process_tree raised")

        # 3. Explicitly close subprocess stdio. The leaked SDK task group may
        #    not, so close FDs ourselves to avoid descriptor growth across
        #    dispatch→reap cycles.
        await self._close_subprocess_fds()

        self._client = None
        self._proc = None
        self._raw_proc = None
        self._alive = False
        self._session_id = None

    async def _close_subprocess_fds(self) -> None:
        """Best-effort close of the claude subprocess stdio streams."""
        raw = self._raw_proc
        if raw is None:
            return
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(raw, name, None)
            if stream is None:
                continue
            try:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
                    continue
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            except Exception:
                logger.debug("SDKTransport: closing %s failed", name, exc_info=True)

    def is_alive(self) -> bool:
        # A finished SDK reader means a dead client even though stop() never
        # ran (see _reader_died). read_stream() keeps using _alive so
        # the dying turn's error and turn_complete still drain.
        if not self._alive:
            return False
        task = self._sdk_read_task()
        return task is None or not task.done()

    def _is_attached_file_read(self, tool_name: str, tool_input) -> bool:
        """A read-category tool (Read/Grep/Glob/LS…) aimed at an attached file."""
        if classify_tool(tool_name) != READ or not isinstance(tool_input, dict):
            return False
        for key in ("file_path", "path", "notebook_path"):
            if self.read_is_granted(tool_input.get(key)):
                return True
        return False

    def update_autonomy(self, preset: Autonomy) -> None:
        """Update the autonomy preset for permission filtering (live settings update)."""
        self._autonomy = preset

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def respond_to_permission(self, permission_id: str, approved: bool) -> None:
        """Resolve a pending permission request."""
        future = self._pending_approvals.pop(permission_id, None)
        self._pending_approval_data.pop(permission_id, None)
        if future is not None and not future.done():
            future.set_result(approved)

    async def _handle_permission(self, tool_name: str, tool_input: dict, context) -> "PermissionResultAllow | PermissionResultDeny":
        """Called by the SDK when a tool needs permission.

        Emits a permission_request TransportEvent and waits for approval
        via respond_to_permission().
        """
        # A file the user attached by its local path: Claude Code asks before
        # reading outside its cwd; the attachment already granted exactly this
        # file, so the read goes through under every autonomy preset.
        if self._is_attached_file_read(tool_name, tool_input):
            logger.debug("SDKTransport: allowed %s of an attached file", tool_name)
            return PermissionResultAllow()

        # Auto-approve if autonomy preset allows this tool
        if self._autonomy is not None and should_auto_approve(tool_name, self._autonomy):
            logger.debug("SDKTransport: auto-approved %s (autonomy=%s)", tool_name, self._autonomy.value)
            return PermissionResultAllow()

        request_id = str(uuid.uuid4())

        # Create a future for the approval response
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_approvals[request_id] = future
        self._pending_approval_data[request_id] = {
            "request_id": request_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }

        # Emit permission request event
        event = TransportEvent(
            event_type="permission_request",
            data={
                "request_id": request_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
            raw_text=f"Permission requested: {tool_name}",
        )
        await self._event_queue.put(event)

        # Wait for approval/denial
        try:
            approved = await future
        except asyncio.CancelledError:
            self._pending_approvals.pop(request_id, None)
            self._pending_approval_data.pop(request_id, None)
            return PermissionResultDeny(message="Request cancelled")

        if approved:
            return PermissionResultAllow()
        else:
            return PermissionResultDeny(message="Denied by user")

    def _message_to_events(self, msg) -> list[TransportEvent]:
        """Convert SDK message types to TransportEvents."""
        events = []

        if isinstance(msg, AssistantMessage):
            # Thread-identity capture: the model actually serving this thread
            # (persisted with the resume record; must match on resume).
            # ``<synthetic>`` marks CLI-generated messages (interrupts, API
            # errors), not a model that served the thread.
            model = getattr(msg, "model", None)
            if model and model != "<synthetic>":
                self._last_model = model
            for block in msg.content:
                if isinstance(block, TextBlock):
                    events.append(TransportEvent(
                        event_type="message",
                        data={"text": block.text},
                        raw_text=block.text,
                    ))
                elif isinstance(block, ToolUseBlock):
                    events.append(TransportEvent(
                        event_type="tool_use",
                        data={
                            "tool_name": block.name,
                            "tool_id": block.id,
                            "tool_input": block.input,
                        },
                        raw_text=f"[Using tool: {block.name}]",
                    ))
                elif isinstance(block, ThinkingBlock):
                    # Spec 100: display-only. Most blocks arrive with empty
                    # text (the model omits it), and an empty row is noise.
                    text = block.thinking or ""
                    if text.strip():
                        events.append(TransportEvent(
                            event_type="thinking",
                            data={"text": text},
                            raw_text=text,
                        ))

        elif isinstance(msg, UserMessage):
            # Spec 100: tool results come back as a user turn. Paired with
            # their tool_use by id; display-only (never the turn's summary).
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        events.append(TransportEvent(
                            event_type="tool_result",
                            data={
                                "tool_call_id": block.tool_use_id,
                                "content": _tool_result_text(block.content),
                                "is_error": bool(block.is_error),
                            },
                        ))

        elif isinstance(msg, ResultMessage):
            self._session_id = getattr(msg, 'session_id', None)
            # Display-only usage capture (P3-B). Never affects enforcement
            # (guard counts management only) and never raises — a capture
            # failure must not break or delay the turn.
            self._capture_usage(msg)
            if msg.is_error and msg.result:
                events.append(TransportEvent(
                    event_type="error",
                    data={"error": msg.result},
                    raw_text=f"Error: {msg.result}",
                ))

        elif isinstance(msg, SystemMessage):
            # BACKLOG 005 §4a — eager resume identity. The CLI's ``init``
            # system message carries the upstream session id and arrives BEFORE
            # the first user-visible chunk of the turn (and well before the
            # turn-closing ResultMessage). Capture it now and surface a
            # ``thread_started`` event so the daemon persists the resume record
            # from turn 1, not only after the turn completes. A concurrent
            # @-mention/dispatch into the same chat session then finds the
            # record even mid-turn. Emitted once per distinct session id (init
            # is sent once; the guard also covers any later system frame that
            # re-states the same id).
            #
            # Gate on subtype == "init" specifically: TaskStarted/TaskProgress/
            # TaskNotification subclass SystemMessage and ALSO carry
            # data["session_id"], so a bare isinstance match would let a
            # mid-turn task frame masquerade as a thread start. Only the genuine
            # init frame establishes the resume identity.
            data = getattr(msg, "data", None) or {}
            sid = data.get("session_id")
            if (getattr(msg, "subtype", None) == "init"
                    and sid and self._session_id != sid):
                self._session_id = sid
                events.append(TransportEvent(
                    event_type="thread_started",
                    data={"session_id": sid, "model": self._last_model},
                    raw_text="",
                ))

        elif isinstance(msg, _RateLimitEvent):
            # Spec 099: the account's subscription quota, sent on every turn.
            # Straight to the quota store (display-only, never raises) and NO
            # TransportEvent: an unmapped event type would default to a
            # "response" chunk and land in the transcript and chat.
            from agent_os.daemon_v2.quota_store import publish_claude_rate_limit
            publish_claude_rate_limit(msg.rate_limit_info)

        return events

    def _capture_usage(self, msg) -> None:
        """Append one ``subagent:claude-code`` ledger event for a result message.

        Display-only (Budget Piece 3 Part B): the captured tokens/cost are shown
        in the cost view but NEVER counted toward budget enforcement (the guard
        passes ``sources=["management"]``; the P3-A pinned regression proves
        isolation). NEVER raises: wrapped in the same never-break pattern as
        ``append_event`` so a capture error cannot fail or delay a turn.

        The SDK ``ResultMessage.usage`` dict follows Anthropic's additive usage
        semantics (``input_tokens`` excludes the cache fields), so we normalize
        with ``ProviderSemantics.ANTHROPIC``. ``total_cost_usd`` is the CLI
        process's RUNNING total for the session, not this turn's cost, so the
        recorded ``reported_cost`` is its increase since the previous result
        (never recomputed from our rates). A total below the last one means a
        process we did not see restart, counted from zero. No increase, or
        absent/zero (subscription auth) → tokens only, no reported_cost.
        """
        try:
            usage = getattr(msg, "usage", None)
            if not isinstance(usage, dict):
                return  # no usage on this result; nothing to capture
            # Lazy import: budget.normalize -> providers.types pulls the agent
            # package __init__; a module-level import here risks a cycle and is
            # paid only once per turn-final result.
            from agent_os.agent.providers.types import TokenUsage
            from agent_os.budget.normalize import (
                ProviderSemantics,
                normalize_usage,
            )
            from agent_os.budget.ledger import (
                SOURCE_SUBAGENT_CLAUDE_CODE,
                LedgerEvent,
                append_event,
            )

            token_usage = TokenUsage(
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                cache_write_tokens=int(
                    usage.get("cache_creation_input_tokens", 0) or 0),
            )
            normalized = normalize_usage(token_usage, ProviderSemantics.ANTHROPIC)

            # Summing the running total verbatim overstated a long session
            # ~5x (spend() adds reported_cost across rows).
            reported_cost = None
            reported_cost_currency = None
            cost = getattr(msg, "total_cost_usd", None)
            if isinstance(cost, (int, float)) and cost > 0:
                increase = (cost - self._session_cost_seen
                            if cost >= self._session_cost_seen else cost)
                self._session_cost_seen = float(cost)
                if increase > 0:
                    reported_cost = float(increase)
                    reported_cost_currency = "USD"

            # ResultMessage carries no model attribute; attribution comes from
            # the last AssistantMessage's model, then a sentinel.
            model = self._last_model or "unknown"
            append_event(
                self._workspace,
                LedgerEvent(
                    session_id=self._session_id or "",
                    source=SOURCE_SUBAGENT_CLAUDE_CODE,
                    provider="anthropic",
                    model=model,
                    usage=normalized,
                    reported_cost=reported_cost,
                    reported_cost_currency=reported_cost_currency,
                ),
            )
        except Exception:  # noqa: BLE001 — capture must never break a turn
            logger.warning(
                "SDKTransport: sub-agent usage capture failed (session=%s); "
                "continuing without it.", self._session_id, exc_info=True)


def _tool_result_text(content) -> str:
    """A ToolResultBlock's content as text: a string as-is, the text parts of
    a content list joined, anything else (images, None) as empty."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") == "text")
    return ""
