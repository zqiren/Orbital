# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sub-agent lifecycle management.

Owns all sub-agent adapters. Provides interface for AgentMessageTool.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.prompt_builder import Autonomy
from agent_os.agent.project_paths import ProjectPaths
from agent_os.daemon_v2.sub_agent_visibility import resolve_visible_sub_agent_slugs
from agent_os.daemon_v2.models import (
    SessionKey,
    make_session_key,
)

logger = logging.getLogger(__name__)


MAX_CONCURRENT_SUBAGENTS = 5  # Max active sub-agents per project


@dataclass
class _QueuedPrompt:
    """One provider turn waiting behind the active turn for a handle."""

    message: str
    dispatch_id: str
    transcript_path: str
    initiator: str = "management_agent"


class SubAgentManager:
    """Owns all sub-agent adapters. Provides interface for AgentMessageTool."""

    # Piece 3 Part E — teardown budgets, ordered OUTER > INNER (the old
    # inversion — caller 5.0s < adapter 6.0s — is what cancelled kills
    # mid-flight into the force-drop leak):
    #
    # ADAPTER_STOP_TIMEOUT bounds adapter.stop() itself and must exceed the
    # transport's real kill budget. The SDK path is the worst case: bg-task
    # cancel wait (5s) + disconnect + kill_process_tree (SIGTERM grace +
    # SIGKILL grace + parent reap ≈ 4s) — 6.0 was BELOW it, so any adapter
    # with descendants timed out by construction (reproduced 3× in the
    # piece-3 probes). On exceeding it we now ESCALATE to a direct
    # snapshotted tree kill and confirm — never force-drop.
    ADAPTER_STOP_TIMEOUT = 15.0
    # Straggler escalation grace per phase (wait → terminate → wait → kill →
    # wait) in _reap_stragglers.
    STOP_CONFIRM_GRACE = 3.0
    # How long stop() WAITS for the (shielded) teardown task. On expiry the
    # caller returns; the kill keeps running to confirmation in the
    # background — waiting is bounded, killing is not abandoned.
    TEARDOWN_WAIT_BUDGET = 30.0

    def __init__(self, process_manager, adapter_configs: dict | None = None,
                 platform_provider=None, registry=None, setup_engine=None,
                 project_store=None, lifecycle_observer=None,
                 ws_manager=None):
        self._process_manager = process_manager
        self._adapter_configs = adapter_configs or {}  # handle -> AdapterConfig (legacy)
        self._platform_provider = platform_provider
        self._registry = registry
        self._setup_engine = setup_engine
        self._project_store = project_store
        self._lifecycle_observer = lifecycle_observer
        # WebSocketManager used to surface CLAUDE.md interference banners.
        # Optional: when None we fall back to process_manager._ws if set.
        self._ws_manager = ws_manager
        # Resume persistence (TASK-resume-persistence): resolver wired at
        # bootstrap to AgentManager.get_session so dispatch can read the
        # per-(SessionKey, handle) thread records without SubAgentManager
        # holding an AgentManager reference. None => resume disabled
        # (every spawn is fresh/first_spawn).
        self._session_resolver = None
        # ``_adapters`` is keyed by ``SessionKey == (project_id, session_id)``
        # so each chat session within a project owns an independent slate of
        # active sub-agents. Single-loop callers route through
        # ``DEFAULT_SESSION_ID`` via ``_resolve_session_id``.
        self._adapters: dict[SessionKey, dict[str, object]] = {}
        # Piece 3 Part E (keying fix): transcripts are SESSION-scoped —
        # (project_id, session_id, handle) -> SubAgentTranscript. Two chat
        # sessions running the same handle must not clobber each other's
        # transcript reference.
        self._transcripts: dict[tuple[str, str, str], object] = {}
        # Per-session lifecycle lock — concurrent sub-agent dispatch from
        # two sessions in the same project must not serialize behind a
        # single project-level lock.
        self._lifecycle_locks: dict[SessionKey, asyncio.Lock] = {}
        self._stopping: set[SessionKey] = set()  # sessions currently in stop_all
        # CLAUDE.md interference state for this orbital session.
        # Maps (project_id, content_hash) -> "warned" | "dismissed".
        # Cleared on daemon restart (in-memory only).
        self._claudemd_warning_state: dict[tuple[str, str], str] = {}
        # Per-(project_id, agent_slug) locks guarding ORBITAL-SIDE writes to a
        # sub-agent's MEMORY.md. Track F4 / dispatch 2026-05-20 §5.
        #
        # PARTIAL COVERAGE: this protects Orbital's own touches (lazy creation
        # via ensure_memory_md, UI overwrite via the PUT endpoint, future
        # Orbital-side reads). It does NOT prevent claude.exe from racing
        # directly against the file — that path is entirely external to the
        # daemon (per TASK/INVESTIGATION-sub-agent-memory-write-path.md Track D
        # 2026-05-20). Intercepting claude.exe's I/O is architectural rework,
        # explicitly out of F4's scope (§5.3).
        self._memory_write_locks: dict[tuple[str, str], asyncio.Lock] = {}
        # Fanout wiring (spec 009, Task 2/6): both None until app startup sets
        # them post-construction (mirrors ``_lifecycle_observer``'s own
        # pattern above). ``_fanout_registry`` is the join/gather core
        # (agent_os/daemon_v2/fanout.py); ``_worker_deps_factory`` is
        # ``(project_id, session_id) -> WorkerDeps``, built by Task 6 from
        # Task 3's ``make_tool_registry``. Either being None means
        # ``dispatch_fanout`` cannot run yet — it fails closed with an
        # "unavailable" error rather than raising.
        self._fanout_registry = None
        self._worker_deps_factory = None
        # Prompt execution is serialized independently for every owning
        # management session and sub-agent handle.  The existing
        # ProcessManager dispatch-id deque labels transcript boundaries; it
        # does not prevent a second provider prompt from cancelling or racing
        # the first.  This queue owns that execution invariant.
        self._prompt_queues: dict[tuple[str, str, str], deque[_QueuedPrompt]] = {}
        self._prompt_active: set[tuple[str, str, str]] = set()
        # "Approve all" is Orbital's existing temporary (10 minute) bypass,
        # never a request for a provider-persistent allow grant.
        self._permission_bypass_until: dict[tuple[str, str, str], float] = {}

        # ProcessManager observes the genuine streaming turn boundary.  Keep
        # this duck-typed so focused tests and older embedders can supply a
        # lightweight process-manager double.
        if hasattr(self._process_manager, "set_turn_closed_callback"):
            self._process_manager.set_turn_closed_callback(
                self._on_prompt_turn_closed)
        else:
            self._process_manager.on_turn_closed = self._on_prompt_turn_closed
        if hasattr(self._process_manager, "set_permission_request_callback"):
            self._process_manager.set_permission_request_callback(
                self._on_permission_request)
        else:
            self._process_manager.on_permission_request = self._on_permission_request

    @staticmethod
    def _resolve_session_id(session_id: str | None) -> str:
        """Resolve a sub-agent's parent session id (seam 3 / D1).

        A sub-agent ALWAYS has a parent session, so ``None`` here is a real
        programming bug — it is a hard raise, never a fallback. This is the
        deliberate None-policy difference from the project-level resolver
        (which maps None → holder / persistent chat); the common non-None
        passthrough is the shared ``resolve_session_id`` rule.
        """
        from agent_os.daemon_v2.models import resolve_session_id

        def _required():
            raise ValueError(
                "sub-agent session_id is required (a sub-agent always has a "
                "parent session); got None"
            )
        return resolve_session_id(session_id, on_none=_required)

    def _get_lock(self, project_id: str,
                  session_id: str | None = None) -> asyncio.Lock:
        """Get or create the per-session lifecycle lock."""
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        lock = self._lifecycle_locks.get(sk)
        if lock is None:
            lock = asyncio.Lock()
            self._lifecycle_locks[sk] = lock
        return lock

    def _get_memory_write_lock(
        self, project_id: str, agent_slug: str,
    ) -> asyncio.Lock:
        """Get or lazily create the per-(project, agent) MEMORY.md write lock.

        Synchronous dict op; the returned lock is acquired via ``async with``
        at the call site. See ``self._memory_write_locks`` docstring above
        for the partial-coverage caveat: this lock only serializes
        Orbital-mediated writes, not claude.exe's own direct file writes.
        """
        key = (project_id, agent_slug)
        lock = self._memory_write_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._memory_write_locks[key] = lock
        return lock

    async def ensure_memory_md_locked(
        self, workspace: str, project_id: str, agent_slug: str,
    ) -> str:
        """Lock-guarded wrapper around ``ensure_memory_md``.

        Used by ``_start_from_registry`` at lazy-create time. Returns the
        MEMORY.md path. PARTIAL COVERAGE caveat applies — see
        ``self._memory_write_locks``.
        """
        from agent_os.agent.sub_agent_prompt import ensure_memory_md
        lock = self._get_memory_write_lock(project_id, agent_slug)
        async with lock:
            return ensure_memory_md(workspace, agent_slug)

    async def write_memory_md(
        self, workspace: str, project_id: str, agent_slug: str, content: str,
    ) -> str:
        """Lock-guarded full-overwrite write of MEMORY.md.

        Used by the UI overwrite endpoint at
        ``agent_os/api/routes/agents_v2.py:update_sub_agent_memory``.
        Creates the parent directory if missing, writes the full content,
        returns the path. PARTIAL COVERAGE caveat applies — see
        ``self._memory_write_locks``.
        """
        from agent_os.agent.sub_agent_prompt import _memory_md_path
        path = _memory_md_path(workspace, agent_slug)
        parent = os.path.dirname(path)
        lock = self._get_memory_write_lock(project_id, agent_slug)
        async with lock:
            os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        return path

    async def start(self, project_id: str, handle: str, depth: int = 0,
                    *, session_id: str | None = None,
                    announce: bool = True, fresh: bool = False) -> str:
        """Create adapter from config, call adapter.start(), register with process_manager.

        ``session_id`` selects which chat session this sub-agent attaches to.
        Defaults to ``DEFAULT_SESSION_ID`` for single-loop back-compat.

        ``announce=False`` suppresses the on_started session injection (WS
        broadcast still fires) — used when the spawn is internal to a
        spawn-on-demand ``send`` so one action yields one session message
        (TASK-collapse-dispatch-to-send H2).

        ``fresh=True`` (BACKLOG 005 §4d) forces a reset for this spawn: no
        resume (a brand-new provider session) and a brand-new transcript file,
        bypassing the §4b ``.latest`` reuse. The prior thread/transcript stay
        archived on disk.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        if sk in self._stopping:
            return "Error: project is shutting down, cannot start new agents"

        # Breadth check: limit concurrent sub-agents per session
        current_count = len(self._adapters.get(sk, {}))
        if current_count >= MAX_CONCURRENT_SUBAGENTS:
            return (
                f"Error: concurrent sub-agent limit reached "
                f"(max {MAX_CONCURRENT_SUBAGENTS} per project). "
                f"Stop an existing sub-agent before starting a new one."
            )

        # New path: use registry + setup_engine if available
        if self._registry is not None and self._setup_engine is not None:
            return await self._start_from_registry(
                project_id, handle, session_id=session_id, announce=announce,
                fresh=fresh,
            )

        # Legacy path: use adapter_configs
        config = self._adapter_configs.get(handle)
        if config is None:
            return f"Error: no adapter config for handle '{handle}'"

        # Configure network isolation for this project. Native workers
        # (handle == "worker:<fanout_id>-<i>") never reach this path today —
        # dispatch_fanout constructs NativeWorkerAdapter directly and never
        # calls start() — but guard anyway (belt-and-braces): a worker handle
        # must never trigger the per-start allowlist rewrite this CLI path
        # does.
        if self._platform_provider is not None and not handle.startswith("worker:"):
            try:
                from agent_os.daemon_v2.network_rules_builder import build_network_rules
                project = self._project_store.get_project(project_id) if self._project_store else {}
                approved = project.get("approved_domains") if isinstance(project, dict) else getattr(project, "approved_domains", None)
                rules = build_network_rules(approved)
                self._platform_provider.configure_network(project_id, rules)
            except RuntimeError as e:
                return f"Error: network configuration failed: {e}"

        adapter = CLIAdapter(
            handle=handle,
            display_name=handle,
            platform_provider=self._platform_provider,
            project_id=project_id,
        )
        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            try:
                await adapter.start(config)
            except Exception as e:
                try:
                    await adapter.stop()
                except Exception:
                    pass
                return f"Error: adapter start failed: {e}"
            if sk not in self._adapters:
                self._adapters[sk] = {}
            self._adapters[sk][handle] = adapter

        # Create transcript if workspace is available
        transcript = None
        if self._project_store is not None:
            project = self._project_store.get_project(project_id) if self._project_store else {}
            workspace = project.get("workspace", "") if project else ""
            if workspace:
                from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
                # §4b: reuse the handle's .latest transcript across respawns
                # (continuous UI); §4d fresh=True mints a new one.
                transcript = SubAgentTranscript.open_for_handle(
                    workspace, handle, fresh=fresh)
                self._transcripts[(project_id, session_id, handle)] = transcript

        await self._process_manager.start(project_id, handle, adapter, transcript=transcript, session_id=session_id)

        if self._lifecycle_observer:
            tp = transcript.filepath if transcript else "unknown"
            await self._lifecycle_observer.on_started(project_id, handle, initiator="management_agent", transcript_path=tp, session_id=session_id, inject=announce)

        return f"Started {handle}"

    @staticmethod
    def _argv_value(args, flag: str) -> str | None:
        """Value following ``flag`` in an argv list (config-store flag
        templates render e.g. ["--model", "sonnet"]), or None."""
        args = args or []
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                return args[i + 1]
        return None

    def _resolve_transport(self, manifest, config_dict, autonomy=None, system_prompt: str | None = None,
                           resume_record: dict | None = None):
        """Resolve the appropriate transport for a manifest.

        system_prompt, when provided, is forwarded to transports that
        support it (SDK, Pipe). PTYTransport does not currently support
        --append-system-prompt-file injection; the caller is responsible
        for the degraded first-turn injection path.
        """
        transport_hint = getattr(manifest.runtime, 'transport', 'auto')
        mode = manifest.runtime.mode

        # Determine effective transport type
        if transport_hint == "auto":
            if mode == "pipe":
                # For pipe mode, try SDK first if available, fallback to pipe
                try:
                    from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
                    if HAS_SDK:
                        transport_type = "sdk"
                    else:
                        transport_type = "pipe"
                except ImportError:
                    transport_type = "pipe"
            else:
                transport_type = "pty"
        else:
            transport_type = transport_hint

        if transport_type == "sdk":
            try:
                from agent_os.agent.transports.sdk_transport import SDKTransport, HAS_SDK
                if HAS_SDK:
                    return SDKTransport(
                        autonomy=autonomy, system_prompt=system_prompt,
                        resume_session_id=(resume_record or {}).get("session_id"),
                        # Model is CONFIG (AMENDS piece 2): resolved live from
                        # the current config-store override — never from the
                        # record. No override -> None -> the CLI serves the
                        # session's last-used model (wire-verified).
                        model=self._argv_value(
                            (config_dict or {}).get("args"), "--model"),
                    )
            except ImportError:
                pass
            # Fallback to pipe if SDK not available
            from agent_os.agent.transports.pipe_transport import PipeTransport
            return PipeTransport(
                config=self._get_pipe_config(manifest.slug),
                system_prompt=system_prompt,
                agent_slug=manifest.slug,
            )
        elif transport_type == "pipe":
            from agent_os.agent.transports.pipe_transport import PipeTransport
            return PipeTransport(
                config=self._get_pipe_config(manifest.slug),
                system_prompt=system_prompt,
                agent_slug=manifest.slug,
            )
        elif transport_type == "pty":
            from agent_os.agent.transports.pty_transport import PTYTransport
            approval_patterns = config_dict.get("approval_patterns", [])
            return PTYTransport(approval_patterns=approval_patterns)
        elif transport_type == "acp-sdk":
            from agent_os.agent.transports.acp_sdk_transport import ACPSDKTransport
            # Permission policy and model are current config args consumed by
            # the transport at start(); only the provider session id belongs
            # to resume identity.
            return ACPSDKTransport(resume_record=resume_record)
        elif transport_type == "codex-appserver":
            from agent_os.agent.transports.codex_transport import CodexTransport
            # Resume identity: threadId from the pre-checked record; the
            # record's model is display metadata only — the transport
            # resolves the model from current config argv at start()
            # (model-is-config, AMENDS piece 2). Rollout pre-check happened
            # in _determine_resume. system_prompt is NOT forwarded — the
            # protocol's developerInstructions param is live-untested.
            return CodexTransport(
                autonomy=autonomy,
                resume_record=resume_record,
            )
        else:
            # Fallback: no transport, use legacy CLIAdapter path
            return None

    def _determine_resume(self, workspace: str, project_id: str, handle: str,
                          session_id: str | None):
        """Resume decision for a (SessionKey, handle) dispatch.

        Returns ``(record_or_None, status, reason)`` where status/reason
        follow the TASK-resume-persistence taxonomy:
        ``("resumed", None)`` | ``("fresh", "first_spawn")`` |
        ``("fresh", "resume_failed")``.

        Pre-checks the on-disk resume source instead of trusting the resume
        call to fail loudly (claude-code prunes its store at ~30 days, and
        it is machine-local). A fresh start must never look like a resume.
        """
        resolver = self._session_resolver
        session = resolver(project_id, session_id) if resolver else None
        getter = getattr(session, "get_sub_agent_thread", None)
        record = getter(handle) if getter else None
        if not record or not record.get("session_id"):
            return None, "fresh", "first_spawn"
        # Piece 3 Part F (minimal resume backstop): if a LIVE process is
        # still attached to this claude session id (the rare force-drop/
        # OS-refused-kill leftover that survives Part E), do NOT
        # double-attach — kill-then-resume, confirmed. Resuming a live
        # session silently shares its store file and loses one branch's
        # history on the next resume (REPORT-piece3-prerequisites.md Q2).
        if not self._ensure_no_live_attachment(project_id, handle, record):
            logger.error(
                "resume backstop: live process attached to claude session "
                "%s for %s/%s could NOT be confirmed dead — starting fresh "
                "instead of double-attaching",
                record.get("session_id"), project_id, handle,
            )
            return None, "fresh", "resume_failed"
        try:
            manifest = self._registry.get(handle) if self._registry else None
            transport_hint = getattr(
                getattr(manifest, "runtime", None), "transport", None)
            if transport_hint == "acp-sdk":
                # ACP session identity is provider-owned. There is no
                # provider-neutral local file whose presence proves the
                # session still exists, so pass the persisted candidate to
                # session/load and let transport.resume_outcome report the
                # authoritative result after start(). In particular, never
                # interpret an ACP id as a Claude ~/.claude session id.
                return record, "resumed", None
            if transport_hint == "codex-appserver":
                from agent_os.agent.transports.codex_transport import CodexTransport
                source_ok = CodexTransport.resume_source_exists(record)
            else:
                from agent_os.agent.transports.sdk_transport import SDKTransport
                source_ok = SDKTransport.resume_source_exists(
                    workspace, record["session_id"])
            if source_ok:
                if record.get("background_loss"):
                    # Part E loud-loss: the prior teardown destroyed live
                    # background work — the resume clause must say so.
                    return record, "resumed_after_loss", None
                return record, "resumed", None
        except Exception:
            logger.exception(
                "resume_source_exists check failed for %s/%s — treating as "
                "resume_failed", project_id, handle,
            )
        return None, "fresh", "resume_failed"

    @staticmethod
    def _provider_confirmed_resume_outcome(
        resume_record: dict | None, resume_status: str,
        resume_reason: str | None, transport,
    ) -> tuple[str, str | None]:
        """Apply a provider-confirmed load outcome after transport start."""
        if resume_record is None:
            return resume_status, resume_reason
        outcome = getattr(transport, "resume_outcome", None)
        if not (isinstance(outcome, tuple) and len(outcome) == 2
                and outcome[0] in ("resumed", "fresh")):
            return resume_status, resume_reason
        status, reason = outcome
        if status == "resumed" and resume_record.get("background_loss"):
            return "resumed_after_loss", None
        return status, reason

    def _ensure_no_live_attachment(self, project_id: str, handle: str,
                                   record: dict) -> bool:
        """Part F backstop body: True when it is safe to resume the record's
        claude session (no live attachment, or the leftover was killed and
        CONFIRMED dead). False = could not confirm — caller must not resume.

        Identity check is pid + create_time (PID reuse reads as dead, not
        alive). Minimal by design: no orphan-sweeping subsystem, just
        don't-double-attach.
        """
        pid = record.get("proc_pid")
        ctime = record.get("proc_create_time")
        if not isinstance(pid, int):
            return True  # no anchor recorded — nothing to check
        import psutil as _psutil
        try:
            proc = _psutil.Process(pid)
            if ctime is not None and abs(proc.create_time() - ctime) > 1.0:
                return True  # PID reused by an unrelated process
        except _psutil.NoSuchProcess:
            return True
        except _psutil.Error:
            return True  # unreadable — treat as gone (can't be attached)
        logger.warning(
            "resume backstop: pid %s is still attached to claude session %s "
            "for %s/%s — killing it (confirmed) before resuming",
            pid, record.get("session_id"), project_id, handle,
        )
        try:
            victims = [proc] + proc.children(recursive=True)
        except _psutil.Error:
            victims = [proc]
        # Short grace: this runs synchronously inside the dispatch path and
        # only on the rare leaked-attachment hit; worst case ~3s.
        return self._reap_stragglers(victims, 1.0)

    @staticmethod
    def _format_resume_clause(status: str, reason: str | None) -> str:
        """Human/LLM-facing resume-status clause for the spawn result.

        Honesty rule (R4): a fresh start must never read as a resume, and a
        failed resume says so explicitly so the orchestrator re-briefs.
        """
        if status == "resumed":
            return ("resumed prior session — context from previous work "
                    "is available")
        if status == "resumed_after_loss":
            return ("resumed prior session — context is available, but "
                    "background processes from the prior run were "
                    "TERMINATED before completing; re-verify any work they "
                    "were doing")
        if reason == "resume_failed":
            return ("fresh session — prior session could not be resumed; "
                    "re-brief any context it needs")
        if reason == "explicit_reset":
            # §4d: the caller passed fresh=True deliberately to drop prior
            # context — say so, so the orchestrator knows the reset took.
            return ("fresh session — reset requested (prior conversation "
                    "intentionally dropped)")
        return "fresh session — first spawn"

    def _get_pipe_config(self, slug: str):
        """Build a PipeTransportConfig for the given agent slug."""
        from agent_os.agent.transports.pipe_transport import PipeTransportConfig, CLAUDE_CODE_PIPE_CONFIG
        if slug == "claude-code":
            return CLAUDE_CODE_PIPE_CONFIG
        return PipeTransportConfig()

    # ------------------------------------------------------------------
    # Workspace CLAUDE.md interference: passive detection + WS banner.
    # See spec §4 / DECISIONS-from-followup.md D4: detect+warn only,
    # no prompt-side defense, no --bare opt-in for v1.
    # ------------------------------------------------------------------

    def _ws(self):
        """Return the WebSocketManager instance, falling back to the one
        owned by the process_manager when none was injected directly.
        """
        if self._ws_manager is not None:
            return self._ws_manager
        return getattr(self._process_manager, "_ws", None)

    def _maybe_emit_claudemd_warning(self, project_id: str, workspace: str,
                                     *, session_id: str | None = None) -> None:
        """Inspect workspace CLAUDE.md and emit a one-time WS banner.

        - Logs INFO when CLAUDE.md is present (with content hash).
        - Emits ``workspace_claudemd_warning`` when conflicting tokens
          match. Re-emits if file content changes (different hash) or
          if the daemon was restarted (state map empty).
        - Suppressed for the (project_id, content_hash) pair when the
          banner has already been emitted or dismissed in this session.
        """
        from agent_os.agent.sub_agent_prompt import (
            detect_claudemd_conflict,
            hash_claudemd,
        )

        # Log INFO event whenever CLAUDE.md is present, regardless of conflict.
        present_hash = hash_claudemd(workspace)
        if present_hash is not None:
            logger.info(
                "workspace CLAUDE.md present for project=%s hash=%s",
                project_id, present_hash[:12],
            )

        info = detect_claudemd_conflict(workspace)
        if info is None:
            return

        key = (project_id, info["content_hash"])
        state = self._claudemd_warning_state.get(key)
        if state in ("warned", "dismissed"):
            return

        logger.warning(
            "workspace CLAUDE.md may conflict with Orbital inheritance "
            "(project=%s, matched_token=%r, hash=%s)",
            project_id, info["matched_token"], info["content_hash"][:12],
        )

        # Mark warned BEFORE broadcasting so a misfired re-call cannot
        # double-emit even if broadcast happens to dispatch reentrantly.
        self._claudemd_warning_state[key] = "warned"

        ws = self._ws()
        if ws is not None:
            try:
                ws.broadcast(project_id, {
                    "type": "workspace_claudemd_warning",
                    "project_id": project_id,
                    "session_id": session_id,
                    "claudemd_path": info["claudemd_path"],
                    "content_hash": info["content_hash"],
                    "matched_token": info["matched_token"],
                })
            except Exception:
                logger.exception(
                    "failed to broadcast workspace_claudemd_warning "
                    "for project=%s",
                    project_id,
                )

    def dismiss_claudemd_warning(self, project_id: str, content_hash: str) -> None:
        """Mark a CLAUDE.md banner as dismissed for the current session.

        Suppresses re-emission for the same (project_id, content_hash)
        until the daemon restarts or the file content changes.
        """
        self._claudemd_warning_state[(project_id, content_hash)] = "dismissed"

    async def _start_from_registry(self, project_id: str, handle: str, depth: int = 0,
                                   *, session_id: str | None = None,
                                   announce: bool = True,
                                   fresh: bool = False) -> str:
        """Start a sub-agent using the manifest registry and setup engine.

        ``fresh=True`` (BACKLOG 005 §4d) forces a reset: skip resume and mint a
        new transcript (bypass §4b reuse). See ``start``.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        if sk in self._stopping:
            return "Error: project is shutting down, cannot start new agents"

        # Breadth check: limit concurrent sub-agents per session
        current_count = len(self._adapters.get(sk, {}))
        if current_count >= MAX_CONCURRENT_SUBAGENTS:
            return (
                f"Error: concurrent sub-agent limit reached "
                f"(max {MAX_CONCURRENT_SUBAGENTS} per project). "
                f"Stop an existing sub-agent before starting a new one."
            )

        from agent_os.agent.adapters.base import AdapterConfig

        manifest = self._registry.get(handle)
        if manifest is None:
            return f"Error: unknown agent '{handle}'"

        if manifest.runtime.adapter == "built_in":
            return f"Error: '{handle}' is a built-in agent, not a sub-agent"

        project = self._project_store.get_project(project_id) if self._project_store else {}
        workspace = project.get("workspace", "") if project else ""

        try:
            config_dict = self._setup_engine.get_adapter_config(
                slug=handle,
                project_workspace=workspace,
            )
        except ValueError as e:
            return f"Error: {e}"

        env = config_dict.get("env") or {}
        env.pop("CLAUDECODE", None)  # Prevent nested Claude Code detection

        config = AdapterConfig(
            command=config_dict["command"],
            workspace=config_dict["workspace"],
            approval_patterns=config_dict.get("approval_patterns", []),
            env=env,
            args=config_dict.get("args"),
        )

        # Configure network isolation. Same worker guard as the legacy start()
        # path above — belt-and-braces, currently unreachable for workers.
        if self._platform_provider is not None and not handle.startswith("worker:"):
            try:
                from agent_os.daemon_v2.network_rules_builder import build_network_rules
                approved = project.get("approved_domains") if isinstance(project, dict) else getattr(project, "approved_domains", None)
                rules = build_network_rules(approved, extra=config_dict.get("network_domains", []))
                self._platform_provider.configure_network(project_id, rules)
            except RuntimeError as e:
                return f"Error: network configuration failed: {e}"

        # Sub-agents always run with HANDS_OFF autonomy, regardless of the
        # project's setting for the management agent. Rationale: dispatching a
        # sub-agent IS the user's approval — the management agent (acting on
        # the user's behalf) already decided this work should happen. Requiring
        # a second per-tool approval inside the sub-agent breaks the
        # unattended-queue workflow that is the product's headline feature.
        # The user's project autonomy still applies to the management agent
        # (see agents_v2.py:651-655).
        autonomy = Autonomy.HANDS_OFF

        # --- Sub-agent inheritance: render prompt + lazily create MEMORY.md ---
        # Re-render fresh per dispatch (per spec: "Do NOT cache the rendered string").
        # MEMORY.md is created LAZILY just before injection, never at project init.
        system_prompt: str | None = None
        if workspace:
            try:
                from agent_os.agent.sub_agent_prompt import (
                    ensure_memory_md,
                    render_sub_agent_prompt,
                )

                # Determine peer slugs (other enabled sub-agents in the project),
                # honouring the project's disabled_sub_agents denylist so a
                # disabled sub-agent is never listed as a peer either.
                enabled_sub_agents = resolve_visible_sub_agent_slugs(
                    enabled_sub_agents=(project.get("enabled_sub_agents") if project else None),
                    disabled_sub_agents=(project.get("disabled_sub_agents") if project else None),
                    setup_engine=self._setup_engine,
                )
                if not enabled_sub_agents:
                    enabled_sub_agents = [handle]

                # Lazily create THIS sub-agent's MEMORY.md (not peers — see
                # regression test 5: peer memory files are not created at
                # dispatch of another sub-agent).
                #
                # Skip the create when the manifest's transport drops
                # system_prompt (PTY / ACP today — see Track D investigation
                # 2026-05-20 at TASK/INVESTIGATION-sub-agent-memory-write-path.md).
                # The sub-agent never learns about the file via Orbital, so
                # the stub is a permanent orphan with zero user benefit.
                # Mirrors the auto-resolution in `_resolve_transport`:
                # `auto` + non-pipe mode → pty, which also drops.
                transport_hint = getattr(manifest.runtime, "transport", "auto")
                runtime_mode = getattr(manifest.runtime, "mode", None)
                skips_system_prompt = (
                    transport_hint in (
                        "pty", "acp-sdk", "codex-appserver")
                    or (transport_hint == "auto" and runtime_mode != "pipe")
                )
                if skips_system_prompt:
                    logger.info(
                        "Skipping ensure_memory_md for project=%s handle=%s "
                        "(transport=%s mode=%s does not forward system_prompt).",
                        project_id, handle, transport_hint, runtime_mode,
                    )
                else:
                    # F4 / dispatch 2026-05-20 §5: lock-guarded variant. Two
                    # concurrent dispatches of the same sub-agent within one
                    # project will serialize through ``_memory_write_locks``.
                    # PARTIAL COVERAGE: does not block claude.exe's own writes.
                    await self.ensure_memory_md_locked(
                        workspace, project_id, handle,
                    )

                system_prompt = render_sub_agent_prompt(
                    workspace=workspace,
                    namespace=None,
                    agent_slug=handle,
                    enabled_sub_agents=enabled_sub_agents,
                )
            except Exception:
                logger.exception(
                    "Failed to render sub-agent inheritance prompt for "
                    "project=%s handle=%s",
                    project_id, handle,
                )
                system_prompt = None

            # Detect workspace CLAUDE.md interference (passive surface only).
            # This is a separate side-channel concern from prompt rendering.
            try:
                self._maybe_emit_claudemd_warning(project_id, workspace, session_id=session_id)
            except Exception:
                logger.exception(
                    "claudemd detection failed for project=%s handle=%s",
                    project_id, handle,
                )

        # Resume decision (TASK-resume-persistence): load the persisted
        # (SessionKey, handle) record and pre-check its on-disk source
        # BEFORE building the transport, so a live record resumes and a
        # dead one starts fresh — and says so.
        #
        # §4d reset: fresh=True force-skips resume entirely. No persisted record
        # is consulted; the provider gets NO resume id, so a brand-new upstream
        # session starts. The old record stays on disk until the new turn's
        # on_thread_update overwrites it.
        if fresh:
            resume_record, resume_status, resume_reason = (
                None, "fresh", "explicit_reset")
        else:
            resume_record, resume_status, resume_reason = self._determine_resume(
                workspace, project_id, handle, session_id,
            )

        # Resolve transport from manifest. An unrecognised transport hint
        # falls through to None and downgrades to the legacy adapter path;
        # the catch is the general guard for a ValueError out of transport
        # construction (e.g. a malformed pipe config), so a bad manifest
        # surfaces as a dispatch error string instead of an unhandled raise.
        try:
            transport = self._resolve_transport(
                manifest, config_dict, autonomy=autonomy, system_prompt=system_prompt,
                resume_record=resume_record,
            )
        except ValueError as e:
            return f"Error: unsupported transport in manifest: {e}"

        # Honesty downgrade: a record existed and its source was live, but
        # the resolved transport cannot resume (e.g. pipe fallback when the
        # SDK is unavailable). Never claim "resumed" for a thread the
        # transport did not actually pick up.
        if resume_record is not None and getattr(
                transport, "_resume_session_id", None) != resume_record.get("session_id"):
            resume_status, resume_reason = "fresh", "resume_failed"
            logger.warning(
                "resume record present for %s/%s but transport %s cannot "
                "resume — starting fresh", project_id, handle,
                type(transport).__name__,
            )

        adapter = CLIAdapter(
            handle=handle,
            display_name=manifest.name,
            platform_provider=self._platform_provider,
            project_id=project_id,
            mode=manifest.runtime.mode,
            prompt_flag=manifest.runtime.prompt_flag,
            resume_flag=manifest.runtime.resume_flag,
            session_id_pattern=manifest.runtime.session_id_pattern,
            transport=transport,
        )

        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            try:
                await adapter.start(config)
            except Exception as e:
                try:
                    await adapter.stop()
                except Exception:
                    pass
                return f"Error: adapter start failed: {e}"
            if sk not in self._adapters:
                self._adapters[sk] = {}
            self._adapters[sk][handle] = adapter

        # ACP session/load is the authority on whether continuity actually
        # happened.  The pre-start resume decision only says an attempt is
        # plausible; after start, prefer the provider-confirmed outcome.
        # Preserve the richer first-spawn / explicit-reset reason when no
        # resume was attempted at all.
        resume_status, resume_reason = self._provider_confirmed_resume_outcome(
            resume_record, resume_status, resume_reason, transport)

        # Create transcript for this sub-agent. §4b: reuse the handle's
        # .latest transcript across respawns so the UI stays continuous; §4d
        # fresh=True mints a new file (prior one archived on disk).
        transcript = None
        if workspace:
            from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
            transcript = SubAgentTranscript.open_for_handle(
                workspace, handle, fresh=fresh)
            self._transcripts[(project_id, session_id, handle)] = transcript

        # Pipe handles responses via send() return value — no streaming consumer needed
        # PTY and legacy paths need process_manager to consume read_stream()
        # ACPSDKTransport is deliberately NOT in this tuple: its
        # permission-request events are drained by the streaming consumer
        # (process_manager.py:429-430), so skipping it would hang approvals.
        from agent_os.agent.transports.pipe_transport import PipeTransport
        if not isinstance(transport, (PipeTransport,)):
            await self._process_manager.start(project_id, handle, adapter, transcript=transcript, session_id=session_id)

        if self._lifecycle_observer:
            tp = transcript.filepath if transcript else "unknown"
            await self._lifecycle_observer.on_started(project_id, handle, initiator="management_agent", transcript_path=tp, session_id=session_id, inject=announce)

        # Resume-status reporting: the spawn result is what the management
        # agent reads, so the outcome travels on it (resumed | fresh +
        # reason). Stashed on the adapter too for list/status surfacing
        # (consumed by the collaboration model, SPEC-multiagent).
        adapter._resume_status = (resume_status, resume_reason)
        return (
            f"Started {manifest.name} "
            f"({self._format_resume_clause(resume_status, resume_reason)})"
        )

    async def send(self, project_id: str, handle: str, message: str,
                   *, session_id: str | None = None, depth: int = 0,
                   fresh: bool = False, dispatch_id: str | None = None,
                   initiator: str = "management_agent") -> str:
        """Dispatch message to adapter, spawning the sub-agent if needed.

        Returns immediately with a transcript path acknowledgement.
        The response will appear asynchronously in the transcript and
        via WebSocket broadcast.

        Spawn-on-demand (TASK-collapse-dispatch-to-send): a send to a
        not-running handle spawns it via ``start()`` (announce=False — the
        dispatch ack is the one announcement, carrying the piece-2
        resume/honesty clause) and then dispatches, all in one call. ``send``
        is the management agent's ONLY dispatch verb; the ``start`` tool
        action was removed so a spawned-but-untasked strand state cannot
        occur. ``depth`` is the spawning agent's nesting depth + 1, forwarded
        to the spawn (the MAX_DEPTH policy gate lives in AgentMessageTool).

        ``session_id`` selects which session's adapter slate to dispatch
        through; defaults to ``DEFAULT_SESSION_ID``.

        ``fresh=True`` (BACKLOG 005 §4d) resets this handle's thread before
        dispatching: any live adapter is torn down so the spawn below starts a
        brand-new provider session (no resume) with a new transcript, rather
        than continuing the in-memory client. Without the teardown, a handle
        that is already running would silently keep its old context and the
        reset would be a no-op.

        ``dispatch_id`` (TASK-dispatch-id-pairing) identifies this ONE
        dispatch across the marker written into the management session and
        the transcript boundary row that closes the turn it starts — the
        chat renderer joins the two by this id instead of by position
        (positional pairing broke once a transcript outlived the chat
        session that started it). Minted here when the caller doesn't
        already have one in scope; the @mention API route mints its own up
        front and passes it in.

        ``initiator`` (backlog #23 D3) is threaded through ``_QueuedPrompt``
        to the ONE ``on_message_routed`` notification this dispatch ever
        gets (fired here, immediately, or later by
        ``_on_prompt_turn_closed`` when a queued prompt drains) — it is how
        that single marker learns whether a human addressed the sub-agent
        directly via @mention (``"user_mention"``) versus the management
        agent dispatching it itself. The @mention route passes
        ``initiator="user_mention"`` and fires no notification of its own.
        """
        session_id = self._resolve_session_id(session_id)
        if dispatch_id is None:
            from uuid import uuid4
            dispatch_id = f"{session_id}:{uuid4().hex[:8]}"
        sk = make_session_key(project_id, session_id)
        prompt_key = (project_id, session_id, handle)

        # §4d reset: drop any live adapter for this handle so the spawn-on-
        # demand block below actually re-spawns fresh (the default branch only
        # spawns when the handle is absent). Done OUTSIDE the dispatch lock —
        # stop() takes the same per-session lock internally.
        if fresh and handle in self._adapters.get(sk, {}):
            await self.stop(project_id, handle, session_id=session_id)

        # Spawn-on-demand pre-step, OUTSIDE the dispatch lock: start() takes
        # the same per-session lifecycle lock internally (non-reentrant), so
        # spawning under it would deadlock. A concurrent double-send race here
        # mirrors the pre-existing @mention try/start/retry race — last
        # registration wins, same as before this change.
        spawn_clause = ""
        if handle not in self._adapters.get(sk, {}):
            start_result = await self.start(
                project_id, handle, depth=depth,
                session_id=session_id, announce=False, fresh=fresh,
            )
            if start_result.startswith("Error"):
                return start_result
            spawned = self._adapters.get(sk, {}).get(handle)
            status, reason = getattr(
                spawned, "_resume_status", (None, None)) if spawned else (None, None)
            if status:
                spawn_clause = f" ({self._format_resume_clause(status, reason)})"

        # Invariant 7 (reap-vs-dispatch): take the per-session lifecycle lock
        # around queue admission / dispatch so turn-start is mutually exclusive
        # with a reap.  Exactly one provider prompt is active for this key; later
        # ordinary sends enter the FIFO instead of cancelling the active turn.
        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            adapters = self._adapters.get(sk, {})
            adapter = adapters.get(handle)
            if adapter is None:
                return f"Error: agent '{handle}' not running for project '{project_id}'"
            if getattr(adapter, "_broken", False) is True:
                return (
                    f"Error: agent '{handle}' transport is broken; stop it "
                    "before dispatching again so it can restart cleanly"
                )

            transcript = self._transcripts.get((project_id, session_id, handle))
            transcript_path = transcript.filepath if transcript else "unknown"
            queued = _QueuedPrompt(
                message=message,
                dispatch_id=dispatch_id,
                transcript_path=transcript_path,
                initiator=initiator,
            )
            if prompt_key in self._prompt_active:
                self._prompt_queues.setdefault(prompt_key, deque()).append(queued)
                position = len(self._prompt_queues[prompt_key])
                return (
                    f"Message queued for {handle}{spawn_clause} "
                    f"(position {position}). Transcript: {transcript_path}"
                )

            self._prompt_active.add(prompt_key)
            try:
                await self._dispatch_prompt_locked(adapter, queued, project_id,
                                                   session_id, handle)
            except Exception:
                self._prompt_active.discard(prompt_key)
                raise

        return f"Message sent to {handle}{spawn_clause}. Transcript: {transcript_path}"

    async def _dispatch_prompt_locked(
        self, adapter, prompt: _QueuedPrompt, project_id: str,
        session_id: str, handle: str,
    ) -> None:
        """Start one queued prompt while the session lifecycle lock is held."""
        self._process_manager.set_active_dispatch(
            project_id, handle, prompt.dispatch_id, session_id=session_id)
        try:
            await self._dispatch_async(
                adapter, project_id, handle, prompt.message,
                session_id=session_id, dispatch_id=prompt.dispatch_id)
        except Exception:
            self._process_manager.clear_dispatch(
                project_id, handle, prompt.dispatch_id, session_id=session_id)
            raise

        if self._lifecycle_observer:
            await self._lifecycle_observer.on_message_routed(
                project_id, handle,
                initiator=prompt.initiator,
                message_preview=prompt.message[:100],
                transcript_path=prompt.transcript_path,
                session_id=session_id,
                dispatch_id=prompt.dispatch_id,
            )

    async def _mark_queued_prompts_dropped(
        self, dropped, project_id: str, handle: str, *, session_id: str,
        why: str,
    ) -> None:
        """Leave a durable timeline row for queued prompts that never dispatch.

        A queued @mention is asymmetric (backlog #26d): the user's own message
        is persisted up front by ``persist_mention_message``, but its dispatch
        marker is only written when the prompt actually fires. Dropping the
        queue entry therefore left an orphaned "You -> handle" bubble with
        nothing after it — no trace in the session JSONL of what became of the
        mention.

        ``why`` carries the honest attribution — a user-initiated stop is not
        a malfunction, so the text must read as an explanation rather than an
        error. #26 borrowed ``on_failed`` to say it because that shape already
        had a renderer rule; backlog #35a gives the event its own amber
        ``on_queue_dropped`` shape, so the chrome around the reason stops
        contradicting it.

        One marker per dropped prompt, not one per drop: each queued entry has
        its own orphaned user bubble upstream, and an aggregate row would
        leave all but one of them unexplained.

        Every ``_prompt_queues`` pop must reach here or be provably empty —
        the marker-parity guard discovers the drop sites from this module's
        AST and fails on any that does neither.
        """
        if not dropped or self._lifecycle_observer is None:
            return
        for _prompt in dropped:
            try:
                await self._lifecycle_observer.on_queue_dropped(
                    project_id, handle, why=why, session_id=session_id,
                )
            except Exception:
                logger.exception(
                    "lifecycle_observer.on_queue_dropped raised while marking "
                    "a dropped queued prompt for project=%s handle=%s",
                    project_id, handle,
                )

    async def _on_prompt_turn_closed(
        self, project_id: str, handle: str, *, session_id: str | None = None,
        cause: str | None = None,
    ) -> None:
        """Advance a handle FIFO after its provider reports a real boundary."""
        session_id = self._resolve_session_id(session_id)
        key = (project_id, session_id, handle)
        sk = make_session_key(project_id, session_id)
        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            self._prompt_active.discard(key)
            if cause in ("stream_ended", "consumer_exception"):
                # No honest provider boundary exists and the consumer is
                # gone. Never drain queued work into this dead/unknown
                # transport; mark it broken so the next send fails honestly
                # until the handle is stopped/restarted.
                dropped = self._prompt_queues.pop(key, None)
                adapter = self._adapters.get(sk, {}).get(handle)
                if adapter is not None:
                    adapter._broken = True
                await self._mark_queued_prompts_dropped(
                    dropped, project_id, handle, session_id=session_id,
                    why="transport ended before dispatch",
                )
                return
            queue = self._prompt_queues.get(key)
            if not queue:
                self._prompt_queues.pop(key, None)
                return
            adapter = self._adapters.get(sk, {}).get(handle)
            if adapter is None or getattr(adapter, "_broken", False) is True:
                dropped = self._prompt_queues.pop(key, None)
                await self._mark_queued_prompts_dropped(
                    dropped, project_id, handle, session_id=session_id,
                    why="the sub-agent was no longer available before dispatch",
                )
                return
            prompt = queue.popleft()
            if not queue:
                self._prompt_queues.pop(key, None)
            self._prompt_active.add(key)
            try:
                await self._dispatch_prompt_locked(
                    adapter, prompt, project_id, session_id, handle)
            except Exception:
                self._prompt_active.discard(key)
                dropped = self._prompt_queues.pop(key, None)
                adapter._broken = True
                logger.exception(
                    "queued sub-agent dispatch failed for %s/%s", project_id,
                    handle,
                )
                if self._lifecycle_observer is not None:
                    await self._lifecycle_observer.on_failed(
                        project_id, handle, reason="queued_dispatch_exception",
                        session_id=session_id,
                    )
                # The fifth drop site (backlog #35b). #26 marked four and
                # missed this one — the on_failed above covers only the prompt
                # whose dispatch raised, while everything still queued behind
                # it was popped here and vanished. Found by the new
                # structural drop-site scan, not by reading.
                await self._mark_queued_prompts_dropped(
                    dropped, project_id, handle, session_id=session_id,
                    why="a prior dispatch failed",
                )

    async def respond_to_interaction(
        self, project_id: str, handle: str, interaction_id: str, *,
        session_id: str | None = None, text: str | None = None,
        selection=None,
    ) -> bool:
        """Resolve a blocked provider request without opening another turn."""
        session_id = self._resolve_session_id(session_id)
        adapter = self._adapters.get(
            make_session_key(project_id, session_id), {}).get(handle)
        transport = getattr(adapter, "_transport", None) if adapter else None
        if transport is None:
            return False
        pending = getattr(transport, "_pending_interactions", None)
        if pending is not None and interaction_id not in pending:
            return False
        responder = getattr(transport, "respond_to_interaction", None)
        if responder is None:
            responder = getattr(transport, "resolve_interaction", None)
        if responder is None:
            return False
        await responder(interaction_id, text=text, selection=selection)
        return True

    async def dispatch_fanout(self, project_id: str, tasks: list[dict], *,
                              session_id: str, max_runtime_s: int = 3600,
                              depth: int = 0) -> str:
        """Spawn N native-worker sub-tasks in parallel and register them as
        one join group (spec 009, W2/W3).

        ``tasks``: ``[{"brief": str, "label": str, "files_scope": {"allowed":
        [...], "forbidden": [...]} | None}, ...]``. Errors return
        ``"Error: ..."`` strings — never raises to the calling tool (mirrors
        every other public verb on this class). ``depth`` is accepted for
        signature parity with ``send``/``start`` (nesting depth + 1 from the
        caller); the MAX_DEPTH gate itself lives in the tool layer
        (``FanoutTool`` mirrors ``AgentMessageTool``), not here — same
        division of responsibility as the existing dispatch verbs.
        """
        if not (2 <= len(tasks) <= MAX_CONCURRENT_SUBAGENTS):
            return (
                f"Error: fanout requires 2-{MAX_CONCURRENT_SUBAGENTS} tasks "
                f"(got {len(tasks)}). A single task should use agent_message "
                f"or be done inline."
            )
        for i, task in enumerate(tasks):
            if not isinstance(task, dict) or not task.get("brief") or not task.get("label"):
                return (
                    f"Error: task {i} is missing a required 'brief' and/or "
                    f"'label'"
                )
        if self._worker_deps_factory is None:
            return "Error: fanout is not available (worker factory unwired)"
        if self._fanout_registry is None:
            return "Error: fanout is not available (fanout registry unwired)"

        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)

        import time
        import uuid as _uuid

        from agent_os.daemon_v2.fanout import FanoutTask
        from agent_os.daemon_v2.native_worker import (
            NativeWorkerAdapter,
            make_worker_handle,
        )
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        fanout_id = _uuid.uuid4().hex[:8]
        adapters: dict[str, object] = {}
        fanout_tasks: list[FanoutTask] = []

        # Atomic cap check + adapter construction/registration/dispatch, ALL
        # under the per-session lock (closes the TOCTOU at :210-216/:621-627
        # for the fanout path). Holding the lock across this whole block is
        # deliberate, not just for the cap check: registering an adapter into
        # ``_adapters[sk]`` and then releasing the lock BEFORE dispatching or
        # creating the join group would open a window where a concurrent
        # ``stop_all``/``stop`` (which also takes this lock) could tear a
        # freshly-registered worker down before its brief was ever sent, or
        # before ``create_group`` existed to absorb that teardown as a
        # fanout event instead of a stray per-worker one. Every call made
        # here is synchronous or a bare ``asyncio.create_task(...)`` (no
        # blocking await) — mirrors ``send()``'s own pattern of dispatching
        # while holding this same lock — and this method never calls
        # ``start()``/``stop()`` (the only other lock acquirers), so holding
        # it across the whole block cannot deadlock.
        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            current_count = len(self._adapters.get(sk, {}))
            if current_count + len(tasks) > MAX_CONCURRENT_SUBAGENTS:
                return (
                    f"Error: fanout would exceed the concurrent sub-agent "
                    f"limit (max {MAX_CONCURRENT_SUBAGENTS} per project; "
                    f"{current_count} already running, {len(tasks)} "
                    f"requested). Stop an existing sub-agent first."
                )

            deps = self._worker_deps_factory(project_id, session_id)
            workspace = deps.workspace

            # IMPORTANT (round-3 review): construction must never raise out
            # of dispatch_fanout — every other public verb on this class
            # returns "Error: ..." on failure (mirrors start()'s convention
            # at :258-266). A mid-batch failure (e.g. task 2 of 4's adapter
            # construction raises) must not leave earlier-registered
            # transcripts as residue, so track them and unwind on failure.
            # ``self._adapters`` itself is updated ONLY after the whole
            # try block succeeds (last line before the except), so a
            # failure here never needs to unwind that dict at all.
            registered_handles: list[str] = []
            # Spec 013 (race #5): the management session owns Layer-1 memory.
            # Workers may READ memory files (scope gates only write/edit) but
            # must never write them — a worker writing DECISIONS.md mid-flight
            # would race the management session's background consolidation.
            _memory_forbidden = ProjectPaths(workspace).orbital_dir
            try:
                for i, task in enumerate(tasks):
                    handle = make_worker_handle(fanout_id, i)
                    files_scope = task.get("files_scope") or {}
                    allowed = files_scope.get("allowed")
                    forbidden = list(files_scope.get("forbidden") or [])
                    if _memory_forbidden not in forbidden:
                        forbidden.append(_memory_forbidden)

                    # WorkerDeps is one shared instance per batch (Task 1);
                    # bind on_activity to THIS handle right before
                    # constructing THIS adapter — the adapter reads it once
                    # at __init__, so mutating the shared deps between
                    # constructions is safe.
                    deps.on_activity = self._make_activity_callback(
                        project_id, handle, session_id)

                    adapter = NativeWorkerAdapter(
                        deps=deps, handle=handle, display_name=task["label"],
                        allowed_paths=allowed, forbidden_paths=forbidden,
                    )
                    adapters[handle] = adapter

                    transcript = SubAgentTranscript.open_for_handle(
                        workspace, handle, fresh=True)
                    self._transcripts[(project_id, session_id, handle)] = transcript
                    registered_handles.append(handle)

                    fanout_tasks.append(FanoutTask(
                        handle=handle,
                        label=task["label"],
                        brief=task["brief"],
                        allowed_paths=allowed,
                        forbidden_paths=forbidden,
                        status="running",
                        transcript_path=transcript.filepath,
                        last_activity=time.monotonic(),
                        session_uuid=adapter.session_uuid,
                    ))

                group = self._fanout_registry.create_group(
                    project_id, session_id, fanout_tasks,
                    max_runtime_s=max_runtime_s,
                )
                self._adapters.setdefault(sk, {}).update(adapters)
            except Exception as e:
                logger.exception(
                    "dispatch_fanout: worker construction failed for "
                    "project=%s fanout=%s", project_id, fanout_id,
                )
                for handle in registered_handles:
                    self._transcripts.pop(
                        (project_id, session_id, handle), None)
                return f"Error: fanout worker construction failed: {e}"

            await self._fanout_registry.start_watchdog(group)

            for task, fanout_task in zip(tasks, fanout_tasks):
                await self._dispatch_async(
                    adapters[fanout_task.handle], project_id,
                    fanout_task.handle, task["brief"], session_id=session_id,
                )

        ws = self._ws()
        if ws is not None:
            try:
                ws.broadcast(project_id, {
                    "type": "fanout.started",
                    "project_id": project_id,
                    "session_id": session_id,
                    "fanout_id": fanout_id,
                    "tasks": [
                        {
                            "handle": t.handle,
                            "label": t.label,
                            "session_uuid": t.session_uuid,
                        }
                        for t in fanout_tasks
                    ],
                })
            except Exception:
                logger.exception(
                    "failed to broadcast fanout.started for project=%s "
                    "fanout=%s", project_id, fanout_id,
                )

        labels = ", ".join(t.label for t in fanout_tasks)
        return (
            f"Fanout {fanout_id} dispatched: {len(fanout_tasks)} tasks — "
            f"{labels}. Results will arrive together when all tasks finish."
        )

    def _make_activity_callback(self, project_id: str, handle: str,
                                session_id: str):
        """Zero-arg closure bound to one worker handle, handed to that
        worker's ``WorkerDeps.on_activity`` (see ``dispatch_fanout``)."""
        def _on_activity() -> None:
            if self._fanout_registry is not None:
                self._fanout_registry.note_activity(
                    project_id, handle, session_id)
        return _on_activity

    async def _dispatch_async(self, adapter, project_id: str, handle: str, message: str,
                              *, session_id: str | None = None,
                              dispatch_id: str | None = None) -> None:
        """Dispatch message to adapter without blocking on response.

        For transports that support non-blocking dispatch (SDK with queue),
        writes to the adapter and returns. For blocking transports (Pipe, ACP)
        and legacy PTY paths, wraps the send in a background task.

        ``dispatch_id`` (TASK-dispatch-id-pairing, FIFO desync guard) is used
        ONLY by the blocking-transport branch below: its failure happens
        asynchronously inside a detached background task, so this function
        must clean up the queued id itself when that task fails before ever
        producing a response. The non-blocking (SDK) branch's failure is a
        synchronous raise straight out of this ``await`` — the caller
        (``send()``) wraps that call and does the equivalent cleanup there.
        """
        transport = getattr(adapter, '_transport', None)

        if transport is not None and hasattr(transport, 'dispatch'):
            adapter._idle = False  # Reset idle on new task
            await transport.dispatch(message)
            return

        # Fallback: wrap send() in background task (covers PTY, Pipe, ACP)
        transcript = self._transcripts.get((project_id, session_id, handle))

        async def _background_send():
            from datetime import datetime, timezone
            # Fanout activity plumbing (Task 2, spec 009 §0.5-5): bump the
            # watchdog's last-activity clock at dispatch time. A no-op for
            # non-fanout handles (owner_group lookup misses) and when no
            # registry is wired yet. This is turn-START coverage from the
            # dispatch side; NativeWorkerAdapter's own on_activity callback
            # (wired by the spawner) covers turn-start/end from inside the
            # loop — belt-and-braces per the brief, not a duplicate bug.
            if self._fanout_registry is not None:
                try:
                    self._fanout_registry.note_activity(
                        project_id, handle, session_id)
                except Exception:
                    logger.exception(
                        "fanout note_activity raised for %s/%s",
                        project_id, handle,
                    )
            try:
                await adapter.send(message)
                boundary_noted = False
                # Pipe/ACP transports store the response in _last_response
                response = getattr(adapter, '_last_response', None)
                if response:
                    ts = datetime.now(timezone.utc).isoformat()
                    if transcript is not None:
                        transcript.append({
                            "source": handle,
                            "content": response,
                            "timestamp": ts,
                            "chunk_type": "response",
                        })
                    self._process_manager._ws.broadcast(project_id, {
                        "type": "chat.sub_agent_message",
                        "project_id": project_id,
                        "session_id": session_id,
                        "content": response,
                        "source": handle,
                        "timestamp": ts,
                    })
                    if self._lifecycle_observer and transcript is not None:
                        # Honest routing (TASK-honest-subagent-completion-
                        # reporting, fix 6): blocking transports surface
                        # failures as "Error:"-prefixed response strings
                        # (pipe timeout / non-zero exit, SDK send errors).
                        # Those are failures, not completion summaries.
                        #
                        # IMPORTANT (round-3 review): a NativeWorkerAdapter
                        # stopped via stop_all/user-stop ALSO produces an
                        # "Error: task was cancelled..." response (its
                        # AgentLoop.run() returns normally after
                        # cancel_turn(), and _read_final_response()'s
                        # exit_reason=="cancelled" fallback is "Error:"-
                        # prefixed) — that's a deliberate stop, not a
                        # failure, and must not be reported to the fanout
                        # join / management session as "broke". Route those
                        # to on_turn_interrupted instead. getattr defaults
                        # to False for adapters with no such flag (CLI
                        # adapters), so this is additive-only for them.
                        if response.startswith("Error") and getattr(
                                adapter, "_stop_requested", False):
                            await self._lifecycle_observer.on_turn_interrupted(
                                project_id, handle,
                                transcript.filepath,
                                session_id=session_id,
                            )
                        elif response.startswith("Error"):
                            await self._lifecycle_observer.on_error(
                                project_id, handle,
                                response[:500],
                                transcript.filepath,
                                session_id=session_id,
                            )
                        else:
                            await self._lifecycle_observer.on_completed(
                                project_id, handle,
                                summary=response[:200] if response else "(no output)",
                                transcript_path=transcript.filepath,
                                session_id=session_id,
                            )
                        # The turn is accounted for — a later clean teardown's
                        # stream-end must not re-report it (these transports
                        # never emit turn_complete).
                        self._process_manager.note_turn_closed(
                            project_id, handle, session_id=session_id)
                        boundary_noted = True
                # A blocking send returning is itself its genuine request /
                # response boundary, including an empty response.  Account
                # for it exactly once, then advance the execution FIFO.
                if not boundary_noted:
                    self._process_manager.note_turn_closed(
                        project_id, handle, session_id=session_id)
                await self._on_prompt_turn_closed(
                    project_id, handle, session_id=session_id,
                    cause="success",
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    "_background_send failed for project=%s handle=%s",
                    project_id, handle, exc_info=True,
                )
                adapter._broken = True
                # FIFO desync guard (Important review finding,
                # TASK-dispatch-id-pairing): adapter.send() failed before a
                # response was ever produced, so note_turn_closed() above
                # never ran — no boundary will pop the id set_active_dispatch
                # enqueued for this dispatch. Left queued it would be handed
                # to the NEXT dispatch's boundary instead of its own.
                if dispatch_id:
                    self._process_manager.clear_dispatch(
                        project_id, handle, dispatch_id, session_id=session_id)
                if self._lifecycle_observer is not None:
                    try:
                        # on_failed injects into the management session (not
                        # just WS) so the dispatcher learns the task died —
                        # path (e) of the honest-reporting contract.
                        await self._lifecycle_observer.on_failed(
                            project_id, handle,
                            reason="background_send_exception",
                            session_id=session_id,
                        )
                    except Exception:
                        logger.exception(
                            "lifecycle_observer.on_failed raised for project=%s handle=%s",
                            project_id, handle,
                        )
                key = (project_id, self._resolve_session_id(session_id), handle)
                self._prompt_active.discard(key)
                dropped = self._prompt_queues.pop(key, None)
                # The on_failed above covers only the send that raised; the
                # prompts still waiting behind it are discarded here and are
                # the same silent-drop class as every other pop of this queue.
                await self._mark_queued_prompts_dropped(
                    dropped, project_id, handle, session_id=session_id,
                    why="a prior send failed before dispatch",
                )
                return

        adapter._background_send_task = asyncio.create_task(
            _background_send(),
            name=f"bgsend-{project_id}-{handle}",
        )
        adapter._background_send_task.add_done_callback(
            lambda t: self._on_bg_send_done(adapter, project_id, handle, t)
        )

    def _on_bg_send_done(self, adapter, project_id: str, handle: str, task: asyncio.Task) -> None:
        """Done-callback for the background send task.

        Clears the strong reference on the adapter and defensively marks the
        adapter broken if the task raised an exception that the inner handler
        somehow missed. Must not itself raise — it runs on the event loop
        during task finalization.
        """
        try:
            if task.cancelled():
                logger.debug(
                    "_background_send cancelled for project=%s handle=%s",
                    project_id, handle,
                )
            else:
                exc = task.exception()
                if exc is not None:
                    # Inner handler should already have set _broken. Defensive
                    # fallback in case the exception escaped the inner try.
                    adapter._broken = True
                    logger.error(
                        "_background_send task ended with exception for "
                        "project=%s handle=%s",
                        project_id, handle, exc_info=exc,
                    )
        except Exception:
            logger.exception(
                "_on_bg_send_done callback raised for project=%s handle=%s",
                project_id, handle,
            )
        finally:
            adapter._background_send_task = None

    async def stop(self, project_id: str, handle: str, *,
                   session_id: str | None = None) -> str:
        """Stop adapter with CONFIRMED death, then deregister the consumer.

        Piece 3 Part E: no path pops the adapter and walks away while the
        process may be alive — the old force-drop on timeout is what leaked
        a live process with a stale resume record (the double-attach
        precondition). The kill-to-confirmation runs in its OWN task and is
        shielded from caller cancellation: an impatient caller stops
        WAITING, never the KILL. The adapter slot is released only after
        the snapshotted process tree is verified dead; an unkillable tree
        keeps the slot (marked broken — dispatch refuses it) and logs
        loudly.

        ``session_id`` selects which session's adapter slate to stop in.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        prompt_key = (project_id, session_id, handle)
        # A stop is also cancellation of work that has not reached the
        # provider yet.  Pending in-flight interactions are rejected by the
        # transport's stop() implementation so the original JSON-RPC request
        # cannot remain stranded.
        dropped = self._prompt_queues.pop(prompt_key, None)
        self._prompt_active.discard(prompt_key)
        self._permission_bypass_until.pop(prompt_key, None)
        # Mark before the kill, not after: the teardown below can exceed its
        # wait budget and return early, and these rows are the only record
        # that the queued messages existed at all.
        await self._mark_queued_prompts_dropped(
            dropped, project_id, handle, session_id=session_id,
            why="agent stopped before dispatch",
        )
        teardown = asyncio.create_task(
            self._kill_confirm_and_release(project_id, session_id, handle),
            name=f"teardown-{project_id}-{handle}",
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(teardown), timeout=self.TEARDOWN_WAIT_BUDGET,
            )
        except asyncio.TimeoutError:
            logger.error(
                "stop: teardown of %s exceeded the %.0fs wait budget; the "
                "kill CONTINUES in the background until confirmed dead "
                "(adapter slot retained meanwhile — no force-drop)",
                handle, self.TEARDOWN_WAIT_BUDGET,
            )
            return f"Stopping {handle} (kill continuing in background)"
        except asyncio.CancelledError:
            # Caller cancelled — the shielded teardown still runs to
            # completion on its own; never abandon a kill mid-flight.
            raise
        if result == "not_running":
            return f"Agent '{handle}' not running"
        # NOTE (shutdown-ordering coupling, TASK-honest-completion fix 4 /
        # trigger-fix piece 3): adapter.stop() inside the teardown cancels
        # the SDK bg task, whose finally pushes a turn_complete tagged
        # cause="stopped" that the still-alive consumer ingests during the
        # SIGTERM grace window. ProcessManager routes cause="stopped" to NO
        # lifecycle event — that tag is the only thing standing between a
        # clean reap and a phantom "[Sub-agent] completed". The consumer is
        # deregistered only AFTER the teardown completed.
        await self._process_manager.stop(
            project_id, handle, session_id=session_id,
        )
        if result == "unkillable":
            return (f"Stop FAILED for {handle}: process not confirmed dead; "
                    f"slot retained (broken)")
        return f"Stopped {handle}"

    @staticmethod
    def _snapshot_kill_targets(adapter) -> list:
        """psutil handles for the adapter's process tree, captured BEFORE
        adapter.stop() (which clears the transport's proc references)."""
        import psutil as _psutil
        roots = []
        transport = getattr(adapter, "_transport", None)
        proc = getattr(transport, "_proc", None)  # SDK: psutil.Process
        # Strict isinstance: partially-mocked adapters (tests) and foreign
        # transports must not feed non-psutil objects into the kill path.
        if isinstance(proc, _psutil.Process):
            roots.append(proc)
        for raw in (getattr(transport, "_process", None),
                    getattr(adapter, "_process", None)):
            pid = getattr(raw, "pid", None)
            if isinstance(pid, int):
                try:
                    roots.append(_psutil.Process(pid))
                except _psutil.Error:
                    pass
        victims = []
        for r in roots:
            victims.append(r)
            try:
                victims.extend(r.children(recursive=True))
            except Exception:
                pass
        return victims

    @staticmethod
    def _reap_stragglers(victims: list, grace_s: float) -> bool:
        """Blocking (run in a thread): escalate terminate→kill on whatever in
        ``victims`` is still alive and CONFIRM death. True iff all dead."""
        import psutil as _psutil
        if not victims:
            return True
        _, alive = _psutil.wait_procs(victims, timeout=grace_s)
        for p in alive:
            try:
                p.terminate()
            except _psutil.Error:
                pass
        _, alive = _psutil.wait_procs(alive, timeout=grace_s)
        for p in alive:
            try:
                p.kill()
            except _psutil.Error:
                pass
        _, alive = _psutil.wait_procs(alive, timeout=grace_s)
        return not alive

    async def _kill_confirm_and_release(self, project_id: str,
                                        session_id: str, handle: str) -> str:
        """The teardown body: kill → confirm → release-or-retain.

        Holds the per-session lifecycle lock across the whole sequence
        (invariant 7: a concurrent dispatch cannot open a turn mid-kill).
        Returns 'stopped' | 'unkillable' | 'not_running'.
        """
        sk = make_session_key(project_id, session_id)
        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            adapters = self._adapters.get(sk, {})
            adapter = adapters.get(handle)
            if adapter is None:
                return "not_running"

            # Loud-loss accounting BEFORE the kill (Part E): live tracked
            # background work that this teardown will destroy.
            from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
            registry = getattr(self._process_manager, "background_work", None)
            doomed: list[str] = []
            if isinstance(registry, BackgroundWorkRegistry):
                doomed = registry.live_commands(project_id, session_id, handle)

            victims = self._snapshot_kill_targets(adapter)
            try:
                await asyncio.wait_for(
                    adapter.stop(), timeout=self.ADAPTER_STOP_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "stop: adapter %s stop() exceeded %.0fs; escalating to "
                    "direct tree kill", handle, self.ADAPTER_STOP_TIMEOUT,
                )
            except Exception:
                logger.exception(
                    "stop: adapter %s stop() raised; escalating to direct "
                    "tree kill", handle,
                )
            confirmed = await asyncio.to_thread(
                self._reap_stragglers, victims, self.STOP_CONFIRM_GRACE,
            )
            if confirmed:
                adapters.pop(handle, None)
                if not adapters:
                    self._adapters.pop(sk, None)
                if isinstance(registry, BackgroundWorkRegistry):
                    registry.clear(project_id, session_id, handle)
            else:
                # NEVER pop-and-leak: keep the slot so the process stays
                # addressable; broken refuses reuse; Part F's backstop
                # protects resume.
                adapter._broken = True
                logger.error(
                    "stop: adapter %s NOT confirmed dead after escalation; "
                    "slot retained (broken) — pids may include %s",
                    handle,
                    [getattr(v, 'pid', '?') for v in victims][:8],
                )
                return "unkillable"

        # Outside the lock: loud loss reporting + resume-record note.
        if doomed:
            await self._report_background_loss(
                project_id, session_id, handle, doomed,
            )
        return "stopped"

    async def _report_background_loss(self, project_id: str, session_id: str,
                                      handle: str, commands: list[str]) -> None:
        """A teardown destroyed live background work — report LOUDLY (Part E):
        management-session event + WS broadcast + a note on the resume record
        so the next resume is briefed about the loss."""
        if self._lifecycle_observer is not None:
            try:
                await self._lifecycle_observer.on_background_work_lost(
                    project_id, handle, commands=commands,
                    session_id=session_id,
                )
            except Exception:
                logger.exception(
                    "on_background_work_lost raised for %s/%s",
                    project_id, handle,
                )
        # Flag the persisted thread record (cleared naturally by the next
        # completed turn's on_thread_update overwrite).
        try:
            resolver = self._session_resolver
            session = resolver(project_id, session_id) if resolver else None
            record = (session.get_sub_agent_thread(handle)
                      if session is not None else None)
            if session is not None and record and record.get("session_id"):
                session.set_sub_agent_thread(
                    handle,
                    session_id=record["session_id"],
                    model=record.get("model"),
                    background_loss=True,
                )
        except Exception:
            logger.exception(
                "failed to flag background loss on thread record for %s/%s",
                project_id, handle,
            )

    def _adapter_status(self, adapter, project_id: str, session_id: str,
                        handle: str) -> str:
        """Three-state status (Piece 3 Part A): 'running' (turn open),
        'background-running' (turn done but registered background work is
        live), 'idle' (turn done, nothing real underneath).

        Three-state applies ONLY to transports that declare
        ``supports_background_status`` (SDK claude-code) AND only when the
        registry's platform gate is open (the Windows liveness shape is
        unverified — degraded two-state there). Warm infrastructure (MCP
        servers, browsers) is never consulted: classification is by
        provenance records, not by any-live-child.
        """
        if not adapter.is_idle():
            return "running"
        transport = getattr(adapter, "_transport", None)
        if transport is not None and getattr(
                transport, "supports_background_status", False):
            from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
            registry = getattr(self._process_manager, "background_work", None)
            if (isinstance(registry, BackgroundWorkRegistry)
                    and registry.three_state_supported
                    and registry.has_live(project_id, session_id, handle)):
                return "background-running"
        return "idle"

    def status(self, project_id: str, handle: str, *,
               session_id: str | None = None) -> str:
        """Return 'running' | 'background-running' | 'idle' | 'stopped' | 'unknown'."""
        session_id = self._resolve_session_id(session_id)
        adapters = self._adapters.get(make_session_key(project_id, session_id), {})
        adapter = adapters.get(handle)
        if adapter is None:
            return "unknown"
        if not adapter.is_alive():
            return "stopped"
        return self._adapter_status(adapter, project_id, session_id, handle)

    def get_pending_sub_agent_approval(self, project_id: str, *,
                                       session_id: str | None = None) -> dict | None:
        """Return the first pending sub-agent approval for a session, or None.

        Used by the REST recovery endpoint so clients can fetch sub-agent
        approval card data when they miss the WebSocket event.

        This is a LOOKUP/RECOVERY read (seam 3 / D1): ``session_id`` is
        optional. The REST poll (``GET /agents/{pid}/pending-approval``) has no
        session in hand, so it passes ``None`` — a legitimate "not yet known"
        state, NOT a bug. None must therefore degrade GRACEFULLY (scan), never
        hard-raise like the sub-agent lifecycle resolver. When ``session_id`` is
        None we scan every adapter slate in the project (a pending approval is
        unique to its transport, so a project-wide scan is unambiguous),
        mirroring the None-tolerance of ``resolve_sub_agent_approval``. A
        provided ``session_id`` narrows to that one slate.
        """
        if session_id is None:
            slates = [
                adapters for (pid, _sid), adapters in self._adapters.items()
                if pid == project_id
            ]
        else:
            slates = [self._adapters.get(make_session_key(project_id, session_id), {})]
        for adapters in slates:
            for handle, adapter in adapters.items():
                transport = getattr(adapter, '_transport', None)
                if transport is None:
                    continue
                pending = getattr(transport, '_pending_approvals', {})
                if not pending:
                    continue
                # Get metadata from _pending_approval_data if available
                approval_data = getattr(transport, '_pending_approval_data', {})
                for request_id in pending:
                    data = approval_data.get(request_id, {})
                    raw_tool_args = data.get("tool_input", {})
                    tool_args = (
                        dict(raw_tool_args)
                        if isinstance(raw_tool_args, dict)
                        else {"input": raw_tool_args}
                    )
                    if data.get("options"):
                        tool_args["permission_options"] = data["options"]
                    return {
                        "tool_call_id": data.get("request_id", request_id),
                        "tool_name": data.get("tool_name", ""),
                        "tool_args": tool_args,
                        "what": f"Sub-agent {handle} requests approval: {data.get('tool_name', 'unknown')}",
                        "source": handle,
                    }
        return None

    async def _on_permission_request(
        self, project_id: str, handle: str, request_id: str, *,
        session_id: str | None = None, metadata: dict | None = None,
    ) -> bool:
        """Auto-resolve a request covered by Orbital's temporary bypass.

        Returns True when ProcessManager should suppress the permission card.
        A provider's ordinary ``auto`` mode normally resolves before emitting
        an event; this callback is the ask-mode, 10-minute bypass backstop.
        """
        session_id = self._resolve_session_id(session_id)
        key = (project_id, session_id, handle)
        if self._permission_bypass_until.get(key, 0.0) <= time.monotonic():
            self._permission_bypass_until.pop(key, None)
            return False
        return await self.resolve_sub_agent_approval(
            project_id, request_id, approved=True, session_id=session_id,
        )

    async def resolve_sub_agent_approval(self, project_id: str, tool_call_id: str,
                                         approved: bool, *,
                                         session_id: str | None = None,
                                         decision: str | None = None,
                                         reply_text: str | None = None,
                                         approve_all: bool = False) -> bool:
        """Try to resolve a permission request on any sub-agent transport.

        Returns True if the approval was routed to a sub-agent, False if not found.

        Unlike ``start``/``send``/``stop`` (which act on one named session and so
        require a session_id), approval routing is a *lookup by tool_call_id*: the
        REST approve/deny endpoints may not know which session owns the pending
        request. So ``session_id`` is optional here — when omitted (None), every
        adapter slate in the project is scanned. The tool_call_id is globally
        unique, so a project-wide scan is unambiguous. This is the one sub-agent
        path that legitimately tolerates None (seam 3 / D1: no "default" sentinel —
        a None search just widens to all sessions, it does not route to a phantom).

        Optional ``decision`` passes a richer provider vocabulary string (e.g.
        ``"cancel"`` to end the turn as ``interrupted``) to transports that
        implement ``respond_to_permission_decision``.  When ``decision`` is None
        the legacy boolean ``respond_to_permission`` path is used unchanged.
        ``reply_text`` is forwarded only to transports whose permission
        surface accepts guidance (ACP itself currently does not). ``approve_all`` activates Orbital's existing
        ten-minute, in-memory bypass; it is deliberately not translated into
        a provider-persistent ``allow_always`` grant.
        """
        if session_id is None:
            slates = [
                adapters for (pid, _sid), adapters in self._adapters.items()
                if pid == project_id
            ]
        else:
            slates = [self._adapters.get(make_session_key(project_id, session_id), {})]
        for adapters in slates:
            for handle, adapter in adapters.items():
                transport = getattr(adapter, '_transport', None)
                if transport is not None and hasattr(transport, 'respond_to_permission'):
                    # Check if this transport has the pending approval
                    pending = getattr(transport, '_pending_approvals', {})
                    if tool_call_id in pending:
                        if approve_all and approved:
                            if session_id is None:
                                # Locate the concrete owning slate for the
                                # project-wide lookup case.
                                owning_sid = next(
                                    (sid for (pid, sid), slate in self._adapters.items()
                                     if pid == project_id and adapter in slate.values()),
                                    None,
                                )
                            else:
                                owning_sid = session_id
                            if owning_sid is not None:
                                self._permission_bypass_until[
                                    (project_id, owning_sid, handle)
                                ] = time.monotonic() + 600.0

                        # Check the advertised surface, not just getattr():
                        # MagicMock fabricates arbitrary attributes and would
                        # otherwise look like it implements this optional API.
                        rich = (
                            getattr(transport, "respond_to_permission_response", None)
                            if "respond_to_permission_response" in dir(transport)
                            else None
                        )
                        if rich is not None:
                            await rich(
                                tool_call_id,
                                approved=approved,
                                reply_text=reply_text,
                                # Temporary duration is explicit so a
                                # transport cannot mistake it for permanent
                                # provider authorization.
                                temporary_allow_s=600 if approve_all and approved else None,
                                decision=decision,
                            )
                        elif decision is not None and hasattr(
                                transport, "respond_to_permission_decision"):
                            # Richer codex vocabulary: "cancel" ends the turn
                            # `interrupted` (Deny & stop); "decline" lets it
                            # continue. Transports without the method (SDK)
                            # fall back to the boolean wire.
                            await transport.respond_to_permission_decision(
                                tool_call_id, decision)
                        else:
                            await transport.respond_to_permission(
                                tool_call_id, approved)
                        return True
        return False

    def update_sub_agent_autonomy(self, project_id: str, preset, *,
                                  session_id: str | None = None) -> None:
        """Propagate autonomy preset change to all active SDK sub-agent transports."""
        session_id = self._resolve_session_id(session_id)
        adapters = self._adapters.get(make_session_key(project_id, session_id), {})
        for handle, adapter in adapters.items():
            transport = getattr(adapter, '_transport', None)
            if transport is not None and hasattr(transport, 'update_autonomy'):
                transport.update_autonomy(preset)

    async def stop_all(self, project_id: str, *,
                       session_id: str | None = None) -> None:
        """Stop all sub-agent adapters for a session.

        ``session_id`` selects which session's adapter slate to drain;
        defaults to ``DEFAULT_SESSION_ID``. Other sessions in the same
        project are unaffected.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        self._stopping.add(sk)
        try:
            lock = self._get_lock(project_id, session_id=session_id)
            async with lock:
                adapters = self._adapters.get(sk, {})
                handles = list(adapters.keys())
            for handle in handles:
                try:
                    await self.stop(project_id, handle, session_id=session_id)
                except Exception as e:
                    logger.warning("Failed to stop sub-agent %s: %s", handle, e)
        finally:
            self._stopping.discard(sk)
            self._lifecycle_locks.pop(sk, None)

    def list_active(self, project_id: str, *,
                    session_id: str | None = None) -> list[dict]:
        """Return [{'handle', 'display_name', 'status'}, ...] for a session.

        Lazily evicts dead adapters: an adapter is otherwise removed only by
        ``stop()``, so a sub-agent process that exits on its own would leave a
        stale entry forever (REPORT-is-idle-and-adapter-lifecycle.md Q6).
        ``is_alive()==False`` means the process is gone, so the entry is popped
        as we scan; an emptied SessionKey bucket is dropped too.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        adapters = self._adapters.get(sk, {})
        result = []
        dead: list[str] = []
        for handle, adapter in adapters.items():
            if adapter.is_alive():
                result.append({
                    "handle": handle,
                    "display_name": getattr(adapter, "display_name", handle),
                    "status": self._adapter_status(
                        adapter, project_id, session_id, handle),
                })
            else:
                dead.append(handle)
        for handle in dead:
            adapters.pop(handle, None)
            logger.debug("cleaned stale adapter %s for %s", handle, sk)
        if not adapters:
            self._adapters.pop(sk, None)
        return result

    # User-facing honesty: what the stop button can and cannot reach
    # (Piece 3 Part D; accepted limitation per
    # REPORT-piece3-child-classification.md — raw `&`-detached work
    # reparents away from the tree and equally escapes the reap kill).
    RAW_DETACH_WARNING = (
        "Stopping cancels the agent's current turn and terminates its "
        "tracked background work. Work the agent detached via a raw shell "
        "'&'/'nohup' runs outside the agent's process tree and cannot be "
        "guaranteed stopped."
    )

    async def stop_for_user(self, project_id: str, handle: str, *,
                            session_id: str | None = None) -> dict:
        """User stop button (Piece 3 Part D): cancel the open turn, terminate
        the agent's TRACKED background work (confirmed), stop the adapter
        (tree kill), and report honestly what was terminated and what the
        mechanism cannot reach."""
        session_id = self._resolve_session_id(session_id)
        from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
        registry = getattr(self._process_manager, "background_work", None)
        terminated: list[str] = []
        if isinstance(registry, BackgroundWorkRegistry):
            try:
                terminated = await registry.terminate_all(
                    project_id, session_id, handle,
                )
            except Exception:
                logger.exception(
                    "stop_for_user: background-work termination raised for "
                    "%s/%s/%s", project_id, session_id, handle,
                )
        # stop() cancels the in-flight send (the open turn) and tears the
        # process tree down; Part E hardens it to confirmed-death.
        stop_result = await self.stop(project_id, handle, session_id=session_id)
        if self._lifecycle_observer is not None:
            try:
                await self._lifecycle_observer.on_user_stopped(
                    project_id, handle,
                    terminated=terminated, session_id=session_id,
                )
            except Exception:
                logger.exception("on_user_stopped raised for %s/%s",
                                 project_id, handle)
        return {
            "handle": handle,
            "status": "stopped",
            "stop_result": stop_result,
            "background_terminated": terminated,
            "warning": self.RAW_DETACH_WARNING,
        }

    def seconds_since_background_activity(self, project_id: str, *,
                                          session_id: str | None = None) -> float | None:
        """Seconds since the most recent registered background-work activity
        across this session's sub-agents, or None if none was ever observed.
        The eviction timer resets on this (Piece 3 Part B)."""
        import time as _time
        from agent_os.daemon_v2.background_work import BackgroundWorkRegistry
        registry = getattr(self._process_manager, "background_work", None)
        if not isinstance(registry, BackgroundWorkRegistry):
            return None
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        deltas = []
        for handle in self._adapters.get(sk, {}):
            t = registry.last_background_activity(project_id, session_id, handle)
            if t is not None:
                deltas.append(_time.monotonic() - t)
        return min(deltas) if deltas else None

    def has_live_descendants_degraded(self, project_id: str, *,
                                      session_id: str | None = None) -> bool:
        """Conservative eviction gate for degraded-path transports (Piece 3).

        For adapters whose transport does NOT support background-status
        classification (Pipe/ACP/PTY — their event streams discard
        tool_input), we cannot distinguish real work from infrastructure, so
        ANY live descendant blocks eviction. May over-keep agents holding
        warm infra — accepted (costs only RAM). SDK adapters are skipped
        here; the provenance gate covers them.
        """
        import psutil as _psutil
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        for handle, adapter in self._adapters.get(sk, {}).items():
            transport = getattr(adapter, "_transport", None)
            if transport is not None and getattr(
                    transport, "supports_background_status", False):
                continue  # SDK — provenance-gated, not descendant-gated
            pid = None
            proc = getattr(transport, "_proc", None)  # psutil handle
            if proc is not None:
                pid = getattr(proc, "pid", None)
            if pid is None:
                raw = (getattr(transport, "_process", None)
                       or getattr(adapter, "_process", None)
                       or getattr(adapter, "_proc_handle", None))
                pid = getattr(raw, "pid", None)
            if not isinstance(pid, int):
                continue  # no reachable process — nothing to gate on
            try:
                if _psutil.Process(pid).children(recursive=True):
                    return True
            except _psutil.Error:
                continue
        return False

    def get_transcript(self, project_id: str, handle: str,
                       session_id: str | None = None):
        """Return the transcript for a sub-agent, or None.

        Session-scoped key (Part E). A session-less caller (e.g. the
        @mention REST path) gets the most recently created transcript for
        (project, handle) across sessions.
        """
        if session_id is not None:
            return self._transcripts.get((project_id, session_id, handle))
        for (pid, _sid, h), transcript in reversed(list(self._transcripts.items())):
            if pid == project_id and h == handle:
                return transcript
        return None

    def fanout_session_uuid_for_handle(self, project_id: str, handle: str,
                                       session_id: str | None = None) -> str | None:
        """Return the live worker session_uuid for a fanout task ``handle``,
        or None.

        Read-only accessor onto the (otherwise-private) ``_fanout_registry``:
        the transcript endpoint (agents_v2.py) needs the worker's real chat
        session id to resolve its ``/chat`` transcript for drill-in, but must
        not reach into registry internals directly.

        This ONLY works mid-batch: ``FanoutRegistry.resolve_group`` pops a
        group's routing entries once it resolves (fanout.py), so a completed
        fanout's handle is unknown to the registry moments after the batch
        finishes. Callers MUST also have a disk fallback (the transcript
        endpoint's session-file-stem scan) — this accessor is not sufficient
        alone.
        """
        if self._fanout_registry is None:
            return None
        group = self._fanout_registry.owner_group(project_id, handle, session_id)
        if group is None:
            return None
        task = group.tasks.get(handle)
        if task is None:
            return None
        return task.session_uuid

    def get_all_transcript_entries(self, project_id: str) -> list[dict]:
        """Read all sub-agent transcript entries for a project.

        Uses disk scan as primary method (survives daemon restarts),
        with in-memory transcript paths as supplementary source.
        """
        import glob as globmod
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        seen_paths: set[str] = set()
        entries: list[dict] = []

        # 1. Disk scan: find all transcript JSONL files in workspace
        workspace = ""
        if self._project_store is not None:
            project = self._project_store.get_project(project_id)
            workspace = (project.get("workspace", "") if project else "")

        if workspace:
            from agent_os.agent.project_paths import ProjectPaths
            base = ProjectPaths(workspace).sub_agents_dir
            if os.path.isdir(base):
                for jsonl_path in globmod.glob(os.path.join(base, "*", "*.jsonl")):
                    norm = os.path.normpath(jsonl_path)
                    seen_paths.add(norm)
                    try:
                        entries.extend(SubAgentTranscript.read(norm))
                    except (OSError, json.JSONDecodeError):
                        pass

        # 2. In-memory transcripts (covers cases where workspace lookup fails)
        for (pid, _sid, handle), transcript in self._transcripts.items():
            if pid == project_id:
                norm = os.path.normpath(transcript.filepath)
                if norm not in seen_paths:
                    try:
                        entries.extend(SubAgentTranscript.read(norm))
                    except (OSError, json.JSONDecodeError):
                        pass

        return entries

    def read_transcript_entries(self, project_id: str, handle: str,
                                session_id: str | None = None) -> "list[dict] | None":
        """Read one sub-agent's transcript entries for the drill-in endpoint.

        Returns ``None`` when no transcript exists for (project, handle) —
        the caller's 404 signal. Returns ``[]`` (not None) for a transcript
        file that exists but has produced zero chunks yet (started, nothing
        streamed) — that is a legitimate empty transcript, not a 404.

        Works for BOTH live and non-live handles:
        - live / recently-reset: prefers the in-memory ``Transcript`` for
          this exact (project_id, session_id, handle) via ``get_transcript``
          — accurate even right after a §4d ``fresh=True`` reset, where the
          on-disk ``.latest`` pointer and an in-memory handle from a
          different session could otherwise disagree.
        - non-live (daemon restarted, sub-agent already stopped, or no
          in-memory entry at all): falls back to the on-disk ``.latest``
          pointer for (workspace, handle) — the same continuity convention
          ``SubAgentTranscript.open_for_handle`` uses to resume a handle's
          transcript across respawns. Storage is keyed by (workspace,
          handle) only, not by session (see BACKLOG 005 §4b) — this is why
          the disk fallback needs no session_id.
        """
        from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript

        filepath: str | None = None
        transcript = self.get_transcript(project_id, handle, session_id)
        if transcript is not None:
            filepath = transcript.filepath
        else:
            workspace = ""
            if self._project_store is not None:
                project = self._project_store.get_project(project_id)
                workspace = (project.get("workspace", "") if project else "")
            if workspace:
                from agent_os.agent.project_paths import ProjectPaths
                sub_dir = ProjectPaths(workspace).sub_agent_dir(handle)
                latest_id = SubAgentTranscript._read_latest_id(sub_dir)
                if latest_id is not None:
                    filepath = os.path.join(sub_dir, f"{latest_id}.jsonl")

        if filepath is None or not os.path.isfile(filepath):
            return None
        try:
            return SubAgentTranscript.read(filepath)
        except (OSError, json.JSONDecodeError):
            return None
