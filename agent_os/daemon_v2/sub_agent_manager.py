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

from agent_os.agent.adapters.cli_adapter import CLIAdapter
from agent_os.agent.prompt_builder import Autonomy
from agent_os.daemon_v2.models import (
    DEFAULT_SESSION_ID,
    SessionKey,
    make_session_key,
)
from agent_os.platform.types import NetworkRules, DEFAULT_ALLOWLIST_DOMAINS

logger = logging.getLogger(__name__)


MAX_CONCURRENT_SUBAGENTS = 5  # Max active sub-agents per project


class SubAgentManager:
    """Owns all sub-agent adapters. Provides interface for AgentMessageTool."""

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
        # ``_adapters`` is keyed by ``SessionKey == (project_id, session_id)``
        # so each chat session within a project owns an independent slate of
        # active sub-agents. Single-loop callers route through
        # ``DEFAULT_SESSION_ID`` via ``_resolve_session_id``.
        self._adapters: dict[SessionKey, dict[str, object]] = {}
        self._transcripts: dict[tuple[str, str], object] = {}  # (project_id, handle) -> SubAgentTranscript
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

    @staticmethod
    def _resolve_session_id(session_id: str | None) -> str:
        """Normalize optional ``session_id`` to a non-empty string.

        Mirror of ``AgentManager._resolve_session_id`` — both managers
        share the same back-compat sentinel so single-loop callers do
        not need to know about ``DEFAULT_SESSION_ID``.
        """
        return session_id or DEFAULT_SESSION_ID

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
                    *, session_id: str | None = None) -> str:
        """Create adapter from config, call adapter.start(), register with process_manager.

        ``session_id`` selects which chat session this sub-agent attaches to.
        Defaults to ``DEFAULT_SESSION_ID`` for single-loop back-compat.
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
                project_id, handle, session_id=session_id,
            )

        # Legacy path: use adapter_configs
        config = self._adapter_configs.get(handle)
        if config is None:
            return f"Error: no adapter config for handle '{handle}'"

        # Configure network isolation for this project
        if self._platform_provider is not None:
            try:
                rules = NetworkRules(
                    mode="allowlist",
                    domains=list(DEFAULT_ALLOWLIST_DOMAINS),
                )
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
                from uuid import uuid4
                from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
                transcript = SubAgentTranscript(workspace, handle, str(uuid4())[:8])
                self._transcripts[(project_id, handle)] = transcript

        await self._process_manager.start(project_id, handle, adapter, transcript=transcript, session_id=session_id)

        if self._lifecycle_observer:
            tp = transcript.filepath if transcript else "unknown"
            await self._lifecycle_observer.on_started(project_id, handle, initiator="management_agent", transcript_path=tp, session_id=session_id)

        return f"Started {handle}"

    def _resolve_transport(self, manifest, config_dict, autonomy=None, system_prompt: str | None = None):
        """Resolve the appropriate transport for a manifest.

        system_prompt, when provided, is forwarded to transports that
        support it (SDK, Pipe). PTYTransport does not currently support
        --append-system-prompt-file injection; the caller is responsible
        for the degraded first-turn injection path.

        Raises:
            ValueError: if a claude-code manifest requests ``transport: acp``.
                claude-code has no ACP server mode (the binary rejects
                ``claude acp`` with "unknown command"); ACP is for gemini-cli
                and other ACP-compliant agents. See
                ``docs/investigations/FINDINGS-sub-agent-context-and-persistence.md``
                Q4. We refuse silently redirecting to SDK because the user
                wrote ``acp`` deliberately and a silent swap masks the
                misconfiguration.
        """
        transport_hint = getattr(manifest.runtime, 'transport', 'auto')
        mode = manifest.runtime.mode
        command = getattr(manifest.runtime, 'command', None) or ""

        # Guard: ACP transport is not supported by claude-code.
        # Treat as a user manifest error, not a silent redirect.
        if transport_hint == "acp" and command.startswith("claude"):
            manifest_path = f"agent_os/agents/manifests/{manifest.slug.replace('-', '_')}.yaml"
            msg = (
                "ACP transport is not supported by claude-code. "
                "Use 'auto' or 'sdk' transport for claude. "
                "ACP is for gemini-cli and other ACP-compliant agents. "
                f"Edit manifest: {manifest_path}"
            )
            logger.warning(msg)
            raise ValueError(msg)

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
                    return SDKTransport(autonomy=autonomy, system_prompt=system_prompt)
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
        elif transport_type == "acp":
            from agent_os.agent.transports.acp_transport import ACPTransport
            return ACPTransport()
        else:
            # Fallback: no transport, use legacy CLIAdapter path
            return None

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
                                   *, session_id: str | None = None) -> str:
        """Start a sub-agent using the manifest registry and setup engine."""
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

        # Configure network isolation
        if self._platform_provider is not None:
            try:
                domains = config_dict.get("network_domains", []) + list(DEFAULT_ALLOWLIST_DOMAINS)
                rules = NetworkRules(mode="allowlist", domains=domains)
                self._platform_provider.configure_network(project_id, rules)
            except RuntimeError as e:
                return f"Error: network configuration failed: {e}"

        # Resolve autonomy preset for SDK transport filtering
        autonomy = None
        if project:
            autonomy_str = project.get("autonomy", "check_in")
            try:
                autonomy = Autonomy(autonomy_str)
            except ValueError:
                autonomy = Autonomy.CHECK_IN

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

                # Determine peer slugs (other enabled sub-agents in the project)
                enabled_sub_agents = project.get("enabled_sub_agents", None) if project else None
                if not enabled_sub_agents and self._setup_engine is not None:
                    try:
                        available = self._setup_engine.check_all()
                        enabled_sub_agents = [
                            a.slug for a in available
                            if getattr(a, "installed", False) and a.slug != "built-in"
                        ]
                    except Exception:
                        enabled_sub_agents = [handle]
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
                    transport_hint in ("pty", "acp")
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

        # Resolve transport from manifest. May raise ValueError for invalid
        # manifest combinations (e.g. claude-code with transport: acp).
        try:
            transport = self._resolve_transport(
                manifest, config_dict, autonomy=autonomy, system_prompt=system_prompt,
            )
        except ValueError as e:
            return f"Error: unsupported transport in manifest: {e}"

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

        # Create transcript for this sub-agent
        transcript = None
        if workspace:
            from uuid import uuid4
            from agent_os.daemon_v2.sub_agent_transcript import SubAgentTranscript
            transcript = SubAgentTranscript(workspace, handle, str(uuid4())[:8])
            self._transcripts[(project_id, handle)] = transcript

        # ACP and Pipe handle responses via send() return value — no streaming consumer needed
        # PTY and legacy paths need process_manager to consume read_stream()
        from agent_os.agent.transports.acp_transport import ACPTransport
        from agent_os.agent.transports.pipe_transport import PipeTransport
        if not isinstance(transport, (ACPTransport, PipeTransport)):
            await self._process_manager.start(project_id, handle, adapter, transcript=transcript, session_id=session_id)

        if self._lifecycle_observer:
            tp = transcript.filepath if transcript else "unknown"
            await self._lifecycle_observer.on_started(project_id, handle, initiator="management_agent", transcript_path=tp, session_id=session_id)

        return f"Started {manifest.name}"

    async def send(self, project_id: str, handle: str, message: str,
                   *, session_id: str | None = None) -> str:
        """Dispatch message to adapter without blocking on response.

        Returns immediately with a transcript path acknowledgement.
        The response will appear asynchronously in the transcript and
        via WebSocket broadcast.

        ``session_id`` selects which session's adapter slate to dispatch
        through; defaults to ``DEFAULT_SESSION_ID``.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        adapters = self._adapters.get(sk, {})
        adapter = adapters.get(handle)
        if adapter is None:
            return f"Error: agent '{handle}' not running for project '{project_id}'"

        transcript = self._transcripts.get((project_id, handle))
        transcript_path = transcript.filepath if transcript else "unknown"

        await self._dispatch_async(adapter, project_id, handle, message, session_id=session_id)

        if self._lifecycle_observer:
            await self._lifecycle_observer.on_message_routed(
                project_id, handle,
                initiator="management_agent",
                message_preview=message[:100],
                transcript_path=transcript_path,
                session_id=session_id,
            )

        return f"Message sent to {handle}. Transcript: {transcript_path}"

    async def _dispatch_async(self, adapter, project_id: str, handle: str, message: str,
                              *, session_id: str | None = None) -> None:
        """Dispatch message to adapter without blocking on response.

        For transports that support non-blocking dispatch (SDK with queue),
        writes to the adapter and returns. For blocking transports (Pipe, ACP)
        and legacy PTY paths, wraps the send in a background task.
        """
        transport = getattr(adapter, '_transport', None)

        if transport is not None and hasattr(transport, 'dispatch'):
            adapter._idle = False  # Reset idle on new task
            await transport.dispatch(message)
            return

        # Fallback: wrap send() in background task (covers PTY, Pipe, ACP)
        transcript = self._transcripts.get((project_id, handle))

        async def _background_send():
            from datetime import datetime, timezone
            try:
                await adapter.send(message)
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
                        await self._lifecycle_observer.on_completed(
                            project_id, handle,
                            summary=response[:200] if response else "(no output)",
                            transcript_path=transcript.filepath,
                            session_id=session_id,
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(
                    "_background_send failed for project=%s handle=%s",
                    project_id, handle, exc_info=True,
                )
                adapter._broken = True
                if self._lifecycle_observer is not None:
                    try:
                        self._lifecycle_observer.on_failed(
                            project_id, handle,
                            reason="background_send_exception",
                            session_id=session_id,
                        )
                    except Exception:
                        logger.exception(
                            "lifecycle_observer.on_failed raised for project=%s handle=%s",
                            project_id, handle,
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
        """Stop adapter, deregister from process_manager.

        ``session_id`` selects which session's adapter slate to stop in;
        defaults to ``DEFAULT_SESSION_ID``.
        """
        session_id = self._resolve_session_id(session_id)
        sk = make_session_key(project_id, session_id)
        lock = self._get_lock(project_id, session_id=session_id)
        async with lock:
            adapters = self._adapters.get(sk, {})
            adapter = adapters.pop(handle, None)
        if adapter is None:
            return f"Agent '{handle}' not running"
        await adapter.stop()
        await self._process_manager.stop(project_id, handle)

        return f"Stopped {handle}"

    def status(self, project_id: str, handle: str, *,
               session_id: str | None = None) -> str:
        """Return 'running' | 'idle' | 'stopped' | 'unknown'."""
        session_id = self._resolve_session_id(session_id)
        adapters = self._adapters.get(make_session_key(project_id, session_id), {})
        adapter = adapters.get(handle)
        if adapter is None:
            return "unknown"
        if not adapter.is_alive():
            return "stopped"
        if adapter.is_idle():
            return "idle"
        return "running"

    def get_pending_sub_agent_approval(self, project_id: str, *,
                                       session_id: str | None = None) -> dict | None:
        """Return the first pending sub-agent approval for a session, or None.

        Used by the REST recovery endpoint so clients can fetch sub-agent
        approval card data when they miss the WebSocket event.
        """
        session_id = self._resolve_session_id(session_id)
        adapters = self._adapters.get(make_session_key(project_id, session_id), {})
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
                return {
                    "tool_call_id": data.get("request_id", request_id),
                    "tool_name": data.get("tool_name", ""),
                    "tool_args": data.get("tool_input", {}),
                    "what": f"Sub-agent {handle} requests approval: {data.get('tool_name', 'unknown')}",
                    "source": handle,
                }
        return None

    async def resolve_sub_agent_approval(self, project_id: str, tool_call_id: str,
                                         approved: bool, *,
                                         session_id: str | None = None) -> bool:
        """Try to resolve a permission request on any sub-agent transport.

        Returns True if the approval was routed to a sub-agent, False if not found.
        """
        session_id = self._resolve_session_id(session_id)
        adapters = self._adapters.get(make_session_key(project_id, session_id), {})
        for handle, adapter in adapters.items():
            transport = getattr(adapter, '_transport', None)
            if transport is not None and hasattr(transport, 'respond_to_permission'):
                # Check if this transport has the pending approval
                pending = getattr(transport, '_pending_approvals', {})
                if tool_call_id in pending:
                    await transport.respond_to_permission(tool_call_id, approved)
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
                    "status": "running" if not adapter.is_idle() else "idle",
                })
            else:
                dead.append(handle)
        for handle in dead:
            adapters.pop(handle, None)
            logger.debug("cleaned stale adapter %s for %s", handle, sk)
        if not adapters:
            self._adapters.pop(sk, None)
        return result

    def get_transcript(self, project_id: str, handle: str):
        """Return the transcript for a sub-agent, or None."""
        return self._transcripts.get((project_id, handle))

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
        for (pid, handle), transcript in self._transcripts.items():
            if pid == project_id:
                norm = os.path.normpath(transcript.filepath)
                if norm not in seen_paths:
                    try:
                        entries.extend(SubAgentTranscript.read(norm))
                    except (OSError, json.JSONDecodeError):
                        pass

        return entries
