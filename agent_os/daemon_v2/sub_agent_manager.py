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
from agent_os.platform.types import NetworkRules, DEFAULT_ALLOWLIST_DOMAINS

logger = logging.getLogger(__name__)


MAX_CONCURRENT_SUBAGENTS = 5  # Max active sub-agents per project


class SubAgentManager:
    """Owns all sub-agent adapters. Provides interface for AgentMessageTool."""

    def __init__(self, process_manager, adapter_configs: dict | None = None,
                 platform_provider=None, registry=None, setup_engine=None,
                 project_store=None, lifecycle_observer=None):
        self._process_manager = process_manager
        self._adapter_configs = adapter_configs or {}  # handle -> AdapterConfig (legacy)
        self._platform_provider = platform_provider
        self._registry = registry
        self._setup_engine = setup_engine
        self._project_store = project_store
        self._lifecycle_observer = lifecycle_observer
        self._adapters: dict[str, dict[str, object]] = {}  # project_id -> {handle -> adapter}
        self._transcripts: dict[tuple[str, str], object] = {}  # (project_id, handle) -> SubAgentTranscript
        self._lifecycle_locks: dict[str, asyncio.Lock] = {}  # project_id -> lock
        self._stopping: set[str] = set()  # project_ids currently in stop_all

    def _get_lock(self, project_id: str) -> asyncio.Lock:
        """Get or create the per-project lifecycle lock."""
        lock = self._lifecycle_locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            self._lifecycle_locks[project_id] = lock
        return lock

    async def start(self, project_id: str, handle: str, depth: int = 0) -> str:
        """Create adapter from config, call adapter.start(), register with process_manager."""
        if project_id in self._stopping:
            return "Error: project is shutting down, cannot start new agents"

        # Breadth check: limit concurrent sub-agents per project
        current_count = len(self._adapters.get(project_id, {}))
        if current_count >= MAX_CONCURRENT_SUBAGENTS:
            return (
                f"Error: concurrent sub-agent limit reached "
                f"(max {MAX_CONCURRENT_SUBAGENTS} per project). "
                f"Stop an existing sub-agent before starting a new one."
            )

        # New path: use registry + setup_engine if available
        if self._registry is not None and self._setup_engine is not None:
            return await self._start_from_registry(project_id, handle)

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
        lock = self._get_lock(project_id)
        async with lock:
            try:
                await adapter.start(config)
            except Exception as e:
                try:
                    await adapter.stop()
                except Exception:
                    pass
                return f"Error: adapter start failed: {e}"
            if project_id not in self._adapters:
                self._adapters[project_id] = {}
            self._adapters[project_id][handle] = adapter

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

        await self._process_manager.start(project_id, handle, adapter, transcript=transcript)

        if self._lifecycle_observer:
            tp = transcript.filepath if transcript else "unknown"
            await self._lifecycle_observer.on_started(project_id, handle, initiator="management_agent", transcript_path=tp)

        return f"Started {handle}"

    def _resolve_transport(self, manifest, config_dict):
        """Resolve the appropriate transport for a manifest."""
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
                    return SDKTransport()
            except ImportError:
                pass
            # Fallback to pipe if SDK not available
            from agent_os.agent.transports.pipe_transport import PipeTransport
            return PipeTransport(config=self._get_pipe_config(manifest.slug))
        elif transport_type == "pipe":
            from agent_os.agent.transports.pipe_transport import PipeTransport
            return PipeTransport(config=self._get_pipe_config(manifest.slug))
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

    async def _start_from_registry(self, project_id: str, handle: str, depth: int = 0) -> str:
        """Start a sub-agent using the manifest registry and setup engine."""
        if project_id in self._stopping:
            return "Error: project is shutting down, cannot start new agents"

        # Breadth check: limit concurrent sub-agents per project
        current_count = len(self._adapters.get(project_id, {}))
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

        # Resolve transport from manifest
        transport = self._resolve_transport(manifest, config_dict)

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

        lock = self._get_lock(project_id)
        async with lock:
            try:
                await adapter.start(config)
            except Exception as e:
                try:
                    await adapter.stop()
                except Exception:
                    pass
                return f"Error: adapter start failed: {e}"
            if project_id not in self._adapters:
                self._adapters[project_id] = {}
            self._adapters[project_id][handle] = adapter

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
            await self._process_manager.start(project_id, handle, adapter, transcript=transcript)

        if self._lifecycle_observer:
            tp = transcript.filepath if transcript else "unknown"
            await self._lifecycle_observer.on_started(project_id, handle, initiator="management_agent", transcript_path=tp)

        return f"Started {manifest.name}"

    async def send(self, project_id: str, handle: str, message: str) -> str:
        """Dispatch message to adapter without blocking on response.

        Returns immediately with a transcript path acknowledgement.
        The response will appear asynchronously in the transcript and
        via WebSocket broadcast.
        """
        adapters = self._adapters.get(project_id, {})
        adapter = adapters.get(handle)
        if adapter is None:
            return f"Error: agent '{handle}' not running for project '{project_id}'"

        transcript = self._transcripts.get((project_id, handle))
        transcript_path = transcript.filepath if transcript else "unknown"

        await self._dispatch_async(adapter, project_id, handle, message)

        if self._lifecycle_observer:
            await self._lifecycle_observer.on_message_routed(
                project_id, handle,
                initiator="management_agent",
                message_preview=message[:100],
                transcript_path=transcript_path,
            )

        return f"Message sent to {handle}. Transcript: {transcript_path}"

    async def _dispatch_async(self, adapter, project_id: str, handle: str, message: str) -> None:
        """Dispatch message to adapter without blocking on response.

        For transports that support non-blocking dispatch (SDK with queue),
        writes to the adapter and returns. For blocking transports (Pipe, ACP)
        and legacy PTY paths, wraps the send in a background task.
        """
        transport = getattr(adapter, '_transport', None)

        if transport is not None and hasattr(transport, 'dispatch'):
            await transport.dispatch(message)
            return

        # Fallback: wrap send() in background task (covers PTY, Pipe, ACP)
        async def _background_send():
            try:
                await adapter.send(message)
            except Exception as e:
                logger.warning("Background send to %s failed: %s", handle, e)

        asyncio.create_task(_background_send())

    async def stop(self, project_id: str, handle: str) -> str:
        """Stop adapter, deregister from process_manager."""
        lock = self._get_lock(project_id)
        async with lock:
            adapters = self._adapters.get(project_id, {})
            adapter = adapters.pop(handle, None)
        if adapter is None:
            return f"Agent '{handle}' not running"
        await adapter.stop()
        await self._process_manager.stop(project_id, handle)

        return f"Stopped {handle}"

    def status(self, project_id: str, handle: str) -> str:
        """Return 'running' | 'idle' | 'stopped' | 'unknown'."""
        adapters = self._adapters.get(project_id, {})
        adapter = adapters.get(handle)
        if adapter is None:
            return "unknown"
        if not adapter.is_alive():
            return "stopped"
        if adapter.is_idle():
            return "idle"
        return "running"

    async def resolve_sub_agent_approval(self, project_id: str, tool_call_id: str, approved: bool) -> bool:
        """Try to resolve a permission request on any sub-agent transport.

        Returns True if the approval was routed to a sub-agent, False if not found.
        """
        adapters = self._adapters.get(project_id, {})
        for handle, adapter in adapters.items():
            transport = getattr(adapter, '_transport', None)
            if transport is not None and hasattr(transport, 'respond_to_permission'):
                # Check if this transport has the pending approval
                pending = getattr(transport, '_pending_approvals', {})
                if tool_call_id in pending:
                    await transport.respond_to_permission(tool_call_id, approved)
                    return True
        return False

    async def stop_all(self, project_id: str) -> None:
        """Stop all sub-agent adapters for a project."""
        self._stopping.add(project_id)
        try:
            lock = self._get_lock(project_id)
            async with lock:
                adapters = self._adapters.get(project_id, {})
                handles = list(adapters.keys())
            for handle in handles:
                try:
                    await self.stop(project_id, handle)
                except Exception as e:
                    logger.warning("Failed to stop sub-agent %s: %s", handle, e)
        finally:
            self._stopping.discard(project_id)
            self._lifecycle_locks.pop(project_id, None)

    def list_active(self, project_id: str) -> list[dict]:
        """Return [{'handle', 'display_name', 'status'}, ...]"""
        adapters = self._adapters.get(project_id, {})
        result = []
        for handle, adapter in adapters.items():
            if adapter.is_alive():
                result.append({
                    "handle": handle,
                    "display_name": getattr(adapter, "display_name", handle),
                    "status": "running" if not adapter.is_idle() else "idle",
                })
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
            base = os.path.join(workspace, ".agent-os", "sub_agents")
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
