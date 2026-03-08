# Orbital — An operating system for AI agents
# Copyright (C) 2026 Orbital Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""WindowsPlatformProvider — assembles all isolation components into a single provider."""

import logging
from typing import Literal

from agent_os.platform.base import PlatformProvider
from agent_os.platform.shared.network import NetworkProxy
from agent_os.platform.types import (
    CommandResult,
    DEFAULT_ALLOWLIST_DOMAINS,
    BlockedCallback,
    FolderInfo,
    NetworkRules,
    PermissionResult,
    PlatformCapabilities,
    ProcessHandle,
    SetupResult,
)
from agent_os.platform.windows.credentials import CredentialStore
from agent_os.platform.windows.permissions import PermissionManager
from agent_os.platform.windows.process import ProcessLauncher
from agent_os.platform.windows.sandbox import SandboxAccountManager
from agent_os.platform.windows.setup import SetupOrchestrator

logger = logging.getLogger("agent_os.platform.windows.provider")


class WindowsPlatformProvider(PlatformProvider):
    """Windows platform provider using sandbox user isolation."""

    def __init__(self, on_network_blocked: BlockedCallback | None = None) -> None:
        self._credential_store = CredentialStore()
        self._account_manager = SandboxAccountManager(self._credential_store)
        self._permission_manager = PermissionManager()
        self._process_launcher = ProcessLauncher(self._credential_store)
        self._setup_orchestrator = SetupOrchestrator(
            self._account_manager, self._permission_manager
        )
        self._on_network_blocked = on_network_blocked

        # Per-project proxy instances: {project_id: NetworkProxy}
        self._proxies: dict[str, NetworkProxy] = {}

        # Per-project process handles: {project_id: ProcessHandle}
        self._processes: dict[str, ProcessHandle] = {}

        # Pending network rules for projects whose proxy hasn't started yet
        self._pending_rules: dict[str, NetworkRules] = {}

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    async def setup(self) -> SetupResult:
        try:
            return self._setup_orchestrator.run_setup()
        except Exception as exc:
            logger.error("setup() failed: %s", exc)
            return SetupResult(success=False, error=str(exc))

    async def teardown(self) -> SetupResult:
        try:
            # Stop all proxies
            for project_id, proxy in list(self._proxies.items()):
                try:
                    await proxy.stop()
                except Exception:
                    logger.warning("Failed to stop proxy for project %s", project_id)
            self._proxies.clear()

            # Terminate all running processes
            for project_id, handle in list(self._processes.items()):
                try:
                    self._process_launcher.terminate(handle)
                except Exception:
                    logger.warning("Failed to terminate process for project %s", project_id)
            self._processes.clear()

            return self._setup_orchestrator.run_teardown()
        except Exception as exc:
            logger.error("teardown() failed: %s", exc)
            return SetupResult(success=False, error=str(exc))

    def is_setup_complete(self) -> bool:
        try:
            return self._setup_orchestrator.check_setup_status().is_complete
        except Exception as exc:
            logger.error("is_setup_complete() failed: %s", exc)
            return False

    @staticmethod
    def _check_conpty_available() -> bool:
        try:
            import ctypes
            return hasattr(ctypes.windll.kernel32, "CreatePseudoConsole")
        except Exception:
            return False

    def get_capabilities(self) -> PlatformCapabilities:
        try:
            status = self._setup_orchestrator.check_setup_status()
            issues = list(status.issues)
            if not self._check_conpty_available():
                issues.append("ConPTY not available — external agent PTY mode disabled")
            return PlatformCapabilities(
                platform="windows",
                isolation_method="sandbox_user",
                setup_complete=status.is_complete,
                setup_issues=issues,
                supports_network_restriction=True,
                supports_folder_access=True,
                sandbox_username=(
                    self._account_manager.get_username()
                    if status.sandbox_user_exists
                    else None
                ),
            )
        except Exception as exc:
            logger.error("get_capabilities() failed: %s", exc)
            return PlatformCapabilities(
                platform="windows",
                isolation_method="sandbox_user",
                setup_complete=False,
                setup_issues=[str(exc)],
                supports_network_restriction=True,
                supports_folder_access=True,
                sandbox_username=None,
            )

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    async def run_process(
        self,
        project_id: str,
        command: str,
        args: list[str],
        working_dir: str,
        extra_env: dict[str, str] | None = None,
        use_pty: bool = False,
    ) -> ProcessHandle:
        # 0. Terminate existing process for this project if still running
        existing = self._processes.get(project_id)
        if existing and self._process_launcher.is_running(existing):
            logger.warning("Terminating existing process for %s before new launch", project_id)
            self._process_launcher.terminate(existing, timeout_sec=5)

        # 1. Ensure proxy is running for this project
        proxy = await self._ensure_proxy(project_id)

        # 2. Build environment with proxy settings
        env: dict[str, str] = {}
        if proxy:
            env["HTTP_PROXY"] = proxy.proxy_url
            env["HTTPS_PROXY"] = proxy.proxy_url
            env["NO_PROXY"] = "localhost,127.0.0.1"
        if extra_env:
            env.update(extra_env)

        # 3. Launch process — PTY (current user) or sandbox user
        if use_pty:
            handle = self._process_launcher.launch_with_pty(
                command=command,
                args=args,
                working_dir=working_dir,
                env_vars=env,
                inherit_env=True,
            )
        else:
            handle = self._process_launcher.launch(
                command=command,
                args=args,
                working_dir=working_dir,
                env_vars=env,
                inherit_env=True,
            )

        # 4. Track process
        self._processes[project_id] = handle
        return handle

    async def run_command(
        self,
        project_id: str,
        command: str,
        args: list[str],
        working_dir: str,
        timeout_sec: int = 300,
        extra_env: dict[str, str] | None = None,
    ) -> CommandResult:
        proxy = await self._ensure_proxy(project_id)
        env: dict[str, str] = {}
        if proxy:
            env["HTTP_PROXY"] = proxy.proxy_url
            env["HTTPS_PROXY"] = proxy.proxy_url
            env["NO_PROXY"] = "localhost,127.0.0.1"
        if extra_env:
            env.update(extra_env)
        return self._process_launcher.run_and_capture(
            command, args, working_dir, env_vars=env, timeout_sec=timeout_sec
        )

    async def stop_process(self, project_id: str, timeout_sec: int = 10) -> bool:
        try:
            handle = self._processes.get(project_id)
            if handle is None or not self._process_launcher.is_running(handle):
                return False

            # Close ConPTY handle before termination if present
            hpc = handle._native_handles.get("hpc")
            if hpc is not None:
                try:
                    from agent_os.platform.windows.process import _ClosePseudoConsole
                    _ClosePseudoConsole(hpc)
                except Exception:
                    logger.warning("Failed to close ConPTY handle for project %s", project_id)

            # Close I/O streams
            for stream in (handle.stdin, handle.stdout, handle.stderr):
                if stream and not getattr(stream, "closed", True):
                    try:
                        stream.close()
                    except Exception:
                        pass

            self._process_launcher.terminate(handle, timeout_sec)
            self._processes.pop(project_id, None)

            # Stop the project's proxy
            proxy = self._proxies.pop(project_id, None)
            if proxy is not None:
                try:
                    await proxy.stop()
                except Exception:
                    logger.warning("Failed to stop proxy for project %s", project_id)

            return True
        except Exception as exc:
            logger.error("stop_process(%s) failed: %s", project_id, exc)
            return False

    # ------------------------------------------------------------------
    # Folder access
    # ------------------------------------------------------------------

    def grant_folder_access(
        self, path: str, mode: Literal["read_only", "read_write"]
    ) -> PermissionResult:
        try:
            username = self._account_manager.get_username()
            return self._permission_manager.grant_access(username, path, mode)
        except Exception as exc:
            logger.error("grant_folder_access() failed: %s", exc)
            return PermissionResult(success=False, path=path, error=str(exc))

    def revoke_folder_access(self, path: str) -> PermissionResult:
        try:
            username = self._account_manager.get_username()
            return self._permission_manager.revoke_access(username, path)
        except Exception as exc:
            logger.error("revoke_folder_access() failed: %s", exc)
            return PermissionResult(success=False, path=path, error=str(exc))

    def get_available_folders(self) -> list[FolderInfo]:
        try:
            return self._permission_manager.get_available_folders()
        except Exception as exc:
            logger.error("get_available_folders() failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    def configure_network(self, project_id: str, rules: NetworkRules) -> None:
        if project_id in self._proxies:
            self._proxies[project_id].set_rules(rules)
        # Store rules for deferred application when proxy starts
        self._pending_rules[project_id] = rules

    # ------------------------------------------------------------------
    # Sleep prevention
    # ------------------------------------------------------------------

    def prevent_sleep(self, reason: str) -> object:
        """Prevent system sleep using SetThreadExecutionState."""
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        logger.info("Sleep prevention enabled: %s", reason)
        return {"reason": reason}

    def allow_sleep(self, handle: object) -> None:
        """Re-allow system sleep."""
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logger.info("Sleep prevention disabled")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_proxy(self, project_id: str) -> NetworkProxy:
        """Ensure a proxy is running for the given project, creating one if needed."""
        if project_id not in self._proxies:
            proxy = NetworkProxy(
                project_id=project_id,
                on_blocked=self._on_network_blocked,
            )

            # Apply pending rules if any, otherwise use default allowlist
            if project_id in self._pending_rules:
                proxy.set_rules(self._pending_rules[project_id])
            else:
                proxy.set_rules(NetworkRules(
                    mode="allowlist",
                    domains=list(DEFAULT_ALLOWLIST_DOMAINS),
                ))

            await proxy.start()
            self._proxies[project_id] = proxy

        return self._proxies[project_id]
